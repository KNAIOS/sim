"""Counterfactual reference returns on strictly separated external trace streams."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.baseline_anchor.control import action_at as old_action
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.high_confirmation import ROOT as PREVIOUS, intact as previous_intact
from joint_power.model import Environment
from joint_power.rollout.verify import independent
from joint_power.strong_headroom.study import advance, reference_command, state_labels, totals
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_risk_only')
STREAMS = {'train': (5409220000, 8), 'validation': (5509220000, 4)}
SLOTS = (0, 36, 72, 108)


def intact():
    previous_intact()
    hashes = json.loads((PREVIOUS/'manifest.json').read_text())['input_hashes']
    hashes.update(json.loads((PREVIOUS/'output_hashes.json').read_text()))
    assert all(digest(p) == h for p, h in hashes.items())


def one(task):
    split, scenario, index, behavior = task
    torch.set_num_threads(1)
    cfg = make_config('validation')
    seed = STREAMS[split][0]+10000*scenario+index
    trace = make_trace(cfg, seed, SCENARIOS[scenario])
    unchanged = (trace.channel.copy(), trace.gain_db.copy(), trace.requirements_mbps.copy())
    env, past, payload, errors, checked = Environment(cfg, trace), 0, [], defaultdict(float), 0
    predictor = load_predictor() if behavior == 'myopic' else None
    model_seed = (11 if index % 2 == 0 else 22) if behavior == 'myopic' else 0
    model = models()[0]['Visitation-added', model_seed] if model_seed else None
    while env.t < cfg.episode_steps:
        env.prepare()
        if env.t in SLOTS:
            slot = env.t
            before = (env.hw.active.copy(), env.hw.countdown.copy(), env.age.copy(),
                      env.tracker.channel_estimate.copy())
            item, histories = state_labels(env, False, past)
            assert env.t == slot and item['x'].shape == (17, 342)
            for a, b in zip(before, (env.hw.active, env.hw.countdown, env.age, env.tracker.channel_estimate)):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(item['y'][..., 0], item['y'][..., 2:].sum(-1), atol=1e-9)
            for action in {item['reference'], 0, 7, 15}:
                complete = env.records+histories[action]
                for name, value in independent(complete, trace).items():
                    errors[name] = max(errors[name], value)
                checked += len(complete)
            payload.append(dict(**item, slot=slot))
        if behavior == 'prs':
            command = reference_command(env, False, past)
        else:
            command, _ = old_action(env, predictor, model, 'Myopic', 'Energy', .1, past)
        past = advance(env, command, past)
    if behavior == 'prs':
        for item in payload:
            np.testing.assert_allclose(item['y'][item['reference'], -1],
                totals(env.records[item['slot']:], cfg.users), atol=1e-9)
    for name, value in independent(env.records, trace).items():
        errors[name] = max(errors[name], value)
    checked += len(env.records)
    for actual, expected in zip((trace.channel, trace.gain_db, trace.requirements_mbps), unchanged):
        np.testing.assert_array_equal(actual, expected)
    arrays = {k: np.array([p[k] for p in payload]) for k in payload[0]}
    arrays['trace_seed'] = np.full(len(payload), seed, np.int64)
    arrays['identity'] = np.array([[scenario, index, int(behavior == 'myopic'), p['slot']]
                                   for p in payload], np.int32)
    path = ROOT/f'{split}_{behavior}_{scenario}_{index}.npz'
    np.savez_compressed(path, **arrays)
    print(f'risk-only labels {split}/{behavior} scenario={scenario} trace={index}: '
          f'{len(payload)} roots verified', flush=True)
    return dict(split=split, behavior=behavior, scenario=scenario, trajectory=index,
        trace_seed=seed, behavior_model_seed=model_seed, roots=len(payload),
        path=str(path), **env.metrics()), dict(errors), checked


def main():
    torch.set_num_threads(1)
    intact()
    ROOT.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/risk_only/PLAN.md'),
             Path('joint_power/strong_headroom/study.py'), Path('joint_power/baseline_anchor/control.py'),
             PREVIOUS/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    _, checkpoints = models()
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, behavior_checkpoints=checkpoints,
        streams=STREAMS, root_slots=SLOTS, reference='Expanded PRS minimum1 fixed0.05',
        config=make_config('validation').as_dict(), host_environment_changed=False), indent=2)+'\n')
    tasks = [(split, s, i, b) for split, (_, n) in STREAMS.items()
             for s in range(3) for i in range(n) for b in ('prs', 'myopic')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        collected = list(pool.map(one, tasks))
    rows = [r for r, _, _ in collected]
    for split, (_, number) in STREAMS.items():
        payloads = []
        for row in rows:
            if row['split'] == split:
                with np.load(row['path']) as data:
                    payloads.append({k: data[k] for k in data.files})
        merged = {k: np.concatenate([p[k] for p in payloads]) for k in payloads[0]}
        assert len(merged['x']) == number*3*2*len(SLOTS)
        assert len(set(merged['trace_seed'])) == number*3
        np.savez_compressed(ROOT/f'{split}.npz', **merged)
    train_seeds = {r['trace_seed'] for r in rows if r['split'] == 'train'}
    valid_seeds = {r['trace_seed'] for r in rows if r['split'] == 'validation'}
    assert len(train_seeds) == 24 and len(valid_seeds) == 12 and train_seeds.isdisjoint(valid_seeds)
    errors = {k: max(e[k] for _, e, _ in collected) for k in collected[0][1]}
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    write_rows(ROOT/'trajectories.csv', rows)
    report = dict(unique_training_traces=24, unique_validation_traces=12, behavior_episodes=72,
        training_roots=192, validation_roots=96, candidate_entries=288*17,
        physical_slot_checks=sum(n for _, _, n in collected), maximum_errors=errors,
        reference_suffixes_verified=True, train_validation_disjoint=True,
        outcome_labels_not_deployed=True)
    (ROOT/'collection_report.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(r['path']) == r['sha256'] for r in checkpoints.values())
    (ROOT/'collection_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
