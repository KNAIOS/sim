"""Paired visitation states; all labels use the same causal PRS continuation."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.rollout.absolute import ROOT as PREVIOUS
from joint_power.rollout.analyze import load as old_load
from joint_power.rollout.collect import SLOTS, base_command, state_labels
from joint_power.rollout.control import input_checks, load_models
from joint_power.rollout.probe import select
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_aggregation')
STREAMS = {'train': (4109220000, 8), 'validation': (4209220000, 4)}
KEYS = ('x', 'y', 'reference', 'identity', 'masks', 'raw_action')


def previous_intact():
    input_checks()
    for name in ('validation', 'price_extension'):
        hashes = json.loads((PREVIOUS/name/'output_hashes.json').read_text())
        assert all(digest(p) == h for p, h in hashes.items()), name


def visitor_command(env, predictor, model, price):
    candidates = env.candidates()
    state = decision(env, candidates, predictor)
    baseline = base_command(env)
    reference = int(baseline.sum())-1
    np.testing.assert_array_equal(state.masks[reference], baseline)
    with torch.no_grad():
        prediction = model(torch.from_numpy(state.x[None]), torch.tensor([reference]),
                           torch.tensor([env.cfg.episode_steps-env.t], dtype=torch.float32)).numpy()
    action = int(select(prediction, np.array([reference]), price)[0])
    return state.masks[action]


def collect_one(task):
    split, scenario, index, behavior = task
    torch.set_num_threads(1)
    cfg, predictor = make_config('validation'), load_predictor()
    models, _ = load_models()
    visitor_seed, price = (11, 4.) if index % 2 == 0 else (22, 2.)
    trace_seed = STREAMS[split][0]+scenario*10000+index
    trace = make_trace(cfg, trace_seed, SCENARIOS[scenario])
    original_channel, original_demand = trace.channel.copy(), trace.requirements_mbps.copy()
    env, rows, counts = Environment(cfg, trace), [], []
    while env.t < cfg.episode_steps:
        env.prepare()
        if env.t in SLOTS:
            slot = env.t
            count = int(env.hw.active.sum())
            age, active, countdown = env.age.copy(), env.hw.active.copy(), env.hw.countdown.copy()
            labels = state_labels(env, predictor)
            assert env.t == slot
            np.testing.assert_array_equal(env.age, age)
            np.testing.assert_array_equal(env.hw.active, active)
            np.testing.assert_array_equal(env.hw.countdown, countdown)
            rows.append(dict(slot=slot, **labels))
            counts.append(count)
        command = base_command(env) if behavior == 'prs' else visitor_command(
            env, predictor, models['MC absolute', visitor_seed], price)
        env.block(command)
    if behavior == 'prs':
        for row in rows:
            suffix = env.records[row['slot']:]
            expected = [sum(r['energy_j'] for r in suffix), sum(round(3*r['outage']) for r in suffix)]
            np.testing.assert_allclose(row['y'][row['reference'], -1], expected, atol=1e-9)
    np.testing.assert_array_equal(trace.channel, original_channel)
    np.testing.assert_array_equal(trace.requirements_mbps, original_demand)
    payload = {k: np.array([r[k] for r in rows]) for k in KEYS if k != 'identity'}
    payload['identity'] = np.array([[scenario, index, r['slot']] for r in rows], dtype=np.int32)
    payload['active_before'] = np.array(counts, dtype=np.int32)
    payload['trajectory_seed'] = np.full(len(rows), trace_seed, dtype=np.int64)
    payload['visitor_seed'] = np.full(len(rows), 0 if behavior == 'prs' else visitor_seed, dtype=np.int32)
    np.savez_compressed(ROOT/f'{split}_{behavior}_{scenario}_{index}.npz', **payload)
    print(f'visitation {split} {behavior} scenario={scenario} trace={index} '
          f'below4={sum(n < 4 for n in counts)}/{len(counts)}', flush=True)
    return dict(split=split, behavior=behavior, scenario=scenario, trajectory=index,
        trace_seed=trace_seed, visitor_seed=0 if behavior == 'prs' else visitor_seed,
        states=len(rows), below4=sum(n < 4 for n in counts), minimum_active=min(counts),
        **env.metrics())


def concatenate(payloads, keys=KEYS):
    return {k: np.concatenate([p[k] for p in payloads]) for k in keys}


def load(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def main():
    previous_intact()
    ROOT.mkdir(exist_ok=False)
    files = [*Path('joint_power/aggregation').glob('*.py'), Path('joint_power/aggregation/PLAN.md'),
             *Path('joint_power/rollout').glob('*.py'), PREVIOUS/'price_extension/output_hashes.json',
             Path('results/joint_power_20260922_rollout/collection_hashes.json')]
    hashes = {str(p): digest(p) for p in files}
    _, checkpoints = load_models()
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, streams=STREAMS,
        checkpoints=checkpoints, slots=SLOTS, config=make_config('validation').as_dict(),
        host_environment_changed=False), indent=2)+'\n')
    tasks = [(split, s, i, behavior) for split, (_, number) in STREAMS.items()
             for s in range(3) for i in range(number) for behavior in ('prs', 'learned')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(collect_one, tasks))
    for split, (_, number) in STREAMS.items():
        for behavior in ('prs', 'learned'):
            parts = [load(ROOT/f'{split}_{behavior}_{s}_{i}.npz')
                     for s in range(3) for i in range(number)]
            np.savez_compressed(ROOT/f'{split}_{behavior}.npz', **concatenate(parts, tuple(parts[0])))
    old = old_load('train')
    for arm, behavior in (('prs_added', 'prs'), ('visitation_added', 'learned')):
        added = load(ROOT/f'train_{behavior}.npz')
        assert len(old['x']) == len(added['x']) == 192
        np.savez_compressed(ROOT/f'{arm}.npz', **concatenate([old, added]))
    validation = concatenate([load(ROOT/f'validation_{b}.npz') for b in ('prs', 'learned')])
    assert len(validation['x']) == 192
    np.savez_compressed(ROOT/'validation.npz', **validation)
    train_seeds = {r['trace_seed'] for r in rows if r['split'] == 'train'}
    valid_seeds = {r['trace_seed'] for r in rows if r['split'] == 'validation'}
    assert len(train_seeds) == 24 and len(valid_seeds) == 12 and train_seeds.isdisjoint(valid_seeds)
    write_rows(ROOT/'trajectories.csv', rows)
    previous_intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'collection_complete.json').write_text(json.dumps(dict(exogenous_train=24,
        exogenous_validation=12, behavior_trajectories=len(rows), states=sum(r['states'] for r in rows),
        train_states_per_arm=384, common_prediction_validation_states=192), indent=2)+'\n')
    (ROOT/'collection_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
