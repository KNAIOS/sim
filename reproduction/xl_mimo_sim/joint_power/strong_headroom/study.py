"""Hindsight one-action opportunity, never a deployable learned-policy result."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.anchor_confirmation.domain_prs import command_at
from joint_power.anchor_confirmation.study import ROOT as PRIOR, input_checks
from joint_power.calibrated.calibration import branch
from joint_power.model import Environment
from joint_power.postdecision.encoding import hardware_features
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_strong_headroom')
STREAM, NUMBER, SLOTS = 5109220000, 4, (20, 60, 100)
FIELDS = ('energy_j', 'failed', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j')


def failures(records, users):
    return sum(round(users*r['outage']) for r in records)


def reference_command(env, adaptive, past):
    env.prepare()
    cfg = env.cfg
    target = cfg.outage_target
    if adaptive:
        allowance = target*cfg.users*cfg.episode_steps-past
        target = min(1., max(0., allowance)/(cfg.users*(cfg.episode_steps-env.t)))
    return command_at(replace(cfg, outage_target=target), env.tracker, env.hw, env.req)


def advance(env, command, past):
    before = len(env.records)
    env.block(command)
    return past+failures(env.records[before:], env.cfg.users)


def totals(records, users):
    return np.array([failures(records, users) if key == 'failed'
                     else sum(r[key] for r in records) for key in FIELDS])


def rollout(env, command, adaptive, past):
    other = branch(env)
    past = advance(other, command, past)
    while other.t < other.cfg.episode_steps:
        past = advance(other, reference_command(other, adaptive, past), past)
    return np.stack([totals(other.records[:h], env.cfg.users)
                     for h in (4, 16, len(other.records))]), other.records


def select(labels, reference, horizon):
    """Same FULL-tail failure filter at every energy horizon; all are oracles."""
    valid = np.flatnonzero(labels[:, -1, 1] <= labels[reference, -1, 1])
    return min(valid, key=lambda a: (labels[a, horizon, 0], a != reference, a))


def state_labels(env, adaptive, past):
    candidates = env.candidates()
    baseline = reference_command(env, adaptive, past)
    ref = int(baseline.sum())-1
    np.testing.assert_array_equal(candidates.masks[ref], baseline)
    post = hardware_features(env.hw, candidates.masks).reshape(len(candidates.masks), -1)
    # Finalize causal inputs BEFORE any simulator outcome is generated.
    x = np.concatenate((candidates.features, post,
                        np.full((len(post), 1), past/(env.cfg.users*env.cfg.episode_steps))), axis=1)
    cache, labels, records = {}, [], []
    for command in candidates.masks:
        key = command.tobytes()
        if key not in cache:
            cache[key] = rollout(env, command, adaptive, past)
        value, history = cache[key]
        labels.append(value)
        records.append(history)
    return dict(x=x.astype(np.float32), y=np.array(labels), masks=candidates.masks,
                reference=ref, past=past), records


def collect_one(task):
    scenario, index = task
    torch.set_num_threads(1)
    cfg = make_config('validation')
    trace = make_trace(cfg, STREAM+10000*scenario+index, SCENARIOS[scenario])
    unchanged = (trace.channel.copy(), trace.gain_db.copy(), trace.requirements_mbps.copy())
    episodes, roots, payload, errors, checked = [], [], [], defaultdict(float), 0
    for adaptive in (False, True):
        env, past, local = Environment(cfg, trace), 0, []
        name = 'Expanded budgetPRS' if adaptive else 'Expanded PRS'
        while env.t < cfg.episode_steps:
            env.prepare()
            if env.t in SLOTS:
                slot, active, age = env.t, env.hw.active.copy(), env.age.copy()
                countdown, old_command = env.hw.countdown.copy(), env.hw.command.copy()
                item, histories = state_labels(env, adaptive, past)
                assert env.t == slot
                for actual, expected in ((env.hw.active, active), (env.age, age),
                                         (env.hw.countdown, countdown), (env.hw.command, old_command)):
                    np.testing.assert_array_equal(actual, expected)
                y, ref = item['y'], item['reference']
                actions = [select(y, ref, h) for h in range(3)]
                for a in set([ref, *actions]):
                    complete = env.records+histories[a]
                    assert len(complete) == cfg.episode_steps
                    for key, error in independent(complete, trace).items():
                        errors[key] = max(errors[key], error)
                    checked += len(complete)
                local.append((slot, item))
                payload.append(dict(identity=np.array([scenario, index, int(adaptive), slot]), **item))
                for horizon, a in zip((4, 16, -1), actions):
                    row = dict(scenario=scenario, trajectory=index, reference=name, slot=slot,
                        horizon=horizon, reference_action=ref, selected_action=int(a),
                        selected_count=int(item['masks'][a].sum()), reference_count=int(item['masks'][ref].sum()),
                        feasible_actions=int(np.sum(y[:, -1, 1] <= y[ref, -1, 1])),
                        first_block_delta_j=float(y[a, 0, 0]-y[ref, 0, 0]),
                        later_delta_j=float(y[a, -1, 0]-y[ref, -1, 0]-(y[a, 0, 0]-y[ref, 0, 0])))
                    row.update({f'delta_{key}': float(y[a, -1, k]-y[ref, -1, k])
                                for k, key in enumerate(FIELDS)})
                    assert row['delta_failed'] <= 0
                    if horizon == -1:
                        assert row['delta_energy_j'] <= 1e-9
                    roots.append(row)
            past = advance(env, reference_command(env, adaptive, past), past)
        metrics = env.metrics()
        for slot, item in local:
            np.testing.assert_allclose(item['y'][item['reference'], -1],
                                       totals(env.records[slot:], cfg.users), atol=1e-9)
        episodes.append(dict(scenario=scenario, trajectory=index, reference=name, **metrics))
        for row in roots:
            if row['reference'] == name:
                row['reference_episode_j'] = metrics['energy_j']
                row['relative_single_intervention_saving'] = -row['delta_energy_j']/metrics['energy_j']
    for actual, expected in zip((trace.channel, trace.gain_db, trace.requirements_mbps), unchanged):
        np.testing.assert_array_equal(actual, expected)
    np.savez_compressed(ROOT/f'labels_{scenario}_{index}.npz',
                        **{key: np.array([p[key] for p in payload]) for key in payload[0]})
    print(f'headroom scenario={scenario} trace={index}: {len(payload)} roots verified', flush=True)
    return episodes, roots, dict(errors), checked


def intact():
    input_checks()
    for folder in (PRIOR, PRIOR/'expanded_prs', PRIOR/'expanded_comparison'):
        manifest = json.loads((folder/'manifest.json').read_text())
        outputs = json.loads((folder/'output_hashes.json').read_text())
        assert all(digest(p) == h for p, h in (manifest['input_hashes'] | outputs).items())


def summarize(rows):
    rng, result = np.random.default_rng(2026092230), []
    for scenario in range(3):
        for reference in ('Expanded PRS', 'Expanded budgetPRS'):
            group = [r for r in rows if r['scenario'] == scenario and r['reference'] == reference]
            for horizon in (4, 16, -1):
                chosen = [r for r in group if r['horizon'] == horizon]
                delta = np.array([np.mean([r['delta_energy_j'] for r in chosen
                                          if r['trajectory'] == i]) for i in range(NUMBER)])
                draws = delta[rng.integers(0, NUMBER, (4000, NUMBER))].mean(1)
                result.append(dict(scenario=scenario, reference=reference, horizon=horizon,
                    trajectories=NUMBER, roots=len(chosen), mean_delta_j=float(delta.mean()),
                    ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975)),
                    mean_relative_saving=float(np.mean([r['relative_single_intervention_saving'] for r in chosen])),
                    improved_root_fraction=float(np.mean([r['delta_energy_j'] < -1e-8 for r in chosen])),
                    mean_delta_failed=float(np.mean([r['delta_failed'] for r in chosen])),
                    **{f'mean_delta_{key}': float(np.mean([r[f'delta_{key}'] for r in chosen]))
                       for key in FIELDS[2:]},
                    mean_first_block_delta_j=float(np.mean([r['first_block_delta_j'] for r in chosen])),
                    mean_later_delta_j=float(np.mean([r['later_delta_j'] for r in chosen]))))
    return result


def main():
    intact()
    ROOT.mkdir(exist_ok=False)
    files = [*Path('joint_power/strong_headroom').glob('*.py'),
             Path('joint_power/strong_headroom/PLAN.md'),
             Path('joint_power/anchor_confirmation/domain_prs.py'),
             Path('joint_power/model.py'), Path('joint_power/calibrated/calibration.py'),
             Path('joint_power/postdecision/encoding.py'), Path('joint_power/rollout/verify.py'),
             PRIOR/'expanded_comparison/output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        stream=STREAM, trajectories_per_scenario=NUMBER, root_slots=SLOTS,
        config=make_config('validation').as_dict(), training_diagnostic=True,
        oracle_not_deployable=True, host_environment_changed=False), indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        collected = list(pool.map(collect_one, [(s, i) for s in range(3) for i in range(NUMBER)]))
    episodes = [r for e, _, _, _ in collected for r in e]
    rows = [r for _, rs, _, _ in collected for r in rs]
    assert len(episodes) == 24 and len(rows) == 216
    summary = summarize(rows)
    write_rows(ROOT/'episodes.csv', episodes)
    write_rows(ROOT/'roots.csv', rows)
    write_rows(ROOT/'summary.csv', summary)
    nominal = {r['horizon']: r for r in summary
               if r['scenario'] == 1 and r['reference'] == 'Expanded budgetPRS'}
    long_gap = nominal[-1]['mean_relative_saving']-nominal[4]['mean_relative_saving']
    gate = dict(further_causal_probe_justified=bool(nominal[-1]['mean_relative_saving'] >= .02
        and nominal[-1]['improved_root_fraction'] >= .25),
        long_energy_head_motivated=bool(long_gap >= .01), long_energy_gap_fraction=long_gap,
        nominal_full=nominal[-1], diagnostic_only=True, R1_closed=False)
    errors = {k: max(e[k] for _, _, e, _ in collected) for k in collected[0][2]}
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    report = dict(gate=gate, summary=summary, unique_traces=12, baseline_episodes=24,
        root_states=72, candidate_entries=72*17, physical_slot_checks=sum(n for *_, n in collected),
        max_physical_errors=errors, all_reference_suffixes_verified=True,
        future_in_labels_only=True, limitations='Hindsight training diagnostic; roots correlated; '
        'single interventions cannot be summed or claimed as learned closed-loop savings.')
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
