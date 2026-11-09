"""Exact sample-path performance differences at actual policy-visited states."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import read_rows
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment
from joint_power.risk_budget.control import action_at
from joint_power.risk_budget.study import ROOT, controllers, input_checks
from joint_power.rollout.collect import base_command, rollout
from joint_power.rollout.probe import FULL_SLOT
from joint_power.study import SCENARIOS, digest, make_trace, write_rows


def decomposition(cfg, trace, model, predictor, alpha):
    env, past, states = Environment(cfg, trace), 0, []
    while env.t < cfg.episode_steps:
        command, info = action_at(env, predictor, model, 'Value', alpha, past)
        baseline = base_command(env)
        slot = env.t
        active, countdown, age = env.hw.active.copy(), env.hw.countdown.copy(), env.age.copy()
        baseline_value = rollout(env, baseline)[-1]
        chosen_value = baseline_value.copy() if np.array_equal(command, baseline) else rollout(env, command)[-1]
        assert env.t == slot
        np.testing.assert_array_equal(env.hw.active, active)
        np.testing.assert_array_equal(env.hw.countdown, countdown)
        np.testing.assert_array_equal(env.age, age)
        advantage = chosen_value-baseline_value
        predicted_e = info['predicted_relative_energy']*FULL_SLOT
        predicted_f = info['selected_count_prediction']-info['reference_count_prediction']
        states.append(dict(**info, baseline_remaining_j=float(baseline_value[0]),
            baseline_remaining_failed=int(baseline_value[1]), chosen_remaining_j=float(chosen_value[0]),
            chosen_remaining_failed=int(chosen_value[1]), energy_advantage_j=float(advantage[0]),
            failed_advantage=int(advantage[1]), predicted_energy_advantage_j=predicted_e,
            predicted_failed_advantage=predicted_f,
            energy_prediction_error_j=predicted_e-float(advantage[0]),
            failed_prediction_error=predicted_f-float(advantage[1]),
            differs_from_prs=not np.array_equal(command, baseline),
            wake_requests=int(np.sum(command & ~active & (countdown == 0))),
            command_changes=int(np.sum(command != env.hw.command))))
        before = len(env.records)
        env.block(command)
        past += sum(round(cfg.users*r['outage']) for r in env.records[before:])
    total_e = sum(r['energy_advantage_j'] for r in states)
    total_f = sum(r['failed_advantage'] for r in states)
    e_gap = env.metrics()['energy_j']-states[0]['baseline_remaining_j']
    f_gap = past-states[0]['baseline_remaining_failed']
    assert abs(total_e-e_gap) < 1e-8, (total_e, e_gap)
    assert total_f == f_gap, (total_f, f_gap)
    return env.metrics(), states, dict(energy_gap_j=e_gap, failed_gap=f_gap,
        energy_identity_error=abs(total_e-e_gap), failed_identity_error=abs(total_f-f_gap),
        sum_predicted_energy_advantage_j=sum(r['predicted_energy_advantage_j'] for r in states),
        sum_predicted_failed_advantage=sum(r['predicted_failed_advantage'] for r in states))


def one(row):
    torch.set_num_threads(1)
    policies, _ = controllers()
    _, model = policies[row['method'], row['seed']]
    cfg, predictor = make_config('validation'), load_predictor()
    trace = make_trace(cfg, 3909220000+10000*row['scenario']+row['trajectory'], SCENARIOS[row['scenario']])
    metrics, states, audit = decomposition(cfg, trace, model, predictor, row['parameter'])
    for key, value in metrics.items():
        assert abs(value-row[key]) < 1e-8
    ident = {key: row[key] for key in ('method', 'seed', 'scenario', 'trajectory', 'parameter')}
    print(f'performance difference scenario={row["scenario"]} trace={row["trajectory"]} seed={row["seed"]}', flush=True)
    return dict(**ident, **audit), [dict(**ident, **r) for r in states]


def main():
    input_checks()
    old_hashes = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in old_hashes.items())
    output = ROOT/'performance_difference'
    output.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/risk_budget/DIAGNOSIS_PLAN.md'), ROOT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (output/'manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    rows = [r for r in read_rows(ROOT/'selected_episodes.csv') if r['method'] == 'Value Visitation-added']
    assert len(rows) == 24
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, rows))
    episodes = [r[0] for r in results]
    states = [row for _, items in results for row in items]
    assert len(states) == 840
    write_rows(output/'episodes.csv', episodes)
    write_rows(output/'states.csv', states)
    summary = []
    for s in range(3):
        group = [r for r in episodes if r['scenario'] == s]
        roots = [r for r in states if r['scenario'] == s]
        summary.append(dict(scenario=s, trajectories=4, model_seeds=2, roots=len(roots),
            **{key: float(np.mean([r[key] for r in group])) for key in
               ('energy_gap_j', 'failed_gap', 'sum_predicted_energy_advantage_j', 'sum_predicted_failed_advantage')},
            selected_energy_mae_j=float(np.mean([abs(r['energy_prediction_error_j']) for r in roots])),
            selected_failed_mae=float(np.mean([abs(r['failed_prediction_error']) for r in roots])),
            fraction_changed=float(np.mean([r['differs_from_prs'] for r in roots])),
            fraction_predicted_energy_saving_but_actual_cost=float(np.mean([
                r['predicted_energy_advantage_j'] < -1e-6 and r['energy_advantage_j'] > 1e-6 for r in roots])),
            fraction_risk_underprediction=float(np.mean([
                r['selected_count_prediction'] < r['chosen_remaining_failed']-1e-6 for r in roots]))))
    write_rows(output/'summary.csv', summary)
    report = dict(summary=summary, selected_policy_unchanged=True,
        max_energy_identity_error=max(r['energy_identity_error'] for r in episodes),
        max_failed_identity_error=max(r['failed_identity_error'] for r in episodes),
        actual_episodes_match_frozen=True, diagnostic_only=True,
        limitation='Labels use future trace offline; root states are correlated actual-policy visits, '
        'not independent experiments or deployable clairvoyant choices')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in old_hashes.items())
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
