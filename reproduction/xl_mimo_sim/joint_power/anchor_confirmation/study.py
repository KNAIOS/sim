"""Frozen admission policy and ablations on a previously unused trace stream."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models, summaries
from joint_power.baseline_anchor.control import episode as anchored
from joint_power.baseline_anchor.study import ROOT as RESEARCH, analyze, input_checks as old_checks
from joint_power.calibrated.histogram import load_predictor
from joint_power.floor_projection.control import episode as floor_episode
from joint_power.risk_budget.control import episode as budget_episode
from joint_power.rollout.control import episode as old_episode
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_anchor_confirmation')
STREAM, NUMBER = 4409220000, 8
LEARNED = ('Energy Value', 'Empty Value', 'Energy Myopic', 'Floor Value', 'Fixed-price Value')
NONLEARNING = ('PRS original', 'PRS budget', 'Raw full', 'Raw coarse', 'Full power', 'Empty Raw budget', 'Energy Raw budget')


def input_checks():
    old_checks()
    assert json.loads((RESEARCH/'gate.json').read_text())['passed']
    for name in ('manifest.json', 'output_hashes.json'):
        hashes = json.loads((RESEARCH/name).read_text())
        hashes = hashes['input_hashes'] if name == 'manifest.json' else hashes
        assert all(digest(p) == h for p, h in hashes.items())


def specifications():
    rows = [(m, seed, 4. if m == 'Fixed-price Value' else .1) for m in LEARNED for seed in (11, 22)]
    return rows+[(m, 0, p) for m, p in zip(NONLEARNING, (.05, 1., .05, 0., .05, 0., 0.))]


def comparisons(rows):
    rng = np.random.default_rng(2026092226)
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    assert len(lookup) == len(rows)
    result = []
    for s in range(3):
        for reference in (*LEARNED[1:], *NONLEARNING):
            for metric in ('energy_j', 'outage'):
                delta = np.array([[lookup['Energy Value', seed, s, i][metric]-lookup[
                    reference, 0 if reference in NONLEARNING else seed, s, i][metric]
                    for i in range(NUMBER)] for seed in (11, 22)])
                si, ti = rng.integers(0, 2, (4000, 2)), rng.integers(0, NUMBER, (4000, NUMBER))
                draws = delta[si[:, :, None], ti[:, None, :]].mean((1, 2))
                result.append(dict(scenario=s, reference=reference, metric=metric,
                    difference=float(delta.mean()), ci_low=float(np.quantile(draws, .025)),
                    ci_high=float(np.quantile(draws, .975))))
    return result


def main():
    torch.set_num_threads(1)
    input_checks()
    ROOT.mkdir(exist_ok=False)
    loaded, checkpoints = models()
    files = [*Path('joint_power/anchor_confirmation').glob('*.py'),
        Path('joint_power/anchor_confirmation/PLAN.md'), RESEARCH/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        stream=STREAM, trajectories_per_scenario=NUMBER, fixed_specifications=specifications(),
        previously_examined=False, no_retuning=True, host_environment_changed=False), indent=2)+'\n')
    cfg, predictor = make_config('validation'), load_predictor()
    rows, slots, decisions, ledgers, errors = [], [], [], [], defaultdict(float)
    for s, scenario in enumerate(SCENARIOS):
        for i in range(NUMBER):
            trace = make_trace(cfg, STREAM+s*10000+i, scenario)
            for method, seed, parameter in specifications():
                model = loaded['Visitation-added', seed] if seed else None
                ledger = None
                if method.startswith(('Energy ', 'Empty ')):
                    rule, proposer = method.split(' ', 1)
                    metrics, records, ledger = anchored(trace, proposer, rule, parameter, predictor, model)
                elif method == 'Floor Value':
                    metrics, records, _ = floor_episode(trace, 'Value', parameter, predictor, model)
                elif method == 'PRS budget':
                    metrics, records, _ = budget_episode(trace, method, parameter, predictor)
                else:
                    name = 'MC absolute' if method == 'Fixed-price Value' else method
                    metrics, records = old_episode(trace, name, parameter, predictor, model)
                ident = dict(method=method, seed=seed, scenario=s, trajectory=i, parameter=parameter)
                rows.append(dict(**ident, **metrics))
                slots.extend(dict(**ident, **r) for r in records)
                for name, value in independent(records, trace).items():
                    errors[name] = max(errors[name], value)
                failures = sum(round(3*r['outage']) for r in records)
                assert abs(metrics['outage']-failures/420) < 1e-10
                for name in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
                    assert abs(metrics[name]-sum(r[name] for r in records)) < 1e-8
                ledgers.append(dict(**ident, final_failed=failures, pathwise_target_exceeded=int(failures > 21)))
                if ledger is not None:
                    decisions.extend(dict(**ident, **r) for r in ledger)
                    past = 0
                    for j, item in enumerate(ledger):
                        block = records[4*j:4*j+4]
                        floor = round(3*block[0]['outage'])+sum(3 for r in block[1:] if r['active_count'] == 0)
                        failed = sum(round(3*r['outage']) for r in block)
                        assert item['executed_floor'] == floor <= failed
                        assert item['past_failed'] == past and item['block_failed'] == failed
                        assert abs(item['residual_budget']-max(0., 21-past)) < 1e-10
                        if not item['accepted']:
                            assert item['executed_bits'] == item['reference_bits']
                        if method.startswith('Energy ') and item['accepted']:
                            assert item['proposal_relative_energy'] < -1e-8
                        past += failed
                        assert item['cumulative_failed'] == past
            print(f'anchor confirmation scenario={s} trace={i}, episodes={len(rows)}', flush=True)
    assert len(rows) == 408 and len(slots) == 57120
    assert errors['rate_mbps'] < 1e-5 and max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    result = analyze(rows)
    result['independent_confirmation'] = True
    result['scope'] = 'Fresh frozen-policy confirmation; not matched-outage or hard safety proof'
    write_rows(ROOT/'episodes.csv', rows)
    write_rows(ROOT/'slots.csv', slots)
    write_rows(ROOT/'decisions.csv', decisions)
    write_rows(ROOT/'episode_ledgers.csv', ledgers)
    write_rows(ROOT/'summary.csv', summaries(rows))
    write_rows(ROOT/'paired_differences.csv', comparisons(rows))
    (ROOT/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    report = dict(episodes=408, exogenous_traces=24, trajectories_per_scenario=8,
        verified_slots=57120, max_physical_errors=dict(errors), gate=result,
        uncertainty='Whole-trajectory/seed descriptive bootstrap; only two training seeds',
        source_and_weights_frozen=True, old_failures_retained=True)
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
