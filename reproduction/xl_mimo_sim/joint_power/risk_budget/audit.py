"""Independent physical replay, causal budget accounting and paired effects."""
import json
from collections import defaultdict

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import METRICS, read_rows, summaries
from joint_power.calibrated.histogram import load_predictor
from joint_power.risk_budget.control import episode
from joint_power.risk_budget.study import ROOT, controllers, gates, input_checks
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows


def paired(rows):
    rng = np.random.default_rng(2026092222)
    output = []
    for arm in ('PRS-added', 'Visitation-added'):
        method = f'Value {arm}'
        for s in range(3):
            for reference in (f'Myopic {arm}', arm, 'PRS original', 'Raw budget', 'PRS budget'):
                for metric in ('energy_j', 'outage'):
                    delta = np.empty((2, 4))
                    for n, seed in enumerate((11, 22)):
                        for i in range(4):
                            first = next(r for r in rows if (r['method'], r['seed'], r['scenario'], r['trajectory'])
                                         == (method, seed, s, i))
                            ref_seed = 0 if reference in ('PRS original', 'Raw budget', 'PRS budget') else seed
                            second = next(r for r in rows if (r['method'], r['seed'], r['scenario'], r['trajectory'])
                                          == (reference, ref_seed, s, i))
                            delta[n, i] = first[metric]-second[metric]
                    si, ti = rng.integers(0, 2, (4000, 2)), rng.integers(0, 4, (4000, 4))
                    draws = delta[si[:, :, None], ti[:, None, :]].mean((1, 2))
                    output.append(dict(method=method, scenario=s, reference=reference, metric=metric,
                        difference=float(delta.mean()), ci_low=float(np.quantile(draws, .025)),
                        ci_high=float(np.quantile(draws, .975))))
    return output


def main():
    torch.set_num_threads(1)
    input_checks()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    hashes = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in manifest['input_hashes'].items())
    assert all(digest(p) == h for p, h in hashes.items())
    rows = read_rows(ROOT/'selected_episodes.csv')
    candidates = read_rows(ROOT/'candidate_episodes.csv')
    identity = lambda r: (r['method'], r['seed'], r['scenario'], r['trajectory'], r['parameter'])
    assert len(rows) == 276 and len({identity(r) for r in rows}) == 276
    assert len(candidates) == 600 and len({identity(r) for r in candidates}) == 600
    policies, _ = controllers()
    cfg, predictor = make_config('validation'), load_predictor()
    traces = {(s, i): make_trace(cfg, 3909220000+s*10000+i, spec)
              for s, spec in enumerate(SCENARIOS) for i in range(4)}
    errors, slots, decisions, episodes = defaultdict(float), [], [], []
    for row in rows:
        if (row['method'], row['seed']) not in policies:
            continue
        trace = traces[row['scenario'], row['trajectory']]
        rule, model = policies[row['method'], row['seed']]
        metrics, records, ledger = episode(trace, rule, row['parameter'], predictor, model)
        for key, value in metrics.items():
            assert abs(value-row[key]) < 1e-8
        for key, value in independent(records, trace).items():
            errors[key] = max(errors[key], value)
        ident = {key: row[key] for key in ('method', 'seed', 'scenario', 'trajectory', 'parameter')}
        slots.extend(dict(**ident, **r) for r in records)
        decisions.extend(dict(**ident, **r) for r in ledger)
        past = 0
        for i, decision in enumerate(ledger):
            assert decision['past_failed'] == past
            assert abs(decision['residual_budget']-max(0., 21-past)) < 1e-10
            assert abs(decision['applied_bound']-row['parameter']*decision['residual_budget']) < 1e-10
            fail = sum(round(3*r['outage']) for r in records[4*i:4*i+4])
            assert decision['block_failed'] == fail
            past += fail
            assert decision['cumulative_failed'] == past
        episodes.append(dict(**ident, final_failed=past, pathwise_target_exceeded=past > 21,
            no_feasible_fraction=float(np.mean([r['feasible_candidates'] == 0 for r in ledger]))))
        if len(episodes) % 24 == 0:
            print(f'budget physical/ledger replay {len(episodes)}/120', flush=True)
    assert len(episodes) == 120 and len(slots) == 16800
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for actual, expected in zip(read_rows(ROOT/'summary.csv'), summaries(rows)):
        assert (actual['method'], actual['scenario']) == (expected['method'], expected['scenario'])
        for key in METRICS:
            assert abs(actual[key]-expected[key]) < 1e-8
    assert gates(rows) == json.loads((ROOT/'gate.json').read_text())
    ledger_summary = []
    for method in dict.fromkeys(r['method'] for r in episodes):
        for scenario in range(3):
            group = [r for r in episodes if r['method'] == method and r['scenario'] == scenario]
            ledger_summary.append(dict(method=method, scenario=scenario, episodes=len(group),
                mean_final_failed=float(np.mean([r['final_failed'] for r in group])),
                fraction_exceeding_pathwise_target=float(np.mean([r['pathwise_target_exceeded'] for r in group])),
                mean_no_feasible_fraction=float(np.mean([r['no_feasible_fraction'] for r in group]))))
    output = ROOT/'audit'
    output.mkdir(exist_ok=False)
    write_rows(output/'selected_slots.csv', slots)
    write_rows(output/'selected_decisions.csv', decisions)
    write_rows(output/'episode_ledgers.csv', episodes)
    write_rows(output/'ledger_summary.csv', ledger_summary)
    write_rows(output/'paired_differences.csv', paired(rows))
    report = dict(new_selected_episodes=120, independently_replayed_slots=16800,
        candidate_episodes=600, reused_reference_episodes=156, max_physical_errors=dict(errors),
        ledger_accounting_correct=True, gate=json.loads((ROOT/'gate.json').read_text()),
        independent_test=False, source_sha256=digest(__file__),
        uncertainty='Descriptive whole-trajectory/seed bootstrap after validation selection',
        limitations='Expected PRS-tail counts and empirical factors are not hard safety bounds; '
                    'pathwise overspending and empty estimated feasible sets are explicitly retained')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
