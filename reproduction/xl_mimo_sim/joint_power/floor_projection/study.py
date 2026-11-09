"""Fixed-parameter paired check with independent physical verification."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import read_rows, summaries
from joint_power.calibrated.histogram import load_predictor
from joint_power.floor_projection.control import episode
from joint_power.risk_budget.study import ROOT as PREVIOUS, BASELINES, controllers, input_checks
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_floor_projection')


def intact():
    input_checks()
    for part in ('', 'audit', 'performance_difference', 'floor_audit'):
        hashes = json.loads((PREVIOUS/part/'output_hashes.json').read_text())
        assert all(digest(p) == h for p, h in hashes.items()), part
    manifest = json.loads((PREVIOUS/'manifest.json').read_text())
    assert all(digest(p) == h for p, h in manifest['input_hashes'].items())


def paired(rows):
    """Crossed whole-trajectory/seed descriptive bootstrap, not a new test."""
    rng = np.random.default_rng(2026092223)
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    output = []
    for arm in ('PRS-added', 'Visitation-added'):
        method = f'Floor Value {arm}'
        for s in range(3):
            for reference in (f'Value {arm}', f'Floor Myopic {arm}', 'PRS original', 'PRS budget'):
                for metric in ('energy_j', 'outage'):
                    ref_seed = lambda seed: 0 if reference.startswith('PRS ') else seed
                    delta = np.array([[lookup[method, seed, s, i][metric]-
                        lookup[reference, ref_seed(seed), s, i][metric] for i in range(4)] for seed in (11, 22)])
                    si, ti = rng.integers(0, 2, (4000, 2)), rng.integers(0, 4, (4000, 4))
                    draws = delta[si[:, :, None], ti[:, None, :]].mean((1, 2))
                    output.append(dict(method=method, scenario=s, reference=reference, metric=metric,
                        difference=float(delta.mean()), ci_low=float(np.quantile(draws, .025)),
                        ci_high=float(np.quantile(draws, .975))))
    return output


def gate(rows):
    def mean(method, s, key, seed=None):
        group = [r[key] for r in rows if r['method'] == method and r['scenario'] == s
                 and (seed is None or seed == r['seed'])]
        assert group, (method, s, key, seed)
        return float(np.mean(group))
    feasible = []
    for method in (*BASELINES, 'Floor Raw budget'):
        worst = max(mean(method, s, 'outage') for s in range(3))
        if worst <= .05+1e-12:
            feasible.append(dict(method=method, nominal_energy_j=mean(method, 1, 'energy_j'), worst_outage=worst))
    reference = min(feasible, key=lambda r: (r['nominal_energy_j'], r['worst_outage']))
    families = []
    for arm in ('PRS-added', 'Visitation-added'):
        method = f'Floor Value {arm}'
        per_seed = {str(seed): [mean(method, s, 'outage', seed) for s in range(3)] for seed in (11, 22)}
        energies = {str(seed): mean(method, 1, 'energy_j', seed) for seed in (11, 22)}
        gain = 1-mean(method, 1, 'energy_j')/reference['nominal_energy_j']
        families.append(dict(method=method, nominal_energy_j=mean(method, 1, 'energy_j'),
            nominal_saving=gain, per_seed_nominal_energy=energies, per_seed_outage=per_seed,
            passed=bool(max(max(v) for v in per_seed.values()) <= .05+1e-12 and gain >= .01
                        and max(energies.values()) <= reference['nominal_energy_j'])))
    return dict(passed=any(f['passed'] for f in families), reference=reference,
        feasible_baselines=feasible, families=families, required_saving=.01,
        independent_test=False, development_launched=False)


def main():
    torch.set_num_threads(1)
    intact()
    ROOT.mkdir(exist_ok=False)
    policies, checkpoints = controllers()
    selections = json.loads((PREVIOUS/'selection.json').read_text())
    selections = [r for r in selections if r['method'] != 'PRS budget']
    assert len(selections) == 9
    files = [*Path('joint_power/floor_projection').glob('*.py'),
        Path('joint_power/floor_projection/PLAN.md'), Path('joint_power/risk_budget/physical_floor.py'),
        PREVIOUS/'selection.json', PREVIOUS/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        fixed_selections=selections, stream=3909220000, previously_examined_validation=True,
        host_environment_changed=False), indent=2)+'\n')
    cfg, predictor = make_config('validation'), load_predictor()
    rows, slots, decisions, ledgers, errors = [], [], [], [], defaultdict(float)
    for s, scenario in enumerate(SCENARIOS):
        for i in range(4):
            trace = make_trace(cfg, 3909220000+s*10000+i, scenario)
            for selected in selections:
                method, seed, alpha = (selected[k] for k in ('method', 'seed', 'parameter'))
                rule, model = policies[method, seed]
                metrics, records, ledger = episode(trace, rule, alpha, predictor, model)
                ident = dict(method='Floor '+method, seed=seed, scenario=s, trajectory=i, parameter=alpha)
                rows.append(dict(**ident, **metrics))
                slots.extend(dict(**ident, **r) for r in records)
                decisions.extend(dict(**ident, **r) for r in ledger)
                for name, value in independent(records, trace).items():
                    errors[name] = max(errors[name], value)
                past = 0
                for j, item in enumerate(ledger):
                    block = records[4*j:4*j+4]
                    actual_floor = round(3*block[0]['outage'])+sum(3 for r in block[1:] if r['active_count'] == 0)
                    failed = sum(round(3*r['outage']) for r in block)
                    assert item['selected_floor'] == actual_floor <= failed
                    assert item['selected_projected_count'] >= actual_floor
                    assert item['past_failed'] == past and item['block_failed'] == failed
                    assert abs(item['residual_budget']-max(0., 21-past)) < 1e-10
                    assert abs(item['applied_bound']-alpha*item['residual_budget']) < 1e-10
                    past += failed
                    assert item['cumulative_failed'] == past
                ledgers.append(dict(**ident, final_failed=past, pathwise_target_exceeded=int(past > 21),
                    changed_commands=sum(r['changed_command'] for r in ledger),
                    decisions_with_corrected_candidates=sum(r['corrected_candidates'] > 0 for r in ledger),
                    no_feasible_decisions=sum(r['feasible_candidates'] == 0 for r in ledger)))
            print(f'floor-projection scenario={s} trace={i}, episodes={len(rows)}', flush=True)
    assert len(rows) == 108 and len(slots) == 15120 and len(decisions) == 3780
    assert errors['rate_mbps'] < 1e-5 and max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    references = read_rows(PREVIOUS/'selected_episodes.csv')
    assert len(references) == 276
    all_rows = rows+references
    write_rows(ROOT/'new_episodes.csv', rows)
    write_rows(ROOT/'all_episodes.csv', all_rows)
    write_rows(ROOT/'slots.csv', slots)
    write_rows(ROOT/'decisions.csv', decisions)
    write_rows(ROOT/'episode_ledgers.csv', ledgers)
    write_rows(ROOT/'summary.csv', summaries(all_rows))
    write_rows(ROOT/'paired_differences.csv', paired(all_rows))
    result = gate(all_rows)
    (ROOT/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    report = dict(new_episodes=108, reference_episodes=276, verified_slots=15120,
        verified_decisions=3780, max_physical_errors=dict(errors),
        physical_floor_matches_realized_known_failures=True, ledger_correct=True,
        remaining_count_at_least_physical_floor=True,
        uncertainty='Descriptive bootstrap on reused validation; not independent significance',
        caveat='Lower-bound consistency is not a safety upper bound or proof of savings',
        gate=result)
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
