"""Reassess the unchanged primary with strengthened matched-count PRS."""
import json
from pathlib import Path

import numpy as np

from joint_power.aggregation.control import read_rows
from joint_power.anchor_confirmation.frontier import envelope
from joint_power.anchor_confirmation.study import ROOT, NONLEARNING, input_checks
from joint_power.study import digest, write_rows

BASELINES = (*NONLEARNING, 'Expanded PRS', 'Expanded budgetPRS')


def main():
    input_checks()
    groups = [json.loads((ROOT/p/'output_hashes.json').read_text()) for p in ('', 'expanded_prs', 'matched_risk')]
    assert all(digest(p) == h for group in groups for p, h in group.items())
    out = ROOT/'expanded_comparison'
    out.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/anchor_confirmation/DOMAIN_PLAN.md'),
             Path('joint_power/anchor_confirmation/frontier.py'), ROOT/'expanded_prs/output_hashes.json',
             ROOT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (out/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, baselines=BASELINES,
        posthoc_strengthened_baseline_analysis=True, primary_unchanged=True), indent=2)+'\n')
    rows = read_rows(ROOT/'episodes.csv')+read_rows(ROOT/'expanded_prs/episodes.csv')
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    assert len(lookup) == len(rows) == 456
    def mean(method, scenario, key, seed=None):
        return float(np.mean([r[key] for r in rows if r['method'] == method and r['scenario'] == scenario
                             and (seed is None or seed == r['seed'])]))
    feasible = [dict(method=m, energy_j=mean(m, 1, 'energy_j'),
                     worst_outage=max(mean(m, s, 'outage') for s in range(3))) for m in BASELINES]
    feasible = [r for r in feasible if r['worst_outage'] <= .05+1e-12]
    reference = min(feasible, key=lambda r: (r['energy_j'], r['worst_outage']))
    primary_energy = mean('Energy Value', 1, 'energy_j')
    per_seed = {str(seed): dict(energy_j=mean('Energy Value', 1, 'energy_j', seed),
        outage=[mean('Energy Value', s, 'outage', seed) for s in range(3)]) for seed in (11, 22)}
    gain = 1-primary_energy/reference['energy_j']
    gate = dict(passed=bool(gain >= .02 and all(r['energy_j'] <= reference['energy_j'] and
        max(r['outage']) <= .05+1e-12 for r in per_seed.values())),
        nominal_energy_j=primary_energy, nominal_saving=gain, reference=reference, per_seed=per_seed,
        feasible_baselines=feasible, original_policy_and_thresholds_unchanged=True)
    frontier_rows, intervals = [], []
    rng = np.random.default_rng(2026092228)
    for s in range(3):
        e = np.array([[lookup[m, 0, s, i]['energy_j'] for i in range(8)] for m in BASELINES])
        c = np.array([[lookup[m, 0, s, i]['outage'] for i in range(8)] for m in BASELINES])
        pe = np.array([[lookup['Energy Value', seed, s, i]['energy_j'] for i in range(8)] for seed in (11, 22)])
        pc = np.array([[lookup['Energy Value', seed, s, i]['outage'] for i in range(8)] for seed in (11, 22)])
        for seed, le, lc in ((0, pe.mean(0), pc.mean(0)), (11, pe[0], pc[0]), (22, pe[1], pc[1])):
            for name, target in (('matched', float(lc.mean())), ('target5pct', .05)):
                value, risk, first, second, p = envelope(e.mean(1), c.mean(1), target)
                frontier_rows.append(dict(scenario=s, seed=seed, bound=name, learned_energy_j=float(le.mean()),
                    learned_outage=float(lc.mean()), target_outage=target, first=BASELINES[first],
                    second=BASELINES[second], first_probability=p, mixture_expected_energy_j=value,
                    mixture_expected_outage=risk, difference_j=float(le.mean()-value), saving=float(1-le.mean()/value)))
        draws = {name: [] for name in ('matched', 'target5pct', 'Expanded PRS', 'Expanded budgetPRS')}
        for _ in range(2000):
            ids, seeds = rng.integers(0, 8, 8), rng.integers(0, 2, 2)
            le, lc = pe[seeds[:, None], ids[None, :]].mean(), pc[seeds[:, None], ids[None, :]].mean()
            for name, target in (('matched', lc), ('target5pct', .05)):
                draws[name].append(le-envelope(e[:, ids].mean(1), c[:, ids].mean(1), target)[0])
            for name in ('Expanded PRS', 'Expanded budgetPRS'):
                draws[name].append(le-e[BASELINES.index(name), ids].mean())
        for name, values in draws.items():
            intervals.append(dict(scenario=s, reference=name, ci_low=float(np.quantile(values, .025)),
                                  ci_high=float(np.quantile(values, .975))))
    write_rows(out/'frontier.csv', frontier_rows)
    write_rows(out/'bootstrap_intervals.csv', intervals)
    (out/'gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    report = dict(gate=gate, pooled_frontier=[r for r in frontier_rows if r['seed'] == 0],
        intervals=intervals, old_seven_baseline_gate_superseded=True,
        finite_empirical_mixtures_not_deployed=True, R1_closed=False)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for group in groups for p, h in group.items())
    assert all(digest(p) == h for p, h in hashes.items())
    (out/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in out.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
