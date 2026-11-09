"""Empirical expected mixtures at matched mean risk; never deploys hindsight."""
import json
from pathlib import Path

import numpy as np

from joint_power.aggregation.control import read_rows
from joint_power.anchor_confirmation.study import ROOT, LEARNED, NONLEARNING, input_checks
from joint_power.study import digest, write_rows


def envelope(energy, risk, target):
    energy, risk = np.asarray(energy), np.asarray(risk)
    if energy.ndim != 1 or risk.shape != energy.shape or target < 0:
        raise ValueError('Invalid frontier arrays or risk target')
    if not np.isfinite(energy).all() or not np.isfinite(risk).all() or (risk < 0).any():
        raise ValueError('Finite nonnegative risk required')
    options = [(float(energy[i]), float(risk[i]), i, i, 1.)
               for i in range(len(energy)) if risk[i] <= target+1e-12]
    for i in range(len(energy)):
        for j in range(i+1, len(energy)):
            if abs(risk[i]-risk[j]) < 1e-12:
                continue
            weight = (target-risk[j])/(risk[i]-risk[j])
            if 0 <= weight <= 1:
                options.append((float(weight*energy[i]+(1-weight)*energy[j]), float(target), i, j, float(weight)))
    if not options:
        raise ValueError('No empirically feasible pure or pair mixture')
    return min(options)


def main():
    input_checks()
    old_hashes = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in old_hashes.items())
    out = ROOT/'matched_risk'
    out.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/anchor_confirmation/FRONTIER_PLAN.md'), ROOT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (out/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        posthoc_saved_outcome_analysis=True, probabilities_fitted_on_confirmation=True,
        actual_randomized_deployment=False, baselines=NONLEARNING, bootstrap=2000), indent=2)+'\n')
    rows = read_rows(ROOT/'episodes.csv')
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    summaries, paired, intervals = [], [], []
    rng = np.random.default_rng(2026092227)
    for s in range(3):
        e = np.array([[lookup[m, 0, s, i]['energy_j'] for i in range(8)] for m in NONLEARNING])
        c = np.array([[lookup[m, 0, s, i]['outage'] for i in range(8)] for m in NONLEARNING])
        pe = np.array([[lookup['Energy Value', seed, s, i]['energy_j'] for i in range(8)] for seed in (11, 22)])
        pc = np.array([[lookup['Energy Value', seed, s, i]['outage'] for i in range(8)] for seed in (11, 22)])
        for seed, learned_e, learned_c in ((0, pe.mean(0), pc.mean(0)),
                                           (11, pe[0], pc[0]), (22, pe[1], pc[1])):
            for bound_name, target in (('matched', float(learned_c.mean())), ('target5pct', .05)):
                fe, fc, first, second, weight = envelope(e.mean(1), c.mean(1), target)
                mixture = weight*e[first]+(1-weight)*e[second]
                summaries.append(dict(scenario=s, seed=seed, bound=bound_name, target_outage=target,
                    learned_energy_j=float(learned_e.mean()), learned_outage=float(learned_c.mean()),
                    first=NONLEARNING[first], second=NONLEARNING[second], first_probability=weight,
                    mixture_expected_energy_j=fe, mixture_expected_outage=fc,
                    learned_minus_mixture_j=float(learned_e.mean()-fe), saving=float(1-learned_e.mean()/fe)))
                for i in range(8):
                    paired.append(dict(scenario=s, seed=seed, bound=bound_name, trajectory=i,
                        learned_energy_j=float(learned_e[i]), mixture_expected_energy_j=float(mixture[i]),
                        difference_j=float(learned_e[i]-mixture[i])))
        draws = {'matched': [], 'target5pct': []}
        for _ in range(2000):
            ids, seeds = rng.integers(0, 8, 8), rng.integers(0, 2, 2)
            energy, risk = pe[seeds[:, None], ids[None, :]].mean(), pc[seeds[:, None], ids[None, :]].mean()
            for name, target in (('matched', risk), ('target5pct', .05)):
                value = envelope(e[:, ids].mean(1), c[:, ids].mean(1), target)[0]
                draws[name].append(energy-value)
        for name, values in draws.items():
            intervals.append(dict(scenario=s, bound=name,
                ci_low=float(np.quantile(values, .025)), ci_high=float(np.quantile(values, .975))))
    write_rows(out/'summary.csv', summaries)
    write_rows(out/'paired_expectations.csv', paired)
    write_rows(out/'bootstrap_intervals.csv', intervals)
    report = dict(pooled=[r for r in summaries if r['seed'] == 0], intervals=intervals,
        empirical_expectations_only=True, no_retuning=True, no_global_optimality_claim=True,
        caveat='Posthoc finite-policy envelope and refitted plug-in bootstrap, not independent mixture deployment')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in old_hashes.items())
    assert all(digest(p) == h for p, h in hashes.items())
    (out/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in out.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
