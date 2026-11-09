"""Post-hoc scenario price frontier; not a deployable scenario-oracle policy."""
import json

import numpy as np

from joint_power.aggregation.collect import ROOT
from joint_power.aggregation.control import input_checks, read_rows
from joint_power.study import digest, write_rows


def main():
    input_checks()
    source = ROOT/'control_validation'
    hashes = json.loads((source/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    data = read_rows(source/'candidate_episodes.csv')
    curves, relaxed = [], []
    for method in ('PRS-added', 'Visitation-added'):
        for seed in (11, 22):
            for scenario in range(3):
                group = []
                for price in sorted({r['parameter'] for r in data}):
                    rows = [r for r in data if (r['method'], r['seed'], r['scenario'], r['parameter'])
                            == (method, seed, scenario, price)]
                    row = dict(method=method, seed=seed, scenario=scenario, price=price,
                        energy_j=float(np.mean([r['energy_j'] for r in rows])),
                        outage=float(np.mean([r['outage'] for r in rows])))
                    group.append(row)
                    curves.append(row)
                feasible = [r for r in group if r['outage'] <= .05+1e-12]
                best = min(feasible or group, key=lambda r: (max(0., r['outage']-.05), r['energy_j'], r['price']))
                relaxed.append(best)
    nominal_at_one = [r for r in curves if r['method'] == 'Visitation-added' and r['scenario'] == 1 and r['price'] == 1.]
    high_at_one = [r for r in curves if r['method'] == 'Visitation-added' and r['scenario'] == 2 and r['price'] == 1.]
    old = read_rows(source/'selected_episodes.csv')
    prs = float(np.mean([r['energy_j'] for r in old if r['method'] == 'PRS original' and r['scenario'] == 1]))
    nominal_energy = float(np.mean([r['energy_j'] for r in nominal_at_one]))
    output = ROOT/'price_diagnosis'
    output.mkdir(exist_ok=False)
    write_rows(output/'all_scenario_prices.csv', curves)
    write_rows(output/'scenario_separate_choices.csv', relaxed)
    report = dict(post_hoc_validation_diagnostic=True, original_gate_preserved=True,
        formal_policy_changed=False, source_sha256=digest(__file__),
        nominal_at_price_one=nominal_at_one, high_at_price_one=high_at_one,
        nominal_mean_energy_price_one=nominal_energy, nominal_prs_energy=prs,
        nominal_saving_price_one=1-nominal_energy/prs,
        limitation='Scenario-specific hindsight parameter selection is a relaxation, not a causal '
        'adaptive controller or independent result. Fixed price1 is diagnostic, not a newly selected '
        'successful controller. Do not feed hidden speed/load labels at inference.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
