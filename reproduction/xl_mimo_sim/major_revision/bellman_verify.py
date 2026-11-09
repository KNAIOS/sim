"""Recompute the new screen from raw CSV, without its selection/frontier helpers."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from major_revision.bellman_branch_data import ROOT, digest


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        result = list(csv.DictReader(handle))
    for row in result:
        for key, value in row.items():
            if key != 'method':
                row[key] = None if value == '' else float(value)
    return result


def summarize(raw, keys):
    groups = {}
    for row in raw:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in groups.items():
        assert len(group) == 8 and sorted(r['trajectory'] for r in group) == list(range(8))
        result.append(dict(zip(keys, key), energy_j=np.mean([r['energy_j'] for r in group]),
                           outage=np.mean([r['outage'] for r in group])))
    return result


def frontier_energy(points, limit):
    candidates = [p['energy_j'] for p in points if p['outage'] <= limit+1e-12]
    for p, q in itertools.combinations(points, 2):
        if (p['outage']-limit)*(q['outage']-limit) < 0:
            # Solve the two equality constraints rather than calling the screen's
            # mixture helper: weights sum to one and expected outage equals limit.
            weights = np.linalg.solve([[1., 1.], [p['outage'], q['outage']]], [1., limit])
            assert np.min(weights) > -1e-10 and abs(weights.sum()-1) < 1e-10
            candidates.append(float(weights @ [p['energy_j'], q['energy_j']]))
    return min(candidates) if candidates else None


def selected(points):
    feasible = [r for r in points if r['outage'] <= .05]
    key = (lambda r: (r['energy_j'], r['outage'], r['price'])) if feasible else (
        lambda r: (r['outage'], r['energy_j'], r['price']))
    return sorted(feasible or points, key=key)[0]


def main():
    audit_folder = ROOT/'screen_audit'
    assert (audit_folder/'COMPLETE').exists()
    source = json.loads((audit_folder/'audit.json').read_text())
    for path, expected in source['input_hashes'].items():
        assert digest(path) == expected, path
    raw_l = rows(ROOT/'screen_learned/raw.csv')
    raw_b = rows(ROOT/'screen_baselines/raw.csv')
    lm = summarize(raw_l, ('scenario', 'seed', 'method', 'price'))
    bm = summarize(raw_b, ('scenario', 'method', 'minimum', 'margin'))
    assert len(lm) == 78 and len(bm) == 147
    published = rows(audit_folder/'comparisons.csv')
    chosen = {(j, s): selected([r for r in lm if r['scenario'] == j and r['seed'] == s and r['method'] == 'bellman'])
              for j in range(3) for s in (11, 22)}
    checks, max_error = [], 0.
    for j in range(3):
        points = [r for r in bm if r['scenario'] == j]
        pure_points = [r for r in points if r['method'] != 'full' and r['outage'] <= .05]
        pure = min(pure_points, key=lambda r: (r['energy_j'], r['outage'])) if pure_points else None
        for seed in (11, 22):
            candidate = chosen[j, seed]
            frontier = frontier_energy(points, candidate['outage'])
            myopic = [r for r in lm if r['scenario'] == j and r['seed'] == seed and r['method'] == 'myopic']
            myopic += [r for r in points if r['method'] == 'full']
            myopic_front = frontier_energy(myopic, candidate['outage'])
            comparison = next(r for r in published if r['scenario'] == j and r['seed'] == seed)
            calculated = dict(energy_j=candidate['energy_j'], outage=candidate['outage'], price=candidate['price'],
                              matched_search_energy_j=frontier, matched_myopic_energy_j=myopic_front,
                              pure_energy_j=None if pure is None else pure['energy_j'],
                              pure_outage=None if pure is None else pure['outage'])
            for field, value in calculated.items():
                if value is None:
                    assert comparison[field] is None
                else:
                    error = abs(value-comparison[field])
                    max_error = max(max_error, error)
                    assert error < 1e-9
            valid = pure is not None and frontier is not None and myopic_front is not None
            passed = bool(valid and all(chosen[s, seed]['outage'] <= .05 for s in range(3)) and
                          candidate['energy_j'] <= .98*pure['energy_j'] and
                          candidate['outage'] <= pure['outage']+.005 and
                          candidate['energy_j'] <= .98*frontier and
                          candidate['energy_j'] < myopic_front)
            checks.append(dict(scenario=j, seed=seed, passed=passed))
    gate = json.loads((audit_folder/'gate.json').read_text())
    for check in checks:
        reference = next(g for g in gate['rows'] if g['scenario'] == check['scenario'] and g['seed'] == check['seed'])
        assert check['passed'] == reference['passed']
    permit = all(c['passed'] for c in checks if c['scenario'] == 1)
    assert permit == gate['permit_confirmation']
    assert permit == (audit_folder/'PERMIT_CONFIRMATION').exists()
    energy_error = max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in raw_l+raw_b)
    assert energy_error < 1e-9
    folder = ROOT/'screen_verified'
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(records=1800, external_trajectories=24, all_price_and_search_grids_complete=True,
                  independent_frontier_recalculation_error=max_error,
                  energy_accounting_error=energy_error, gate_reproduced=True,
                  permit_confirmation=permit, no_significance_claim=True,
                  raw_hashes=source['input_hashes'], verifier_hash=digest(__file__))
    (folder/'audit.json').write_text(json.dumps(report, indent=2))
    (folder/'COMPLETE').write_text('Raw-data recalculation passed; scientific gate remains as reported.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
