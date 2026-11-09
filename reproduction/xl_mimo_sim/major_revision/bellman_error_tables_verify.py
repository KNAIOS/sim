"""Check every aggregate against immutable arrays and verified root records."""
import csv
import json
from pathlib import Path

import numpy as np

from major_revision.bellman_branch_data import digest
from major_revision.bellman_error_audit import ROOT


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        values = list(csv.DictReader(handle))
    strings = {'dataset', 'selection'}
    return [{k: (v if k in strings else float(v)) for k, v in row.items()} for row in values]


def main():
    assert (ROOT/'verified/COMPLETE').exists()
    verification = json.loads((ROOT/'verified/audit.json').read_text())
    for path, expected in verification['inputs'].items():
        assert digest(path) == expected, path
    risk = read(ROOT/'summary/risk_prediction.csv')
    summary = read(ROOT/'summary/summary.csv')
    roots = read(ROOT/'summary/relative_errors.csv')
    oracle = read(ROOT/'summary/oracle_interventions.csv')
    clusters = read(ROOT/'summary/trajectory_clusters.csv')
    assert len(risk) == 60 and len(summary) == 12 and len(roots) == len(oracle) == 144
    maximum = 0.
    for seed in (11, 22):
        for dataset, name in (('train', 'training_reference.npz'), ('visited', 'roots.npz')):
            with np.load(ROOT/f'seed{seed}'/name, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            for rollin in (0, 1):
                for scenario in range(3):
                    mask = (arrays['keys'][:, 0] == rollin) & (arrays['keys'][:, 1] == scenario)
                    selected = {'all': None,
                        'learned': arrays['selected_action'][mask] if dataset == 'train' else arrays['actions'][mask, 0]}
                    if dataset == 'visited':
                        selected['search'] = arrays['actions'][mask, 1]
                    for selection, actions in selected.items():
                        predicted = arrays['predicted4'][mask, :, 1].astype(np.float64)
                        actual = arrays['actual4'][mask, :, 1].astype(np.float64)
                        if actions is not None:
                            predicted = predicted[np.arange(12), actions]
                            actual = actual[np.arange(12), actions]
                        difference = predicted-actual
                        row = next(r for r in risk if (r['seed'], r['dataset'], r['rollin'], r['scenario'], r['selection']) ==
                                   (seed, dataset, rollin, scenario, selection))
                        expected = dict(predicted=predicted.mean(), actual=actual.mean(), bias=difference.mean(),
                                        mae=np.abs(difference).mean(), rmse=np.sqrt(np.square(difference).mean()))
                        for field, value in expected.items():
                            maximum = max(maximum, abs(row[field]-value))
        for rollin in (0, 1):
            for scenario in range(3):
                group = [r for r in roots if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, scenario)]
                intervention = [r for r in oracle if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, scenario)]
                row = next(r for r in summary if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, scenario))
                counts = dict(roots=12, different_from_search=sum(r['action'] != r['search_action'] for r in group),
                    wrong_sign=sum(r['predicted_delta'] < -1e-7 and r['actual_delta'] > 1e-7 for r in group),
                    oracle_changes=sum(r['changed'] for r in intervention),
                    oracle_better=sum(r['changed'] and r['scalar_change'] < -1e-7 for r in intervention),
                    oracle_worse=sum(r['changed'] and r['scalar_change'] > 1e-7 for r in intervention))
                assert all(row[k] == v for k, v in counts.items())
                expected = {field: np.mean([r[field] for r in group]) for field in ('predicted_delta', 'actual_delta')}
                expected['oracle_scalar_change'] = np.mean([r['scalar_change'] for r in intervention])
                for field in ('immediate_energy_error', 'immediate_risk_error', 'tail_energy_error', 'tail_risk_error'):
                    expected[field] = np.mean([r[field] for r in group])
                    expected[field+'_mae'] = np.mean([abs(r[field]) for r in group])
                for field, value in expected.items():
                    maximum = max(maximum, abs(row[field]-value))
                for i in range(4):
                    local = [r for r in group if r['trajectory'] == i]
                    cluster = next(r for r in clusters if (r['seed'], r['rollin'], r['scenario'], r['trajectory']) ==
                                   (seed, rollin, scenario, i))
                    for field in ('predicted_delta', 'actual_delta'):
                        maximum = max(maximum, abs(cluster[field]-np.mean([r[field] for r in local])))
    assert maximum < 1e-12
    output = ROOT/'tables_verified'
    output.mkdir(parents=True, exist_ok=False)
    files = [*list((ROOT/'summary').glob('*.csv')), ROOT/'verified/audit.json',
             Path(__file__), Path(__file__).with_name('bellman_error_summary.py')]
    report = dict(risk_rows=60, summary_rows=12, root_records=144, trajectory_cluster_rows=len(clusters),
                  aggregate_recalculation_error=maximum,
                  source_hashes={str(p): digest(p) for p in files}, scientific_success=False)
    (output/'audit.json').write_text(json.dumps(report, indent=2))
    (output/'COMPLETE').write_text('All reported diagnostic aggregates checked; no performance claim.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
