"""Audit pilot records and report paired descriptive development differences."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main(directory):
    root=Path(directory)
    assert (root/'COMPLETE').exists(), 'Pilot is not complete'
    manifest=json.loads((root/'manifest.json').read_text())
    with (root/'development_raw.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    keys=[(r['scenario'],r['trajectory'],r['method'],r['seed']) for r in rows]
    assert len(keys)==len(set(keys))
    expected=len(manifest['scenarios'])*manifest['development_per_scenario']*(2*len(manifest['seeds'])+4)
    assert len(rows)==expected,(len(rows),expected)
    for r in rows:
        assert abs(float(r['energy_j'])-sum(float(r[k]) for k in ('operating_j','switching_j','pilot_j')))<1e-7
        assert 0<=float(r['outage'])<=1
    rng=np.random.default_rng(924000000)
    comparisons=[]
    for sc in range(len(manifest['scenarios'])):
        subset=[r for r in rows if int(r['scenario'])==sc]
        ids=sorted({int(r['trajectory']) for r in subset})
        lookup={(r['method'],int(r['seed']),int(r['trajectory'])):r for r in subset}
        for baseline in ('Original-DQN','PRS','Hysteresis PRS','Switch-aware PRS'):
            row=dict(scenario=sc,comparison='Enriched-DQN minus '+baseline)
            for metric in ('energy_j','outage'):
                a=np.array([[float(lookup['Enriched-DQN',seed,i][metric])-
                             float(lookup[baseline,seed if baseline=='Original-DQN' else 0,i][metric])
                             for i in ids] for seed in manifest['seeds']])
                si=rng.integers(a.shape[0],size=(2000,a.shape[0]))
                ti=rng.integers(a.shape[1],size=(2000,a.shape[1]))
                boot=a[si[:,:,None],ti[:,None,:]].mean((1,2))
                row[metric]=dict(mean=float(a.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))
            comparisons.append(row)
    report=dict(unique_records=len(rows),energy_accounting='PASS',
                caveat='Exploratory development only; two seeds/eight trajectories are not confirmatory evidence.',
                paired_differences=comparisons)
    (root/'audit_and_pairs.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory')
    main(p.parse_args().directory)
