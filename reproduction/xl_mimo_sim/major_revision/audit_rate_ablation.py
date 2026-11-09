"""Exploratory crossed bootstrap for the two-seed/four-trajectory ablation."""
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    root=Path('results/major_revision_20260920_rate_ablation')
    ref=Path('results/major_revision_20260920_penalty_sweep')
    assert (root/'COMPLETE').exists() and (ref/'COMPLETE').exists()
    rows=read(root/'development_raw.csv')
    prior=read(ref/'development_raw.csv')
    keys=[(r['scenario'],r['trajectory'],r['seed']) for r in rows]
    assert len(keys)==len(set(keys))==24
    for r in rows:
        assert abs(float(r['energy_j'])-sum(float(r[k]) for k in ('operating_j','switching_j','pilot_j')))<1e-8
    rng=np.random.default_rng(954000000)
    reports=[]
    for j in range(3):
        a={(int(r['seed']),int(r['trajectory'])):r for r in rows if int(r['scenario'])==j}
        b={int(r['trajectory']):r for r in prior if int(r['scenario'])==j and r['method']=='Switch-aware PRS'}
        si=rng.integers(2,size=(10000,2))
        ti=rng.integers(4,size=(10000,4))
        report={'scenario':j}
        for name,field,paired in [('rl_outage','outage',False),('energy_difference_vs_cost_prs','energy_j',True),
                                  ('outage_difference_vs_cost_prs','outage',True)]:
            values=np.array([[float(a[s,i][field])-(float(b[i][field]) if paired else 0.) for i in range(4)]
                             for s in (11,22)])
            boot=values[si[:,:,None],ti[:,None,:]].mean((1,2))
            report[name]=dict(mean=float(values.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))
        reports.append(report)
    result=dict(unique_records=24,energy_accounting='PASS',
                caveat='exploratory only; two training seeds and four trajectories do not establish statistical certification',
                intervals=reports)
    (root/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
