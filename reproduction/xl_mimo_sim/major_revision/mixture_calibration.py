"""Validation-tuned, non-learning episode mixture with exact component integration."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.ready_report import read


def mixture_weight(e0,c0,e1,c1,target=.05):
    if not all(math.isfinite(x) for x in (e0,c0,e1,c1,target)):
        raise ValueError('Finite costs required')
    if min(e0,e1)<0 or not all(0<=x<=1 for x in (c0,c1,target)):
        raise ValueError('Invalid energy or outage')
    candidates=[0.,1.]
    if abs(c1-c0)>1e-15:
        boundary=(target-c0)/(c1-c0)
        if 0<=boundary<=1:
            candidates.append(boundary)
    energy=lambda p:(1-p)*e0+p*e1
    outage=lambda p:(1-p)*c0+p*c1
    feasible=[p for p in candidates if outage(p)<=target+1e-12]
    chosen=min(feasible,key=lambda p:(energy(p),p)) if feasible else \
        min(candidates,key=lambda p:(outage(p),energy(p),p))
    return chosen,bool(feasible)


def run():
    out=Path('results/major_revision_20260920_mixture_calibration')
    out.mkdir(parents=True,exist_ok=False)
    validation=Path('results/major_revision_20260920_reward_audit/raw.csv')
    val=read(validation)
    choices=[]
    for j in range(3):
        means={}
        for name in ('PRS','Fixed residual -1'):
            group=[r for r in val if int(r['scenario'])==j and r['method']==name]
            assert len(group)==12
            means[name]=tuple(float(np.mean([float(r[k]) for r in group])) for k in ('energy_j','outage'))
        e0,c0=means['PRS']
        e1,c1=means['Fixed residual -1']
        p,feasible=mixture_weight(e0,c0,e1,c1)
        choices.append(dict(scenario=j,decrement_probability=p,validation_feasible=feasible,
            validation_energy_j=(1-p)*e0+p*e1,validation_outage=(1-p)*c0+p*c1,
            prs_energy_j=e0,prs_outage=c0,decrement_energy_j=e1,decrement_outage=c1))
    # Lock the probabilities before opening any development component outcomes.
    _write_rows(out/'selected.csv',choices)
    (out/'calibration.json').write_text(json.dumps(dict(validation_sha256=
        hashlib.sha256(validation.read_bytes()).hexdigest(),choices=choices,
        rule='choose the whole-episode component before observations; minimize linear expected energy',
        reliability='validation mean constraint only; no probabilistic guarantee'),indent=2))
    anchor=Path('results/major_revision_20260920_constraint_screen/prs/development_raw.csv')
    decrement=Path('results/major_revision_20260920_predictive_residual/seed11/fixed_raw.csv')
    a={(int(r['scenario']),int(r['trajectory'])):r for r in read(anchor) if r['method']=='PRS'}
    b={(int(r['scenario']),int(r['trajectory'])):r for r in read(decrement) if r['method']=='Fixed residual -1'}
    assert a.keys()==b.keys() and len(a)==48
    fields=('energy_j','outage','operating_j','switching_j','pilot_j','transitions',
            'short_sleep_reactivations','mean_count')
    rows=[]
    for (j,i),first in sorted(a.items()):
        p=choices[j]['decrement_probability']
        rows.append(dict(scenario=j,trajectory=i,seed=0,method='Episode mixture',
            decrement_probability=p,measurement='analytically integrated expectation over episode choice',
            **{k:(1-p)*float(first[k])+p*float(b[j,i][k]) for k in fields}))
    _write_rows(out/'development_raw.csv',rows)
    summary=[dict(scenario=j,method='Episode mixture',**{
        k:float(np.mean([r[k] for r in rows if r['scenario']==j])) for k in fields}) for j in range(3)]
    _write_rows(out/'summary.csv',summary)
    error=max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in rows)
    assert error<1e-9
    (out/'audit.json').write_text(json.dumps(dict(records=48,energy_identity_max_error_j=error,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (validation,anchor,decrement)},
        interpretation='expected mixture performance; no new RL or independent stochastic seeds'),indent=2))
    (out/'COMPLETE').write_text('Non-learning mixture calibrated on validation and evaluated in expectation.\n')
    print(json.dumps(dict(selection=choices,development=summary),indent=2))


if __name__=='__main__':
    run()
