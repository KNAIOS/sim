"""Independent high-load check of the candidate's small apparent energy gain.

No retraining or checkpoint selection on the new data. Fine-grid search tuning
uses validation only. The result is conditional on two trained seeds, not a
population-wide assertion of training stability or a statistical certificate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import episode, trajectory
from major_revision.rate_context import candidate_rate_context

SCENARIO=(3.,1.15)
MARGINS=tuple(round(1+i*.01,2) for i in range(11))+(1.15,1.2)


def load_models(cfg):
    result=[]
    for method,folder,pattern,features in (
        ('Rate-context DQN','major_revision_20260920_rate_ablation','rate_seed*.pt',candidate_rate_context),
        ('Enriched-DQN','major_revision_20260920_penalty_sweep','price4.0_seed*.pt',None)):
        for path in sorted((Path('results')/folder).glob(pattern)):
            saved=torch.load(path,map_location='cpu',weights_only=True)
            seed=int(path.stem.split('seed')[-1])
            controller=ConstrainedDQN(cfg,saved['state_dimension'],seed)
            controller.online.load_state_dict(saved['online'])
            result.append((method,seed,controller,features,path))
    assert len(result)==4
    return result


def summarize(rows,out):
    rng=np.random.default_rng(957000000)
    ids=sorted({int(r['trajectory']) for r in rows})
    methods=sorted({r['method'] for r in rows})
    report=[]
    for method in methods:
        group=[r for r in rows if r['method']==method]
        seeds=sorted({int(r['seed']) for r in group})
        lookup={(int(r['seed']),int(r['trajectory'])):r for r in group}
        si=rng.integers(len(seeds),size=(10000,len(seeds)))
        ti=rng.integers(len(ids),size=(10000,len(ids)))
        result={'method':method}
        for metric in ('energy_j','outage'):
            data=np.array([[float(lookup[s,i][metric]) for i in ids] for s in seeds])
            boot=data[si[:,:,None],ti[:,None,:]].mean((1,2))
            result[metric]=dict(mean=float(data.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))
        report.append(result)
    # Pair identical channels and demand. Resample both trained seeds and traces.
    a={(int(r['seed']),int(r['trajectory'])):r for r in rows if r['method']=='Rate-context DQN'}
    pairs=[]
    for method in ('PRS','Switch-aware PRS'):
        b={int(r['trajectory']):r for r in rows if r['method']==method}
        item={'baseline':method}
        si=rng.integers(2,size=(10000,2))
        ti=rng.integers(len(ids),size=(10000,len(ids)))
        for metric in ('energy_j','outage'):
            data=np.array([[float(a[s,i][metric])-float(b[i][metric]) for i in ids] for s in (11,22)])
            boot=data[si[:,:,None],ti[:,None,:]].mean((1,2))
            item[metric]=dict(mean=float(data.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))
        pairs.append(item)
    (out/'summary.json').write_text(json.dumps(dict(methods=report,paired_rate_dqn_minus_baseline=pairs,
        caveat='conditional on two trained seeds; percentile intervals are exploratory, not simultaneous certification'),indent=2))


def run(args):
    torch.set_num_threads(1)
    cfg=make_config('validation')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    models=load_models(cfg)
    manifest=dict(scenario=SCENARIO,margins=MARGINS,validation_count=args.validation,
        test_count=args.test,validation_stream=955000000,test_stream=956000000,
        selection='minimum energy among margins with validation mean outage <=0.05; otherwise minimum outage',
        purpose='independent check of the apparent high-load energy advantage; frozen models, no test-driven tuning',
        config=cfg.as_dict(),checkpoints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for *_,p in models},
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),Path(__file__).with_name('pilot.py'),Path(__file__).with_name('rate_context.py')]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    val=[trajectory(cfg,955000000+i,SCENARIO) for i in range(args.validation)]
    valrows=[]
    valmeans=[]
    selected=[]
    for method in ('PRS','Switch-aware PRS'):
        means=[]
        for margin in MARGINS:
            group=[episode(cfg,tr,method,rate_margin=margin) for tr in val]
            valrows.extend(dict(trajectory=i,margin=margin,**r) for i,r in enumerate(group))
            mean=dict(method=method,margin=margin,energy_j=float(np.mean([r['energy_j'] for r in group])),
                      outage=float(np.mean([r['outage'] for r in group])))
            means.append(mean)
            valmeans.append(mean)
            _write_rows(out/'validation_raw.csv',valrows)
            _write_rows(out/'validation_summary.csv',valmeans)
            print('validation',method,margin,round(mean['outage'],4),flush=True)
        feasible=[r for r in means if r['outage']<=.05]
        best=min(feasible,key=lambda r:r['energy_j']) if feasible else min(means,key=lambda r:r['outage'])
        selected.append(dict(**best,validation_feasible=bool(feasible)))
        _write_rows(out/'selected.csv',selected)
    # The tuning choices are persisted before the independent test starts.
    rows=[]
    for i in range(args.test):
        tr=trajectory(cfg,956000000+i,SCENARIO)
        for method,seed,controller,features,_ in models:
            r=episode(cfg,tr,'Enriched-DQN',controller,state_features=features)
            r['method']=method
            rows.append(dict(trajectory=i,seed=seed,margin=1.,**r))
        for setting in selected:
            r=episode(cfg,tr,setting['method'],rate_margin=setting['margin'])
            rows.append(dict(trajectory=i,seed=0,margin=setting['margin'],**r))
        _write_rows(out/'test_raw.csv',rows)
        print('test',i+1,'/',args.test,flush=True)
    assert len(rows)==args.test*6
    assert len({(r['trajectory'],r['seed'],r['method']) for r in rows})==len(rows)
    for r in rows:
        assert abs(r['energy_j']-r['operating_j']-r['pilot_j']-r['switching_j'])<1e-7
    summarize(rows,out)
    (out/'COMPLETE').write_text('Independent fixed-candidate check finished; no tuning on test.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--validation',type=int,default=12)
    p.add_argument('--test',type=int,default=40)
    run(p.parse_args())
