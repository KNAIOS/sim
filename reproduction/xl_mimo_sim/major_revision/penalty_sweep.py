"""Development reliability-price sweep with matched trajectory streams.

Fixed prices diagnose the overconservative adaptive-dual pilot. They are not
a new methodological contribution. Validation chooses checkpoints; a fresh
development stream estimates the frontier without silently dropping failures.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS, episode, trajectory

PRICES=(.25,1.,4.)
MARGINS=(1.,1.05,1.1,1.2,1.35,1.5)


def objective(metrics,price,full_energy):
    return float(np.mean([r['energy_j']/full_energy+price*r['outage'] for r in metrics]))


def run(args):
    torch.set_num_threads(1)
    cfg=make_config('validation')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(prices=PRICES,margins=MARGINS,seeds=args.seeds,episodes=args.episodes,
        validation_per_scenario=args.trajectories,development_per_scenario=args.trajectories,
        scenarios=SCENARIOS,config=cfg.as_dict(),train_stream=951000000,
        validation_stream=952000000,development_stream=953000000,
        checkpoint_selection='minimum mean normalized energy + fixed price * outage on validation',
        search_selection='minimum mean energy subject to per-scenario validation mean outage <=0.05',
        status='development only; not confirmatory evidence or new algorithmic contribution',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),Path(__file__).with_name('pilot.py'),*Path('future_vr').glob('*.py')]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validation=[(j,trajectory(cfg,952000000+j*10000+i,sc))
                for j,sc in enumerate(SCENARIOS) for i in range(args.trajectories)]
    full=episode(cfg,validation[0][1],'Full array')['energy_j']
    dim=4*cfg.subarrays+2*cfg.users+2+cfg.subarrays*cfg.users+cfg.subarrays+cfg.users
    models=[]
    history=[]
    val_rows=[]
    for price in PRICES:
        for seed in args.seeds:
            model=ConstrainedDQN(cfg,dim,seed)
            model.dual=price
            rng=np.random.default_rng(seed)
            best_value=float('inf')
            best=None
            selected=0
            for ep in range(args.episodes):
                sc=SCENARIOS[int(rng.integers(len(SCENARIOS)))]
                tr=trajectory(cfg,951000000+seed*10000+ep,sc)
                epsilon=.04+.76*np.exp(-4*ep/max(args.episodes-1,1))
                r=episode(cfg,tr,'Enriched-DQN',model,True,epsilon)
                assert model.dual==price
                history.append(dict(price=price,seed=seed,episode=ep+1,**r))
                if (ep+1)%20==0 or ep+1==args.episodes:
                    metrics=[episode(cfg,tr,'Enriched-DQN',model) for _,tr in validation]
                    value=objective(metrics,price,full)
                    for i,((j,_),r) in enumerate(zip(validation,metrics)):
                        val_rows.append(dict(price=price,seed=seed,episode=ep+1,scenario=j,
                                             trajectory=i%args.trajectories,**r))
                    if value<best_value:
                        best_value=value
                        best=copy.deepcopy(model.online.state_dict())
                        selected=ep+1
                    _write_rows(out/'training.csv',history)
                    _write_rows(out/'validation.csv',val_rows)
                    print('price',price,'seed',seed,'episode',ep+1,'objective',round(value,4),flush=True)
            model.online.load_state_dict(best)
            models.append((price,seed,model))
            torch.save(dict(online=best,state_dimension=dim,selected_episode=selected,
                            validation_objective=best_value,price=price),out/f'price{price}_seed{seed}.pt')
    # Give search its own validation tuning; these margins do not change true demand.
    baseline_rows=[]
    selections=[]
    for j,sc in enumerate(SCENARIOS):
        for method in ('PRS','Switch-aware PRS'):
            means=[]
            for margin in MARGINS:
                group=[]
                for i,(_,tr) in enumerate([v for v in validation if v[0]==j]):
                    r=episode(cfg,tr,method,rate_margin=margin)
                    group.append(r)
                    baseline_rows.append(dict(scenario=j,trajectory=i,margin=margin,**r))
                means.append(dict(scenario=j,method=method,margin=margin,
                                  energy_j=float(np.mean([r['energy_j'] for r in group])),
                                  outage=float(np.mean([r['outage'] for r in group]))))
            feasible=[r for r in means if r['outage']<=cfg.outage_target]
            best=min(feasible,key=lambda r:r['energy_j']) if feasible else min(means,key=lambda r:r['outage'])
            selections.append(dict(**best,validation_feasible=bool(feasible)))
            _write_rows(out/'baseline_validation.csv',baseline_rows)
            _write_rows(out/'baseline_selected.csv',selections)
            print('search selection',j,method,best['margin'],flush=True)
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(args.trajectories):
            tr=trajectory(cfg,953000000+j*10000+i,sc)
            for price,seed,model in models:
                r=episode(cfg,tr,'Enriched-DQN',model)
                rows.append(dict(scenario=j,trajectory=i,price=price,seed=seed,margin=1.,**r))
            for selection in [s for s in selections if s['scenario']==j]:
                r=episode(cfg,tr,selection['method'],rate_margin=selection['margin'])
                rows.append(dict(scenario=j,trajectory=i,price=0.,seed=0,margin=selection['margin'],**r))
            _write_rows(out/'development_raw.csv',rows)
            print('development',j,i+1,flush=True)
    summary=[]
    for j in range(len(SCENARIOS)):
        for method,price in [('Enriched-DQN',p) for p in PRICES]+[(m,0.) for m in ('PRS','Switch-aware PRS')]:
            group=[r for r in rows if r['scenario']==j and r['method']==method and r['price']==price]
            summary.append(dict(scenario=j,method=method,price=price,
                **{k:float(np.mean([r[k] for r in group])) for k in ('energy_j','outage','mean_count','switching_j')}))
    _write_rows(out/'summary.csv',summary)
    (out/'COMPLETE').write_text('Fixed-price development sweep completed.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22])
    p.add_argument('--trajectories',type=int,default=4)
    run(p.parse_args())
