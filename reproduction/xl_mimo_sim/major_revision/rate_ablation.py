"""Bounded rate-context ablation against the completed price-4 enriched run.

Same train/validation/development streams and hyperparameters. Development
data have already been viewed; nothing here is an untouched final test.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS, episode, trajectory
from major_revision.penalty_sweep import objective
from major_revision.rate_context import candidate_rate_context


def run(args):
    torch.set_num_threads(1)
    cfg=make_config('validation')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    reference=Path('results/major_revision_20260920_penalty_sweep')
    assert (reference/'COMPLETE').exists()
    manifest=dict(price=4.,episodes=100,seeds=[11,22],state_dimension=178,
        reference=str(reference),train_stream=951000000,validation_stream=952000000,
        development_stream=953000000,trajectories_per_scenario=4,config=cfg.as_dict(),
        status='development ablation; previously viewed traces; unequal input parameter counts disclosed',
        extra_work='13 causal last-estimate RZF evaluations, no future-CSI access',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__),Path(__file__).with_name('pilot.py'),Path(__file__).with_name('rate_context.py'))})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validation=[(j,trajectory(cfg,952000000+j*10000+i,sc)) for j,sc in enumerate(SCENARIOS) for i in range(4)]
    full=episode(cfg,validation[0][1],'Full array')['energy_j']
    models=[]
    history=[]
    valrows=[]
    for seed in (11,22):
        controller=ConstrainedDQN(cfg,178,seed)
        controller.dual=4.
        rng=np.random.default_rng(seed)
        best_value=float('inf')
        for ep in range(100):
            sc=SCENARIOS[int(rng.integers(3))]
            tr=trajectory(cfg,951000000+seed*10000+ep,sc)
            r=episode(cfg,tr,'Enriched-DQN',controller,True,.04+.76*np.exp(-4*ep/99),
                      state_features=candidate_rate_context)
            history.append(dict(seed=seed,episode=ep+1,**r))
            if (ep+1)%20==0:
                metrics=[episode(cfg,tr,'Enriched-DQN',controller,state_features=candidate_rate_context)
                         for _,tr in validation]
                value=objective(metrics,4.,full)
                if value<best_value:
                    best_value=value
                    best=copy.deepcopy(controller.online.state_dict())
                    selected=ep+1
                for i,((j,_),r) in enumerate(zip(validation,metrics)):
                    valrows.append(dict(seed=seed,episode=ep+1,scenario=j,trajectory=i%4,**r))
                _write_rows(out/'training.csv',history)
                _write_rows(out/'validation.csv',valrows)
                print('Rate-context',seed,ep+1,round(value,4),flush=True)
        controller.online.load_state_dict(best)
        torch.save(dict(online=best,state_dimension=178,selected_episode=selected,price=4.,
                        validation_objective=best_value),out/f'rate_seed{seed}.pt')
        models.append((seed,controller))
    with (reference/'development_raw.csv').open(encoding='utf-8-sig') as f:
        prior=list(csv.DictReader(f))
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(4):
            tr=trajectory(cfg,953000000+j*10000+i,sc)
            for seed,controller in models:
                r=episode(cfg,tr,'Enriched-DQN',controller,state_features=candidate_rate_context)
                r['method']='Rate-context DQN'
                rows.append(dict(scenario=j,trajectory=i,seed=seed,**r))
    _write_rows(out/'development_raw.csv',rows)
    summary=[]
    for j in range(3):
        groups={'Rate-context DQN':[r for r in rows if r['scenario']==j],
                'Enriched-DQN price4':[r for r in prior if int(r['scenario'])==j and r['method']=='Enriched-DQN' and float(r['price'])==4.],
                'Tuned PRS':[r for r in prior if int(r['scenario'])==j and r['method']=='PRS'],
                'Tuned cost PRS':[r for r in prior if int(r['scenario'])==j and r['method']=='Switch-aware PRS']}
        for method,group in groups.items():
            summary.append(dict(scenario=j,method=method,**{k:float(np.mean([float(r[k]) for r in group]))
                for k in ('energy_j','outage','mean_count','switching_j')}))
    _write_rows(out/'summary.csv',summary)
    (out/'COMPLETE').write_text('Bounded development ablation complete.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    run(p.parse_args())
