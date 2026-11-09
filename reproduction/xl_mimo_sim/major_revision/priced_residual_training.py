"""Isolated fixed-price calibration; physical model and residual network unchanged."""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.constraint_training import preserved_exploration_rng,selection_key
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.predictive_residual import ResidualDQN,residual_episode
from major_revision.ready_report import read


def fit(seed,price):
    cfg=replace(make_config('validation'),train_episodes=300)
    extended=replace(cfg,episode_steps=cfg.episode_steps+1)
    out=Path(f'results/major_revision_20260920_price_calibration/price{price}_seed{seed}')
    out.mkdir(parents=True,exist_ok=False)
    anchor_file=Path('results/major_revision_20260920_constraint_screen/prs/selected.csv')
    anchors={int(r['scenario']):r for r in read(anchor_file)}
    margins={j:float(r['margin']) for j,r in anchors.items()}
    model=ResidualDQN(cfg,seed)
    model.dual=float(price)
    sources=[Path(__file__).with_name(name) for name in ('priced_residual_training.py',
        'predictive_residual.py','audit_rollout.py','pilot.py','rate_context.py',
        'ready_rate_context.py','constraint_training.py')]+list(Path('future_vr').glob('*.py'))
    manifest=dict(seed=seed,price=price,episodes=300,config=cfg.as_dict(),state_dimension=218,
        anchor_margins=margins,train_stream=971000000,validation_stream=972000000,
        development_stream=973000000,validation_per_scenario=12,development_per_scenario=16,
        checkpoint_episodes=[0,50,100,150,200,250,300],
        selection='worst-scenario mean-outage violation then mean energy; anchor wins ties',
        changed_from_price4='only fixed outage coefficient; same network, dynamics and sampling schedule',
        status='bounded development calibration; not final confirmation',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        anchor_hash=hashlib.sha256(anchor_file.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validation=[(j,i,trajectory(cfg,972000000+j*10000+i,sc))
                for j,sc in enumerate(SCENARIOS) for i in range(12)]
    validation_rows,history,selections=[],[],[]
    best_key=(float('inf'),float('inf'))
    chosen_episode=None

    def checkpoint(ep):
        nonlocal best_key,chosen_episode
        current=[]
        with preserved_exploration_rng(model):
            for j,i,tr in validation:
                current.append(dict(scenario=j,trajectory=i,
                    **residual_episode(cfg,tr,model,margins[j])))
        violation,energy,worst=selection_key(current)
        if ep==0:
            for j in range(3):
                group=[r for r in current if r['scenario']==j]
                for field in ('energy_j','outage'):
                    assert abs(np.mean([r[field] for r in group])-float(anchors[j][field]))<1e-9
        key=(violation,energy)
        saved=dict(online=copy.deepcopy(model.online.state_dict()),
            target=copy.deepcopy(model.target.state_dict()),seed=seed,price=price,episode=ep,
            validation_key=key,worst_validation_outage=worst,validation_feasible=violation==0.,
            state_dimension=218,anchor_margins=margins,bootstrap_truncation=True)
        torch.save(saved,out/f'checkpoint_{ep:03d}.pt')
        if key<best_key:
            best_key,chosen_episode=key,ep
            torch.save(saved,out/'selected.pt')
        validation_rows.extend(dict(seed=seed,price=price,episode=ep,**r) for r in current)
        selections.append(dict(episode=ep,violation=violation,energy_j=energy,
                               worst_outage=worst,selected_episode=chosen_episode))
        _write_rows(out/'validation.csv',validation_rows)
        _write_rows(out/'selection.csv',selections)
        print('priced validation',price,seed,ep,'worst',round(worst,5),
              'energy',round(energy,3),'selected',chosen_episode,flush=True)

    checkpoint(0)
    rng=np.random.default_rng(seed)
    for ep in range(300):
        j=int(rng.integers(3))
        tr=trajectory(extended,971000000+seed*10000+ep,SCENARIOS[j])
        epsilon=.04+.76*np.exp(-4*ep/99)
        result=residual_episode(cfg,tr,model,margins[j],True,epsilon)
        history.append(dict(seed=seed,price=price,episode=ep+1,scenario=j,epsilon=epsilon,**result))
        if (ep+1)%10==0:
            _write_rows(out/'training.csv',history)
            print('priced train',price,seed,ep+1,flush=True)
        if (ep+1)%50==0:
            checkpoint(ep+1)
    selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    model.online.load_state_dict(selected['online'])
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(16):
            tr=trajectory(cfg,973000000+j*10000+i,sc)
            result=residual_episode(cfg,tr,model,margins[j])
            result['method']=f'Residual-DQN price{price}'
            rows.append(dict(scenario=j,trajectory=i,seed=seed,margin=margins[j],**result))
            _write_rows(out/'development_raw.csv',rows)
        print('priced development',price,seed,j,'complete',flush=True)
    (out/'COMPLETE').write_text(f'Fixed price {price} completed; selected checkpoint {chosen_episode}.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(11,22),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    for price in (1,2):
        fit(args.seed,price)
