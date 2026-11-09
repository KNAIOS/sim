"""Frozen, bounded stage-B protocol; no development-driven tuning or resweeps."""
import argparse
import copy
import hashlib
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import ActionValueDQN,action_episode,PRICE
from major_revision.constraint_training import preserved_exploration_rng,selection_key
from major_revision.pilot import SCENARIOS,trajectory

ROOT=Path('results/major_revision_20260920_action_value')
MARGINS=(.90,1.,1.04,1.08,1.16,1.24)


def initialize(part):
    cfg=make_config('validation')
    out=ROOT/part
    out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('ACTION_VALUE_PLAN.md','action_value.py','gram_control.py','full_domain.py',
         'audit_rollout.py','pilot.py','spatial_actions.py','constraint_training.py')],
        *Path('future_vr').glob('*.py')]
    manifest=dict(config=cfg.as_dict(),part=part,price=PRICE,episodes=200,
        checkpoints=[0,50,100,150,200],training_stream=996000000,
        validation_stream=997000000,development_stream=998000000,
        validation_trajectories=12,development_trajectories=16,
        common_dimension=362,action_dimension=67,serialized_dimension=1434,
        role='bounded method development, not untouched final confirmation',
        process_id=os.getpid(),python=platform.python_version(),torch=str(torch.__version__),
        numpy=np.__version__,threads=1,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validation=[(j,i,trajectory(cfg,997000000+j*10000+i,sc))
                for j,sc in enumerate(SCENARIOS) for i in range(12)]
    return cfg,out,validation,files,manifest


def verify_sources(files,manifest):
    for path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['source_hashes'][str(path)],path


def choose_baseline(rows):
    feasible=[r for r in rows if r['outage']<=.05]
    if feasible:
        return min(feasible,key=lambda r:(r['energy_j'],r['minimum'],r['margin']))
    return min(rows,key=lambda r:(r['outage'],r['energy_j'],r['minimum'],r['margin']))


def baselines():
    cfg,out,validation,files,manifest=initialize('baselines')
    rows,means,selected=[],[],[]
    for mode in ('prs','cost'):
        for j in range(3):
            choices=[]
            for minimum in (1,4):
                for margin in MARGINS:
                    group=[dict(scenario=j,trajectory=i,minimum=minimum,margin=margin,
                        **action_episode(cfg,tr,mode=mode,minimum=minimum,margin=margin))
                        for scenario,i,tr in validation if scenario==j]
                    mean=dict(scenario=j,method=mode,minimum=minimum,margin=margin,
                        **{k:float(np.mean([r[k] for r in group])) for k in ('energy_j','outage')})
                    rows.extend(group)
                    means.append(mean)
                    choices.append(mean)
                    _write_rows(out/'validation_raw.csv',rows)
                    _write_rows(out/'validation_summary.csv',means)
            chosen=choose_baseline(choices)
            selected.append(chosen)
            print('action-value baseline',mode,j,'selected',chosen,flush=True)
    for mode in ('prior','full'):
        for j in range(3):
            group=[dict(scenario=j,trajectory=i,minimum=1,margin=1.,
                **action_episode(cfg,tr,mode=mode)) for scenario,i,tr in validation if scenario==j]
            rows.extend(group)
            means.append(dict(scenario=j,method=mode,minimum=1,margin=1.,
                **{k:float(np.mean([r[k] for r in group])) for k in ('energy_j','outage')}))
    assert len(rows)==936
    _write_rows(out/'validation_raw.csv',rows)
    _write_rows(out/'validation_summary.csv',means)
    _write_rows(out/'selected.csv',selected)
    # Freeze search choices before opening any development trajectory.
    development=[]
    for j,sc in enumerate(SCENARIOS):
        controls=[r for r in selected if r['scenario']==j]+[
            dict(method=m,margin=1.,minimum=1) for m in ('prior','full')]
        for i in range(16):
            tr=trajectory(cfg,998000000+j*10000+i,sc)
            for setting in controls:
                result=action_episode(cfg,tr,mode=setting['method'],margin=setting['margin'],minimum=setting['minimum'])
                development.append(dict(scenario=j,trajectory=i,seed=0,**result))
            _write_rows(out/'development.csv',development)
        print('action-value baseline development',j,'complete',flush=True)
    verify_sources(files,manifest)
    (out/'COMPLETE').write_text('All frozen baseline calibration and development records complete.\n')


def training(seed,architecture):
    cfg,out,validation,files,manifest=initialize(f'{architecture}_seed{seed}')
    model=ActionValueDQN(cfg,seed,architecture)
    manifest['parameters']=sum(p.numel() for p in model.online.parameters())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    history,valrows,choices=[],[],[]
    best=(float('inf'),float('inf'))
    def validate(episode):
        nonlocal best
        with preserved_exploration_rng(model):
            current=[dict(scenario=j,trajectory=i,**action_episode(cfg,tr,model)) for j,i,tr in validation]
        violation,energy,worst=selection_key(current)
        key=(violation,energy)
        checkpoint=dict(online=copy.deepcopy(model.online.state_dict()),
            target=copy.deepcopy(model.target.state_dict()),seed=seed,architecture=architecture,
            episode=int(episode),validation_key=key,worst_outage=worst,price=PRICE,
            serialized_dimension=1434,common_dimension=362,action_dimension=67,
            parameters=manifest['parameters'],updates=model.steps)
        torch.save(checkpoint,out/f'checkpoint_{episode:03d}.pt')
        if key<best:
            best=key
            torch.save(checkpoint,out/'selected.pt')
        valrows.extend(dict(seed=seed,episode=episode,**r) for r in current)
        choices.append(dict(episode=episode,violation=violation,energy_j=energy,worst_outage=worst,updates=model.steps))
        _write_rows(out/'validation.csv',valrows)
        _write_rows(out/'selection.csv',choices)
        print('action-value validation',architecture,seed,episode,'outage',round(worst,5),'energy',round(energy,3),flush=True)
    validate(0)
    rng=np.random.default_rng(seed)
    extended=replace(cfg,episode_steps=cfg.episode_steps+1)
    for ep in range(200):
        j=int(rng.integers(3))
        tr=trajectory(extended,996000000+seed*10000+ep,SCENARIOS[j])
        epsilon=.04+.76*np.exp(-4*ep/199)
        result=action_episode(cfg,tr,model,training=True,epsilon=epsilon)
        history.append(dict(seed=seed,episode=ep+1,scenario=j,epsilon=epsilon,**result))
        if (ep+1)%10==0:
            _write_rows(out/'training.csv',history)
            print('action-value training',architecture,seed,ep+1,flush=True)
        if (ep+1)%50==0:
            validate(ep+1)
    checkpoint=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    model.online.load_state_dict(checkpoint['online'])
    development=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(16):
            tr=trajectory(cfg,998000000+j*10000+i,sc)
            result=action_episode(cfg,tr,model)
            development.append(dict(seed=seed,scenario=j,trajectory=i,selected_episode=checkpoint['episode'],**result))
            _write_rows(out/'development.csv',development)
        print('action-value learned development',architecture,seed,j,'complete',flush=True)
    verify_sources(files,manifest)
    (out/'COMPLETE').write_text('Bounded training, all endpoints and development completed.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=('baselines','shared11','shared22','flat11','flat22'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.part=='baselines':
        baselines()
    else:
        architecture='shared' if args.part.startswith('shared') else 'flat'
        if architecture=='flat':
            assert (ROOT/'PERMIT_FLAT').exists(),'Shared-model validation gate must be audited first'
        training(int(args.part[-2:]),architecture)
