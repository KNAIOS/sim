"""Prespecified two-seed pilot with exact PRS checkpoint-zero fallback."""
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
from major_revision.predictive_residual import ResidualDQN,FixedResidual,residual_episode
from major_revision.ready_report import read


def run(args):
    torch.set_num_threads(1)
    cfg = replace(make_config('validation'),train_episodes=300)
    assert cfg.episode_steps % 4 == 0
    extended = replace(cfg,episode_steps=cfg.episode_steps+1)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    anchor_file = Path('results/major_revision_20260920_constraint_screen/prs/selected.csv')
    anchors = {int(r['scenario']):r for r in read(anchor_file)}
    margins = {j:float(r['margin']) for j,r in anchors.items()}
    model = ResidualDQN(cfg,args.seed)
    sources = [Path(__file__).with_name(name) for name in (
        'residual_training.py','predictive_residual.py','audit_rollout.py','pilot.py',
        'rate_context.py','ready_rate_context.py','constraint_training.py')]
    sources += list(Path('future_vr').glob('*.py'))
    manifest = dict(seed=args.seed,config=cfg.as_dict(),price=4.,episodes=300,
        state_dimension=218,residual_actions=[-1,0,1],anchor_margins=margins,
        train_stream=971000000,validation_stream=972000000,development_stream=973000000,
        checkpoint_episodes=[0,50,100,150,200,250,300],validation_per_scenario=12,
        development_per_scenario=16,scenarios=SCENARIOS,
        selection='minimum worst-scenario mean-outage violation, then mean energy',
        epsilon='0.04 + 0.76 * exp(-4 * zero_based_episode / 99)',
        objective='continuing discounted control, real successor bootstrapping at sampled truncation',
        initialization='zero output weights; biases [-0.01,0,-0.01]; initial greedy policy exactly PRS',
        boundary='illegal residuals masked in exploration, policy and Bellman targets',
        status='development pilot only; checkpoint-zero selection is NOT learned improvement',
        parameters=sum(p.numel() for p in model.online.parameters()),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        anchor_hash=hashlib.sha256(anchor_file.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validation = [(j,i,trajectory(cfg,972000000+j*10000+i,scenario))
                  for j,scenario in enumerate(SCENARIOS) for i in range(12)]
    rows,history,selections = [],[],[]
    best_key = (float('inf'),float('inf'))
    selected = None

    def checkpoint(ep):
        nonlocal best_key,selected
        current=[]
        with preserved_exploration_rng(model):
            for j,i,tr in validation:
                current.append(dict(scenario=j,trajectory=i,
                    **residual_episode(cfg,tr,model,margins[j])))
        violation,energy,worst=selection_key(current)
        if ep==0:
            for j in range(3):
                group=[r for r in current if r['scenario']==j]
                for metric in ('energy_j','outage'):
                    assert abs(np.mean([r[metric] for r in group])-float(anchors[j][metric]))<1e-9
        key=(violation,energy)
        saved=dict(online=copy.deepcopy(model.online.state_dict()),
            target=copy.deepcopy(model.target.state_dict()),seed=args.seed,episode=ep,
            validation_key=key,worst_validation_outage=worst,validation_feasible=violation==0.,
            state_dimension=218,anchor_margins=margins,price=4.,bootstrap_truncation=True)
        torch.save(saved,out/f'checkpoint_{ep:03d}.pt')
        if key<best_key:
            best_key,selected=key,ep
            torch.save(saved,out/'selected.pt')
        rows.extend(dict(seed=args.seed,episode=ep,**r) for r in current)
        selections.append(dict(episode=ep,violation=violation,energy_j=energy,
                               worst_outage=worst,selected_episode=selected))
        _write_rows(out/'validation.csv',rows)
        _write_rows(out/'selection.csv',selections)
        print('residual validation',args.seed,ep,'worst',round(worst,5),
              'energy',round(energy,3),'selected',selected,flush=True)

    checkpoint(0)
    rng=np.random.default_rng(args.seed)
    for ep in range(300):
        j=int(rng.integers(3))
        tr=trajectory(extended,971000000+args.seed*10000+ep,SCENARIOS[j])
        epsilon=.04+.76*np.exp(-4*ep/99)
        result=residual_episode(cfg,tr,model,margins[j],True,epsilon)
        history.append(dict(seed=args.seed,episode=ep+1,scenario=j,epsilon=epsilon,**result))
        if (ep+1)%10==0:
            _write_rows(out/'training.csv',history)
            print('residual train',args.seed,ep+1,flush=True)
        if (ep+1)%50==0:
            checkpoint(ep+1)
    chosen=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    model.online.load_state_dict(chosen['online'])
    development=[]
    for j,scenario in enumerate(SCENARIOS):
        for i in range(16):
            tr=trajectory(cfg,973000000+j*10000+i,scenario)
            result=residual_episode(cfg,tr,model,margins[j])
            development.append(dict(scenario=j,trajectory=i,seed=args.seed,margin=margins[j],**result))
            _write_rows(out/'development_raw.csv',development)
        print('residual development',args.seed,j,'complete',flush=True)
    if args.seed==11:
        fixed=[]
        for delta in (-1,1):
            policy=FixedResidual(cfg,delta)
            for j,scenario in enumerate(SCENARIOS):
                for i in range(16):
                    tr=trajectory(cfg,973000000+j*10000+i,scenario)
                    result=residual_episode(cfg,tr,policy,margins[j])
                    result['method']=f'Fixed residual {delta:+d}'
                    fixed.append(dict(scenario=j,trajectory=i,seed=0,margin=margins[j],**result))
                    _write_rows(out/'fixed_raw.csv',fixed)
                print('fixed residual',delta,j,'complete',flush=True)
    (out/'COMPLETE').write_text(f'Bounded residual pilot complete; selected checkpoint {selected}.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(11,22),required=True)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
