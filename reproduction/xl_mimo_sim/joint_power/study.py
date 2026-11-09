"""Bounded two-seed joint-control study. No installs or frozen-source writes."""
import argparse
import copy
import csv
import hashlib
import json
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from future_vr.channel import generate_trajectory
from future_vr.config import make_config
from major_revision.budget_search import search as legacy_search
from joint_power.model import Environment, search_action

ROOT=Path('results/joint_power_20260922_phase1')
SCENARIOS=((.5,.85),(2.,1.),(3.,1.15))
SEEDS=(11,22)
TRAIN,VALIDATION,DEVELOPMENT=2209220000,2309220000,2409220000
PRICE=8.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_trace(cfg, seed, scenario):
    speed,load=scenario
    trace=generate_trajectory(cfg,seed,speed)
    return replace(trace,requirements_mbps=trace.requirements_mbps*load)


class ResidualNetwork(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, prior):
        return prior+self.net(features).squeeze(-1)


class Learner:
    def __init__(self, width, seed):
        torch.manual_seed(seed)
        self.rng=random.Random(seed)
        self.online=ResidualNetwork(width)
        self.target=copy.deepcopy(self.online)
        self.optimizer=torch.optim.AdamW(self.online.parameters(),lr=.0004,weight_decay=1e-5)
        self.memory=deque(maxlen=5000)
        self.updates=0

    def choose(self, c, epsilon=0.):
        if self.rng.random()<epsilon:
            return self.rng.randrange(len(c.masks))
        with torch.no_grad():
            value=self.online(torch.from_numpy(c.features),torch.from_numpy(c.prior))
        return int(value.argmin())

    def learn(self, state, action, cost, following, done):
        self.memory.append((state.features,state.prior,action,cost,following.features,following.prior,done))
        if len(self.memory)<64:
            return None
        batch=self.rng.sample(list(self.memory),32)
        x,prior,a,c,nx,nprior,done=map(np.asarray,zip(*batch))
        x,prior,nx,nprior=map(torch.from_numpy,(x,prior,nx,nprior))
        a=torch.tensor(a,dtype=torch.long)
        costs=torch.tensor(c,dtype=torch.float32)
        terminal=torch.tensor(done,dtype=torch.float32)
        pred=self.online(x,prior).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            next_actions=self.online(nx,nprior).argmin(1)
            continuation=self.target(nx,nprior).gather(1,next_actions[:,None]).squeeze(1)
            target=costs+.98*(1-terminal)*continuation
        loss=nn.functional.smooth_l1_loss(pred,target)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite RL loss; do not silently replace this seed')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),5.)
        self.optimizer.step()
        self.updates+=1
        if self.updates%100==0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())


def episode(cfg, trace, method, learner=None, epsilon=0., train=False):
    variable=method not in ('Full fixed','PRS fixed')
    env=Environment(cfg,trace,variable_power=variable)
    total_loss=[]
    c=env.candidates(PRICE) if method in ('Joint full','Joint coarse','Prior','Joint RL') else None
    while env.t<cfg.episode_steps:
        env.prepare()
        if method.startswith('Full'):
            mask=np.ones(cfg.subarrays,bool)
        elif method.startswith('PRS'):
            mask=legacy_search(cfg,env.tracker,env.hw,env.req)
        else:
            if method=='Joint RL':
                action=learner.choose(c,epsilon)
            elif method=='Prior':
                action=int(c.prior.argmin())
            else:
                action=search_action(c,stride=3 if method=='Joint coarse' else 1)
            mask=c.masks[action]
        energy,outage=env.block(mask)
        done=env.t==cfg.episode_steps
        if c is not None:
            following=c if done else env.candidates(PRICE)
            if train:
                loss=learner.learn(c,action,float(energy+PRICE*outage),following,done)
                if loss is not None:
                    total_loss.append(loss)
            c=following
    return env.metrics()|{'loss':float(np.mean(total_loss)) if total_loss else 0.},env.records


def validation(cfg, learner):
    values=[]
    for scenario,spec in enumerate(SCENARIOS):
        for i in range(4):
            trace=make_trace(cfg,VALIDATION+scenario*10000+i,spec)
            metrics,_=episode(cfg,trace,'Joint RL',learner)
            values.append(dict(scenario=scenario,trajectory=i,**metrics))
    worst=max(np.mean([r['outage'] for r in values if r['scenario']==j]) for j in range(3))
    energy=np.mean([r['energy_j'] for r in values])
    key=(max(0.,float(worst)-.05),float(energy),float(worst))
    return key,values


def train_seed(seed):
    torch.set_num_threads(1)
    cfg=make_config('validation')
    folder=ROOT/f'seed{seed}'
    folder.mkdir(exist_ok=False)
    sample=Environment(cfg,make_trace(cfg,TRAIN+seed*100000,SCENARIOS[1])).candidates()
    learner=Learner(sample.features.shape[1],seed)
    rng=np.random.default_rng(seed)
    best_key,initial=validation(cfg,learner)
    best=copy.deepcopy(learner.online.state_dict())
    selected_episode=0
    valid_rows=[dict(episode=0,**r) for r in initial]
    torch.save(dict(state=best,width=sample.features.shape[1],episode=0,price=PRICE),folder/'initial.pt')
    history=[]
    for ep in range(120):
        scenario=int(rng.integers(3))
        trace=make_trace(cfg,TRAIN+seed*100000+ep,SCENARIOS[scenario])
        epsilon=.04+.46*np.exp(-4*ep/119)
        metrics,_=episode(cfg,trace,'Joint RL',learner,epsilon,train=True)
        history.append(dict(episode=ep+1,scenario=scenario,epsilon=epsilon,updates=learner.updates,**metrics))
        if (ep+1)%30==0:
            key,values=validation(cfg,learner)
            valid_rows.extend(dict(episode=ep+1,**r) for r in values)
            if key<best_key:
                best_key,best,selected_episode=key,copy.deepcopy(learner.online.state_dict()),ep+1
            write_rows(folder/'history.csv',history)
            write_rows(folder/'validation.csv',valid_rows)
            print(f'seed {seed} ep {ep+1}: validation energy={key[1]:.3f} J, worst outage={key[2]:.5f}, selected={selected_episode}',flush=True)
    torch.save(dict(state=learner.online.state_dict(),width=sample.features.shape[1],episode=120,price=PRICE),folder/'last.pt')
    torch.save(dict(state=best,width=sample.features.shape[1],episode=selected_episode,price=PRICE),folder/'selected.pt')
    metadata=dict(seed=seed,selected_episode=selected_episode,validation_key=best_key,
        training_episodes=120,updates=learner.updates,feature_width=sample.features.shape[1],
        parameters=sum(p.numel() for p in learner.online.parameters()))
    (folder/'selection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def train():
    ROOT.mkdir(parents=True,exist_ok=False)
    files=[*Path('joint_power').glob('*.py'),Path('joint_power/PLAN.md'),
           *Path('future_vr').glob('*.py'),Path('major_revision/budget_search.py')]
    files += [Path('../ICECAI2026_4pages_20260922')/name for name in
              ('icecai2026_energy_rl.tex','output/pdf/clean.pdf','output/pdf/icecai2026_energy_rl.pdf')]
    hashes={str(p):digest(p) for p in files}
    manifest=dict(input_hashes=hashes,config=make_config('validation').as_dict(),seeds=SEEDS,
        streams=dict(train=TRAIN,validation=VALIDATION,development=DEVELOPMENT),
        stages='bounded development only',host_environment_changed=False)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        metadata=list(pool.map(train_seed,SEEDS))
    assert hashes=={str(p):digest(p) for p in files},'Inputs changed while training'
    (ROOT/'training_complete.json').write_text(json.dumps(metadata,indent=2)+'\n')


def evaluate():
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert all(digest(path)==value for path,value in manifest['input_hashes'].items())
    (ROOT/'evaluation').mkdir(exist_ok=False)
    cfg=make_config('validation')
    learners={}
    for seed in SEEDS:
        data=torch.load(ROOT/f'seed{seed}/selected.pt',weights_only=True)
        learner=Learner(data['width'],seed)
        learner.online.load_state_dict(data['state'])
        learners[seed]=learner
    policies=[(m,0) for m in ('Full fixed','Full power','PRS fixed','PRS power','Joint full','Joint coarse','Prior')]
    policies += [('Joint RL',seed) for seed in SEEDS]
    episodes,slots=[] ,[]
    for scenario,spec in enumerate(SCENARIOS):
        for i in range(4):
            trace_seed=DEVELOPMENT+scenario*10000+i
            trace=make_trace(cfg,trace_seed,spec)
            before=digest_bytes(trace.channel.tobytes()+trace.requirements_mbps.tobytes())
            for method,seed in policies:
                metrics,records=episode(cfg,trace,method,learners.get(seed))
                identity=dict(scenario=scenario,trajectory=i,trajectory_seed=trace_seed,method=method,seed=seed)
                episodes.append(dict(**identity,**metrics))
                slots.extend(dict(**identity,**r) for r in records)
            assert before==digest_bytes(trace.channel.tobytes()+trace.requirements_mbps.tobytes())
            print(f'development scenario={scenario}, trajectory={i}: {len(episodes)} episodes',flush=True)
    write_rows(ROOT/'evaluation/episodes.csv',episodes)
    write_rows(ROOT/'evaluation/slots.csv',slots)
    summary=[]
    for scenario in range(3):
        for method in dict.fromkeys(m for m,_ in policies):
            group=[r for r in episodes if r['scenario']==scenario and r['method']==method]
            summary.append(dict(scenario=scenario,method=method,trajectories=4,
                training_seeds=2 if method=='Joint RL' else 0,
                **{k:float(np.mean([r[k] for r in group])) for k in
                   ('energy_j','outage','circuit_j','pa_j','switching_j','pilot_j','mean_active','numerical_fallbacks')}))
    write_rows(ROOT/'evaluation/summary.csv',summary)
    gate=screen(episodes)
    (ROOT/'evaluation/screen.json').write_text(json.dumps(gate,indent=2)+'\n')
    assert all(digest(path)==value for path,value in manifest['input_hashes'].items())
    outputs={str(p):digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
    (ROOT/'output_hashes.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,screen=gate),indent=2),flush=True)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def screen(episodes):
    means=lambda scenario,method: float(np.mean([r['energy_j'] for r in episodes if r['scenario']==scenario and r['method']==method]))
    outage=[float(np.mean([r['outage'] for r in episodes if r['scenario']==j and r['method']=='Joint RL'])) for j in range(3)]
    search=means(1,'Joint full')
    saving=1-means(1,'Joint RL')/search
    seed_energy={str(seed):float(np.mean([r['energy_j'] for r in episodes if r['scenario']==1 and r['method']=='Joint RL' and r['seed']==seed])) for seed in SEEDS}
    selected=[json.loads((ROOT/f'seed{s}/selection.json').read_text())['selected_episode'] for s in SEEDS]
    passed=max(outage)<=.05 and saving>=.02 and all(e<=search for e in seed_energy.values()) and all(ep>0 for ep in selected)
    return dict(passed=bool(passed),rl_outage_by_scenario=outage,nominal_energy_saving_vs_full_search=saving,
        nominal_search_energy_j=search,nominal_rl_energy_by_seed=seed_energy,selected_episodes=selected,
        independent_confirmation=False,manuscript_replacement_authorized_by_result=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('train','evaluate'))
    args=parser.parse_args()
    train() if args.phase=='train' else evaluate()
