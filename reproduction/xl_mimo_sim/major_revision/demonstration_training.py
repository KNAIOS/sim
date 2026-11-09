"""Bounded behavioral cloning plus two own-state data-aggregation rounds.

This is imitation, not reinforcement learning. Classification logits must not
be reported as calibrated Q-values. All endpoints and failed gates are retained.
"""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.demonstration_data import settings,collect_episode,concatenate,source_files
from major_revision.full_domain import FullDomainDQN,full_domain_episode
from major_revision.nested_search import ROOT
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read


def agreement(model,data):
    with torch.no_grad():
        logits=model.online(torch.from_numpy(data['state']))
        targets=torch.as_tensor(data['expert_action'],dtype=torch.long)
        predictions=logits.argmax(1).numpy()
        loss=float(nn.functional.cross_entropy(logits,targets))
    return dict(cross_entropy=loss,accuracy=float(np.mean(predictions==data['expert_action'])),
        count_mae=float(np.mean(np.abs(predictions-data['expert_action']))))


def imitation_key(metrics,reference):
    excess=[]
    for j in range(3):
        group=[r for r in metrics if r['scenario']==j]
        excess.append(np.mean([r['outage'] for r in group])-reference[j]['outage'])
    return max(0.,max(excess)-.01),float(np.mean([r['energy_j'] for r in metrics]))


def fit(seed):
    assert (ROOT/'data/COMPLETE').exists()
    cfg=make_config('validation')
    extended=replace(cfg,episode_steps=cfg.episode_steps+1)
    teacher=settings()
    out=ROOT/f'imitation_seed{seed}'
    out.mkdir(exist_ok=False)
    files=source_files()+[Path(__file__)]
    inputs=[ROOT/'data/train.npz',ROOT/'data/validation.npz',ROOT/'data/expert_rollouts.csv',
            ROOT/'teacher/selected.csv']
    manifest=dict(seed=seed,config=cfg.as_dict(),teacher=teacher,
        stages=['teacher-only','aggregation-1','aggregation-2'],epochs=[100,50,50],
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        role='classification/own-state imitation; no TD loss or RL performance claim',
        validation_stream=992000000,development_stream=993000000,
        selection='worst-scenario excess outage above teacher beyond .01, then mean energy')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    train=dict(np.load(inputs[0],allow_pickle=False))
    validation=dict(np.load(inputs[1],allow_pickle=False))
    teacher_rows=[r for r in read(inputs[2]) if r['split']=='validation']
    reference={j:{k:float(np.mean([float(r[k]) for r in teacher_rows if int(r['scenario'])==j]))
                  for k in ('energy_j','outage')} for j in range(3)}
    val_tr=[(j,i,trajectory(extended,992000000+j*10000+i,sc))
            for j,sc in enumerate(SCENARIOS) for i in range(12)]
    model=FullDomainDQN(cfg,seed)
    rng=np.random.default_rng(seed)
    history,selections,valrows,aggregation_rows=[],[],[],[]
    best=(float('inf'),float('inf'))
    selected_stage=None
    for stage,epochs in enumerate((100,50,50)):
        if stage:
            additions=[]
            for j,sc in enumerate(SCENARIOS):
                for i in range(12):
                    stream=994000000+seed*100000+stage*10000+j*1000+i
                    tr=trajectory(extended,stream,sc)
                    arrays,result=collect_episode(cfg,tr,teacher[j],model)
                    arrays['scenario']=np.full(len(arrays['action']),j,dtype=np.int64)
                    arrays['trajectory']=np.full(len(arrays['action']),i,dtype=np.int64)
                    additions.append(arrays)
                    aggregation_rows.append(dict(stage=stage,scenario=j,trajectory=i,stream=stream,**result))
                print('imitation aggregation',seed,stage,j,flush=True)
            added=concatenate(additions)
            np.savez_compressed(out/f'aggregation_{stage}.npz',**added)
            train=concatenate([train,added])
            _write_rows(out/'aggregation_rollouts.csv',aggregation_rows)
        features=torch.from_numpy(train['state'])
        targets=torch.as_tensor(train['expert_action'],dtype=torch.long)
        for epoch in range(epochs):
            order=rng.permutation(len(features))
            for start in range(0,len(order),256):
                index=order[start:start+256]
                loss=nn.functional.cross_entropy(model.online(features[index]),targets[index])
                model.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.online.parameters(),5.)
                model.optimizer.step()
            if (epoch+1)%10==0:
                for split,data in (('train',train),('validation',validation)):
                    history.append(dict(seed=seed,stage=stage,epoch=epoch+1,split=split,
                        labels=len(data['action']),**agreement(model,data)))
                _write_rows(out/'classification.csv',history)
                print('imitation fit',seed,stage,epoch+1,'heldout agreement',
                      round(history[-1]['accuracy'],4),flush=True)
        current=[dict(scenario=j,trajectory=i,**full_domain_episode(cfg,tr,model)) for j,i,tr in val_tr]
        key=imitation_key(current,reference)
        agree=agreement(model,validation)
        by_scenario=[]
        for j in range(3):
            group=[r for r in current if r['scenario']==j]
            by_scenario.append(dict(scenario=j,
                energy_ratio=float(np.mean([r['energy_j'] for r in group]))/reference[j]['energy_j'],
                outage_excess=float(np.mean([r['outage'] for r in group]))-reference[j]['outage']))
        gate=agree['accuracy']>=.90 and all(r['energy_ratio']<=1.10 and r['outage_excess']<=.01
                                          for r in by_scenario)
        saved=dict(online=copy.deepcopy(model.online.state_dict()),seed=seed,stage=stage,
            state_dimension=235,actions=16,role='imitation logits, not TD-trained Q-values',
            validation_key=key,agreement=agree,by_scenario=by_scenario,initialization_gate=bool(gate))
        torch.save(saved,out/f'stage{stage}.pt')
        if key<best:
            best,selected_stage=key,stage
            torch.save(saved,out/'selected.pt')
        selections.append(dict(stage=stage,violation=key[0],energy_j=key[1],
            accuracy=agree['accuracy'],initialization_gate=gate,selected_stage=selected_stage))
        valrows.extend(dict(seed=seed,stage=stage,**r) for r in current)
        _write_rows(out/'validation.csv',valrows)
        _write_rows(out/'selection.csv',selections)
        print('imitation endpoint',seed,stage,'gate',gate,'selected',selected_stage,flush=True)
    # Development is first inspected after the complete stage selection.
    models=[]
    for name,path in (('Teacher-only imitation',out/'stage0.pt'),('Selected imitation',out/'selected.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        policy=FullDomainDQN(cfg,seed)
        policy.online.load_state_dict(saved['online'])
        models.append((name,saved['stage'],policy))
    dev=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(16):
            tr=trajectory(cfg,993000000+j*10000+i,sc)
            for name,stage,policy in models:
                result=full_domain_episode(cfg,tr,policy)
                result['method']=name
                dev.append(dict(seed=seed,scenario=j,trajectory=i,stage=stage,**result))
            _write_rows(out/'development_raw.csv',dev)
        print('imitation development',seed,j,'complete',flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in files)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['input_hashes'][str(p)] for p in inputs)
    (out/'COMPLETE').write_text('Fixed imitation diagnostic finished; inspect initialization gates before any RL.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(11,22),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    fit(args.seed)
