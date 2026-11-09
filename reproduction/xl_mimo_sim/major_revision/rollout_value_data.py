"""Paired Monte Carlo cost-to-go labels under a causal search continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.action_value import observation, baseline_command, full_slot_energy
from major_revision.action_value_report import numeric_rows
from major_revision.action_value_study import ROOT as OLD_ROOT
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS, trajectory

ROOT=Path('results/major_revision_20260921_rollout_value')
TIMES=tuple(range(0,113,16))
BRANCH_SLOTS=16


def branch_return(root, command, setting, slots=BRANCH_SLOTS):
    """Only label generation calls this; clone owns its observations and hardware."""
    if slots%4 or root.t+slots>root.cfg.episode_steps:
        raise ValueError('Each label needs complete four-slot blocks within the trajectory')
    trial=root.clone()
    start=trial.t
    blocks=[]
    while trial.t<start+slots:
        energy=trial.energy
        if trial.t==start:
            action=command
        else:
            action=baseline_command(trial,'cost',setting['margin'],setting['minimum'])
        trial.step(action)
        for _ in range(3):
            trial.step()
        blocks.append([(trial.energy-energy)/(4*full_slot_energy(root.cfg)),
                       float(np.mean(trial.outages[-4:]))])
    blocks=np.asarray(blocks)
    discounted=np.sum(blocks*root.cfg.rl_gamma**np.arange(len(blocks))[:,None],axis=0)
    return np.stack((blocks[0],discounted)),blocks


def collect_trace(cfg,tr,setting,times=TIMES):
    """Freeze all causal inputs first; evaluate future returns in separate clones."""
    run=FullDomainRollout(cfg,tr)
    snapshots,features,references,commands=[],[],[],[]
    while run.t<cfg.episode_steps:
        if run.t%4:
            run.step()
            continue
        if run.t in times:
            state,score=observation(run)
            command=baseline_command(run,'cost',setting['margin'],setting['minimum'])
            snapshots.append(run.clone())
            features.append(np.concatenate((state,np.array([setting['margin'],
                setting['minimum']/cfg.subarrays],dtype=np.float32))))
            references.append(int(command.sum())-1)
            commands.append(np.array([select_top(score,n) for n in range(1,cfg.subarrays+1)]))
        else:
            command=baseline_command(run,'cost',setting['margin'],setting['minimum'])
        run.step(command)
    costs=[]
    replay_error=0.
    for root,ref,masks in zip(snapshots,references,commands):
        original_state,_=observation(root)
        returns=[]
        for action,mask in enumerate(masks):
            value,blocks=branch_return(root,mask,setting)
            returns.append(value)
            if action==ref:
                expected=[]
                for offset in range(0,BRANCH_SLOTS,4):
                    t=root.t+offset
                    expected.append([sum(run.slot_energy[t:t+4])/(4*full_slot_energy(cfg)),
                                     float(np.mean(run.outages[t:t+4]))])
                error=float(np.max(np.abs(blocks-np.asarray(expected))))
                assert error<1e-10
                replay_error=max(replay_error,error)
        after,_=observation(root)
        np.testing.assert_array_equal(after,original_state)
        costs.append(returns)
    costs=np.asarray(costs)
    references=np.asarray(references,dtype=np.int64)
    difference=costs-costs[np.arange(len(references)),references][:,None]
    assert np.all(difference[np.arange(len(references)),references]==0.)
    assert np.all(np.isfinite(costs))
    metrics=run.metrics()
    metrics['method']='cost'
    return np.asarray(features),references,costs,difference,metrics,replay_error


def main(split):
    cfg=make_config('validation')
    old=json.loads((OLD_ROOT/'baselines/manifest.json').read_text())
    for path,digest in old['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    audit=json.loads((OLD_ROOT/'audit.json').read_text())
    for path,digest in audit['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    settings=[r for r in numeric_rows(OLD_ROOT/'baselines/selected.csv') if r['method']=='cost']
    folder=ROOT/split
    folder.mkdir(parents=True,exist_ok=False)
    stream,count=(1010000000,24) if split=='train' else (1020000000,12)
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('ROLLOUT_VALUE_PLAN.md','test_rollout_value_data.py','action_value.py','action_value_report.py',
         'action_value_study.py','gram_control.py','full_domain.py','audit_rollout.py','pilot.py',
         'spatial_actions.py','constraint_training.py')],*Path('future_vr').glob('*.py'),
         OLD_ROOT/'baselines/manifest.json',OLD_ROOT/'baselines/selected.csv',OLD_ROOT/'audit.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(config=cfg.as_dict(),split=split,
        stream=stream,trajectories_per_scenario=count,times=TIMES,branch_slots=BRANCH_SLOTS,
        return_axis=['C4','C16'],cost_axis=['normalized_energy','user_slot_outage'],
        threads=1,process_id=os.getpid(),numpy=np.__version__,source_hashes=hashes,
        role='single-sample counterfactual Monte Carlo labels; not online policy results'),indent=2))
    features,references,costs,differences,keys,episodes=[],[],[],[],[],[]
    error=0.
    for j,sc in enumerate(SCENARIOS):
        setting=next(r for r in settings if r['scenario']==j)
        for i in range(count):
            tr=trajectory(cfg,stream+j*10000+i,sc)
            x,ref,c,d,metrics,local=collect_trace(cfg,tr,setting)
            features.extend(x)
            references.extend(ref)
            costs.extend(c)
            differences.extend(d)
            keys.extend((j,i,t) for t in TIMES)
            episodes.append(dict(scenario=j,trajectory=i,**metrics))
            error=max(error,local)
            print('rollout labels',split,j,i+1,'/',count,flush=True)
    expected=3*count*len(TIMES)
    features=np.asarray(features)
    costs=np.asarray(costs)
    differences=np.asarray(differences)
    references=np.asarray(references,dtype=np.int64)
    assert features.shape==(expected,1436)
    assert costs.shape==differences.shape==(expected,16,2,2)
    assert len(set(keys))==expected
    np.savez_compressed(folder/'labels.npz',features=features,reference=references,
        costs=costs,difference=differences,keys=np.asarray(keys,dtype=np.int64))
    _write_rows(folder/'episodes.csv',episodes)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (folder/'audit.json').write_text(json.dumps(dict(root_states=expected,branches=expected*16,
        replay_error=error,hashes_match=True,baseline_differences_exactly_zero=True,
        data_hash=hashlib.sha256((folder/'labels.npz').read_bytes()).hexdigest()),indent=2))
    (folder/'COMPLETE').write_text('Label collection complete; no learned policy or superiority claim.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('train','validation'),required=True)
    main(parser.parse_args().split)
