"""Causal expert labels and real transitions; no future CSI enters a decision."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.nested_search import ROOT,teacher_command
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read


def settings():
    path=ROOT/'teacher'
    assert (path/'COMPLETE').exists()
    return {int(r['scenario']):dict(minimum=int(r['minimum']),margin=float(r['margin']))
            for r in read(path/'selected.csv')}


def collect_episode(cfg,tr,setting,student=None):
    if len(tr.channel)<=cfg.episode_steps:
        raise ValueError('A real successor observation is needed')
    run=FullDomainRollout(replace(cfg,episode_steps=cfg.episode_steps+1),tr,1)
    full=run.system.evaluate(tr.channel[0],np.ones(cfg.subarrays,bool),
        np.zeros(cfg.subarrays,bool),0.,False,tr.requirements_mbps[0],
        observed_count=cfg.subarrays).energy_j
    records=[]
    pending=None

    def complete(successor):
        obs,label,action,start,energy=pending
        n=run.t-start
        records.append(dict(state=obs,expert_action=label,action=action,next_state=successor,
            energy=np.float32((run.energy-energy)/(n*full)),
            outage=np.float32(np.mean(run.outages[start:run.t])),slot=start))

    while run.t<cfg.episode_steps:
        if run.t%4:
            run.step()
            continue
        obs,score=run.observation()
        if pending is not None:
            complete(obs)
        expert_mask=teacher_command(run,setting)
        label=int(expert_mask.sum())-1
        action=label if student is None else student.choose(obs,0.)[0]
        pending=(obs,label,action,run.t,run.energy)
        run.step(select_top(score,action+1))
    successor,_=run.observation()
    complete(successor)
    assert len(records)==cfg.episode_steps//4
    arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
    metrics=run.metrics()
    metrics['method']='expert' if student is None else 'student'
    return arrays,metrics


def concatenate(blocks):
    return {k:np.concatenate([b[k] for b in blocks],axis=0) for k in blocks[0]}


def source_files():
    return [Path(__file__),*[Path(__file__).with_name(n) for n in
        ('DEMONSTRATION_PLAN.md','nested_search.py','full_domain.py','audit_rollout.py',
         'pilot.py','spatial_actions.py')],*Path('future_vr').glob('*.py')]


def main():
    cfg=make_config('validation')
    extended=replace(cfg,episode_steps=cfg.episode_steps+1)
    teacher=settings()
    out=ROOT/'data'
    out.mkdir(exist_ok=False)
    files=source_files()
    manifest=dict(config=cfg.as_dict(),teacher=teacher,
        train_stream=991000000,validation_stream=992000000,training_per_scenario=32,
        validation_per_scenario=12,decision_labels_per_episode=35,
        next_observation='one extra causal sample only, never executed or charged',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        teacher_sha256=hashlib.sha256((ROOT/'teacher/selected.csv').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for split,stream,n in (('train',991000000,32),('validation',992000000,12)):
        blocks=[]
        for j,sc in enumerate(SCENARIOS):
            for i in range(n):
                tr=trajectory(extended,stream+j*10000+i,sc)
                arrays,result=collect_episode(cfg,tr,teacher[j])
                arrays['scenario']=np.full(len(arrays['action']),j,dtype=np.int64)
                arrays['trajectory']=np.full(len(arrays['action']),i,dtype=np.int64)
                blocks.append(arrays)
                rows.append(dict(split=split,scenario=j,trajectory=i,**result))
                _write_rows(out/'expert_rollouts.csv',rows)
            print('expert dataset',split,j,'trajectories',n,flush=True)
        combined=concatenate(blocks)
        assert len(combined['action'])==3*n*35
        np.savez_compressed(out/f'{split}.npz',**combined)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in files)
    (out/'COMPLETE').write_text('3360 training and 1260 held-out expert-state labels completed.\n')


if __name__=='__main__':
    main()
