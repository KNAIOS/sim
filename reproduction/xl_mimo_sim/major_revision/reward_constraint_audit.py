"""Audit fixed-price reward versus outage-constrained energy selection.

No retraining or test tuning. Replays the best nonzero validation checkpoint
from each residual seed, the anchor, and both fixed residuals on validation only.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.audit_rollout import AuditRollout
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.predictive_residual import ResidualDQN,FixedResidual,anchored_observation
from major_revision.ready_report import read

ROOT=Path('results/major_revision_20260920_reward_audit')
TRAINING=Path('results/major_revision_20260920_predictive_residual')


def discounted_metrics(run):
    cfg=run.cfg
    if len(run.slot_energy)%4:
        raise ValueError('Only complete decision blocks are supported')
    full=run.system.evaluate(run.tr.channel[0],np.ones(cfg.subarrays,bool),
        np.zeros(cfg.subarrays,bool),0.,False,run.tr.requirements_mbps[0],
        observed_count=cfg.subarrays).energy_j
    energy=np.asarray(run.slot_energy).reshape(-1,4).mean(1)/full
    outage=np.asarray(run.outages).reshape(-1,4).mean(1)
    weights=cfg.rl_gamma**np.arange(len(energy))
    eg=float(np.average(energy,weights=weights))
    cg=float(np.average(outage,weights=weights))
    return dict(discounted_energy_cost=eg,discounted_outage_cost=cg,
        discounted_price4_cost=eg+4*cg,undiscounted_energy_cost=float(energy.mean()),
        undiscounted_price4_cost=float(energy.mean()+4*outage.mean()))


def evaluate(cfg,tr,policy,margin):
    run=AuditRollout(cfg,tr)
    while run.t<cfg.episode_steps:
        if run.t%4==0:
            observation,score=anchored_observation(run,margin)
            _,count=policy.choose(observation,0.)
            run.step(select_top(score,count))
        else:
            run.step()
    return dict(**run.metrics(),**discounted_metrics(run))


def main():
    torch.set_num_threads(1)
    cfg=make_config('validation')
    ROOT.mkdir(parents=True,exist_ok=False)
    anchor_file=Path('results/major_revision_20260920_constraint_screen/prs/selected.csv')
    margins={int(r['scenario']):float(r['margin']) for r in read(anchor_file)}
    models={'PRS':FixedResidual(cfg,0),'Fixed residual -1':FixedResidual(cfg,-1),
            'Fixed residual +1':FixedResidual(cfg,1)}
    references={}
    checkpoints={}
    sources=[Path(__file__),Path(__file__).with_name('audit_rollout.py'),
             Path(__file__).with_name('predictive_residual.py'),*Path('future_vr').glob('*.py')]
    for seed in (11,22):
        folder=TRAINING/f'seed{seed}'
        options=[r for r in read(folder/'selection.csv') if int(r['episode'])>0]
        best=min(options,key=lambda r:(float(r['violation']),float(r['energy_j']),int(r['episode'])))
        ep=int(best['episode'])
        path=folder/f'checkpoint_{ep:03d}.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        policy=ResidualDQN(cfg,seed)
        policy.online.load_state_dict(checkpoint['online'])
        name=f'Learned seed{seed} ep{ep}'
        models[name]=policy
        checkpoints[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               criterion='best nonzero checkpoint under original feasibility-first rule')
        references[name]={(int(r['scenario']),int(r['trajectory'])):r
                         for r in read(folder/'validation.csv') if int(r['episode'])==ep}
        if seed==11:
            references['PRS']={(int(r['scenario']),int(r['trajectory'])):r
                              for r in read(folder/'validation.csv') if int(r['episode'])==0}
    manifest=dict(config=cfg.as_dict(),checkpoints=checkpoints,stream=972000000,
        per_scenario=12,price=4.,scenarios=SCENARIOS,margins=margins,
        scope='validation-only reward audit; no new method or policy selection',
        discount='normalized finite-window gamma=0.98 per four-slot decision block',
        limitation='does not include infinite-horizon continuation value or certify constraint satisfaction',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    max_error=0.
    for j,scenario in enumerate(SCENARIOS):
        for i in range(12):
            tr=trajectory(cfg,972000000+j*10000+i,scenario)
            for name,policy in models.items():
                r=evaluate(cfg,tr,policy,margins[j])
                r['method']=name
                if name in references:
                    for field in ('energy_j','outage','operating_j','switching_j','pilot_j','transitions','mean_count'):
                        error=abs(r[field]-float(references[name][j,i][field]))
                        max_error=max(max_error,error)
                        assert error<1e-9,(name,j,i,field,error)
                rows.append(dict(scenario=j,trajectory=i,**r))
            _write_rows(ROOT/'raw.csv',rows)
            print('reward audit',j,i+1,'/12',flush=True)
    fields=('energy_j','outage','discounted_energy_cost','discounted_outage_cost',
            'discounted_price4_cost','undiscounted_energy_cost','undiscounted_price4_cost')
    summaries=[]
    for scenario in (-1,0,1,2):
        for name in models:
            group=[r for r in rows if r['method']==name and (scenario==-1 or r['scenario']==scenario)]
            summaries.append(dict(scenario=scenario,method=name,episodes=len(group),
                **{field:float(np.mean([r[field] for r in group])) for field in fields}))
    _write_rows(ROOT/'summary.csv',summaries)
    global_rows={r['method']:r for r in summaries if r['scenario']==-1}
    crossings=[]
    for name,row in global_rows.items():
        if name=='PRS':
            continue
        de=row['discounted_energy_cost']-global_rows['PRS']['discounted_energy_cost']
        dc=row['discounted_outage_cost']-global_rows['PRS']['discounted_outage_cost']
        crossings.append(dict(method=name,energy_cost_delta=de,outage_cost_delta=dc,
                              break_even_price=-de/dc if abs(dc)>1e-12 else None,
                              price4_cost_delta=de+4*dc))
    _write_rows(ROOT/'price_crossings.csv',crossings)
    assert len(rows)==180
    (ROOT/'audit.json').write_text(json.dumps(dict(records=180,reference_max_error=max_error,
        split='existing validation only',training_runs=0,completion='audit, not new superiority evidence'),indent=2))
    (ROOT/'COMPLETE').write_text('Reward/constraint audit completed without retraining.\n')
    print(json.dumps(crossings,indent=2))


if __name__=='__main__':
    main()
