"""Replay saved masks with original antenna-domain RZF, not batched Gram rates."""
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _prediction, _method_command
from major_revision.audit_rollout import AuditRollout
from major_revision.compute_budget_study import ROOT, EVAL_STREAM, SPECS
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.power_policy_study import controllers, read, SEEDS


def reference_counts(spec,current):
    _,_,family,stride=spec
    grid=list(range(4,17,stride))
    if grid[-1]!=16: grid.append(16)
    if family=='uniform': return grid
    count=len(grid)
    choices=[4,16]
    candidates=[current]
    for distance in range(1,13): candidates.extend([current-distance,current+distance])
    for value in candidates:
        if 4<=value<=16 and value not in choices: choices.append(value)
        if len(choices)==count: break
    return sorted(choices)


def reference_command(run,spec,models):
    _,seed,family,_=spec
    if family in ('rl','full'):
        return _method_command('RA-FVV' if family=='rl' else 'Full array',run.cfg,
            run.tracker,run.hw,run.req,models.get(seed),run.tr,run.t,
            np.zeros(16,bool),1.,run.last_outage)[0]
    cfg=run.cfg
    score,_=_prediction(cfg,run.tracker,run.hw,run.req)
    gain,_,_=run.tracker.predict(cfg.horizon)
    estimate=run.tracker.channel_estimate
    power=np.mean(abs(estimate.reshape(cfg.users,cfg.subarrays,cfg.antennas_per_subarray))**2,axis=2).T
    ratio=np.sqrt(10.**(gain/10.)/np.maximum(power[None],1e-20))
    forecast=estimate[None]*np.repeat(ratio.transpose(0,2,1),cfg.antennas_per_subarray,axis=2)
    candidates=[]
    for count in reference_counts(spec,int(run.hw.command.sum())):
        mask=select_top(score,count)
        rates=np.array([run.system.achievable_rates(forecast[h-1],mask&(run.hw.readiness()<=h))
                        for h in range(1,cfg.horizon+1)])
        outage=float(np.mean(rates<run.req))
        shortfall=float(np.mean(np.maximum(run.req-rates,0.)/run.req))
        candidates.append(((outage>cfg.outage_target,count if outage<=cfg.outage_target else shortfall,count),mask))
    return min(candidates,key=lambda pair:pair[0])[1]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for phase in ('calibration','evaluation'): verify(ROOT/phase)
    out=ROOT/'verified'
    out.mkdir(exist_ok=False)
    paths=[Path(__file__),*[p for phase in ('calibration','evaluation') for p in (ROOT/phase).glob('*') if p.is_file()]]
    hashes={str(p):digest(p) for p in paths}
    (out/'manifest.json').write_text(json.dumps(dict(hashes=hashes,batched_gram_not_used=True),indent=2))
    raw=read(ROOT/'evaluation/slots.csv')
    episodes=read(ROOT/'evaluation/episodes.csv')
    slotmap={(int(r['scenario']),int(r['trajectory']),r['method'],int(r['controller_seed']),int(r['slot'])):r for r in raw}
    epmap={(int(r['scenario']),int(r['trajectory']),r['method'],int(r['controller_seed'])):r for r in episodes}
    assert len(raw)==len(slotmap)==53760 and len(episodes)==len(epmap)==384
    cfg=make_config('validation'); models=controllers(cfg)
    error=0.; roots=slots=0
    for j,scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr=trajectory(cfg,EVAL_STREAM+j*10000+i,scenario)
            for spec in SPECS:
                run=AuditRollout(cfg,tr)
                key=(j,i,spec[0],spec[1])
                while run.t<cfg.episode_steps:
                    run.prepare()
                    if run.t%4==0:
                        mask=reference_command(run,spec,models)
                        roots+=1
                    else: mask=run.hw.command.copy()
                    row=slotmap[(*key,run.t)]
                    assert int(row['command_bits'])==sum(int(v)<<k for k,v in enumerate(mask)),(*key,run.t)
                    run.step(mask if run.t%4==0 else None)
                    assert int(row['command_count'])==int(mask.sum())
                    assert float(row['outage'])==run.outages[-1]
                    error=max(error,abs(float(row['energy_j'])-run.slot_energy[-1]))
                    slots+=1
                expected=epmap[key]
                for field,value in [('energy_j',run.energy),('outage',np.mean(run.outages)),
                    ('mean_count',np.mean(run.counts)),('switching_j',run.switching),('pilot_j',float(run.pilot))]:
                    error=max(error,abs(float(expected[field])-value))
            print('budget independent replay',j,i,flush=True)
    statistic_error=0.
    for row in read(ROOT/'evaluation/summary.csv'):
        group=[r for r in episodes if r['scenario']==row['scenario'] and r['method']==row['method']]
        assert len(group)==int(row['episodes'])
        for field in ('energy_j','outage','mean_count'):
            statistic_error=max(statistic_error,abs(np.mean([float(r[field]) for r in group])-float(row[field])))
    for phase,raw_name,summary_name in [('calibration','raw.csv','summary.csv'),
                                       ('evaluation','timing_raw.csv','timing_summary.csv')]:
        timing=read(ROOT/phase/raw_name)
        assert all(float(r['milliseconds'])>0 for r in timing)
        for row in read(ROOT/phase/summary_name):
            values=[float(r['milliseconds']) for r in timing if r['method']==row['method'] and r['controller_seed']==row['controller_seed']]
            assert len(values)==int(row['n'])
            for field,value in [('median_ms',np.median(values)),('p95_ms',np.quantile(values,.95)),
                                ('p99_ms',np.quantile(values,.99)),('max_ms',max(values))]:
                statistic_error=max(statistic_error,abs(value-float(row[field])))
    rng=np.random.default_rng(20260924)
    seeds=rng.integers(0,5,(10000,5)); traces=rng.integers(0,8,(10000,8))
    for row in read(ROOT/'evaluation/paired.csv'):
        j=int(row['scenario']); method=row['baseline']
        metric='energy_j' if row['metric']=='energy_gap_j' else 'outage'
        scale=1. if metric=='energy_j' else 100.
        x=np.array([[scale*(float(epmap[j,i,'RA-DQN',s][metric])-float(epmap[j,i,method,0][metric]))
                     for i in range(8)] for s in SEEDS])
        sample=x[seeds[:,:,None],traces[:,None,:]].mean((1,2))
        for field,value in zip(('mean','ci95_low','ci95_high'),(x.mean(),*np.quantile(sample,[.025,.975]))):
            statistic_error=max(statistic_error,abs(value-float(row[field])))
    assert error<1e-9 and statistic_error<1e-9 and roots==13440 and slots==53760
    result=dict(episodes=384,independent_trajectories=24,decision_masks_reproduced=roots,
        slot_records_reproduced=slots,maximum_replay_error=error,maximum_statistic_error=statistic_error,
        reference='original antenna-domain RZF and original RL decision path',
        artificial_noise_added=False,training_updates=0,hard_realtime_guarantee=False,
        timing_measurements_reproduced=False,timing_aggregates_recomputed=True,scientific_completion=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    finish(out,hashes)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
