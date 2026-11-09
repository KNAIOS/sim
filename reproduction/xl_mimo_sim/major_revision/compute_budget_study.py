"""Disjoint latency calibration and all-variant frozen-policy evaluation."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from future_vr.experiment import _method_command, _write_rows
from future_vr.config import make_config
from major_revision.audit_rollout import AuditRollout
from major_revision.budget_search import search, learned, stride_counts
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.power_policy_study import SEEDS, CHECKPOINTS, controllers, read

ROOT=Path('results/major_revision_20260922_compute_budget')
CAL_STREAM, EVAL_STREAM=1600000000,1610000000
# (public method name, seed, family, stride)
SPECS=[('RA-DQN',s,'rl',0) for s in SEEDS]
SPECS += [(f'Uniform-d{d}',0,'uniform',d) for d in (1,2,3,4,6,12)]
SPECS += [(f'Local-d{d}',0,'local',d) for d in (2,3,4,6)]
SPECS += [('Full array',0,'full',0)]
LEGACY=[('Legacy PRS',0,'legacy_prs',1),('Legacy RA-DQN',11,'legacy_rl',0)]


def decide(run,spec,models):
    _,seed,family,stride=spec
    if family=='rl':
        return learned(run.cfg,run.tracker,run.hw,run.req,run.last_outage,models[seed])
    if family=='full':
        return np.ones(run.cfg.subarrays,bool)
    if family.startswith('legacy'):
        method='RA-FVV' if family=='legacy_rl' else 'Predictive rate search'
        return _method_command(method,run.cfg,run.tracker,run.hw,run.req,models.get(seed),
            run.tr,run.t,np.zeros(run.cfg.subarrays,bool),1.,run.last_outage)[0]
    return search(run.cfg,run.tracker,run.hw,run.req,family,stride)


def initialize(phase):
    folder=ROOT/phase
    folder.mkdir(parents=True,exist_ok=False)
    paths=[*Path('future_vr').glob('*.py'),*Path('major_revision').glob('*.py'),
        Path('major_revision/COMPUTE_BUDGET_PLAN.md'),*[CHECKPOINTS/f'RA-FVV_{s}.pt' for s in SEEDS]]
    if phase=='evaluation':
        paths += [ROOT/'calibration/budget_selection.json',ROOT/'calibration/summary.csv']
    hashes={str(p):digest(p) for p in paths}
    manifest=dict(hashes=hashes,phase=phase,config=make_config('validation').as_dict(),
        specifications=SPECS,calibration_stream=CAL_STREAM,evaluation_stream=EVAL_STREAM,
        platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
        numpy=np.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads(),
        process_local_thread_settings={k:os.environ.get(k) for k in
            ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        training_updates=0,original_decision_period_ms=200,hard_deadline_enforced=False)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return folder,hashes


def timing_summary(rows):
    keys=sorted({(r['method'],int(r['controller_seed'])) for r in rows})
    result=[]
    for method,seed in keys:
        x=np.array([float(r['milliseconds']) for r in rows
                    if r['method']==method and int(r['controller_seed'])==seed])
        result.append(dict(method=method,controller_seed=seed,n=len(x),median_ms=float(np.median(x)),
            p95_ms=float(np.quantile(x,.95)),p99_ms=float(np.quantile(x,.99)),max_ms=float(x.max())))
    return result


def choose_budgets(summary):
    times={(r['method'],r['controller_seed']):r['p95_ms'] for r in summary}
    primary=max(times['RA-DQN',seed] for seed in SEEDS)
    budgets=[.25,.5,1.,2.,5.,10.,20.,200.,primary]
    selected=[]
    for b in sorted(set(budgets)):
        row=dict(budget_ms=b,primary=(b==primary),rl_eligible=primary<=b)
        for family in ('uniform','local'):
            choices=[spec for spec in SPECS if spec[2]==family and times[spec[0],0]<=b]
            best=max(choices,key=lambda s:len(stride_counts(4,16,s[3]))) if choices else None
            row[family]=best[0] if best else None
        row['full_search_eligible']=times['Uniform-d1',0]<=b
        selected.append(row)
    return dict(primary_budget_ms=primary,definition='maximum original RL seed calibration p95',
                frozen_before_energy_evaluation=True,choices=selected)


def calibrate():
    folder,hashes=initialize('calibration')
    cfg=make_config('validation')
    models=controllers(cfg)
    specs=SPECS+LEGACY
    rows=[]
    equivalent=0
    for j,scenario in enumerate(SCENARIOS):
        for i in range(2):
            run=AuditRollout(cfg,trajectory(cfg,CAL_STREAM+j*10000+i,scenario))
            while run.t<cfg.episode_steps:
                if run.t%4:
                    run.step()
                    continue
                run.prepare()
                masks={}
                for spec in specs:
                    run.tracker._forecast_cache.clear()
                    masks[spec[0],spec[1]]=decide(run,spec,models)
                np.testing.assert_array_equal(masks['Uniform-d1',0],masks['Legacy PRS',0])
                np.testing.assert_array_equal(masks['RA-DQN',11],masks['Legacy RA-DQN',11])
                equivalent+=1
                for rep in range(3):
                    offset=(j*70+i*35+run.t//4+rep)%len(specs)
                    for spec in specs[offset:]+specs[:offset]:
                        run.tracker._forecast_cache.clear()
                        start=time.perf_counter_ns()
                        mask=decide(run,spec,models)
                        milliseconds=(time.perf_counter_ns()-start)/1e6
                        np.testing.assert_array_equal(mask,masks[spec[0],spec[1]])
                        rows.append(dict(scenario=j,trajectory=i,slot=run.t,repeat=rep,
                            method=spec[0],controller_seed=spec[1],milliseconds=milliseconds))
                driver=masks['Legacy PRS',0] if i==0 else masks['RA-DQN',11]
                run.step(driver)
            _write_rows(folder/'raw.csv',rows)
            print('budget calibration',j,i,'timings',len(rows),flush=True)
    assert equivalent==210 and len(rows)==210*3*len(specs)
    summary=timing_summary(rows)
    _write_rows(folder/'summary.csv',summary)
    selection=choose_budgets(summary)
    (folder/'budget_selection.json').write_text(json.dumps(selection,indent=2))
    (folder/'audit.json').write_text(json.dumps(dict(paired_states=equivalent,
        exact_legacy_vs_optimized_commands=True,timings=len(rows),outliers_deleted=0),indent=2))
    finish(folder,hashes)
    print(json.dumps(selection,indent=2),flush=True)


def performance_summary(rows):
    groups=sorted({(int(r['scenario']),r['method']) for r in rows})
    result=[]
    for scenario,method in groups:
        group=[r for r in rows if int(r['scenario'])==scenario and r['method']==method]
        result.append(dict(scenario=scenario,method=method,episodes=len(group),trajectories=8,
            energy_j=float(np.mean([float(r['energy_j']) for r in group])),
            outage=float(np.mean([float(r['outage']) for r in group])),
            mean_count=float(np.mean([float(r['mean_count']) for r in group]))))
    return result


def paired_summary(rows):
    lookup={(int(r['scenario']),int(r['trajectory']),r['method'],int(r['controller_seed'])):r for r in rows}
    rng=np.random.default_rng(20260924)
    sd=rng.integers(0,5,(10000,5)); td=rng.integers(0,8,(10000,8))
    result=[]
    for j in range(3):
        for method,seed,family,_ in SPECS:
            if seed or family=='full':
                continue
            for metric,scale in [('energy_j',1.),('outage',100.)]:
                difference=np.array([[scale*(float(lookup[j,i,'RA-DQN',s][metric])-
                    float(lookup[j,i,method,0][metric])) for i in range(8)] for s in SEEDS])
                boot=difference[sd[:,:,None],td[:,None,:]].mean((1,2))
                lo,hi=np.quantile(boot,[.025,.975])
                result.append(dict(scenario=j,baseline=method,metric='energy_gap_j' if metric=='energy_j' else 'outage_gap_pp',
                    mean=float(difference.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    return result


def evaluate():
    verify(ROOT/'calibration')
    folder,hashes=initialize('evaluation')
    cfg=make_config('validation')
    models=controllers(cfg)
    rows,slots,timings,traces=[],[],[],[]
    for j,scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr=trajectory(cfg,EVAL_STREAM+j*10000+i,scenario)
            trace_hash=hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            traces.append(dict(scenario=j,trajectory=i,sha256=trace_hash))
            offset=(j*8+i)%len(SPECS)
            for spec in SPECS[offset:]+SPECS[:offset]:
                run=AuditRollout(cfg,tr)
                run.prepare()
                decide(run,spec,models)  # same warm-start convention for every implementation
                while run.t<cfg.episode_steps:
                    run.prepare()
                    if run.t%4==0:
                        run.tracker._forecast_cache.clear()
                        start=time.perf_counter_ns()
                        mask=decide(run,spec,models)
                        ms=(time.perf_counter_ns()-start)/1e6
                        timings.append(dict(scenario=j,trajectory=i,slot=run.t,method=spec[0],
                            controller_seed=spec[1],milliseconds=ms))
                    else:
                        mask=run.hw.command.copy()
                    row=dict(scenario=j,trajectory=i,method=spec[0],controller_seed=spec[1],slot=run.t,
                             command_bits=sum(int(v)<<k for k,v in enumerate(mask)))
                    run.step(mask if run.t%4==0 else None)
                    row.update(energy_j=run.slot_energy[-1],outage=run.outages[-1],command_count=int(mask.sum()))
                    slots.append(row)
                rows.append(dict(scenario=j,trajectory=i,method=spec[0],controller_seed=spec[1],
                    energy_j=run.energy,outage=float(np.mean(run.outages)),
                    mean_count=float(np.mean(run.counts)),switching_j=run.switching,pilot_j=float(run.pilot)))
            assert trace_hash==hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            _write_rows(folder/'episodes.csv',rows)
            print('budget evaluation',j,i,'episodes',len(rows),flush=True)
    assert len(rows)==384 and len(slots)==53760 and len(timings)==13440
    for name,data in [('slots.csv',slots),('timing_raw.csv',timings),('trajectories.csv',traces),
        ('summary.csv',performance_summary(rows)),('timing_summary.csv',timing_summary(timings)),
        ('paired.csv',paired_summary(rows))]:
        _write_rows(folder/name,data)
    (folder/'audit.json').write_text(json.dumps(dict(episodes=len(rows),independent_trajectories=24,
        slots=len(slots),timings=len(timings),new_training=False,retuned_from_evaluation=False,
        scientific_completion=False,hard_deadline_enforced=False),indent=2))
    finish(folder,hashes)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['calibrate','evaluate'])
    args=parser.parse_args()
    if args.phase=='calibrate': calibrate()
    else: evaluate()
