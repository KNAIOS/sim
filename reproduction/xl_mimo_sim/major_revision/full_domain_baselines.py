"""Exact full-domain search calibration, preserving the old filtered-current forecast.

The original launch was stopped before its cost-search branch. The first
forecast of that adapter used raw last-observed CSI; pilot.cost_search instead
rescales it to the tracker's current filtered gain. This module preserves the
historical convention without editing the running learner's hashed sources.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainRollout, full_domain_episode
from major_revision.full_domain_study import initialize, verify_sources, MARGINS
from major_revision.pilot import SCENARIOS, trajectory


def cost_command(run, margin):
    _,score=run.base_observation()
    cfg,tracker=run.cfg,run.tracker
    gain,_,_=tracker.predict(cfg.horizon)
    gain=np.concatenate((tracker.gain_db[None],gain[:-1]))
    estimate=tracker.channel_estimate
    power=np.mean(abs(estimate.reshape(cfg.users,cfg.subarrays,
                                      cfg.antennas_per_subarray))**2,axis=2).T
    ratio=np.sqrt(10**(gain/10)/np.maximum(power[None],1e-20))
    forecasts=estimate[None]*np.repeat(ratio.transpose(0,2,1),cfg.antennas_per_subarray,axis=2)
    masks=[select_top(score,n) for n in range(run.minimum,cfg.subarrays+1)]+[run.hw.command.copy()]
    keys=[]
    for mask in masks:
        hardware=copy.deepcopy(run.hw)
        energy=outage=shortfall=0.
        for h in range(cfg.horizon):
            observed=int(hardware.active.sum())
            active,transition,_=hardware.apply(mask,1.)
            result=run.system.evaluate(forecasts[h],active,hardware.countdown>0,
                transition,False,run.req*margin,observed_count=observed)
            energy+=result.energy_j
            outage+=result.outage/cfg.horizon
            shortfall+=result.shortfall/cfg.horizon
            hardware.tick()
        keys.append((outage>cfg.outage_target,
                     energy if outage<=cfg.outage_target else shortfall,energy))
    return masks[min(range(len(masks)),key=lambda i:keys[i])]


def baseline_episode(cfg,tr,mode,margin,minimum=1):
    if not np.isfinite(margin) or margin<=0:
        raise ValueError('Positive finite margin required')
    if mode!='cost':
        return full_domain_episode(cfg,tr,mode=mode,margin=margin,minimum=minimum)
    run=FullDomainRollout(cfg,tr,minimum)
    while run.t<cfg.episode_steps:
        run.step(cost_command(run,margin) if run.t%4==0 else None)
    result=run.metrics()
    result['method']='cost'
    result['below_four_fraction']=float(np.mean(np.asarray(run.counts)<4))
    return result


def main():
    cfg,out,validation,files,manifest=initialize('baselines_v2')
    files += [Path(__file__),Path(__file__).with_name('FULL_DOMAIN_BASELINE_CORRECTION.md')]
    manifest['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest['supersedes']='baselines: interrupted before cost-search branch; no selected/development outputs'
    manifest['correction']='current cost forecast rescaled to filtered tracker gain, matching original cost_search'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    raw,means,selected=[],[],[]
    for mode in ('prs','cost'):
        for j in range(3):
            candidates=[]
            for margin in MARGINS:
                group=[dict(scenario=j,trajectory=i,margin=margin,
                    **baseline_episode(cfg,tr,mode,margin))
                    for scenario,i,tr in validation if scenario==j]
                raw.extend(group)
                mean=dict(scenario=j,mode=mode,margin=margin,
                    energy_j=float(np.mean([r['energy_j'] for r in group])),
                    outage=float(np.mean([r['outage'] for r in group])))
                candidates.append(mean)
                means.append(mean)
                _write_rows(out/'validation_raw.csv',raw)
                _write_rows(out/'validation_summary.csv',means)
                print('full-domain baseline v2',mode,j,margin,round(mean['outage'],5),flush=True)
            feasible=[r for r in candidates if r['outage']<=cfg.outage_target]
            chosen=min(feasible,key=lambda r:r['energy_j']) if feasible else min(
                candidates,key=lambda r:(r['outage'],r['energy_j']))
            selected.append(dict(**chosen,validation_feasible=bool(feasible)))
    _write_rows(out/'selected.csv',selected)
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(16):
            tr=trajectory(cfg,983000000+j*10000+i,sc)
            for setting in [r for r in selected if r['scenario']==j]+[dict(mode='full',margin=1.)]:
                result=baseline_episode(cfg,tr,setting['mode'],setting['margin'])
                rows.append(dict(scenario=j,trajectory=i,seed=0,margin=setting['margin'],**result))
            _write_rows(out/'development_raw.csv',rows)
        print('full-domain baseline v2 development',j,'complete',flush=True)
    verify_sources(files,manifest)
    (out/'COMPLETE').write_text('Corrected full-domain baselines finished.\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
