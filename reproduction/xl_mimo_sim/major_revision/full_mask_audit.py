"""Exhaustive causal hold-mask audit; physical channels are label-only inputs."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from future_vr.controllers import select_top
from future_vr.system import XLSystem
from major_revision.action_value import hold_predictions, slot_energy, baseline_command
from major_revision.direct_policy_screen import read_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import subarray_grams, rates_from_gram
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260921_full_mask')
SOURCE = Path('results/major_revision_20260921_direct_policy/validation_baselines/raw.csv')
TIMES = (0,32,64,96)
MARGINS = (1.,1.08,1.04)
VARIANTS = ('reference','exchange','same_count','all_masks')
STREAM = 1130000000
BATCH = 2048


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_id(mask):
    return int(np.asarray(mask,dtype=np.uint32) @ (1 << np.arange(len(mask),dtype=np.uint32)))


def masks_from_ids(ids, q):
    return (np.asarray(ids,dtype=np.uint32)[:,None] & (1 << np.arange(q,dtype=np.uint32))) != 0


def causal_forecast(cfg, tracker):
    """Same filtered-current convention as the frozen cost-prefix search."""
    estimate = tracker.channel_estimate
    grams = subarray_grams(cfg,estimate)
    power = np.diagonal(grams,axis1=-2,axis2=-1).real/cfg.antennas_per_subarray
    future,_,_ = tracker.predict(cfg.horizon)
    gain = np.concatenate((tracker.gain_db[None],future[:-1]))
    scale = np.sqrt(10.**(gain/10.)/np.maximum(power[None],1e-20))
    channels = estimate[None]*np.repeat(scale.transpose(0,2,1),cfg.antennas_per_subarray,axis=2)
    forecasts = grams[None]*scale[..., :,None]*scale[...,None,:]
    return forecasts, channels


def predicted_batch(cfg, hardware, forecast_grams, requirements, masks):
    """Inputs contain no trajectory; return [energy,outage,shortfall] per mask."""
    masks = np.asarray(masks,dtype=bool)
    if masks.ndim != 2 or masks.shape[1] != cfg.subarrays or np.any(masks.sum(1)==0):
        raise ValueError('Only nonempty matching-width masks are admissible')
    if np.any(requirements<=0):
        raise ValueError('Positive prediction requirements required')
    active = np.broadcast_to(hardware.active,masks.shape).copy()
    countdown = np.broadcast_to(hardware.countdown,masks.shape).copy()
    energy = np.zeros(len(masks))
    failed = np.zeros(len(masks),dtype=np.int64)
    shortfall = np.zeros(len(masks))
    for h in range(cfg.horizon):
        observed = active.sum(1)
        off = (~masks)&(active|(countdown>0))
        on = masks&(~active)&(countdown==0)
        active[off] = False
        countdown[off] = 0
        countdown[on] = cfg.wake_delay_slots
        if cfg.wake_delay_slots == 0:
            active[on] = True
        na,nw = active.sum(1),(countdown>0).sum(1)
        transition = cfg.sleep_energy_j*off.sum(1)+cfg.wake_energy_j*on.sum(1)
        energy += slot_energy(cfg,na,nw,observed,transition)
        g = np.einsum('aq,qkl->akl',active,forecast_grams[h])
        rates = rates_from_gram(cfg,g,na*cfg.antennas_per_subarray)
        failed += np.sum(rates<requirements,axis=1)
        shortfall += np.sum(np.maximum(requirements-rates,0.)/requirements,axis=1)
        completing = countdown==1
        countdown[countdown>0] -= 1
        active[completing] = True
    return np.column_stack((energy,failed/(cfg.horizon*cfg.users),
                            shortfall/(cfg.horizon*cfg.users)))


def enumerate_costs(cfg, hardware, forecast_grams, requirements, batch=BATCH):
    if not 1 <= cfg.subarrays <= 16 or batch <= 0:
        raise ValueError('Bounded audit requires 1--16 subarrays and a positive batch')
    ids = np.arange(1,1 << cfg.subarrays,dtype=np.uint32)
    costs = np.empty((len(ids),3),dtype=np.float64)
    for start in range(0,len(ids),batch):
        values = ids[start:start+batch]
        costs[start:start+len(values)] = predicted_batch(cfg,hardware,forecast_grams,
            requirements,masks_from_ids(values,cfg.subarrays))
    assert np.isfinite(costs).all()
    return ids,costs


def choose_masks(cfg, ids, costs, reference):
    masks = masks_from_ids(ids,cfg.subarrays)
    changed = np.sum(masks!=reference[None],axis=1)
    same = masks.sum(1)==reference.sum()
    infeasible = costs[:,1]>cfg.outage_target
    primary = np.where(infeasible,costs[:,2],costs[:,0])
    order = np.lexsort((ids,changed,costs[:,0],primary,infeasible))
    ref = mask_id(reference)
    assert np.array_equal(ids,np.arange(1,1 << cfg.subarrays))
    choices = [ref]
    for eligible in (same&(changed<=2),same,np.ones(len(ids),bool)):
        choices.append(int(ids[order[np.flatnonzero(eligible[order])[0]]]))
    return np.asarray(choices,dtype=np.uint32)


def scalar_hold(cfg, hardware, channels, requirements, command):
    """Independent original scalar hardware/precoder simulation of a held mask."""
    hw,system = copy.deepcopy(hardware),XLSystem(cfg)
    values = []
    for h,channel in enumerate(channels):
        observed = int(hw.active.sum())
        available,switch,_ = hw.apply(command,1.)
        result = system.evaluate(channel,available,hw.countdown>0,switch,False,
            requirements[h] if np.ndim(requirements)==2 else requirements,observed_count=observed)
        values.append([result.energy_j,result.outage,result.shortfall])
        hw.tick()
    array = np.asarray(values)
    return np.array([array[:,0].sum(),array[:,1].mean(),array[:,2].mean()])


def intervention(root, command, margin):
    run,start = root.clone(),root.t
    while run.t<run.cfg.episode_steps:
        if run.t%4:
            run.step()
        else:
            chosen = command if run.t==start else baseline_command(run,'cost',margin,1)
            run.step(chosen)
    return dict(energy_j=run.energy,outage=run.metrics()['outage'],
                suffix_energy_j=run.energy-root.energy,
                suffix_outage=float(np.mean(run.outages[start:])))


def initialize(part, extras=()):
    folder = ROOT/part
    folder.mkdir(parents=True,exist_ok=False)
    files = [*Path('major_revision').glob('*.py'),*Path('future_vr').glob('*.py'),
             Path(__file__).with_name('FULL_MASK_PLAN.md'),SOURCE,*map(Path,extras)]
    hashes = {str(p):digest(p) for p in files}
    manifest = dict(part=part,pid=os.getpid(),threads=1,config=make_config('validation').as_dict(),
        stream=STREAM,times=TIMES,margins=MARGINS,hashes=hashes,training_updates=0,
        independent_confirmation=False,role='causal single-intervention diagnostic on viewed data')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return folder,hashes


def finish(folder,hashes):
    for path,expected in hashes.items():
        assert digest(path)==expected,path
    outputs = {str(p):digest(p) for p in folder.rglob('*') if p.is_file()}
    (folder/'output_hashes.json').write_text(json.dumps(outputs,indent=2))
    (folder/'COMPLETE').write_text('Diagnostic stage complete, no learned superiority.\n')


def verify(folder):
    assert (folder/'COMPLETE').exists()
    for name,key in (('manifest.json','hashes'),('output_hashes.json',None)):
        hashes = json.loads((folder/name).read_text())
        if key:
            hashes = hashes[key]
        for path,expected in hashes.items():
            assert digest(path)==expected,path


def collect(index):
    folder,hashes = initialize(f'trajectory{index}')
    cfg,rows = make_config('validation'),[]
    old = read_rows(SOURCE)
    errors = dict(episode=0.,prefix=0.,scalar=0.,energy=0.,reference=0.)
    distinct = 0
    for j,scenario in enumerate(SCENARIOS):
        tr = trajectory(cfg,STREAM+j*10000+index,scenario)
        run,roots = FullDomainRollout(cfg,tr),[]
        while run.t<cfg.episode_steps:
            if run.t%4:
                run.step()
                continue
            command = baseline_command(run,'cost',MARGINS[j],1)
            if run.t in TIMES:
                roots.append((run.clone(),command.copy()))
            run.step(command)
        archived = next(r for r in old if r['scenario']==j and r['trajectory']==index and
            r['method']=='cost' and r['minimum']==1 and r['margin']==MARGINS[j])
        for key in ('energy_j','outage','switching_j','pilot_j','operating_j','transitions','mean_count'):
            errors['episode'] = max(errors['episode'],abs(run.metrics()[key]-archived[key]))
        for root,reference in roots:
            started = time.monotonic()
            before,score = root.base_observation()
            grams,channels = causal_forecast(cfg,root.tracker)
            ids,costs = enumerate_costs(cfg,root.hw,grams,root.req*MARGINS[j])
            selected = choose_masks(cfg,ids,costs,reference)
            # All choices are frozen here, before physical future labels.
            prefixes = hold_predictions(cfg,root.tracker,root.hw,score)
            pids = np.array([mask_id(mask) for mask in prefixes['masks']])
            req = root.req[None,None,:]*MARGINS[j]
            expected = np.column_stack((prefixes['energy'].sum(1),
                (prefixes['rates']<req).mean((1,2)),
                (np.maximum(req-prefixes['rates'],0.)/req).mean((1,2))))
            errors['prefix'] = max(errors['prefix'],float(np.max(np.abs(costs[pids-1]-expected))))
            labels = {}
            for identifier in sorted(set(map(int,selected))):
                mask = masks_from_ids([identifier],cfg.subarrays)[0]
                predicted = scalar_hold(cfg,root.hw,channels,root.req*MARGINS[j],mask)
                errors['scalar'] = max(errors['scalar'],float(np.max(np.abs(predicted-costs[identifier-1]))))
                held = scalar_hold(cfg,root.hw,tr.channel[root.t:root.t+cfg.horizon],
                                   tr.requirements_mbps[root.t:root.t+cfg.horizon],mask)
                errors['energy'] = max(errors['energy'],abs(held[0]-costs[identifier-1,0]))
                labels[identifier] = dict(hold_energy_j=held[0],hold_outage=held[1],hold_shortfall=held[2],
                                         **intervention(root,mask,MARGINS[j]))
                distinct += 1
            ref = labels[int(selected[0])]
            errors['reference'] = max(errors['reference'],abs(ref['energy_j']-run.energy),
                                      abs(ref['outage']-run.metrics()['outage']))
            for variant,identifier in zip(VARIANTS,selected):
                mask = masks_from_ids([identifier],cfg.subarrays)[0]
                pred = costs[int(identifier)-1]
                result = labels[int(identifier)]
                rows.append(dict(scenario=j,trajectory=index,t=root.t,variant=variant,mask_id=int(identifier),
                    reference_id=int(selected[0]),count=int(mask.sum()),reference_count=int(reference.sum()),
                    changed_bits=int(np.sum(mask!=reference)),pred_energy_j=pred[0],pred_outage=pred[1],
                    pred_shortfall=pred[2],energy_difference_j=result['energy_j']-ref['energy_j'],
                    outage_difference=result['outage']-ref['outage'],**result))
            np.testing.assert_array_equal(before,root.base_observation()[0])
            np.savez_compressed(folder/f'root_s{j}_t{root.t}.npz',ids=ids,costs=costs,selected_ids=selected,
                prefix_ids=pids,reference=reference,score=score,requirements=root.req,
                active=root.hw.active,countdown=root.hw.countdown,command=root.hw.command,
                channel_estimate=root.tracker.channel_estimate,gain_db=root.tracker.gain_db,
                velocity=root.tracker.velocity,probability=root.tracker.probability,
                uncertainty=root.tracker.uncertainty,age=root.age,features=before)
            _write_rows(folder/'raw.csv',rows)
            print('full-mask',index,j,root.t,'changed',sum(selected!=selected[0]),
                  'seconds',round(time.monotonic()-started,2),flush=True)
    assert len(rows)==48 and max(errors.values())<1e-8,errors
    (folder/'audit.json').write_text(json.dumps(dict(errors=errors,roots=12,masks_per_root=65535,
        candidate_evaluations=12*65535,distinct_interventions=distinct,training_updates=0,
        independent_confirmation=False),indent=2))
    finish(folder,hashes)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',type=int,choices=(0,1),required=True)
    torch.set_num_threads(1)
    collect(parser.parse_args().index)
