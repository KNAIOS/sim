"""Rebuild causal roots, independently select saved masks, replay, summarize."""
import json
import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from future_vr.system import XLSystem
from major_revision.action_value import baseline_command
from major_revision.direct_policy_screen import read_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.full_mask_audit import (ROOT,SOURCE,STREAM,TIMES,MARGINS,VARIANTS,
    masks_from_ids,initialize,finish,verify)
from major_revision.pilot import SCENARIOS,trajectory


def root_arrays(root):
    features,score=root.base_observation()
    return dict(score=score,requirements=root.req,active=root.hw.active,
        countdown=root.hw.countdown,command=root.hw.command,
        channel_estimate=root.tracker.channel_estimate,gain_db=root.tracker.gain_db,
        velocity=root.tracker.velocity,probability=root.tracker.probability,
        uncertainty=root.tracker.uncertainty,age=root.age,features=features)


def replay(root,mask,margin):
    run=root.clone()
    run.step(mask)
    for _ in range(3):
        run.step()
    while run.t<run.cfg.episode_steps:
        run.step(baseline_command(run,'cost',margin,1) if run.t%4==0 else None)
    return dict(energy_j=run.energy,outage=run.metrics()['outage'],
        suffix_energy_j=run.energy-root.energy,
        suffix_outage=float(np.mean(run.outages[root.t:])))


def true_hold_replay(root,mask):
    # Use the complete original rollout rather than the collector's hardware-
    # only scalar helper. Recompute rates before each step for shortfall.
    run=root.clone()
    values=[]
    for _ in range(root.cfg.horizon):
        run.prepare()
        before=run.energy
        available=run.hw.active&mask
        if root.cfg.wake_delay_slots==0:
            available=mask
        rates=run.system.achievable_rates(run.tr.channel[run.t],available)
        shortage=np.mean(np.maximum(run.req-rates,0.)/run.req)
        run.step(mask if run.t%4==0 else None)
        values.append([run.energy-before,run.outages[-1],shortage])
    values=np.asarray(values)
    return dict(hold_energy_j=values[:,0].sum(),hold_outage=values[:,1].mean(),
                hold_shortfall=values[:,2].mean())


def main():
    cfg=make_config('validation')
    raw=[]
    for index in (0,1):
        verify(ROOT/f'trajectory{index}')
        raw.extend(read_rows(ROOT/f'trajectory{index}/raw.csv'))
    assert len(raw)==96
    assert len({(r['scenario'],r['trajectory'],r['t'],r['variant']) for r in raw})==96
    errors=dict(replay=0.,hold=0.,root=0.,delta=0.)
    replayed=0
    for index in (0,1):
        for j,scenario in enumerate(SCENARIOS):
            run=FullDomainRollout(cfg,trajectory(cfg,STREAM+j*10000+index,scenario))
            while run.t<cfg.episode_steps:
                if run.t%4:
                    run.step()
                    continue
                reference=baseline_command(run,'cost',MARGINS[j],1)
                if run.t in TIMES:
                    path=ROOT/f'trajectory{index}/root_s{j}_t{run.t}.npz'
                    with np.load(path,allow_pickle=False) as z:
                        saved={k:z[k].copy() for k in z.files}
                    for key,value in root_arrays(run).items():
                        np.testing.assert_array_equal(value,saved[key])
                    np.testing.assert_array_equal(reference,saved['reference'])
                    ids,costs=saved['ids'],saved['costs']
                    assert costs.shape==(65535,3)
                    np.testing.assert_array_equal(ids,np.arange(1,65536))
                    masks=masks_from_ids(ids,cfg.subarrays)
                    changes=np.count_nonzero(masks!=reference[None],axis=1)
                    counts=masks.sum(1)
                    refid=sum(int(bit)*(2**q) for q,bit in enumerate(reference))
                    def key(n):
                        e,o,s=costs[n]
                        bad=o>cfg.outage_target
                        return (bool(bad),s if bad else e,e,int(changes[n]),int(ids[n]))
                    reconstructed=[refid]
                    restrictions=((counts==reference.sum())&(changes<=2),
                                  counts==reference.sum(),np.ones(len(ids),bool))
                    for eligible in restrictions:
                        selected=min(np.flatnonzero(eligible),key=key)
                        reconstructed.append(int(ids[selected]))
                    np.testing.assert_array_equal(reconstructed,saved['selected_ids'])
                    labels={}
                    for identifier in set(reconstructed):
                        mask=masks[identifier-1]
                        labels[identifier]={**replay(run,mask,MARGINS[j]),**true_hold_replay(run,mask)}
                        replayed+=1
                    ref=labels[refid]
                    for variant,identifier in zip(VARIANTS,reconstructed):
                        row=next(r for r in raw if (r['scenario'],r['trajectory'],r['t'],r['variant'])==
                                 (j,index,run.t,variant))
                        assert int(row['mask_id'])==identifier
                        assert int(row['changed_bits'])==int(changes[identifier-1])
                        for name,value in labels[identifier].items():
                            category='hold' if name.startswith('hold_') else 'replay'
                            errors[category]=max(errors[category],abs(row[name]-value))
                        for stored,metric in (('energy_difference_j','energy_j'),('outage_difference','outage')):
                            errors['delta']=max(errors['delta'],abs(row[stored]-(row[metric]-ref[metric])))
                run.step(reference)
    assert max(errors.values())<1e-9,errors
    summaries=[]
    for j in range(3):
        for variant in VARIANTS:
            group=[r for r in raw if r['scenario']==j and r['variant']==variant]
            assert len(group)==8
            changed=[r for r in group if r['changed_bits']>0]
            favorable=[r for r in changed if r['energy_difference_j']<-1e-9 and r['outage_difference']<=1e-12]
            summaries.append(dict(scenario=j,variant=variant,roots=8,changed=len(changed),
                changed_below_four=sum(r['reference_count']<4 for r in changed),
                lower_count=sum(r['count']<r['reference_count'] for r in group),
                predicted_feasible=sum(r['pred_outage']<=cfg.outage_target for r in group),
                favorable_changed=len(favorable),
                mean_energy_difference_j=float(np.mean([r['energy_difference_j'] for r in group])),
                mean_outage_difference_pp=100*float(np.mean([r['outage_difference'] for r in group])),
                mean_hold_outage=float(np.mean([r['hold_outage'] for r in group])),
                mean_pred_energy_j=float(np.mean([r['pred_energy_j'] for r in group]))))
    extras=[p for i in (0,1) for p in (ROOT/f'trajectory{i}').rglob('*') if p.is_file()]
    folder,hashes=initialize('verified',extras)
    _write_rows(folder/'all_roots.csv',raw)
    _write_rows(folder/'summary.csv',summaries)
    report=dict(errors=errors,roots=24,external_trajectories=6,candidate_evaluations=24*65535,
        selected_choice_records=96,distinct_interventions=replayed,
        all_masks_selected_causally=True,independent_confirmation=False,
        training_updates=0,scientific_completion=False)
    (folder/'audit.json').write_text(json.dumps(report,indent=2))
    finish(folder,hashes)
    print(json.dumps(dict(audit=report,summary=summaries),indent=2),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
