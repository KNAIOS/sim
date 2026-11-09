"""Post-hoc diagnosis of the frozen four-slot physical prior; no new training.

This audit uses already-viewed development traces and is not a superiority test.
For adjacent ranked prefixes differing by a sleeping module, the first four
slots cannot receive that module's signal when wake_delay=4. Counterfactual
prediction differences are retained for every such pair, not only useful ones.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import hold_predictions,baseline_command,PRICE,full_slot_energy
from major_revision.action_value_study import ROOT
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read


def pair_checks(cfg,hardware,prediction,requirements):
    if cfg.wake_delay_slots<4:
        raise ValueError('This four-slot no-service identity needs delay >=4')
    masks=prediction['masks']
    rows=[]
    for a in range(cfg.subarrays-1):
        added=np.flatnonzero(masks[a+1]&~masks[a])
        assert len(added)==1
        q=int(added[0])
        if hardware.active[q] or hardware.countdown[q]>0:
            continue
        rate_delta=prediction['rates'][a+1]-prediction['rates'][a]
        energy_delta=prediction['energy'][a+1]-prediction['energy'][a]
        expected=cfg.wake_energy_j+4*cfg.slot_s*(cfg.waking_subarray_w-cfg.sleeping_subarray_w)
        assert np.max(np.abs(rate_delta[:4]))<1e-4
        assert abs(energy_delta[:4].sum()-expected)<1e-10
        outage=(prediction['rates'][[a,a+1]]<requirements[None,None,:])
        d4=float(outage[1,:4].mean()-outage[0,:4].mean())
        d8=float(outage[1].mean()-outage[0].mean())
        rows.append(dict(lower_count=a+1,added_subarray=q,
            first_block_rate_max_difference_mbps=float(np.max(np.abs(rate_delta[:4]))),
            first_block_energy_delta_j=float(energy_delta[:4].sum()),
            first_block_outage_delta=d4,
            second_block_outage_delta=float(outage[1,4:].mean()-outage[0,4:].mean()),
            first_block_cost_delta=float(energy_delta[:4].mean()/full_slot_energy(cfg)+PRICE*d4),
            eight_slot_cost_delta=float(energy_delta.mean()/full_slot_energy(cfg)+PRICE*d8)))
    return rows


def main():
    out=ROOT/'wake_horizon_audit'
    out.mkdir(exist_ok=False)
    cfg=make_config('validation')
    selected=read(ROOT/'baselines/selected.csv')
    files=[Path(__file__),Path(__file__).with_name('action_value.py'),
           Path(__file__).with_name('gram_control.py'),Path(__file__).with_name('pilot.py'),
           Path(__file__).with_name('audit_rollout.py'),*Path('future_vr').glob('*.py')]
    manifest=dict(role='post-hoc algebra/initial-prior diagnosis; viewed development traces',
        policies=['prior','prs'],trajectories_per_scenario=4,all_decisions=True,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows,states=[] ,[]
    for mode in ('prior','prs'):
        for j,sc in enumerate(SCENARIOS):
            setting=next(r for r in selected if int(r['scenario'])==j and r['method']=='prs')
            for i in range(4):
                tr=trajectory(cfg,998000000+j*10000+i,sc)
                run=FullDomainRollout(cfg,tr)
                while run.t<cfg.episode_steps:
                    if run.t%4:
                        run.step()
                        continue
                    _,score=run.base_observation()
                    prediction=hold_predictions(cfg,run.tracker,run.hw,score)
                    checked=pair_checks(cfg,run.hw,prediction,run.req)
                    rows.extend(dict(method=mode,scenario=j,trajectory=i,slot=run.t,**r) for r in checked)
                    states.append(dict(method=mode,scenario=j,trajectory=i,slot=run.t,pairs=len(checked)))
                    command=baseline_command(run,mode,float(setting['margin']),int(setting['minimum']))
                    run.step(command)
            print('wake-horizon diagnosis',mode,j,'complete',flush=True)
    _write_rows(out/'pairs.csv',rows)
    _write_rows(out/'states.csv',states)
    assert len(states)==840
    summary=[]
    for mode in ('prior','prs'):
        for j in range(3):
            group=[r for r in rows if r['method']==mode and r['scenario']==j]
            summary.append(dict(method=mode,scenario=j,states=140,adjacent_sleep_pairs=len(group),
                rate_identity_max_error=max([r['first_block_rate_max_difference_mbps'] for r in group],default=0.),
                first_block_energy_delta_j=float(np.mean([r['first_block_energy_delta_j'] for r in group])) if group else 0.,
                future_outage_improving_pairs=sum(r['second_block_outage_delta']<0 for r in group),
                eight_slot_cost_improving_pairs=sum(r['eight_slot_cost_delta']<0 for r in group)))
    _write_rows(out/'summary.csv',summary)
    audit=dict(states=len(states),pairs=len(rows),summary=summary,training_repeated=False,
        interpretation='4-slot prior penalizes isolated wakeups before any service; this does not prove why learned policies fail or that extending the horizon wins',
        data_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    for p in files:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)]
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    (out/'COMPLETE').write_text('Post-hoc frozen-prior diagnosis complete, no method change.\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
