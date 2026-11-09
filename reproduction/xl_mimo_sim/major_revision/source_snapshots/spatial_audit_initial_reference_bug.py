"""Same-count spatial interventions and causal rule rollouts on fixed pilot traces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read
from major_revision.search_headroom import SearchAuditRollout
from major_revision.spatial_actions import alternative_command


def mask_bits(mask):
    return sum(1<<int(i) for i in np.flatnonzero(mask))


def finish_spatial(run,margin,mode):
    while run.t<run.cfg.episode_steps:
        if run.t%4:
            run.step()
        else:
            anchor,score=run.command(margin)
            command,_,_=alternative_command(run.cfg,run.tracker,run.hw,
                run.req*margin,score,anchor,mode)
            run.step(command)
    return run.metrics()


def run(mode):
    cfg=make_config('validation')
    root=Path(f'results/major_revision_20260920_spatial_actions/{mode}')
    root.mkdir(parents=True,exist_ok=False)
    baseline=Path('results/major_revision_20260920_constraint_screen/prs')
    selected=baseline/'selected.csv'
    reference_file=baseline/'development_raw.csv'
    margins={int(r['scenario']):float(r['margin']) for r in read(selected)}
    reference={(int(r['scenario']),int(r['trajectory'])):r for r in read(reference_file)}
    sources=[Path(__file__),Path(__file__).with_name('spatial_actions.py'),
        Path(__file__).with_name('audit_rollout.py'),Path(__file__).with_name('search_headroom.py'),
        Path(__file__).with_name('pilot.py'),*Path('future_vr').glob('*.py')]
    manifest=dict(mode=mode,config=cfg.as_dict(),margins=margins,trajectories_per_scenario=4,
        intervention_slots=[0,32,64,96],stream=973000000,
        scope='already-viewed diagnostic; 4 independent trajectories per scenario, not 16 independent intervention starts',
        selection='no checkpoint, parameter or favorable-start selection',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (selected,reference_file)})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pairs,online=[],[]
    replay_error=0.
    for j,sc in enumerate(SCENARIOS):
        margin=margins[j]
        for i in range(4):
            tr=trajectory(cfg,973000000+j*10000+i,sc)
            base=SearchAuditRollout(cfg,tr)
            factual=[]
            while base.t<cfg.episode_steps:
                if base.t%4:
                    base.step()
                    continue
                command,score=base.command(margin)
                if base.t in (0,32,64,96):
                    t=base.t
                    alternative,predicted_base,predicted_alt=alternative_command(
                        cfg,base.tracker,base.hw,base.req*margin,score,command,mode)
                    assert int(alternative.sum())==int(command.sum())
                    a,b=base.clone(),base.clone()
                    x,y=a.finish_search(margin,command),b.finish_search(margin,alternative)
                    factual.append(x)
                    de,dc=y['energy_j']-x['energy_j'],y['outage']-x['outage']
                    pairs.append(dict(scenario=j,trajectory=i,slot=t,mode=mode,
                        count=int(command.sum()),previous_mask=mask_bits(base.hw.command),
                        anchor_mask=mask_bits(command),alternative_mask=mask_bits(alternative),
                        changed_bits=int(np.count_nonzero(command!=alternative)),
                        baseline_energy_j=x['energy_j'],alternative_energy_j=y['energy_j'],
                        baseline_outage=x['outage'],alternative_outage=y['outage'],
                        energy_delta_j=de,outage_delta=dc,
                        realized_lower_energy_no_extra_outage=bool(de<-1e-8 and dc<=1e-12),
                        immediate_energy_delta_j=sum(b.slot_energy[t:t+4])-sum(a.slot_energy[t:t+4]),
                        immediate_outage_delta=float(np.mean(b.outages[t:t+4])-np.mean(a.outages[t:t+4])),
                        **{f'predicted_base_{k}':v for k,v in predicted_base.items()},
                        **{f'predicted_alt_{k}':v for k,v in predicted_alt.items()}))
                    if np.array_equal(command,alternative):
                        assert x==y
                base.step(command)
            actual=base.metrics()
            assert all(actual==x for x in factual)
            for key,value in actual.items():
                if key!='method':
                    error=abs(value-float(reference[j,i][key]))
                    replay_error=max(replay_error,error)
                    assert error<1e-9,(j,i,key,error)
            candidate=finish_spatial(SearchAuditRollout(cfg,tr),margin,mode)
            for name,metrics in (('PRS',actual),(f'Always {mode}',candidate)):
                row=dict(scenario=j,trajectory=i,mode=mode,**metrics)
                row['method']=name
                online.append(row)
            _write_rows(root/'pairs.csv',pairs)
            _write_rows(root/'online.csv',online)
            print('spatial audit',mode,j,i,'pairs',len(pairs),flush=True)
    summary=[]
    for j in range(3):
        group=[r for r in pairs if r['scenario']==j]
        alt=[r for r in online if r['scenario']==j and r['method']!= 'PRS']
        anchor=[r for r in online if r['scenario']==j and r['method']=='PRS']
        summary.append(dict(scenario=j,mode=mode,trajectories=4,correlated_starts=16,
            no_op_starts=sum(r['changed_bits']==0 for r in group),
            favorable_starts=sum(r['realized_lower_energy_no_extra_outage'] for r in group),
            intervention_energy_delta_j=float(np.mean([r['energy_delta_j'] for r in group])),
            intervention_outage_delta=float(np.mean([r['outage_delta'] for r in group])),
            online_energy_j=float(np.mean([r['energy_j'] for r in alt])),
            online_outage=float(np.mean([r['outage'] for r in alt])),
            prs_energy_j=float(np.mean([r['energy_j'] for r in anchor])),
            prs_outage=float(np.mean([r['outage'] for r in anchor]))))
    assert len(pairs)==len({(r['scenario'],r['trajectory'],r['slot']) for r in pairs})==48
    assert len(online)==24
    error=max(abs(r['energy_j']-r['operating_j']-r['pilot_j']-r['switching_j']) for r in online)
    assert error<1e-9
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in sources)
    _write_rows(root/'summary.csv',summary)
    (root/'audit.json').write_text(json.dumps(dict(paired_starts=48,online_rows=24,
        baseline_replay_max_error=replay_error,energy_identity_max_error_j=error,
        source_hashes='match',interpretation='causal heuristic diagnostic, not RL or independent confirmation'),indent=2))
    (root/'COMPLETE').write_text('All prespecified spatial interventions and complete heuristic rollouts finished.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('retain','exchange'),required=True)
    run(parser.parse_args().mode)
