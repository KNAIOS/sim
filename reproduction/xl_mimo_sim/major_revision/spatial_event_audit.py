"""Census every causal PRS decision; simulate every changed spatial alternative."""
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
from major_revision.spatial_audit import prs_reference,mask_bits


def run():
    cfg=make_config('validation')
    root=Path('results/major_revision_20260920_spatial_event_census')
    root.mkdir(parents=True,exist_ok=False)
    baseline=Path('results/major_revision_20260920_constraint_screen/prs')
    margins={int(r['scenario']):float(r['margin']) for r in read(baseline/'selected.csv')}
    reference=prs_reference(read(baseline/'development_raw.csv'))
    sources=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('spatial_actions.py','spatial_audit.py','audit_rollout.py','search_headroom.py','pilot.py')],
        *Path('future_vr').glob('*.py')]
    manifest=dict(config=cfg.as_dict(),margins=margins,stream=973000000,
        trajectories_per_scenario=4,slots=list(range(0,140,4)),
        amendment='all-epoch coverage added after sparse pilot no-ops; not prespecified confirmation',
        selection='all states and both causal alternatives retained; simulate every changed action',
        interpretation='same viewed development trajectories; no-op outcome is exact baseline reuse',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(4):
            base=SearchAuditRollout(cfg,trajectory(cfg,973000000+j*10000+i,sc))
            while base.t<cfg.episode_steps:
                if base.t%4:
                    base.step()
                    continue
                command,score=base.command(margins[j])
                for mode in ('retain','exchange'):
                    alternative,pb,pa=alternative_command(cfg,base.tracker,base.hw,
                        base.req*margins[j],score,command,mode)
                    changed=not np.array_equal(alternative,command)
                    outcome=base.clone().finish_search(margins[j],alternative) if changed else reference[j,i]
                    de=float(outcome['energy_j'])-float(reference[j,i]['energy_j'])
                    dc=float(outcome['outage'])-float(reference[j,i]['outage'])
                    rows.append(dict(scenario=j,trajectory=i,slot=base.t,mode=mode,
                        count=int(command.sum()),anchor_mask=mask_bits(command),
                        alternative_mask=mask_bits(alternative),changed=changed,
                        energy_delta_j=de,outage_delta=dc,
                        favorable=bool(de<-1e-8 and dc<=1e-12),
                        predicted_shortfall_delta=pa['shortfall']-pb['shortfall'],
                        predicted_switch_delta_j=pa['transition_j']-pb['transition_j']))
                base.step(command)
            for key,value in base.metrics().items():
                if key!='method':
                    assert abs(value-float(reference[j,i][key]))<1e-9
            _write_rows(root/'all_states.csv',rows)
            print('spatial census',j,i,'rows',len(rows),flush=True)
    assert len(rows)==len({(r['scenario'],r['trajectory'],r['slot'],r['mode']) for r in rows})==840
    summary=[]
    for j in range(3):
        for mode in ('retain','exchange'):
            group=[r for r in rows if r['scenario']==j and r['mode']==mode]
            changed=[r for r in group if r['changed']]
            summary.append(dict(scenario=j,mode=mode,trajectories=4,decision_states=len(group),
                changed_actions=len(changed),favorable_actions=sum(r['favorable'] for r in group),
                mean_all_state_energy_delta_j=float(np.mean([r['energy_delta_j'] for r in group])),
                mean_all_state_outage_delta=float(np.mean([r['outage_delta'] for r in group])),
                mean_changed_energy_delta_j=float(np.mean([r['energy_delta_j'] for r in changed])) if changed else 0.,
                mean_changed_outage_delta=float(np.mean([r['outage_delta'] for r in changed])) if changed else 0.))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in sources)
    _write_rows(root/'summary.csv',summary)
    (root/'audit.json').write_text(json.dumps(dict(rows=840,unique_states=420,
        counterfactual_continuations=sum(r['changed'] for r in rows),
        baseline_replay_max_error='less than 1e-9',source_hashes='match',
        sampling='complete census of the declared 12 trajectories; correlated within trajectories'),indent=2))
    (root/'COMPLETE').write_text('Complete causal action census; exploratory, not a learned policy.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run()
