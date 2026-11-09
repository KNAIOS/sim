"""Diagnose the imposed four-module floor without changing frozen sources."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read
from major_revision.search_headroom import SearchAuditRollout
from major_revision.spatial_audit import prs_reference
from major_revision.spatial_actions import causal_forecasts,predicted_metrics


class MinimumSearchRollout(SearchAuditRollout):
    def __init__(self,cfg,tr,minimum):
        super().__init__(cfg,tr)
        if not 1<=minimum<=cfg.subarrays:
            raise ValueError('Invalid search minimum')
        self.minimum=minimum
        self.command_counts=[]

    def command(self,margin):
        _,score=self.observation()
        forecasts=causal_forecasts(self.cfg,self.tracker)
        candidates=[]
        for count in range(self.minimum,self.cfg.subarrays+1):
            command=select_top(score,count)
            metrics=predicted_metrics(self.cfg,self.hw,self.req*margin,forecasts,command)
            infeasible=metrics['outage']>self.cfg.outage_target
            candidates.append((infeasible,metrics['shortfall'] if infeasible else count,count,command))
        chosen=min(candidates,key=lambda item:item[:3])[3]
        self.command_counts.append(int(chosen.sum()))
        return chosen,score


def run():
    cfg=make_config('validation')
    root=Path('results/major_revision_20260920_count_domain_audit')
    root.mkdir(parents=True,exist_ok=False)
    baseline=Path('results/major_revision_20260920_constraint_screen/prs')
    margins={int(r['scenario']):float(r['margin']) for r in read(baseline/'selected.csv')}
    reference=prs_reference(read(baseline/'development_raw.csv'))
    sources=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('spatial_actions.py','spatial_audit.py','audit_rollout.py','search_headroom.py','pilot.py')],
        *Path('future_vr').glob('*.py')]
    manifest=dict(config=cfg.as_dict(),minimum_counts=[1,4],margins=margins,
        stream=973000000,trajectories_per_scenario=4,
        hypothesis='K+1 modules is an imposed action restriction, not the full-digital antenna-rank lower bound',
        scope='same viewed pilot traces; no learning, no test-set tuning, no frozen source changes',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(4):
            tr=trajectory(cfg,973000000+j*10000+i,sc)
            for minimum in (4,1):
                simulation=MinimumSearchRollout(cfg,tr,minimum)
                result=simulation.finish_search(margins[j])
                if minimum==4:
                    for key,value in result.items():
                        if key!='method':
                            assert abs(value-float(reference[j,i][key]))<1e-9
                result['method']=f'PRS minimum {minimum}'
                rows.append(dict(scenario=j,trajectory=i,minimum=minimum,
                    decisions=len(simulation.command_counts),
                    below_four_decisions=sum(m<4 for m in simulation.command_counts),
                    smallest_command=min(simulation.command_counts),**result))
            _write_rows(root/'raw.csv',rows)
            print('count domain',j,i,'rows',len(rows),flush=True)
    summary=[]
    for j in range(3):
        for minimum in (4,1):
            group=[r for r in rows if r['scenario']==j and r['minimum']==minimum]
            summary.append(dict(scenario=j,minimum=minimum,trajectories=4,
                energy_j=float(np.mean([r['energy_j'] for r in group])),
                outage=float(np.mean([r['outage'] for r in group])),
                below_four_decisions=sum(r['below_four_decisions'] for r in group),
                total_decisions=sum(r['decisions'] for r in group)))
    assert len(rows)==24
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in sources)
    _write_rows(root/'summary.csv',summary)
    (root/'COMPLETE').write_text('Count-domain diagnostic complete; not a learned method or final test.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run()
