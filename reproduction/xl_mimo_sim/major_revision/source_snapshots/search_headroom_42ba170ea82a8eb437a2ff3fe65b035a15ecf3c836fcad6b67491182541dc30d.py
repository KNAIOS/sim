"""One-count-decrement audit on frozen PRS; future traces are scoring-only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from future_vr.rate_search import rate_search_command
from major_revision.audit_rollout import AuditRollout
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.ready_report import read

ROOT = Path('results/major_revision_20260920_search_headroom')


class SearchAuditRollout(AuditRollout):
    def command(self, margin):
        _,score = self.observation()
        return rate_search_command(self.cfg,self.tracker,self.hw,self.req*margin,score),score

    def finish_search(self, margin, first_command=None):
        first = True
        while self.t < self.cfg.episode_steps:
            if self.t % 4 == 0:
                command,_ = self.command(margin)
                if first and first_command is not None:
                    command = first_command.copy()
                self.step(command)
            else:
                if first and first_command is not None:
                    raise ValueError('Intervene only at a decision epoch')
                self.step()
            first = False
        return self.metrics()


def run():
    cfg = make_config('validation')
    ROOT.mkdir(parents=True,exist_ok=False)
    baseline = Path('results/major_revision_20260920_constraint_screen/prs')
    selected = baseline/'selected.csv'
    raw = baseline/'development_raw.csv'
    margins = {int(r['scenario']):float(r['margin']) for r in read(selected)}
    reference = {(int(r['scenario']),int(r['trajectory'])):r for r in read(raw) if r['method']=='PRS'}
    files = [Path(__file__),Path(__file__).with_name('audit_rollout.py'),
             Path(__file__).with_name('pilot.py'),*Path('future_vr').glob('*.py')]
    manifest = dict(config=cfg.as_dict(),margins=margins,trajectories_per_scenario=4,
        slots=[0,32,64,96],stream=973000000,intervention='one fewer ranked module, floored at four',
        scope='bounded already-viewed development diagnostic; not online-policy performance',
        selection='all first four trajectories and all four specified times; no favorable filtering',
        future_access='outcome scoring only; baseline actions remain causal',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (selected,raw)})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows = []
    replay_error = 0.
    for j,scenario in enumerate(SCENARIOS):
        margin = margins[j]
        for i in range(4):
            tr = trajectory(cfg,973000000+j*10000+i,scenario)
            base = SearchAuditRollout(cfg,tr)
            factual = []
            while base.t < cfg.episode_steps:
                if base.t % 4:
                    base.step()
                    continue
                command,score = base.command(margin)
                if base.t in (0,32,64,96):
                    t,n = base.t,int(command.sum())
                    fewer = max(cfg.users+1,n-1)
                    a,b = base.clone(),base.clone()
                    x,y = a.finish_search(margin,command),b.finish_search(margin,select_top(score,fewer))
                    factual.append(x)
                    de,do = y['energy_j']-x['energy_j'],y['outage']-x['outage']
                    rows.append(dict(scenario=j,trajectory=i,slot=t,prs_count=n,reduced_count=fewer,
                        same_action=n==fewer,prs_energy_j=x['energy_j'],reduced_energy_j=y['energy_j'],
                        prs_outage=x['outage'],reduced_outage=y['outage'],energy_delta_j=de,outage_delta=do,
                        dominance_on_realized_trace=bool(de < -1e-8 and do <= 1e-12),
                        immediate_energy_delta_j=sum(b.slot_energy[t:t+4])-sum(a.slot_energy[t:t+4]),
                        immediate_outage_delta=float(np.mean(b.outages[t:t+4])-np.mean(a.outages[t:t+4]))))
                    if n==fewer:
                        assert x==y
                base.step(command)
            actual = base.metrics()
            for x in factual:
                assert actual == x
            for key,value in actual.items():
                if key!='method':
                    error = abs(value-float(reference[j,i][key]))
                    replay_error=max(replay_error,error)
                    assert error < 1e-9,(j,i,key,error)
            _write_rows(ROOT/'pairs.csv',rows)
            print('search headroom',j,i,'pairs',len(rows),flush=True)
    summary=[]
    for j in range(3):
        group=[r for r in rows if r['scenario']==j]
        summary.append(dict(scenario=j,paired_starts=len(group),
            at_minimum_count=sum(r['same_action'] for r in group),
            fewer_realized_dominance=sum(r['dominance_on_realized_trace'] for r in group),
            any_outage_increase=sum(r['outage_delta']>1e-12 for r in group),
            **{key:float(np.mean([r[key] for r in group])) for key in
               ('energy_delta_j','outage_delta','immediate_energy_delta_j','immediate_outage_delta')}))
    assert len(rows)==len({(r['scenario'],r['trajectory'],r['slot']) for r in rows})==48
    _write_rows(ROOT/'summary.csv',summary)
    (ROOT/'audit.json').write_text(json.dumps(dict(pairs=48,reference_max_error=replay_error,
        warning='Correlated diagnostic starts; cannot sum individual intervention savings.'),indent=2))
    (ROOT/'COMPLETE').write_text('PRS local headroom diagnostic complete; not a policy result.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
