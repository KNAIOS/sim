"""Extend both search baselines below unit prediction margin, without changing true QoS."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS,trajectory,cost_search
from major_revision.search_headroom import SearchAuditRollout
from major_revision.ready_report import read


class CostAuditRollout(SearchAuditRollout):
    def command(self,margin):
        if not np.isfinite(margin) or margin<=0:
            raise ValueError('Prediction margin must be finite and positive')
        _,score=self.observation()
        return cost_search(self.cfg,self.tracker,self.hw,self.req*margin,score),score


def run(part):
    cfg=make_config('validation')
    out=Path(f'results/major_revision_20260920_margin_calibration/{part}')
    out.mkdir(parents=True,exist_ok=False)
    old=Path(f'results/major_revision_20260920_constraint_screen/{part}')
    old_raw=old/'validation_raw.csv'
    rows=read(old_raw)
    engine=SearchAuditRollout if part=='prs' else CostAuditRollout
    name='Calibrated PRS' if part=='prs' else 'Calibrated switch-aware PRS'
    sources=[Path(__file__),Path(__file__).with_name('audit_rollout.py'),
             Path(__file__).with_name('search_headroom.py'),Path(__file__).with_name('pilot.py'),
             *Path('future_vr').glob('*.py')]
    manifest=dict(part=part,config=cfg.as_dict(),extra_margins=[.90,.95,.98],
        validation_stream=972000000,development_stream=973000000,
        validation_per_scenario=12,development_per_scenario=16,
        scope='prediction-threshold calibration only; true requirements and power stay unchanged',
        selection='minimum energy among mean-outage-feasible candidates; then closest margin to one',
        old_validation_sha256=hashlib.sha256(old_raw.read_bytes()).hexdigest(),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    extra=[]
    for j,sc in enumerate(SCENARIOS):
        traces=[trajectory(cfg,972000000+j*10000+i,sc) for i in range(12)]
        for margin in (.90,.95,.98):
            for i,tr in enumerate(traces):
                result=engine(cfg,tr).finish_search(margin)
                result['method']='PRS' if part=='prs' else 'Switch-aware PRS'
                extra.append(dict(scenario=j,trajectory=i,margin=margin,**result))
            _write_rows(out/'extra_validation_raw.csv',extra)
            print('margin calibration',part,j,margin,'complete',flush=True)
    rows.extend(extra)
    _write_rows(out/'validation_all.csv',rows)
    choices,summary,development=[],[],[]
    for j,sc in enumerate(SCENARIOS):
        group=[r for r in rows if int(r['scenario'])==j]
        candidates=[]
        for margin in sorted({float(r['margin']) for r in group}):
            subset=[r for r in group if float(r['margin'])==margin]
            assert len(subset)==12
            candidates.append(dict(scenario=j,margin=margin,method=name,
                energy_j=float(np.mean([float(r['energy_j']) for r in subset])),
                outage=float(np.mean([float(r['outage']) for r in subset]))))
        assert len(candidates)==16
        summary.extend(candidates)
        feasible=[r for r in candidates if r['outage']<=.05]
        selected=min(feasible,key=lambda r:(r['energy_j'],abs(r['margin']-1),r['margin'])) if feasible else \
            min(candidates,key=lambda r:(r['outage'],r['energy_j'],abs(r['margin']-1)))
        choices.append(dict(**selected,validation_feasible=bool(feasible)))
        _write_rows(out/'selected.csv',choices)
        _write_rows(out/'validation_summary.csv',summary)
        for i in range(16):
            tr=trajectory(cfg,973000000+j*10000+i,sc)
            result=engine(cfg,tr).finish_search(selected['margin'])
            result['method']=name
            development.append(dict(scenario=j,trajectory=i,seed=0,margin=selected['margin'],**result))
            _write_rows(out/'development_raw.csv',development)
        print('calibrated development',part,j,selected['margin'],'complete',flush=True)
    keys={(int(r['scenario']),int(r['trajectory']),float(r['margin'])) for r in rows}
    assert len(rows)==len(keys)==576
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in sources)
    (out/'audit.json').write_text(json.dumps(dict(validation_records=576,new_validation_records=108,
        development_records=48,unique_validation_keys=len(keys),source_hashes='match'),indent=2))
    (out/'COMPLETE').write_text('Extended search calibration complete; development only.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=('prs','cost'),required=True)
    run(parser.parse_args().part)
