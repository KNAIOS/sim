"""Preserve the old count-floor policy inside the new feasible action domain."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.full_domain_baselines import baseline_episode
from major_revision.full_domain_study import MARGINS
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read
from major_revision.spatial_actions import causal_forecasts,predicted_metrics

ROOT=Path('results/major_revision_20260920_demonstration')


def choose_setting(rows,target=.05):
    feasible=[r for r in rows if float(r['outage'])<=target]
    if feasible:
        return min(feasible,key=lambda r:(float(r['energy_j']),int(r['minimum']),float(r['margin'])))
    return min(rows,key=lambda r:(float(r['outage']),float(r['energy_j']),int(r['minimum']),float(r['margin'])))


def teacher_command(run,setting):
    _,score=run.base_observation()
    forecasts=causal_forecasts(run.cfg,run.tracker)
    results=[]
    for count in range(int(setting['minimum']),run.cfg.subarrays+1):
        mask=select_top(score,count)
        metric=predicted_metrics(run.cfg,run.hw,run.req*float(setting['margin']),forecasts,mask)
        infeasible=metric['outage']>run.cfg.outage_target
        results.append(((infeasible,metric['shortfall'] if infeasible else count,count),mask))
    return min(results,key=lambda r:r[0])[1]


def main():
    out=ROOT/'teacher'
    out.mkdir(parents=True,exist_ok=False)
    cfg=make_config('validation')
    old=Path('results/major_revision_20260920_full_domain/baselines_v2')
    assert (old/'COMPLETE').exists()
    old_raw=old/'validation_raw.csv'
    raw=[dict(r,minimum=1) for r in read(old_raw) if r['method']=='prs']
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('DEMONSTRATION_PLAN.md','full_domain.py','full_domain_baselines.py','pilot.py','spatial_actions.py')],
        *Path('future_vr').glob('*.py')]
    manifest=dict(config=cfg.as_dict(),margins=MARGINS,minimum_counts=[1,4],
        stream=982000000,trajectories_per_scenario=12,
        status='validation-only nested PRS calibration; no development-driven teacher choice',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_hashes={str(old_raw):hashlib.sha256(old_raw.read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for j,sc in enumerate(SCENARIOS):
        trajectories=[trajectory(cfg,982000000+j*10000+i,sc) for i in range(12)]
        for margin in MARGINS:
            group=[dict(scenario=j,trajectory=i,minimum=4,margin=margin,
                **baseline_episode(cfg,tr,'prs',margin,minimum=4)) for i,tr in enumerate(trajectories)]
            raw.extend(group)
            # Normalize archived and new row fields before writing.
            fields=('scenario','trajectory','minimum','margin','method','energy_j','outage',
                'switching_j','operating_j','pilot_j','transitions','short_sleep_reactivations',
                'mean_count','below_four_fraction')
            _write_rows(out/'validation_raw.csv',[{k:r[k] for k in fields} for r in raw])
            print('nested PRS',j,margin,'outage',round(np.mean([r['outage'] for r in group]),5),flush=True)
    means=[]
    for j in range(3):
        for minimum in (1,4):
            for margin in MARGINS:
                group=[r for r in raw if int(r['scenario'])==j and int(r['minimum'])==minimum
                       and float(r['margin'])==margin]
                assert len(group)==12
                means.append(dict(scenario=j,minimum=minimum,margin=margin,
                    energy_j=float(np.mean([float(r['energy_j']) for r in group])),
                    outage=float(np.mean([float(r['outage']) for r in group]))))
    selected=[choose_setting([r for r in means if r['scenario']==j]) for j in range(3)]
    _write_rows(out/'validation_summary.csv',means)
    _write_rows(out/'selected.csv',selected)
    assert len(raw)==432
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in files)
    (out/'COMPLETE').write_text('Nested teacher selected using calibration validation only.\n')
    print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':
    main()
