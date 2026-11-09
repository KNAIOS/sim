"""Stage-A exactness audit on every decision of twelve fresh causal trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.full_domain_baselines import baseline_episode
from major_revision.gram_control import prefix_forecast_rates,gram_belief_features
from major_revision.demonstration_data import settings
from major_revision.nested_search import teacher_command
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.spatial_actions import causal_forecasts
from major_revision.test_gram_control import forecast_alias

ROOT=Path('results/major_revision_20260920_gram_audit')


def main():
    ROOT.mkdir(exist_ok=False)
    cfg=make_config('validation')
    teacher=settings()
    source=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('CORE_REBUILD_PLAN.md','gram_control.py','test_gram_control.py',
         'full_domain.py','full_domain_baselines.py','audit_rollout.py','pilot.py',
         'spatial_actions.py','nested_search.py')],*Path('future_vr').glob('*.py')]
    selected=Path('results/major_revision_20260920_demonstration/teacher/selected.csv')
    manifest=dict(config=cfg.as_dict(),teacher=teacher,stream=995000000,
        trajectories_per_scenario=4,role='numerical/representation audit, not new-policy performance',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        input_hashes={str(selected):hashlib.sha256(selected.read_bytes()).hexdigest()})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows,rollouts=[],[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(4):
            tr=trajectory(cfg,995000000+j*10000+i,sc)
            run=FullDomainRollout(cfg,tr)
            while run.t<cfg.episode_steps:
                if run.t%4:
                    run.step()
                    continue
                _,score=run.base_observation()
                physical_before={k:getattr(run.hw,k).copy() for k in ('active','command','countdown')}
                actual=prefix_forecast_rates(cfg,run.tracker,run.hw,score)
                channels=causal_forecasts(cfg,run.tracker)
                expected=np.array([[run.system.achievable_rates(channels[h-1],
                    select_top(score,count)&(run.hw.readiness()<=h))
                    for h in range(1,cfg.horizon+1)] for count in range(1,cfg.subarrays+1)])
                features=gram_belief_features(cfg,run.tracker,run.hw,run.req,
                    run.previous_req,run.age,run.last_outage)
                assert features.shape==(362,) and np.isfinite(features).all()
                for k,value in physical_before.items():
                    np.testing.assert_array_equal(getattr(run.hw,k),value)
                error=np.abs(actual-expected)
                req=run.req[None,None,:]
                row=dict(scenario=j,trajectory=i,slot=run.t,scalar_rates=actual.size,
                    max_error_mbps=float(error.max()),mean_error_mbps=float(error.mean()),
                    physical_threshold_mismatches=int(np.sum((actual<req)!=(expected<req))),
                    teacher_threshold_mismatches=int(np.sum((actual<req*teacher[j]['margin'])!=
                                                           (expected<req*teacher[j]['margin']))))
                rows.append(row)
                assert row['max_error_mbps']<1e-4
                assert row['physical_threshold_mismatches']==row['teacher_threshold_mismatches']==0
                run.step(teacher_command(run,teacher[j]))
            result=run.metrics()
            reference=baseline_episode(cfg,tr,'prs',teacher[j]['margin'],teacher[j]['minimum'])
            for k in ('energy_j','outage','switching_j','operating_j','pilot_j','mean_count'):
                assert abs(result[k]-reference[k])<1e-9,(j,i,k)
            rollouts.append(dict(scenario=j,trajectory=i,**result))
            _write_rows(ROOT/'states.csv',rows)
            _write_rows(ROOT/'reference_replays.csv',rollouts)
            print('Gram exactness trajectory',j,i,'complete',flush=True)
    a,b=forecast_alias()
    old_a,score_a=a.observation()
    old_b,score_b=b.observation()
    np.testing.assert_array_equal(old_a,old_b)
    predicted=[prefix_forecast_rates(cfg,r.tracker,r.hw,score_a) for r in (a,b)]
    features=[gram_belief_features(cfg,r.tracker,r.hw,r.req,r.previous_req,r.age,r.last_outage)
              for r in (a,b)]
    index=np.unravel_index(np.argmax(np.abs(predicted[0]-predicted[1])),predicted[0].shape)
    n,h,k=map(int,index)
    constructed=[dict(count=count+1,horizon=step+1,user=user,
        rate_a_mbps=float(predicted[0][count,step,user]),
        rate_b_mbps=float(predicted[1][count,step,user]))
        for count in range(16) for step in range(8) for user in range(3)]
    _write_rows(ROOT/'constructed_forecast_rates.csv',constructed)
    alias=dict(old_feature_dimension=235,old_inputs_exactly_equal=True,
        new_feature_dimension=362,max_new_feature_difference=float(np.max(np.abs(features[0]-features[1]))),
        largest_rate_difference_mbps=float(abs(predicted[0][index]-predicted[1][index])),
        count=n+1,horizon=h+1,user=k,rate_a_mbps=float(predicted[0][index]),
        rate_b_mbps=float(predicted[1][index]),
        scope='constructed complex-channel/tracker states, not observed frequency, policy superiority or physical calibration')
    assert len(rows)==420 and sum(r['scalar_rates'] for r in rows)==161280
    for p in source:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)]
    audit=dict(states=len(rows),scalar_rates=sum(r['scalar_rates'] for r in rows),
        max_error_mbps=max(r['max_error_mbps'] for r in rows),
        threshold_mismatches=sum(r['physical_threshold_mismatches']+r['teacher_threshold_mismatches'] for r in rows),
        reference_replay_count=len(rollouts),reference_replay_matches=True,
        constructed_alias=alias,training_launched=False,
        data_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.csv')})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Stage A completed. No learning or energy-superiority claim.\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
