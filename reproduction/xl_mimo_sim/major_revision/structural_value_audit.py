"""Freeze choices; separate known physics from conditional future variability."""
import hashlib
import json
from dataclasses import fields
from pathlib import Path
import numpy as np
from future_vr.channel import Trajectory
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.action_value import observation,baseline_command,hold_predictions,full_slot_energy,action_size
from major_revision.action_value_report import numeric_rows
from major_revision.action_value_study import ROOT as BASE_ROOT
from major_revision.branching_channel import BranchingChannel,replace_future
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import representation_size
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.rollout_value_data import ROOT as VALUE_ROOT,branch_return
from major_revision.rollout_value_fit import select_action

ROOT=Path('results/major_revision_20260921_structural_value')
TIMES=(0,32,64,96)
DRAWS=16


def known_costs(run,score):
    """C4 energy and current-slot outage, from causal state only, at D>0."""
    if run.cfg.wake_delay_slots<=0:
        raise ValueError('New instant-wake branches need CSI not covered by this identity')
    prediction=hold_predictions(run.cfg,run.tracker,run.hw,score)
    energy=prediction['energy'][:,:4].mean(1)/full_slot_energy(run.cfg)
    outages=[]
    for mask in prediction['masks']:
        available=mask&run.hw.active
        assert not np.any(available&~run.observed)
        rates=run.system.achievable_rates(run.tracker.channel_estimate,available)
        outages.append(float(np.mean(rates<run.req)))
    return energy,np.asarray(outages)


def serialized_energy(cfg,features,references):
    end=representation_size(cfg)+cfg.subarrays*action_size(cfg)
    rows=features[:,representation_size(cfg):end].reshape(-1,cfg.subarrays,action_size(cfg)).astype(np.float64)
    costs=rows[:,:,cfg.subarrays:cfg.subarrays+4].mean(-1)
    return costs-costs[np.arange(len(costs)),references][:,None]


def main():
    assert (VALUE_ROOT/'closed_audit_v2/COMPLETE').exists()
    assert not (VALUE_ROOT/'PERMIT_CONFIRMATION').exists()
    for name in ('data_audit','fit_audit','closed_audit_v2'):
        audit=json.loads((VALUE_ROOT/name/'audit.json').read_text())
        for path,digest in audit['input_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    cfg=make_config('validation')
    files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('STRUCTURAL_VALUE_AUDIT_PLAN.md','branching_channel.py','test_branching_channel.py',
         'test_structural_value_audit.py','rollout_value_data.py','rollout_value_fit.py',
         'action_value.py','gram_control.py','full_domain.py','audit_rollout.py','pilot.py',
         'spatial_actions.py','constraint_training.py')],*Path('future_vr').glob('*.py'),
         VALUE_ROOT/'data_audit/audit.json',VALUE_ROOT/'fit_audit/audit.json',
         VALUE_ROOT/'closed_audit_v2/audit.json',VALUE_ROOT/'closed_loop/learned/selected.csv',
         BASE_ROOT/'baselines/selected.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    ROOT.mkdir(exist_ok=False)
    (ROOT/'manifest.json').write_text(json.dumps(dict(config=cfg.as_dict(),hashes=hashes,
        root_stream=1020000000,root_times=TIMES,trajectories_per_scenario=4,future_draws=DRAWS,
        future_stream=1040000000,threads=1,
        role='post-hoc latent-state conditional diagnostic; no online hidden-state access'),indent=2))
    analytic=[]
    for split in ('train','validation'):
        with np.load(VALUE_ROOT/split/'labels.npz',allow_pickle=False) as data:
            estimate=serialized_energy(cfg,data['features'],data['reference'])
            target=data['difference'][:,:,0,0]
            error=float(np.max(np.abs(estimate-target)))
            assert error<5e-7
            for seed in (11,22):
                pred=np.load(VALUE_ROOT/f'fit_seed{seed}/C4_predictions.npz',allow_pickle=False)[split][:,:,0]
                analytic.append(dict(split=split,seed=seed,roots=len(target),
                    analytic_max_error=error,analytic_mse=float(np.mean((estimate-target)**2)),
                    learned_mse=float(np.mean((pred-target)**2))))
    _write_rows(ROOT/'analytic_energy.csv',analytic)
    data=np.load(VALUE_ROOT/'validation/labels.npz',allow_pickle=False)
    predictions={seed:np.load(VALUE_ROOT/f'fit_seed{seed}/C16_predictions.npz',allow_pickle=False)['validation'] for seed in (11,22)}
    chosen=numeric_rows(VALUE_ROOT/'closed_loop/learned/selected.csv')
    settings=numeric_rows(BASE_ROOT/'baselines/selected.csv')
    keys={tuple(key):index for index,key in enumerate(data['keys'])}
    raw,summaries,decomposition=[],[],[]
    # Full saved validation set, not only the smaller conditional-future sample.
    for seed in (11,22):
        for j in range(3):
            price=float(next(r for r in chosen if r['seed']==seed and r['horizon']=='C16' and r['scenario']==j)['price'])
            for index,key in enumerate(data['keys']):
                if key[0]!=j:
                    continue
                action=select_action(predictions[seed][index],int(data['reference'][index]),price)
                p=predictions[seed][index,action].astype(np.float64)
                y=data['difference'][index,action,1]
                de=float(p[0]-y[0])
                do=float(price*(p[1]-y[1]))
                decomposition.append(dict(seed=seed,scenario=j,trajectory=int(key[1]),step=int(key[2]),
                    energy_error=de,priced_outage_error=do,total_error=de+do))
    _write_rows(ROOT/'selected_error.csv',decomposition)
    r=0
    known_error=0.
    original_error=0.
    branch_evaluations=0
    for j,sc in enumerate(SCENARIOS):
        setting=next(row for row in settings if row['method']=='cost' and row['scenario']==j)
        prices={seed:float(next(row for row in chosen if row['seed']==seed and row['horizon']=='C16' and row['scenario']==j)['price']) for seed in (11,22)}
        for i in range(4):
            seed=1020000000+j*10000+i
            tr=trajectory(cfg,seed,sc)
            copied,snapshots=BranchingChannel(cfg,seed,sc[0]).generate(cfg.episode_steps,capture=TIMES)
            for field in fields(Trajectory):
                expected=getattr(tr,field.name)
                actual=getattr(copied,field.name)*(sc[1] if field.name=='requirements_mbps' else 1.)
                np.testing.assert_array_equal(actual,expected)
            run=FullDomainRollout(cfg,tr)
            while run.t<cfg.episode_steps:
                if run.t%4:
                    run.step()
                    continue
                state,score=observation(run)
                command=baseline_command(run,'cost',setting['margin'],setting['minimum'])
                if run.t not in TIMES:
                    run.step(command)
                    continue
                t=run.t
                index=keys[j,i,t]
                feature=np.concatenate((state,np.array([setting['margin'],setting['minimum']/cfg.subarrays],dtype=np.float32)))
                np.testing.assert_array_equal(feature,data['features'][index])
                ref=int(command.sum())-1
                assert ref==data['reference'][index]
                actions={s:select_action(predictions[s][index],ref,prices[s]) for s in (11,22)}
                candidate_actions=sorted({ref,*actions.values()})
                exact_energy,exact_outage=known_costs(run,score)
                for action in range(cfg.subarrays):
                    trial=run.clone()
                    initial_energy=trial.energy
                    trial.step(select_top(score,action+1))
                    assert abs(trial.outages[-1]-exact_outage[action])<1e-12
                    for _ in range(3):
                        trial.step()
                    observed=(trial.energy-initial_energy)/(4*full_slot_energy(cfg))
                    known_error=max(known_error,abs(observed-exact_energy[action]))
                assert known_error<1e-10
                # Original stochastic continuation must reproduce every stored field.
                original_future,_=snapshots[t].fork().generate(15)
                replay=replace_future(tr,t,original_future,sc[1])
                for field in fields(Trajectory):
                    np.testing.assert_array_equal(getattr(replay,field.name),getattr(tr,field.name))
                for action in candidate_actions:
                    value,_=branch_return(run,select_top(score,action+1),setting)
                    original_error=max(original_error,float(np.max(np.abs(value-data['costs'][index,action]))))
                assert original_error<1e-10
                per_seed={s:[] for s in (11,22)}
                for m in range(DRAWS):
                    future_seed=1040000000+1000*r+m
                    future,_=snapshots[t].fork(seed=future_seed).generate(15)
                    branch=run.clone()
                    branch.tr=replace_future(tr,t,future,sc[1])
                    np.testing.assert_array_equal(observation(branch)[0],state)
                    returns={a:branch_return(branch,select_top(score,a+1),setting)[0][1]
                             for a in candidate_actions}
                    branch_evaluations+=len(candidate_actions)
                    for s in (11,22):
                        difference=returns[actions[s]]-returns[ref]
                        cost=float(difference[0]+prices[s]*difference[1])
                        per_seed[s].append(cost)
                        raw.append(dict(scenario=j,trajectory=i,step=t,root=r,draw=m,future_seed=future_seed,
                            model_seed=s,action_count=actions[s]+1,reference_count=ref+1,price=prices[s],
                            energy_difference=float(difference[0]),outage_difference=float(difference[1]),cost_difference=cost))
                for s in (11,22):
                    samples=np.asarray(per_seed[s])
                    p=predictions[s][index,actions[s]].astype(np.float64)
                    old=data['difference'][index,actions[s],1]
                    summaries.append(dict(scenario=j,trajectory=i,step=t,root=r,model_seed=s,
                        action_count=actions[s]+1,reference_count=ref+1,price=prices[s],
                        predicted_cost=float(p[0]+prices[s]*p[1]),original_sample_cost=float(old[0]+prices[s]*old[1]),
                        conditional_mean_cost=float(samples.mean()),mc_standard_error=float(samples.std(ddof=1)/np.sqrt(DRAWS)),
                        positive_samples=int((samples>1e-6).sum()),draws=DRAWS))
                r+=1
                run.step(command)
            _write_rows(ROOT/'conditional_raw.csv',raw)
            _write_rows(ROOT/'conditional_roots.csv',summaries)
            print('structural value audit',j,i+1,'/ 4',flush=True)
    assert r==48 and len(raw)==1536 and len(summaries)==96 and len(decomposition)==576
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (ROOT/'audit.json').write_text(json.dumps(dict(roots=48,conditional_future_paths=768,
        paired_records=1536,distinct_action_branch_evaluations=branch_evaluations,
        known_energy_max_error=known_error,original_return_replay_error=original_error,
        bitwise_channel_trajectories=12,source_hashes_match=True,models_frozen=True),indent=2))
    (ROOT/'COMPLETE').write_text('Structural and conditional-future diagnostics complete; no policy training or promotion.\n')
    print(json.dumps(analytic,indent=2))


if __name__=='__main__':
    main()
