"""Bounded causal closed-loop check of fixed differential-value models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.action_value import observation,baseline_command,action_episode
from major_revision.action_value_report import numeric_rows,means
from major_revision.action_value_study import ROOT as OLD_ROOT,MARGINS,choose_baseline
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.rollout_value_data import ROOT
from major_revision.rollout_value_fit import DifferenceNetwork,select_action

PRICES=(1.,2.,4.,8.,16.)


def learned_episode(cfg,tr,model,setting,price):
    run=FullDomainRollout(cfg,tr)
    changes=0
    while run.t<cfg.episode_steps:
        if run.t%4:
            run.step()
            continue
        state,score=observation(run)
        command=baseline_command(run,'cost',setting['margin'],setting['minimum'])
        reference=int(command.sum())-1
        x=np.concatenate((state,np.array([setting['margin'],setting['minimum']/cfg.subarrays],dtype=np.float32)))
        with torch.no_grad():
            difference=model(torch.from_numpy(x).unsqueeze(0),torch.tensor([reference]))[0].numpy()
        action=select_action(difference,reference,price)
        changes+=int(action!=reference)
        run.step(select_top(score,action+1))
    result=run.metrics()
    result['method']='paired rollout value'
    result['changed_from_mu_fraction']=changes/(cfg.episode_steps/4)
    return result


def initialize(part):
    assert (ROOT/'PERMIT_CLOSED_LOOP').exists()
    gate=json.loads((ROOT/'fit_audit/gate.json').read_text())
    assert gate['permit_closed_loop']
    for audit_name in ('data_audit','fit_audit'):
        audit=json.loads((ROOT/audit_name/'audit.json').read_text())
        for path,digest in audit['input_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('ROLLOUT_CLOSED_LOOP_PROTOCOL.md','ROLLOUT_VALUE_PLAN.md','rollout_value_fit.py',
         'test_rollout_value_closed_loop.py','action_value.py','action_value_study.py','gram_control.py',
         'full_domain.py','audit_rollout.py','pilot.py','spatial_actions.py','constraint_training.py')],
         *Path('future_vr').glob('*.py'),ROOT/'fit_audit/gate.json',ROOT/'fit_audit/audit.json',
         OLD_ROOT/'baselines/selected.csv']
    cfg=make_config('validation')
    folder=ROOT/'closed_loop'/part
    folder.mkdir(parents=True,exist_ok=False)
    manifest=dict(part=part,config=cfg.as_dict(),stream=1020000000,trajectories_per_scenario=12,
        prices=PRICES,margins=MARGINS,minimum_counts=[1,4],threads=1,
        role='validation-only method screen; no independent superiority confirmation',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    traces=[(j,i,trajectory(cfg,1020000000+j*10000+i,sc)) for j,sc in enumerate(SCENARIOS) for i in range(12)]
    old=numeric_rows(OLD_ROOT/'baselines/selected.csv')
    settings={j:next(r for r in old if r['scenario']==j and r['method']=='cost') for j in range(3)}
    return cfg,folder,manifest,traces,settings


def finish(folder,manifest):
    for path,digest in manifest['hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (folder/'COMPLETE').write_text('All frozen validation settings completed without development feedback.\n')


def baselines():
    cfg,folder,manifest,traces,settings=initialize('baselines')
    rows,summaries,selections=[],[],[]
    for mode in ('prs','cost'):
        for j in range(3):
            candidates=[]
            for minimum in (1,4):
                for margin in MARGINS:
                    group=[dict(scenario=j,trajectory=i,minimum=minimum,margin=margin,
                        **action_episode(cfg,tr,mode=mode,minimum=minimum,margin=margin))
                        for scenario,i,tr in traces if scenario==j]
                    row=dict(scenario=j,method=mode,minimum=minimum,margin=margin,**means(group))
                    candidates.append(row)
                    summaries.append(row)
                    rows.extend(group)
            selection=choose_baseline(candidates)
            selections.append(selection)
            _write_rows(folder/'raw.csv',rows)
            _write_rows(folder/'summary.csv',summaries)
            print('rollout reference calibration',selection,flush=True)
    for mode in ('mu','full'):
        for j in range(3):
            setting=settings[j] if mode=='mu' else dict(margin=1.,minimum=1)
            group=[]
            for scenario,i,tr in traces:
                if scenario!=j:
                    continue
                result=action_episode(cfg,tr,mode='cost' if mode=='mu' else 'full',
                    margin=setting['margin'],minimum=setting['minimum'])
                result['method']=mode
                group.append(dict(scenario=j,trajectory=i,minimum=setting['minimum'],margin=setting['margin'],**result))
            rows.extend(group)
            summaries.append(dict(scenario=j,method=mode,minimum=setting['minimum'],margin=setting['margin'],**means(group)))
    assert len(rows)==936
    _write_rows(folder/'raw.csv',rows)
    _write_rows(folder/'summary.csv',summaries)
    _write_rows(folder/'selected.csv',selections)
    finish(folder,manifest)


def learned():
    cfg,folder,manifest,traces,settings=initialize('learned')
    rows,summaries,selections=[],[],[]
    for seed in (11,22):
        for horizon in ('C4','C16'):
            checkpoint=torch.load(ROOT/f'fit_seed{seed}'/f'{horizon}_final.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==100
            model=DifferenceNetwork(cfg)
            model.load_state_dict(checkpoint['online'])
            model.eval()
            for j in range(3):
                choices=[]
                for price in PRICES:
                    group=[dict(seed=seed,horizon=horizon,scenario=j,trajectory=i,price=price,
                        **learned_episode(cfg,tr,model,settings[j],price)) for scenario,i,tr in traces if scenario==j]
                    row=dict(seed=seed,horizon=horizon,scenario=j,price=price,**means(group),
                        changed_from_mu_fraction=float(np.mean([r['changed_from_mu_fraction'] for r in group])))
                    rows.extend(group)
                    summaries.append(row)
                    choices.append(row)
                feasible=[r for r in choices if r['outage']<=.05]
                chosen=min(feasible,key=lambda r:(r['energy_j'],r['price'])) if feasible else min(choices,key=lambda r:(r['outage'],r['energy_j'],r['price']))
                selections.append(chosen)
                _write_rows(folder/'raw.csv',rows)
                _write_rows(folder/'summary.csv',summaries)
                _write_rows(folder/'selected.csv',selections)
                print('rollout learned validation',chosen,flush=True)
    assert len(rows)==720
    finish(folder,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=('baselines','learned'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    baselines() if args.part=='baselines' else learned()
