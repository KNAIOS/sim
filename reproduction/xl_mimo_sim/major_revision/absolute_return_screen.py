"""Fixed free-value policy screen with matched-information strong searches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.action_value import baseline_command,hold_predictions
from major_revision.action_value_report import numeric_rows,means
from major_revision.action_value_study import ROOT as BASE_ROOT,MARGINS,choose_baseline
from major_revision.factored_value import ROOT as FIT_ROOT,FactoredNetwork,extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import subarray_grams,rates_from_gram
from major_revision.pilot import trajectory,SCENARIOS
from major_revision.rollout_value_fit import select_action

ROOT=Path('results/major_revision_20260921_absolute_return_screen')
METHODS=('prs','cost','current-prs','current-cost')
COUNT=8
STREAM=1060000000


def current_table(run):
    """Only current observed active CSI; future predictions remain unchanged."""
    _,score=run.base_observation()
    cfg=run.cfg
    if cfg.wake_delay_slots<=0:
        raise ValueError('Unobserved instant-wake CSI is outside this current-rate identity')
    prediction=hold_predictions(cfg,run.tracker,run.hw,score)
    active=prediction['masks']&run.hw.active[None,:]
    assert not np.any(active&~run.observed[None,:])
    sums=np.einsum('aq,qkl->akl',active,subarray_grams(cfg,run.tracker.channel_estimate))
    prediction['rates'][:,0]=rates_from_gram(cfg,sums,active.sum(1)*cfg.antennas_per_subarray)
    return prediction,score


def search_command(run,mode,margin=1.,minimum=1):
    if mode in ('prs','cost','full'):
        return baseline_command(run,mode,margin,minimum)
    if mode not in ('current-prs','current-cost'):
        raise ValueError('Unknown search mode')
    prediction,score=current_table(run)
    req=run.req[None,None,:]*margin
    outage=np.mean(prediction['rates']<req,axis=(1,2))
    shortfall=np.mean(np.maximum(req-prediction['rates'],0.)/req,axis=(1,2))
    objective=(np.arange(1,run.cfg.subarrays+1,dtype=float) if mode=='current-prs'
               else prediction['energy'].sum(1))
    def key(i):
        feasible=outage[i]<=run.cfg.outage_target
        return (not feasible,objective[i] if feasible else shortfall[i],objective[i],i)
    action=min(range(minimum-1,run.cfg.subarrays),key=key)
    return select_top(score,action+1)


def episode(cfg,tr,mode='cost',setting=None,model=None,horizon='C16'):
    setting=setting or dict(margin=1.,minimum=1)
    run=FullDomainRollout(cfg,tr)
    changed=0
    while run.t<cfg.episode_steps:
        if run.t%4:
            run.step()
            continue
        if model is None:
            command=search_command(run,mode,setting['margin'],setting['minimum'])
        else:
            x,score=extend_observation(run,setting)
            reference=int(baseline_command(run,'cost',setting['margin'],setting['minimum']).sum())-1
            with torch.no_grad():
                costs=model(torch.from_numpy(x[None]))[0,:,0 if horizon=='C4' else 1].numpy()
            action=select_action(costs-costs[reference],reference,2.)
            changed+=int(action!=reference)
            command=select_top(score,action+1)
        run.step(command)
    result=run.metrics()
    result.update(method=mode if model is None else horizon,
                  changed_from_mu_fraction=changed/(cfg.episode_steps/4) if model is not None else 0.)
    return result


def initialize(part):
    assert (FIT_ROOT/'fit_audit/COMPLETE').exists()
    gate=json.loads((FIT_ROOT/'fit_audit/gate.json').read_text())
    # The failed structural candidate remains failed; this is a new frozen screen.
    assert not gate['permit_closed_loop']
    assert all(r['fit_pass'] and r['decision_pass'] for r in gate['rows'] if r['mode']=='free')
    audit=json.loads((FIT_ROOT/'fit_audit/audit.json').read_text())
    for path,digest in audit['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for seed in (11,22):
        old=json.loads((FIT_ROOT/f'fit_seed{seed}'/'manifest.json').read_text())
        for path,digest in old['hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    cfg=make_config('validation')
    files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('ABSOLUTE_RETURN_SCREEN_PLAN.md','test_absolute_return_screen.py','factored_value.py',
         'action_value.py','gram_control.py','full_domain.py','audit_rollout.py','pilot.py',
         'spatial_actions.py','constraint_training.py')],*Path('future_vr').glob('*.py'),
         FIT_ROOT/'fit_audit/audit.json',FIT_ROOT/'fit_audit/gate.json',
         BASE_ROOT/'baselines/selected.csv',*[FIT_ROOT/f'fit_seed{s}/free_final.pt' for s in (11,22)]]
    folder=ROOT/part
    folder.mkdir(parents=True,exist_ok=False)
    manifest=dict(part=part,config=cfg.as_dict(),stream=STREAM,count=COUNT,price=2.,
        methods=METHODS,margins=MARGINS,minimum_counts=[1,4],threads=1,
        role='new validation screen of fixed free candidate, not final confirmation',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    traces=[(j,i,trajectory(cfg,STREAM+j*10000+i,sc)) for j,sc in enumerate(SCENARIOS) for i in range(COUNT)]
    old=numeric_rows(BASE_ROOT/'baselines/selected.csv')
    settings={j:next(r for r in old if r['method']=='cost' and r['scenario']==j) for j in range(3)}
    return cfg,folder,manifest,traces,settings


def finish(folder,manifest):
    for path,digest in manifest['hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (folder/'COMPLETE').write_text('All frozen configurations finished, no retraining or adaptive stopping.\n')


def baselines():
    cfg,folder,manifest,traces,settings=initialize('baselines')
    raw,summary,selected=[],[],[]
    for mode in METHODS:
        for j in range(3):
            choices=[]
            for minimum in (1,4):
                for margin in MARGINS:
                    rows=[dict(scenario=j,trajectory=i,minimum=minimum,margin=margin,
                        **episode(cfg,tr,mode,dict(margin=margin,minimum=minimum))) for sc,i,tr in traces if sc==j]
                    cell=dict(scenario=j,method=mode,minimum=minimum,margin=margin,**means(rows))
                    raw.extend(rows)
                    summary.append(cell)
                    choices.append(cell)
            choice=choose_baseline(choices)
            selected.append(choice)
            _write_rows(folder/'raw.csv',raw)
            _write_rows(folder/'summary.csv',summary)
            _write_rows(folder/'selected.csv',selected)
            print('absolute screen baseline',choice,flush=True)
    for j in range(3):
        for mode in ('mu','full'):
            setting=settings[j] if mode=='mu' else dict(margin=1.,minimum=1)
            rows=[]
            for sc,i,tr in traces:
                if sc!=j:
                    continue
                result=episode(cfg,tr,'cost' if mode=='mu' else 'full',setting)
                result['method']=mode
                rows.append(dict(scenario=j,trajectory=i,minimum=setting['minimum'],margin=setting['margin'],**result))
            raw.extend(rows)
            summary.append(dict(scenario=j,method=mode,minimum=setting['minimum'],margin=setting['margin'],**means(rows)))
    assert len(raw)==1200
    _write_rows(folder/'raw.csv',raw)
    _write_rows(folder/'summary.csv',summary)
    finish(folder,manifest)


def learned():
    cfg,folder,manifest,traces,settings=initialize('learned')
    raw,summary=[],[]
    for seed in (11,22):
        checkpoint=torch.load(FIT_ROOT/f'fit_seed{seed}/free_final.pt',map_location='cpu',weights_only=True)
        assert checkpoint['mode']=='free' and checkpoint['epoch']==100
        model=FactoredNetwork(cfg,structured=False)
        model.load_state_dict(checkpoint['online'],strict=True)
        model.eval()
        for horizon in ('C4','C16'):
            for j in range(3):
                rows=[dict(seed=seed,horizon=horizon,scenario=j,trajectory=i,price=2.,
                    **episode(cfg,tr,model=model,setting=settings[j],horizon=horizon)) for sc,i,tr in traces if sc==j]
                raw.extend(rows)
                cell=dict(seed=seed,horizon=horizon,scenario=j,price=2.,**means(rows),
                    changed_from_mu_fraction=float(np.mean([r['changed_from_mu_fraction'] for r in rows])))
                summary.append(cell)
                _write_rows(folder/'raw.csv',raw)
                _write_rows(folder/'summary.csv',summary)
                print('absolute screen learned',cell,flush=True)
    assert len(raw)==96
    finish(folder,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=('baselines','learned'),required=True)
    torch.set_num_threads(1)
    baselines() if parser.parse_args().part=='baselines' else learned()
