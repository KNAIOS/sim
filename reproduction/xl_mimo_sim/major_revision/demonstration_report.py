"""Verify imitation choices and gates, then evaluate expert on matched new traces."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.demonstration_data import settings
from major_revision.demonstration_training import imitation_key,agreement
from major_revision.full_domain import FullDomainDQN
from major_revision.full_domain_baselines import baseline_episode
from major_revision.nested_search import ROOT,choose_setting
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read


def verify_source_files(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for key in ('source_hashes','input_hashes'):
        for file,digest in manifest.get(key,{}).items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file


def verify_conversion(folder):
    """Check both preserved checkpoints and the strictly loadable copies."""
    converted=folder/'portable_checkpoints'
    receipts=json.loads((converted/'CONVERSION.json').read_text())
    assert set(receipts)=={'stage0.pt','stage1.pt','stage2.pt','selected.pt'}
    paths=[converted/'CONVERSION.json',folder/'RECOVERY.json']
    recovery=json.loads((folder/'RECOVERY.json').read_text())
    assert recovery['training_repeated'] is False
    assert recovery['selection_changed'] is False
    assert recovery['strict_weights_only'] is True
    recovery_source=Path(__file__).with_name('demonstration_recovery.py')
    assert hashlib.sha256(recovery_source.read_bytes()).hexdigest()==recovery['recovery_source_hash']
    for name,receipt in receipts.items():
        assert receipt['tensor_identity'] is True
        for path,key in ((folder/name,'original_sha256'),(converted/name,'portable_sha256')):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt[key],path
            paths.append(path)
    return paths


def initialization_gate(accuracy,ratios,excesses):
    """Recompute the frozen per-scenario gate; never trust a saved Boolean."""
    return bool(accuracy>=.90 and all(r<=1.10 for r in ratios)
                and all(e<=.01 for e in excesses))


def main():
    cfg=make_config('validation')
    teacher=settings()
    all_inputs=[]
    for folder in (ROOT/'teacher',ROOT/'data',ROOT/'imitation_seed11',ROOT/'imitation_seed22'):
        assert (folder/'COMPLETE').exists(),folder
        verify_source_files(folder)
        all_inputs.append(folder/'manifest.json')
    teacher_means=read(ROOT/'teacher/validation_summary.csv')
    for j in range(3):
        chosen=choose_setting([r for r in teacher_means if int(r['scenario'])==j])
        assert teacher[j]['minimum']==int(chosen['minimum'])
        assert teacher[j]['margin']==float(chosen['margin'])
    data=dict(np.load(ROOT/'data/validation.npz',allow_pickle=False))
    train=dict(np.load(ROOT/'data/train.npz',allow_pickle=False))
    assert data['state'].shape==(1260,235) and train['state'].shape==(3360,235)
    assert np.array_equal(train['expert_action'],train['action'])
    reference_rows=[r for r in read(ROOT/'data/expert_rollouts.csv') if r['split']=='validation']
    reference={j:{k:float(np.mean([float(r[k]) for r in reference_rows if int(r['scenario'])==j]))
                   for k in ('outage','energy_j')} for j in range(3)}
    choices,class_rows,validation_summary=[],[],[]
    for seed in (11,22):
        folder=ROOT/f'imitation_seed{seed}'
        all_inputs.extend(verify_conversion(folder))
        validation=[dict(r,scenario=int(r['scenario']),stage=int(r['stage']),
                        energy_j=float(r['energy_j']),outage=float(r['outage']))
                    for r in read(folder/'validation.csv')]
        assert len(validation)==108
        selected_stage=min(range(3),key=lambda stage:imitation_key(
            [r for r in validation if r['stage']==stage],reference))
        saved=torch.load(folder/'portable_checkpoints/selected.pt',map_location='cpu',weights_only=True)
        assert saved['stage']==selected_stage
        selected_key=imitation_key([r for r in validation if r['stage']==selected_stage],reference)
        assert np.allclose(saved['validation_key'],selected_key,atol=1e-12,rtol=0)
        choices.append(dict(seed=seed,selected_stage=selected_stage,
            initialization_gate=saved['initialization_gate'],agreement=saved['agreement'],
            by_scenario=saved['by_scenario']))
        for stage in range(3):
            model=FullDomainDQN(cfg,seed)
            checkpoint=torch.load(folder/f'portable_checkpoints/stage{stage}.pt',map_location='cpu',weights_only=True)
            model.online.load_state_dict(checkpoint['online'])
            measured=agreement(model,data)
            assert abs(measured['accuracy']-checkpoint['agreement']['accuracy'])<1e-12
            with torch.no_grad():
                predicted=model.online(torch.from_numpy(data['state'])).argmax(1).numpy()
            ratios,excesses=[],[]
            for j in range(3):
                selected=data['scenario']==j
                group=[r for r in validation if r['stage']==stage and r['scenario']==j]
                mean={k:float(np.mean([r[k] for r in group])) for k in ('energy_j','outage')}
                ratios.append(mean['energy_j']/reference[j]['energy_j'])
                excesses.append(mean['outage']-reference[j]['outage'])
                assert np.isclose(ratios[-1],checkpoint['by_scenario'][j]['energy_ratio'],rtol=0,atol=1e-12)
                assert np.isclose(excesses[-1],checkpoint['by_scenario'][j]['outage_excess'],rtol=0,atol=1e-12)
                validation_summary.append(dict(seed=seed,stage=stage,scenario=j,
                    action_accuracy=float(np.mean(predicted[selected]==data['expert_action'][selected])),
                    count_mae=float(np.mean(np.abs(predicted[selected]-data['expert_action'][selected]))),
                    **mean,expert_energy_j=reference[j]['energy_j'],expert_outage=reference[j]['outage'],
                    energy_ratio=mean['energy_j']/reference[j]['energy_j'],
                    outage_excess=mean['outage']-reference[j]['outage']))
            assert checkpoint['initialization_gate']==initialization_gate(measured['accuracy'],ratios,excesses)
            if stage==selected_stage:
                assert saved['initialization_gate']==checkpoint['initialization_gate']
                assert all(torch.equal(saved['online'][k],v) for k,v in checkpoint['online'].items())
            for action in range(cfg.subarrays):
                mask=data['expert_action']==action
                class_rows.append(dict(seed=seed,stage=stage,expert_count=action+1,
                    examples=int(mask.sum()),accuracy=float(np.mean(predicted[mask]==action)) if mask.any() else ''))
        all_inputs.extend(folder/n for n in ('selection.csv','validation.csv','classification.csv'))
    # Selection and the all-seed gate are fixed before new baseline evaluation.
    (ROOT/'imitation_choices.json').write_text(json.dumps(choices,indent=2))
    rows=[]
    for seed in (11,22):
        path=ROOT/f'imitation_seed{seed}/development_raw.csv'
        all_inputs.append(path)
        rows.extend(read(path))
    baseline_rows=[]
    baseline_path=ROOT/'development_baselines.csv'
    if baseline_path.exists():
        baseline_rows=read(baseline_path)
        assert len(baseline_rows)==96,'Do not silently resume an incomplete table'
    else:
        for j,sc in enumerate(SCENARIOS):
            for i in range(16):
                tr=trajectory(cfg,993000000+j*10000+i,sc)
                for name,minimum,margin,mode in (
                    ('Nested PRS',teacher[j]['minimum'],teacher[j]['margin'],'prs'),
                    ('Full array',1,1.,'full')):
                    result=baseline_episode(cfg,tr,mode,margin,minimum)
                    result['method']=name
                    baseline_rows.append(dict(scenario=j,trajectory=i,seed=0,stage=-1,**result))
            print('imitation reference development',j,'complete',flush=True)
        _write_rows(baseline_path,baseline_rows)
    all_inputs.append(baseline_path)
    rows.extend(baseline_rows)
    fields=('scenario','trajectory','seed','stage','method','energy_j','outage','operating_j',
            'switching_j','pilot_j','mean_count','below_four_fraction')
    rows=[{k:r[k] for k in fields} for r in rows]
    assert len(rows)==len({(int(r['scenario']),int(r['trajectory']),int(r['seed']),r['method']) for r in rows})==288
    identity=max(abs(float(r['energy_j'])-float(r['operating_j'])-float(r['switching_j'])-float(r['pilot_j'])) for r in rows)
    assert identity<1e-7
    rng=np.random.default_rng(993900000)
    summaries,pairs=[],[]
    for j in range(3):
        group=[r for r in rows if int(r['scenario'])==j]
        matrices={}
        for method in sorted({r['method'] for r in group}):
            sample=[r for r in group if r['method']==method]
            seeds=sorted({int(r['seed']) for r in sample})
            lookup={(int(r['seed']),int(r['trajectory'])):r for r in sample}
            ti=rng.integers(16,size=(4000,16))
            si=rng.integers(len(seeds),size=(4000,len(seeds)))
            summary=dict(scenario=j,method=method,policy_instances=len(seeds),trajectories=16)
            for metric in ('energy_j','outage'):
                values=np.array([[float(lookup[s,i][metric]) for i in range(16)] for s in seeds])
                matrices[method,metric]=values
                boot=values[si[:,:,None],ti[:,None,:]].mean((1,2))
                summary.update({metric:float(values.mean()),metric+'_low':float(np.quantile(boot,.025)),
                                metric+'_high':float(np.quantile(boot,.975))})
            summaries.append(summary)
        for method in ('Teacher-only imitation','Selected imitation'):
            ti=rng.integers(16,size=(4000,16))
            si=rng.integers(2,size=(4000,2))
            paired=dict(scenario=j,method=method,baseline='Nested PRS')
            for metric in ('energy_j','outage'):
                delta=matrices[method,metric]-matrices['Nested PRS',metric]
                boot=delta[si[:,:,None],ti[:,None,:]].mean((1,2))
                paired.update({metric:float(delta.mean()),metric+'_low':float(np.quantile(boot,.025)),
                               metric+'_high':float(np.quantile(boot,.975))})
            pairs.append(paired)
    for name,content in (('development_all',rows),('summary',summaries),('paired',pairs),
                         ('validation_summary',validation_summary),('class_accuracy',class_rows)):
        _write_rows(ROOT/f'{name}.csv',content)
    audit=dict(choices=choices,all_seeds_initialization_gate=all(r['initialization_gate'] for r in choices),
        role='Imitation only; no RL advantage, no independent final confirmation',
        development_rows=288,energy_identity_max_error_j=identity,
        record_scope='selected and teacher-only are the same policy when selected stage=0; rows are not independent samples',
        bootstrap='4000 crossed policy-seed/trajectory draws; only two trained seeds',
        report_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in all_inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Imitation diagnostic audited; gate is not a manuscript-superiority result.\n')
    print(json.dumps(dict(choices=choices,summary=summaries),indent=2))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
