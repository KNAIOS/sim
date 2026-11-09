"""Lock cross-price choice on validation, then compare every arm with strong controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.ready_report import read

ROOT=Path('results/major_revision_20260920_price_calibration')
SELECTED='Validation-selected residual RL'
BASELINES=('Calibrated PRS','Calibrated switch-aware PRS','Episode mixture',
           'Fixed residual -1','Fixed residual +1','PRS')


def folder(price,seed):
    return Path(f'results/major_revision_20260920_predictive_residual/seed{seed}') if price==4 else \
        ROOT/f'price{price}_seed{seed}'


def run():
    choices=[]
    inputs=[]
    training_audits=[]
    for seed in (11,22):
        candidates=[]
        for price in (1,2,4):
            path=folder(price,seed)
            assert (path/'COMPLETE').exists(),path
            selections=read(path/'selection.csv')
            training=read(path/'training.csv')
            validation=read(path/'validation.csv')
            assert {int(r['episode']) for r in training}==set(range(1,301))
            assert len(training)==300
            assert len(validation)==252
            assert {(int(r['episode']),int(r['scenario']),int(r['trajectory'])) for r in validation}=={
                (e,j,i) for e in range(0,301,50) for j in range(3) for i in range(12)}
            assert {int(r['episode']) for r in selections}==set(range(0,301,50))
            for selection in selections:
                sample=[r for r in validation if int(r['episode'])==int(selection['episode'])]
                worst=max(np.mean([float(r['outage']) for r in sample if int(r['scenario'])==j]) for j in range(3))
                assert abs(worst-float(selection['worst_outage']))<1e-12
                assert abs(np.mean([float(r['energy_j']) for r in sample])-float(selection['energy_j']))<1e-9
                assert abs(max(0.,worst-.05)-float(selection['violation']))<1e-12
            selected_ep=int(selections[-1]['selected_episode'])
            assert selected_ep==int(min(selections,key=lambda r:(float(r['violation']),float(r['energy_j']),int(r['episode'])))['episode'])
            selected=next(r for r in selections if int(r['episode'])==selected_ep)
            candidates.append(dict(seed=seed,price=price,episode=selected_ep,
                violation=float(selected['violation']),energy_j=float(selected['energy_j']),
                worst_outage=float(selected['worst_outage']),source=str(path)))
            inputs.extend((path/'selection.csv',path/'selected.pt',path/'training.csv',path/'validation.csv'))
            training_audits.append(dict(price=price,seed=seed,training_episodes=300,
                validation_records=252,selection_reconstructed=True))
            manifest=json.loads((path/'manifest.json').read_text())
            for file,digest in manifest['source_hashes'].items():
                assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
        chosen=min(candidates,key=lambda r:(r['violation'],r['energy_j'],r['episode']!=0,r['price'],r['episode']))
        choices.append(chosen)
    # No development metrics are opened until the cross-price choice is fixed.
    (ROOT/'selected_by_validation.json').write_text(json.dumps(choices,indent=2))
    rows=[]
    for price in (1,2,4):
        for seed in (11,22):
            raw=folder(price,seed)/'development_raw.csv'
            inputs.append(raw)
            for original in read(raw):
                row=dict(original,method=f'Residual-DQN price{price}',selection_price=price)
                rows.append(row)
                choice=next(r for r in choices if r['seed']==seed)
                if choice['price']==price:
                    rows.append(dict(row,method=SELECTED))
    old=Path('results/major_revision_20260920_predictive_residual/development_all.csv')
    inputs.append(old)
    rows.extend(r for r in read(old) if r['method'] in ('PRS','Full array','Fixed residual -1','Fixed residual +1'))
    for part in ('prs','cost'):
        path=Path(f'results/major_revision_20260920_margin_calibration/{part}')
        assert (path/'COMPLETE').exists()
        inputs.append(path/'development_raw.csv')
        rows.extend(read(inputs[-1]))
    mixture=Path('results/major_revision_20260920_mixture_calibration')
    assert (mixture/'COMPLETE').exists()
    inputs.append(mixture/'development_raw.csv')
    rows.extend(read(inputs[-1]))
    fields=('scenario','trajectory','seed','method','energy_j','outage','switching_j','pilot_j',
            'operating_j','transitions','short_sleep_reactivations','mean_count','selection_price','measurement')
    rows=[{k:r.get(k,'') for k in fields} for r in rows]
    keys={(int(r['scenario']),int(r['trajectory']),int(r['seed']),r['method']) for r in rows}
    assert len(rows)==len(keys)==720
    error=max(abs(float(r['energy_j'])-float(r['operating_j'])-float(r['switching_j'])-
                  float(r['pilot_j'])) for r in rows)
    assert error<1e-7
    rng=np.random.default_rng(977900000)
    summaries,pairs,gates,seed_summaries=[],[],[],[]
    for j in range(3):
        group=[r for r in rows if int(r['scenario'])==j]
        matrices={}
        for method in sorted({r['method'] for r in group}):
            subset=[r for r in group if r['method']==method]
            seeds=sorted({int(r['seed']) for r in subset})
            lookup={(int(r['seed']),int(r['trajectory'])):r for r in subset}
            assert len(subset)==len(seeds)*16
            si=rng.integers(len(seeds),size=(4000,len(seeds)))
            ti=rng.integers(16,size=(4000,16))
            row=dict(scenario=j,method=method,seeds=len(seeds),trajectories=16,
                     interpretation='expected episode mixture' if method=='Episode mixture' else 'rollout')
            for metric in ('energy_j','outage'):
                data=np.array([[float(lookup[s,i][metric]) for i in range(16)] for s in seeds])
                matrices[method,metric]=data
                boot=data[si[:,:,None],ti[:,None,:]].mean((1,2))
                row.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                            metric+'_high':float(np.quantile(boot,.975))})
            summaries.append(row)
            for index,seed in enumerate(seeds):
                seed_summaries.append(dict(scenario=j,method=method,seed=seed,
                    energy_j=float(matrices[method,'energy_j'][index].mean()),
                    outage=float(matrices[method,'outage'][index].mean())))
        for baseline in BASELINES:
            row=dict(scenario=j,method=SELECTED,baseline=baseline)
            si=rng.integers(2,size=(4000,2))
            ti=rng.integers(16,size=(4000,16))
            for metric in ('energy_j','outage'):
                data=matrices[SELECTED,metric]-matrices[baseline,metric]
                boot=data[si[:,:,None],ti[:,None,:]].mean((1,2))
                row.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                            metric+'_high':float(np.quantile(boot,.975))})
            pairs.append(row)
        feasible=[b for b in BASELINES if matrices[b,'outage'].mean()<=.05]
        best=min(feasible,key=lambda b:matrices[b,'energy_j'].mean()) if feasible else None
        comparison=next((r for r in pairs if r['scenario']==j and r['baseline']==best),None)
        selected_summary=next(r for r in summaries if r['scenario']==j and r['method']==SELECTED)
        gates.append(dict(scenario=j,best_mean_feasible_control=best,
            each_seed_mean_feasible=bool(np.all(matrices[SELECTED,'outage'].mean(1)<=.05)),
            selected_mean_outage=float(matrices[SELECTED,'outage'].mean()),
            selected_outage_ci95_high=selected_summary['outage_high'],
            outage_interval_wholly_below_target=selected_summary['outage_high']<=.05,
            selected_mean_energy_j=float(matrices[SELECTED,'energy_j'].mean()),
            beats_all_mean_feasible_controls=bool(comparison and comparison['energy_j']<-1e-9),
            paired_energy_interval_below_zero=bool(comparison and comparison['energy_j_high']<0),
            any_trained_selection=any(r['episode']>0 for r in choices)))
    _write_rows(ROOT/'development_all.csv',rows)
    _write_rows(ROOT/'summary.csv',summaries)
    _write_rows(ROOT/'paired.csv',pairs)
    _write_rows(ROOT/'per_seed.csv',seed_summaries)
    audit=dict(records=720,unique_keys=len(keys),energy_identity_max_error_j=error,
        validation_choices=choices,development_gates=gates,bootstrap_resamples=4000,bootstrap_seed=977900000,
        training_audits=training_audits,report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='already-viewed development; two training seeds; not independent confirmation',
        record_interpretation='720 aggregate rows include 96 selected-policy duplicates and 48 analytically integrated mixture expectations; they are not 720 independent rollouts',
        interval_interpretation='crossed seed/trajectory bootstrap; two learned seeds provide limited training-variability information; mean feasibility is not a reliability guarantee',
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Bounded price calibration and stronger controls aggregated; no final-test claim.\n')
    print(json.dumps(dict(choices=choices,gates=gates),indent=2))


if __name__=='__main__':
    run()
