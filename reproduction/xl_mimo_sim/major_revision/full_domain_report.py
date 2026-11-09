"""Audit frozen validation choices before summarizing paired development results."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.experiment import _write_rows
from major_revision.ready_report import read
from major_revision.full_domain_study import ROOT, PRICES, MARGINS


def selection_from_validation(rows):
    candidates=[]
    for price in PRICES:
        for episode in (100,200,300):
            group=[r for r in rows if int(r['price'])==price and int(r['episode'])==episode]
            assert len(group)==36
            worst=max(np.mean([float(r['outage']) for r in group if int(r['scenario'])==j])
                      for j in range(3))
            energy=float(np.mean([float(r['energy_j']) for r in group]))
            candidates.append(dict(price=price,episode=episode,violation=max(0.,worst-.05),
                energy_j=energy,worst_outage=float(worst)))
    return min(candidates,key=lambda r:(r['violation'],r['energy_j']))


def summarize(rows):
    rng=np.random.default_rng(983900000)
    summaries,paired,seeds,components,gates=[],[],[],[],[]
    for j in range(3):
        group=[r for r in rows if int(r['scenario'])==j]
        matrices={}
        for method in ('Full-domain DQN','prs','cost','full'):
            sample=[r for r in group if r['method']==method]
            ids=sorted({int(r['seed']) for r in sample})
            lookup={(int(r['seed']),int(r['trajectory'])):r for r in sample}
            assert len(lookup)==len(sample)==len(ids)*16
            ti=rng.integers(16,size=(10000,16))
            si=rng.integers(len(ids),size=(10000,len(ids)))
            summary=dict(scenario=j,method=method,policy_instances=len(ids),
                         training_seeds=len(ids) if method=='Full-domain DQN' else 0,trajectories=16)
            for metric in ('energy_j','outage'):
                data=np.array([[float(lookup[s,i][metric]) for i in range(16)] for s in ids])
                matrices[method,metric]=data
                by_trace=data.mean(0)
                boot=by_trace[ti].mean(1)
                crossed=data[si[:,:,None],ti[:,None,:]].mean((1,2))
                summary.update({metric:float(data.mean()),
                    metric+'_trace_low':float(np.quantile(boot,.025)),
                    metric+'_trace_high':float(np.quantile(boot,.975)),
                    metric+'_trace_upper95':float(np.quantile(boot,.95)),
                    metric+'_crossed_low':float(np.quantile(crossed,.025)),
                    metric+'_crossed_high':float(np.quantile(crossed,.975))})
            summaries.append(summary)
            components.append(dict(scenario=j,method=method,**{
                k:float(np.mean([float(r[k]) for r in sample])) for k in
                ('operating_j','switching_j','pilot_j','mean_count','below_four_fraction')}))
            for index,seed in enumerate(ids):
                seeds.append(dict(scenario=j,method=method,seed=seed,
                    energy_j=float(matrices[method,'energy_j'][index].mean()),
                    outage=float(matrices[method,'outage'][index].mean())))
        for baseline in ('prs','cost','full'):
            ti=rng.integers(16,size=(10000,16))
            si=rng.integers(2,size=(10000,2))
            result=dict(scenario=j,baseline=baseline,method='Full-domain DQN')
            for metric in ('energy_j','outage'):
                data=matrices['Full-domain DQN',metric]-matrices[baseline,metric]
                trace=data.mean(0)[ti].mean(1)
                crossed=data[si[:,:,None],ti[:,None,:]].mean((1,2))
                result.update({metric:float(data.mean()),
                    metric+'_trace_low':float(np.quantile(trace,.025)),
                    metric+'_trace_high':float(np.quantile(trace,.975)),
                    metric+'_crossed_low':float(np.quantile(crossed,.025)),
                    metric+'_crossed_high':float(np.quantile(crossed,.975))})
            paired.append(result)
        selected=next(r for r in summaries if r['scenario']==j and r['method']=='Full-domain DQN')
        baselines_feasible=all(matrices[b,'outage'].mean()<=.05 for b in ('prs','cost'))
        each_seed_better=all(np.all(matrices['Full-domain DQN','energy_j'].mean(1)<
                            matrices[b,'energy_j'].mean()) for b in ('prs','cost'))
        ci_negative=all(next(r for r in paired if r['scenario']==j and r['baseline']==b)
                        ['energy_j_trace_high']<0 for b in ('prs','cost'))
        outage_pass=selected['outage']<=.05 and selected['outage_trace_upper95']<=.05
        gates.append(dict(scenario=j,both_search_controls_mean_feasible=bool(baselines_feasible),
            each_seed_lower_energy_than_both=bool(each_seed_better),
            both_paired_intervals_below_zero=bool(ci_negative),outage_gate=bool(outage_pass),
            promote=bool(baselines_feasible and each_seed_better and ci_negative and outage_pass)))
    return summaries,paired,seeds,components,gates


def run():
    inputs=[]
    choices=[]
    parts=[ROOT/'baselines_v2',ROOT/'seed11',ROOT/'seed22']
    for folder in parts:
        assert (folder/'COMPLETE').exists(),folder
        manifest=json.loads((folder/'manifest.json').read_text())
        for file,digest in manifest['source_hashes'].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
        inputs.append(folder/'manifest.json')
    for seed in (11,22):
        folder=ROOT/f'seed{seed}'
        validation=read(folder/'validation.csv')
        training=read(folder/'training.csv')
        assert len(training)==900
        assert {(int(r['price']),int(r['episode'])) for r in training}=={
            (p,e) for p in PRICES for e in range(1,301)}
        assert len(validation)==324
        assert {(int(r['price']),int(r['episode']),int(r['scenario']),int(r['trajectory'])) for r in validation}=={
            (p,e,j,i) for p in PRICES for e in (100,200,300) for j in range(3) for i in range(12)}
        chosen=selection_from_validation(validation)
        saved=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
        assert saved['price']==chosen['price'] and saved['episode']==chosen['episode']
        assert saved['minimum_count']==1 and saved['actions']==16 and saved['state_dimension']==235
        assert np.allclose(saved['validation_key'],[chosen['violation'],chosen['energy_j']],rtol=0,atol=1e-10)
        choices.append(dict(seed=seed,**chosen))
        inputs.extend(folder/n for n in ('validation.csv','training.csv','selected.pt','selection.csv'))
    baseline=ROOT/'baselines_v2'
    calibration=read(baseline/'validation_raw.csv')
    selections=read(baseline/'selected.csv')
    assert len(calibration)==432
    for setting in selections:
        j,method=int(setting['scenario']),setting['mode']
        candidates=[]
        for margin in MARGINS:
            sample=[r for r in calibration if int(r['scenario'])==j and r['method']==method
                    and float(r['margin'])==margin]
            assert len(sample)==12
            candidates.append(dict(margin=margin,energy=float(np.mean([float(r['energy_j']) for r in sample])),
                outage=float(np.mean([float(r['outage']) for r in sample]))))
        feasible=[r for r in candidates if r['outage']<=.05]
        chosen=min(feasible,key=lambda r:r['energy']) if feasible else min(candidates,key=lambda r:(r['outage'],r['energy']))
        assert chosen['margin']==float(setting['margin'])
    inputs.extend((baseline/'validation_raw.csv',baseline/'selected.csv'))
    # Validation selection is independently reconstructed before development data.
    (ROOT/'choices_verified.json').write_text(json.dumps(choices,indent=2))
    rows=[]
    for folder in parts:
        path=folder/'development_raw.csv'
        inputs.append(path)
        rows.extend(read(path))
    fields=('scenario','trajectory','seed','method','energy_j','outage','operating_j','switching_j',
            'pilot_j','mean_count','below_four_fraction','transitions','short_sleep_reactivations')
    rows=[{k:r[k] for k in fields} for r in rows]
    assert len(rows)==len({(r['scenario'],r['trajectory'],r['seed'],r['method']) for r in rows})==240
    error=max(abs(float(r['energy_j'])-float(r['operating_j'])-float(r['switching_j'])-float(r['pilot_j'])) for r in rows)
    assert error<1e-7
    summary,paired,seeds,components,gates=summarize(rows)
    for name,records in (('development_all',rows),('summary',summary),('paired',paired),
                         ('per_seed',seeds),('components',components)):
        _write_rows(ROOT/f'{name}.csv',records)
    audit=dict(choices=choices,development_gates=gates,records=240,
        baseline_validation_records=432,learned_validation_records=648,training_episodes=1800,
        energy_identity_max_error_j=error,bootstrap_replicates=10000,bootstrap_seed=983900000,
        interpretation='New development traces, not independent confirmation after promotion; two training seeds only',
        baseline_correction='Use baselines_v2 only; interrupted initial baseline is preserved, not included',
        statistics='Trajectory-paired percentile intervals conditional on the two selected models; crossed seed/trajectory intervals also shown; no guarantee of reliability',
        sources_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Bounded full-domain study audited; inspect promotion gates before any manuscript change.\n')
    print(json.dumps(dict(choices=choices,gates=gates,summary=summary),indent=2))


if __name__=='__main__':
    run()
