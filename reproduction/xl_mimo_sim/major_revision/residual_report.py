"""Development-only report retaining anchor fallback and fixed-residual controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.ready_report import read

ROOT=Path('results/major_revision_20260920_predictive_residual')
REFERENCE=Path('results/major_revision_20260920_time_limit')
METHOD='Predictive-residual DQN'


def run():
    inputs=[REFERENCE/'development_all.csv']
    rows=[r for r in read(inputs[0]) if r['method'] in
          ('Continuing-ready DQN','PRS','Switch-aware PRS','Full array')]
    selections={}
    for seed in (11,22):
        folder=ROOT/f'seed{seed}'
        assert (folder/'COMPLETE').exists()
        inputs.append(folder/'development_raw.csv')
        rows.extend(read(inputs[-1]))
        selections[seed]=int(read(folder/'selection.csv')[-1]['selected_episode'])
    inputs.append(ROOT/'seed11/fixed_raw.csv')
    rows.extend(read(inputs[-1]))
    keys=[(r['scenario'],r['trajectory'],r['seed'],r['method']) for r in rows]
    assert len(rows)==len(set(keys))==432
    reference={(r['scenario'],r['trajectory']):r for r in rows if r['method']=='PRS'}
    fallback_checked=0
    for r in rows:
        if r['method']==METHOD and selections[int(r['seed'])]==0:
            anchor=reference[r['scenario'],r['trajectory']]
            for metric in ('energy_j','outage','operating_j','switching_j','pilot_j','transitions','mean_count'):
                assert abs(float(r[metric])-float(anchor[metric]))<1e-9
            fallback_checked+=1
    for seed in (11,22):
        manifest=json.loads((ROOT/f'seed{seed}/manifest.json').read_text())
        for source,expected in manifest['source_hashes'].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,source
    residual=max(abs(float(r['energy_j'])-float(r['operating_j'])-
                     float(r['pilot_j'])-float(r['switching_j'])) for r in rows)
    assert residual<1e-7
    rng=np.random.default_rng(976900000)
    summary,pairs=[],[]
    for j in range(3):
        group=[r for r in rows if int(r['scenario'])==j]
        matrices={}
        for method in sorted({r['method'] for r in group}):
            selected=[r for r in group if r['method']==method]
            seeds=sorted({int(r['seed']) for r in selected})
            lookup={(int(r['seed']),int(r['trajectory'])):r for r in selected}
            assert len(selected)==16*len(seeds)
            si=rng.integers(len(seeds),size=(4000,len(seeds)))
            ti=rng.integers(16,size=(4000,16))
            row=dict(scenario=j,method=method,seeds=len(seeds),trajectories=16)
            for metric in ('energy_j','outage'):
                data=np.array([[float(lookup[s,i][metric]) for i in range(16)] for s in seeds])
                matrices[method,metric]=data
                boot=data[si[:,:,None],ti[:,None,:]].mean((1,2))
                row.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                            metric+'_high':float(np.quantile(boot,.975))})
            summary.append(row)
        for baseline in ('PRS','Switch-aware PRS','Fixed residual -1','Fixed residual +1',
                         'Continuing-ready DQN'):
            row=dict(scenario=j,method=METHOD,baseline=baseline)
            si=rng.integers(2,size=(4000,2))
            ti=rng.integers(16,size=(4000,16))
            for metric in ('energy_j','outage'):
                data=matrices[METHOD,metric]-matrices[baseline,metric]
                boot=data[si[:,:,None],ti[:,None,:]].mean((1,2))
                row.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                            metric+'_high':float(np.quantile(boot,.975))})
            row['mean_feasible_energy_signal']=bool(matrices[METHOD,'outage'].mean()<=.05 and
                matrices[baseline,'outage'].mean()<=.05 and row['energy_j']<-1e-9)
            pairs.append(row)
    _write_rows(ROOT/'development_all.csv',rows)
    _write_rows(ROOT/'summary.csv',summary)
    _write_rows(ROOT/'paired.csv',pairs)
    audit=dict(rows=len(rows),unique_keys=len(keys),selected_episodes=selections,
        anchor_fallback_seeds=[s for s,e in selections.items() if e==0],
        fallback_records_checked=fallback_checked,training_source_hashes='match',
        energy_identity_max_error_j=residual,bootstrap_resamples=4000,bootstrap_seed=976900000,
        raw_search_mean_signals=[r for r in pairs if r['mean_feasible_energy_signal'] and
                                r['baseline'] in ('PRS','Switch-aware PRS')],
        search_energy_signals=[r for r in pairs if any(ep>0 for ep in selections.values()) and
                              r['mean_feasible_energy_signal'] and
                              r['baseline'] in ('PRS','Switch-aware PRS')],
        all_models_reverted_to_anchor=all(ep==0 for ep in selections.values()),
        interpretation='A fallback-versus-other-baseline difference is NOT a learning gain.',
        scope='two seeds; reused development trajectories; not final superiority evidence',
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    completion=('Residual pilot completed; all seeds use anchor fallback; no learning gain or promotion.\n'
                if audit['all_models_reverted_to_anchor'] else
                'Residual development pilot completed; not independent confirmation.\n')
    (ROOT/'COMPLETE').write_text(completion)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    run()
