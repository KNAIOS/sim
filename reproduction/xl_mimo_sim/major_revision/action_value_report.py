"""Audit frozen choices before summarizing stage B; development never selects."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.experiment import _write_rows
from major_revision.action_value_study import ROOT,choose_baseline
from major_revision.constraint_training import selection_key
from major_revision.ready_report import read


def screening_gate(episode,candidate,reference,prior):
    return bool(episode>0 and reference is not None and candidate['outage']<=.05
        and candidate['energy_j']<=.98*reference['energy_j']
        and candidate['energy_j']<=.98*prior['energy_j']
        and candidate['outage']<=reference['outage']+.005)


def numeric_rows(path):
    rows=read(path)
    for r in rows:
        for k in ('scenario','trajectory','seed','episode','minimum','selected_episode'):
            if k in r:
                r[k]=int(r[k])
        for k in ('energy_j','outage','margin','operating_j','switching_j','pilot_j','td_loss'):
            if k in r:
                r[k]=float(r[k])
    return rows


def means(group):
    return {k:float(np.mean([r[k] for r in group])) for k in ('energy_j','outage')}


def main():
    parts=['baselines','shared_seed11','shared_seed22']
    inputs=[]
    for part in parts:
        folder=ROOT/part
        assert (folder/'COMPLETE').exists(),folder
        manifest=json.loads((folder/'manifest.json').read_text())
        for path,digest in manifest['source_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        inputs.extend([folder/'manifest.json',*folder.glob('*.csv'),*folder.glob('*.pt')])
    baseline=numeric_rows(ROOT/'baselines/validation_raw.csv')
    chosen=numeric_rows(ROOT/'baselines/selected.csv')
    assert len(baseline)==936 and len(chosen)==6
    baseline_means=[]
    for j in range(3):
        for mode in ('prs','cost'):
            group=[r for r in baseline if r['scenario']==j and r['method']==mode]
            candidates=[]
            for minimum in (1,4):
                for margin in (.9,1.,1.04,1.08,1.16,1.24):
                    cell=[r for r in group if r['minimum']==minimum and r['margin']==margin]
                    assert len(cell)==12
                    candidates.append(dict(scenario=j,method=mode,minimum=minimum,margin=margin,**means(cell)))
            selected=choose_baseline(candidates)
            saved=next(r for r in chosen if r['scenario']==j and r['method']==mode)
            for k in ('minimum','margin','energy_j','outage'):
                assert abs(selected[k]-saved[k])<1e-10
            baseline_means.append(selected)
        for mode in ('prior','full'):
            group=[r for r in baseline if r['scenario']==j and r['method']==mode]
            assert len(group)==12
            baseline_means.append(dict(scenario=j,method=mode,minimum=1,margin=1.,**means(group)))
    validation_means,choices,gates=[],[],[]
    for seed in (11,22):
        folder=ROOT/f'shared_seed{seed}'
        history=numeric_rows(folder/'training.csv')
        assert len(history)==200 and {r['episode'] for r in history}==set(range(1,201))
        assert all(np.isfinite(r['td_loss']) for r in history)
        validation=numeric_rows(folder/'validation.csv')
        assert len(validation)==180
        episodes=(0,50,100,150,200)
        selected_episode=min(episodes,key=lambda e:selection_key([r for r in validation if r['episode']==e])[:2])
        selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
        endpoint=torch.load(folder/f'checkpoint_{selected_episode:03d}.pt',map_location='cpu',weights_only=True)
        assert selected['episode']==selected_episode
        assert all(torch.equal(v,endpoint['online'][k]) for k,v in selected['online'].items())
        np.testing.assert_allclose(selected['validation_key'],selection_key(
            [r for r in validation if r['episode']==selected_episode])[:2],rtol=0,atol=1e-12)
        choices.append(dict(seed=seed,episode=selected_episode,parameters=selected['parameters']))
        for e in episodes:
            for j in range(3):
                group=[r for r in validation if r['episode']==e and r['scenario']==j]
                assert len(group)==12
                row=dict(seed=seed,episode=e,scenario=j,**means(group))
                validation_means.append(row)
                prior=next(r for r in baseline_means if r['scenario']==j and r['method']=='prior')
                if e==0:
                    for k in ('energy_j','outage'):
                        assert abs(row[k]-prior[k])<1e-9
                if e==selected_episode:
                    feasible=[r for r in baseline_means if r['scenario']==j and r['method'] in ('prs','cost') and r['outage']<=.05]
                    reference=min(feasible,key=lambda r:r['energy_j']) if feasible else None
                    gates.append(dict(seed=seed,scenario=j,episode=e,
                        passed=screening_gate(e,row,reference,prior),
                        candidate_energy_j=row['energy_j'],candidate_outage=row['outage'],
                        reference_method=reference['method'] if reference else 'none',
                        reference_energy_j=reference['energy_j'] if reference else None,
                        reference_outage=reference['outage'] if reference else None,
                        prior_energy_j=prior['energy_j'],prior_outage=prior['outage'],
                        beats_search_energy=bool(reference and row['energy_j']<reference['energy_j']),
                        candidate_feasible=bool(row['outage']<=.05)))
    permitted=any(all(r['passed'] for r in gates if r['scenario']==j) for j in range(3))
    (ROOT/'screening.json').write_text(json.dumps(dict(choices=choices,gates=gates,permit_flat=permitted),indent=2))
    if permitted:
        (ROOT/'PERMIT_FLAT').write_text('Both seeds pass the frozen validation gate in a common scenario.\n')
    else:
        assert not (ROOT/'PERMIT_FLAT').exists()
    # Only after the independent gate reconstruction, summarize viewed development.
    rows=numeric_rows(ROOT/'baselines/development.csv')
    for seed in (11,22):
        extra=numeric_rows(ROOT/f'shared_seed{seed}/development.csv')
        assert len(extra)==48
        for r in extra:
            r['method']='Action-value DQN'
        rows.extend(extra)
    assert len(rows)==len({(r['scenario'],r['trajectory'],r['seed'],r['method']) for r in rows})==288
    identity=max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in rows)
    assert identity<1e-7
    rng=np.random.default_rng(998900000)
    summaries,pairs,per_seed=[],[],[]
    for j in range(3):
        group=[r for r in rows if r['scenario']==j]
        matrices={}
        for method in sorted({r['method'] for r in group}):
            sample=[r for r in group if r['method']==method]
            seeds=sorted({r['seed'] for r in sample})
            lookup={(r['seed'],r['trajectory']):r for r in sample}
            ti=rng.integers(16,size=(4000,16))
            si=rng.integers(len(seeds),size=(4000,len(seeds)))
            summary=dict(scenario=j,method=method,training_seeds=len(seeds) if method=='Action-value DQN' else 0,trajectories=16)
            for metric in ('energy_j','outage'):
                values=np.array([[lookup[s,i][metric] for i in range(16)] for s in seeds])
                matrices[method,metric]=values
                boot=values[si[:,:,None],ti[:,None,:]].mean((1,2))
                summary.update({metric:float(values.mean()),metric+'_low':float(np.quantile(boot,.025)),metric+'_high':float(np.quantile(boot,.975))})
            summaries.append(summary)
            for seed in seeds:
                per_seed.append(dict(scenario=j,method=method,seed=seed,**means([r for r in sample if r['seed']==seed])))
        for reference in ('prs','cost','prior'):
            ti=rng.integers(16,size=(4000,16))
            si=rng.integers(2,size=(4000,2))
            pair=dict(scenario=j,method='Action-value DQN',reference=reference)
            for metric in ('energy_j','outage'):
                delta=matrices['Action-value DQN',metric]-matrices[reference,metric]
                boot=delta[si[:,:,None],ti[:,None,:]].mean((1,2))
                pair.update({metric:float(delta.mean()),metric+'_low':float(np.quantile(boot,.025)),metric+'_high':float(np.quantile(boot,.975))})
            pairs.append(pair)
    for name,data in (('summary',summaries),('paired',pairs),('per_seed',per_seed),
                      ('selected_validation',validation_means),('baseline_validation',baseline_means)):
        _write_rows(ROOT/f'{name}.csv',data)
    audit=dict(choices=choices,screening_pass=permitted,training_episodes=400,
        baseline_validation_records=936,learned_validation_records=360,development_records=288,
        energy_identity_max_error_j=identity,role='method-development screen; not final confirmation',
        bootstrap='4000 crossed training-seed/paired-trajectory resamples; only two trained seeds',
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Stage-B shared-model screening audited; inspect gate before any ablation.\n')
    print(json.dumps(dict(choices=choices,gate=permitted,validation=gates,development=summaries),indent=2))


if __name__=='__main__':
    main()
