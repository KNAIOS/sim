"""Recompute all choices, paired differences, and the frozen screen gate."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.action_value_study import MARGINS,choose_baseline
from major_revision.absolute_return_screen import ROOT,METHODS,COUNT


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return [{k:(v if k in ('method','horizon') else float(v)) for k,v in r.items()} for r in csv.DictReader(stream)]


def mean(group):
    return {k:float(np.mean([r[k] for r in group])) for k in ('energy_j','outage')}


def screen_gate(candidate,reference,local):
    core=bool(reference is not None and candidate['outage']<=.05 and
        candidate['energy_j']<=.98*reference['energy_j'] and
        candidate['outage']<=reference['outage']+.005)
    ablation=bool(candidate['outage']<=.05 and (local['outage']>.05 or
        (candidate['energy_j']<=.98*local['energy_j'] and candidate['outage']<=local['outage']+.005)))
    return dict(core_pass=core,ablation_pass=ablation,passed=core and ablation)


def rectangular(records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    return [{k:r.get(k,'') for k in fields} for r in records]


def main():
    files=[Path(__file__),Path(__file__).with_name('test_absolute_return_audit.py'),
        Path(__file__).with_name('ABSOLUTE_RETURN_SCREEN_PLAN.md')]
    for part in ('baselines','learned'):
        folder=ROOT/part
        assert (folder/'COMPLETE').exists()
        manifest=json.loads((folder/'manifest.json').read_text())
        for path,digest in manifest['hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        files.extend(p for p in folder.iterdir() if p.is_file())
    baseline=rows(ROOT/'baselines/raw.csv')
    learned=rows(ROOT/'learned/raw.csv')
    assert len(baseline)==1200 and len(learned)==96
    assert len({(r['method'],r['scenario'],r['minimum'],r['margin'],r['trajectory']) for r in baseline})==1200
    assert len({(r['seed'],r['horizon'],r['scenario'],r['trajectory']) for r in learned})==96
    for r in baseline+learned:
        assert np.all(np.isfinite([r[k] for k in ('energy_j','outage','operating_j','switching_j','pilot_j')]))
        assert 0<=r['outage']<=1
    identity=max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in baseline+learned)
    assert identity<1e-8
    saved=rows(ROOT/'baselines/selected.csv')
    selections,selected_raw=[],[]
    for j in range(3):
        for mode in METHODS:
            choices=[]
            for minimum in (1,4):
                for margin in MARGINS:
                    group=[r for r in baseline if r['method']==mode and r['scenario']==j and
                           r['minimum']==minimum and r['margin']==margin]
                    assert len(group)==COUNT and {r['trajectory'] for r in group}==set(range(COUNT))
                    choices.append(dict(scenario=j,method=mode,minimum=minimum,margin=margin,**mean(group)))
            choice=choose_baseline(choices)
            stored=next(r for r in saved if r['method']==mode and r['scenario']==j)
            for key in ('minimum','margin','energy_j','outage'):
                assert abs(choice[key]-stored[key])<1e-10
            selections.append(choice)
            selected_raw.extend(r for r in baseline if r['method']==mode and r['scenario']==j and
                r['minimum']==choice['minimum'] and r['margin']==choice['margin'])
        for mode in ('mu','full'):
            group=[r for r in baseline if r['method']==mode and r['scenario']==j]
            assert len(group)==COUNT
            selections.append(dict(scenario=j,method=mode,**mean(group)))
            selected_raw.extend(group)
        for seed in (11,22):
            for horizon in ('C4','C16'):
                group=[r for r in learned if r['scenario']==j and r['seed']==seed and r['horizon']==horizon]
                assert len(group)==COUNT
                selections.append(dict(scenario=j,method=horizon,seed=seed,price=2.,**mean(group),
                    changed_from_mu_fraction=float(np.mean([r['changed_from_mu_fraction'] for r in group]))))
                selected_raw.extend(group)
    assert len(selected_raw)==240
    gates,paired,paired_raw=[],[],[]
    for j in range(3):
        feasible=[r for r in selections if r['scenario']==j and r['method'] in METHODS and r['outage']<=.05]
        reference=min(feasible,key=lambda r:(r['energy_j'],r['method'])) if feasible else None
        for seed in (11,22):
            candidate=next(r for r in selections if r['scenario']==j and r.get('seed')==seed and r['method']=='C16')
            local=next(r for r in selections if r['scenario']==j and r.get('seed')==seed and r['method']=='C4')
            gates.append(dict(scenario=j,seed=seed,candidate=candidate,reference=reference,local=local,
                **screen_gate(candidate,reference,local)))
            if reference is None:
                continue
            deltas=[]
            for i in range(COUNT):
                proposed=next(r for r in selected_raw if r['scenario']==j and r.get('seed')==seed and r['method']=='C16' and r['trajectory']==i)
                comparator=next(r for r in selected_raw if r['scenario']==j and r['method']==reference['method'] and r['trajectory']==i)
                row=dict(scenario=j,seed=seed,trajectory=i,reference=reference['method'],
                    energy_difference_j=proposed['energy_j']-comparator['energy_j'],
                    outage_difference=proposed['outage']-comparator['outage'])
                paired_raw.append(row)
                deltas.append([row['energy_difference_j'],row['outage_difference']])
            d=np.asarray(deltas)
            paired.append(dict(scenario=j,seed=seed,reference=reference['method'],trajectories=COUNT,
                energy_difference_mean_j=float(d[:,0].mean()),energy_difference_sd_j=float(d[:,0].std(ddof=1)),
                outage_difference_mean=float(d[:,1].mean()),outage_difference_sd=float(d[:,1].std(ddof=1)),
                lower_energy_trajectories=int((d[:,0]<0).sum())))
    permit=any(all(r['passed'] for r in gates if r['scenario']==j) for j in range(3))
    output=ROOT/'audit'
    output.mkdir(exist_ok=False)
    _write_rows(output/'selections.csv',rectangular(selections))
    _write_rows(output/'selected_raw.csv',rectangular(selected_raw))
    _write_rows(output/'paired.csv',paired)
    _write_rows(output/'paired_raw.csv',paired_raw)
    summary=[]
    for j in range(3):
        for method in (*METHODS,'mu','full','C4','C16'):
            group=[r for r in selected_raw if r['scenario']==j and r['method']==method]
            summary.append(dict(scenario=j,method=method,**mean(group)))
    _write_rows(output/'summary.csv',summary)
    (output/'gate.json').write_text(json.dumps(dict(permit_confirmation=permit,gates=gates),indent=2))
    (output/'audit.json').write_text(json.dumps(dict(input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        baseline_records=1200,learned_records=96,selected_records=240,energy_identity_error=identity,
        shared_external_trajectories=24,price=2.,no_new_training=True,
        role='validation-selected small screen; paired SD is not a confidence interval'),indent=2))
    (output/'COMPLETE').write_text('All settings, paired differences and gate audited.\n')
    if permit:
        (ROOT/'PERMIT_CONFIRMATION').write_text('Freeze independent confirmation protocol before new evaluations.\n')
    print(json.dumps(dict(summary=summary,gates=gates,paired=paired,permit_confirmation=permit),indent=2))


if __name__=='__main__':
    main()
