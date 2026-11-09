"""Reconstruct closed-loop choices and reject post-hoc gate relaxation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.action_value_report import numeric_rows,means
from major_revision.action_value_study import choose_baseline,MARGINS
from major_revision.rollout_value_data import ROOT
from major_revision.rollout_value_closed_loop import PRICES


def promotion(candidate,reference,local):
    core=bool(reference is not None and candidate['outage']<=.05
        and candidate['energy_j']<=.98*reference['energy_j']
        and candidate['outage']<=reference['outage']+.005)
    ablation=bool(candidate['outage']<=.05 and (local['outage']>.05 or
        (candidate['energy_j']<=.98*local['energy_j'] and candidate['outage']<=local['outage']+.005)))
    return dict(passed=core and ablation,core_pass=core,ablation_pass=ablation)


def rectangular(rows):
    """Report-only union schema for mixed learned/baseline records."""
    fields=list(dict.fromkeys(key for row in rows for key in row))
    return [{key:row.get(key,'') for key in fields} for row in rows]


def main():
    folder=ROOT/'closed_loop'
    hashes={}
    for part in ('baselines','learned'):
        source=folder/part
        assert (source/'COMPLETE').exists(),part
        manifest=json.loads((source/'manifest.json').read_text())
        for path,digest in manifest['hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        for path in source.iterdir():
            if path.is_file():
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=numeric_rows(folder/'baselines/raw.csv')
    learned=numeric_rows(folder/'learned/raw.csv')
    for row in learned:
        row['price']=float(row['price'])
        row['changed_from_mu_fraction']=float(row['changed_from_mu_fraction'])
    assert len(raw)==936 and len(learned)==720
    assert len({(r['scenario'],r['trajectory'],r['method'],r['minimum'],r['margin']) for r in raw})==936
    assert len({(r['scenario'],r['trajectory'],r['seed'],r['horizon'],r['price']) for r in learned})==720
    identity=max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in raw+learned)
    assert identity<1e-8
    saved_baselines=numeric_rows(folder/'baselines/selected.csv')
    saved_learned=numeric_rows(folder/'learned/selected.csv')
    selections,selected_raw=[],[]
    for j in range(3):
        for mode in ('prs','cost'):
            candidates=[]
            for minimum in (1,4):
                for margin in MARGINS:
                    group=[r for r in raw if r['scenario']==j and r['method']==mode and r['minimum']==minimum and r['margin']==margin]
                    assert len(group)==12
                    candidates.append(dict(method=mode,scenario=j,minimum=minimum,margin=margin,**means(group)))
            chosen=choose_baseline(candidates)
            saved=next(r for r in saved_baselines if r['scenario']==j and r['method']==mode)
            for key in ('minimum','margin','energy_j','outage'):
                assert abs(saved[key]-chosen[key])<1e-10
            selections.append(chosen)
            selected_raw.extend(dict(**r) for r in raw if r['scenario']==j and r['method']==mode
                and r['minimum']==chosen['minimum'] and r['margin']==chosen['margin'])
        for mode in ('mu','full'):
            group=[r for r in raw if r['scenario']==j and r['method']==mode]
            assert len(group)==12
            selections.append(dict(method=mode,scenario=j,**means(group)))
            selected_raw.extend(group)
        for seed in (11,22):
            for horizon in ('C4','C16'):
                candidates=[]
                for price in PRICES:
                    group=[r for r in learned if r['scenario']==j and r['seed']==seed and r['horizon']==horizon and r['price']==price]
                    assert len(group)==12
                    candidates.append(dict(method=horizon,scenario=j,seed=seed,price=price,**means(group),
                        changed_from_mu_fraction=float(np.mean([r['changed_from_mu_fraction'] for r in group]))))
                feasible=[r for r in candidates if r['outage']<=.05]
                chosen=min(feasible,key=lambda r:(r['energy_j'],r['price'])) if feasible else min(candidates,key=lambda r:(r['outage'],r['energy_j'],r['price']))
                saved=next(r for r in saved_learned if r['scenario']==j and r['seed']==seed and r['horizon']==horizon)
                for key in ('price','energy_j','outage','changed_from_mu_fraction'):
                    assert abs(float(saved[key])-chosen[key])<1e-10
                selections.append(chosen)
                for row in learned:
                    if row['scenario']==j and row['seed']==seed and row['horizon']==horizon and row['price']==chosen['price']:
                        selected_raw.append(dict(row,method=horizon))
    assert len(selected_raw)==288
    gates=[]
    for j in range(3):
        refs=[r for r in selections if r['scenario']==j and r['method'] in ('prs','cost') and r['outage']<=.05]
        reference=min(refs,key=lambda r:r['energy_j']) if refs else None
        for seed in (11,22):
            candidate=next(r for r in selections if r['scenario']==j and r.get('seed')==seed and r['method']=='C16')
            local=next(r for r in selections if r['scenario']==j and r.get('seed')==seed and r['method']=='C4')
            gates.append(dict(scenario=j,seed=seed,candidate=candidate,reference=reference,local=local,
                **promotion(candidate,reference,local)))
    permitted=any(all(r['passed'] for r in gates if r['scenario']==j) for j in range(3))
    # Preserve the first report's partial CSV/source after its union-schema error.
    # Policy data, selection rules and gate are unchanged; only serialization differs.
    output=ROOT/'closed_audit_v2'
    output.mkdir(exist_ok=False)
    _write_rows(output/'selected.csv',rectangular(selections))
    _write_rows(output/'selected_raw.csv',rectangular(selected_raw))
    summary=[]
    for j in range(3):
        for method in ('C4','C16','prs','cost','mu','full'):
            group=[r for r in selected_raw if r['scenario']==j and r['method']==method]
            summary.append(dict(scenario=j,method=method,**means(group)))
    _write_rows(output/'summary.csv',summary)
    (output/'gate.json').write_text(json.dumps(dict(permit_confirmation=permitted,gates=gates),indent=2))
    (output/'audit.json').write_text(json.dumps(dict(input_hashes=hashes,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baseline_records=936,
        learned_records=720,selected_records=288,energy_identity_error=identity,
        role='validation-selected descriptive results, not independent confirmation'),indent=2))
    (output/'COMPLETE').write_text('Closed-loop screen fully audited; inspect gate before further experiments.\n')
    if permitted:
        (ROOT/'PERMIT_CONFIRMATION').write_text('Both C16 seeds pass in at least one common scenario.\n')
    print(json.dumps(dict(permit_confirmation=permitted,summary=summary,gates=gates),indent=2))


if __name__=='__main__':
    main()
