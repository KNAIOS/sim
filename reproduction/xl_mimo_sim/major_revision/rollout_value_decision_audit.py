"""Post-screen check: average regression quality is not action-ranking quality."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.action_value_report import numeric_rows
from major_revision.rollout_value_data import ROOT
from major_revision.rollout_value_fit import select_action


def action_result(prediction,truth,reference,price):
    action=select_action(prediction,reference,price)
    predicted=float(prediction[action,0]+price*prediction[action,1])
    realized=float(truth[action,0]+price*truth[action,1])
    return dict(action=action+1,reference=reference+1,changed=int(action!=reference),
        predicted_cost_difference=predicted,sampled_cost_difference=realized,
        sampled_energy_difference=float(truth[action,0]),sampled_outage_difference=float(truth[action,1]),
        predicted_improvement_sample_worse=int(predicted < -1e-6 and realized > 1e-6))


def main():
    gate=json.loads((ROOT/'closed_audit_v2/gate.json').read_text())
    assert not gate['permit_confirmation']
    paths=[Path(__file__),Path(__file__).with_name('DECISION_VALUE_AUDIT_PLAN.md'),
        Path(__file__).with_name('test_rollout_value_decision_audit.py'),
        ROOT/'closed_audit_v2/gate.json',ROOT/'validation/labels.npz',
        ROOT/'closed_loop/learned/selected.csv']
    for name in ('data_audit','fit_audit','closed_audit_v2'):
        manifest=json.loads((ROOT/name/'audit.json').read_text())
        for path,digest in manifest['input_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        paths.append(ROOT/name/'audit.json')
    folder=ROOT/'decision_audit'
    folder.mkdir(exist_ok=False)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (folder/'manifest.json').write_text(json.dumps(dict(hashes=hashes,
        role='post-hoc ranking diagnosis on reference-policy validation states; no policy change'),indent=2))
    data=np.load(ROOT/'validation/labels.npz',allow_pickle=False)
    choices=numeric_rows(ROOT/'closed_loop/learned/selected.csv')
    rows,summary=[],[]
    for seed in (11,22):
        for h,horizon in enumerate(('C4','C16')):
            predictions=np.load(ROOT/f'fit_seed{seed}'/f'{horizon}_predictions.npz',allow_pickle=False)['validation']
            for j in range(3):
                price=float(next(r for r in choices if r['seed']==seed and r['horizon']==horizon and r['scenario']==j)['price'])
                group=[]
                for index,key in enumerate(data['keys']):
                    if key[0]!=j:
                        continue
                    result=action_result(predictions[index],data['difference'][index,:,h,:],int(data['reference'][index]),price)
                    group.append(dict(seed=seed,horizon=horizon,scenario=j,trajectory=int(key[1]),step=int(key[2]),price=price,**result))
                assert len(group)==96
                rows.extend(group)
                summary.append(dict(seed=seed,horizon=horizon,scenario=j,price=price,roots=96,
                    **{k:sum(r[k] for r in group) for k in ('changed','predicted_improvement_sample_worse')},
                    **{k:float(np.mean([r[k] for r in group])) for k in
                       ('predicted_cost_difference','sampled_cost_difference','sampled_energy_difference','sampled_outage_difference')}))
    assert len(rows)==1152
    _write_rows(folder/'raw.csv',rows)
    _write_rows(folder/'summary.csv',summary)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (folder/'COMPLETE').write_text('Post-hoc ranking check complete; choices unchanged.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
