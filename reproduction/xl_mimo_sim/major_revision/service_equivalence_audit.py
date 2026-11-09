"""Audit short-block service invariance on all archived roots and models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.factored_value import ROOT as FIT_ROOT,FactoredNetwork
from major_revision.factored_value_fit import load_data
from major_revision.rollout_value_fit import select_action
from major_revision.service_equivalence import service_signatures,service_projection,project_risk

ROOT=Path('results/major_revision_20260921_service_equivalence')


def summarize_choices(group):
    return dict(roots=len(group),changed_from_reference=sum(r['action']!=r['reference'] for r in group),
        chosen_same_service_as_reference=sum(r['same_service_as_reference'] for r in group),
        impossible_chosen_block_advantage=sum(r['impossible_block_advantage'] for r in group),
        **{key:float(np.mean([r[key] for r in group])) for key in
           ('predicted_energy_difference','predicted_outage_difference','sample_energy_difference',
            'sample_outage_difference','predicted_cost_difference','sample_cost_difference')})


def main():
    data=load_data()
    cfg=make_config('validation')
    previous=json.loads((FIT_ROOT/'fit_audit/audit.json').read_text())
    for path,digest in previous['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('SERVICE_EQUIVALENCE_AUDIT_PLAN.md','service_equivalence.py','test_service_equivalence.py',
         'factored_value.py','factored_value_fit.py','gram_control.py','action_value.py')],
        Path('future_vr/system.py'),FIT_ROOT/'fit_audit/audit.json']
    files.extend(FIT_ROOT/f'fit_seed{s}/{mode}_{suffix}' for s in (11,22)
        for mode in ('structured','free') for suffix in ('final.pt','predictions.npz'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    ROOT.mkdir(exist_ok=False)
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,threads=1,price=2.,
        role='post-hoc local physical consistency check, no new training or policy result'),indent=2))
    physical,metrics,choices,summaries=[],[],[],[]
    label_error=0.
    identity_error=0.
    for split,ds in data.items():
        signatures=service_signatures(cfg,ds['features'])
        projection=service_projection(signatures)
        same=projection>0
        same_pairs=np.triu(same,k=1)
        truth=ds['costs']
        n=len(truth)
        reference=ds['reference']
        counts=[]
        for i,key in enumerate(ds['keys']):
            paired=same_pairs[i]
            risk=truth[i,:,0,1]
            mismatch=float(np.max(np.abs(risk[:,None]-risk[None,:])[paired])) if paired.any() else 0.
            label_error=max(label_error,mismatch)
            classes=len(np.unique(signatures[i].reshape(cfg.subarrays,-1),axis=0))
            counts.append(classes)
            physical.append(dict(split=split,scenario=int(key[0]),trajectory=int(key[1]),step=int(key[2]),
                service_classes=classes,equivalent_unordered_pairs=int(paired.sum()),
                max_true_C4_outage_difference=mismatch))
        assert label_error<1e-12
        counts=np.asarray(counts)
        for seed in (11,22):
            for mode in ('structured','free'):
                checkpoint=torch.load(FIT_ROOT/f'fit_seed{seed}/{mode}_final.pt',map_location='cpu',weights_only=True)
                model=FactoredNetwork(cfg,structured=mode=='structured')
                model.load_state_dict(checkpoint['online'],strict=True)
                model.eval()
                with torch.no_grad():
                    original=model(torch.from_numpy(ds['features']).float()).numpy()
                saved=np.load(FIT_ROOT/f'fit_seed{seed}/{mode}_predictions.npz',allow_pickle=False)[split]
                np.testing.assert_array_equal(original,saved)
                original=original.astype(np.float64)
                projected=project_risk(original,projection)
                np.testing.assert_array_equal(projected[:,:,:,0],original[:,:,:,0])
                np.testing.assert_allclose(projected[:,:,1,1]-projected[:,:,0,1],
                    original[:,:,1,1]-original[:,:,0,1],rtol=0,atol=1e-15)
                old_error=(original[:,:,0,1]-truth[:,:,0,1])**2
                new_error=(projected[:,:,0,1]-truth[:,:,0,1])**2
                removed=(projected[:,:,0,1]-original[:,:,0,1])**2
                identity_error=max(identity_error,float(np.max(np.abs((old_error-new_error-removed).sum(1)))))
                assert identity_error<1e-12
                for j in (-1,0,1,2):
                    selected=np.ones(n,bool) if j==-1 else ds['keys'][:,0]==j
                    observed_pairs=same_pairs[selected]
                    pred_risk=original[selected,:,0,1]
                    pair_delta=np.abs(pred_risk[:,:,None]-pred_risk[:,None,:])[observed_pairs]
                    row=dict(split=split,seed=seed,mode=mode,scenario=j,roots=int(selected.sum()),
                        mean_service_classes=float(counts[selected].mean()),
                        equivalent_pairs=int(observed_pairs.sum()),
                        predicted_unequal_equivalent_pairs=int((pair_delta>1e-6).sum()),
                        max_predicted_equivalent_pair_gap=float(pair_delta.max()) if len(pair_delta) else 0.,
                        C4_absolute_outage_mse_original=float(old_error[selected].mean()),
                        C4_absolute_outage_mse_projected=float(new_error[selected].mean()),
                        C4_removed_inconsistency_mse=float(removed[selected].mean()))
                    for name,costs in (('original',original),('projected',projected)):
                        delta=costs-costs[np.arange(n),reference][:,None]
                        for h in (0,1):
                            label='C4' if h==0 else 'C16'
                            row[f'{label}_relative_outage_mse_{name}']=float(((delta[selected,:,h,1]-ds['difference'][selected,:,h,1])**2).mean())
                        row[f'C16_absolute_outage_mse_{name}']=float(((costs[selected,:,1,1]-truth[selected,:,1,1])**2).mean())
                    metrics.append(row)
                for name,costs in (('original',original),('projected',projected)):
                    delta=costs-costs[np.arange(n),reference][:,None]
                    for h,horizon in enumerate(('C4','C16')):
                        local=[]
                        for i,key in enumerate(ds['keys']):
                            action=select_action(delta[i,:,h],int(reference[i]),2.)
                            pred=delta[i,action,h]
                            realized=ds['difference'][i,action,h]
                            equal=bool(same[i,action,reference[i]])
                            impossible=equal and delta[i,action,0,1]<-1e-6
                            row=dict(split=split,seed=seed,mode=mode,view=name,horizon=horizon,
                                scenario=int(key[0]),trajectory=int(key[1]),step=int(key[2]),
                                action=action+1,reference=int(reference[i])+1,same_service_as_reference=int(equal),
                                impossible_block_advantage=int(impossible),
                                predicted_energy_difference=float(pred[0]),predicted_outage_difference=float(pred[1]),
                                sample_energy_difference=float(realized[0]),sample_outage_difference=float(realized[1]),
                                predicted_cost_difference=float(pred[0]+2*pred[1]),sample_cost_difference=float(realized[0]+2*realized[1]))
                            local.append(row)
                        choices.extend(local)
                        for j in range(3):
                            summaries.append(dict(split=split,seed=seed,mode=mode,view=name,horizon=horizon,
                                scenario=j,**summarize_choices([r for r in local if r['scenario']==j])))
    assert len(physical)==864 and len(choices)==13824 and len(metrics)==32 and len(summaries)==96
    _write_rows(ROOT/'physical_roots.csv',physical)
    _write_rows(ROOT/'metrics.csv',metrics)
    _write_rows(ROOT/'choices.csv',choices)
    _write_rows(ROOT/'choice_summary.csv',summaries)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    report=dict(root_states=864,physical_trajectories=108,choice_records=13824,
        label_equivalence_max_error=label_error,projection_identity_max_error=identity_error,
        weight_reloaded_predictions_exact=True,new_simulations=0,new_training=0,
        input_hashes=hashes,claim_scope='absolute C4 risk consistency only; not complete-action equivalence or energy superiority')
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2))
    (ROOT/'COMPLETE').write_text('Local equivalence audit completed; no model promoted.\n')
    print(json.dumps(dict(audit={k:v for k,v in report.items() if k!='input_hashes'},
        validation_metrics=[r for r in metrics if r['split']=='validation' and r['scenario']==-1],
        free_C16_nominal=[r for r in summaries if r['split']=='validation' and r['mode']=='free' and r['horizon']=='C16' and r['scenario']==1]),indent=2))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
