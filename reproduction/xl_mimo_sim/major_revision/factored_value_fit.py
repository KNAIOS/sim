"""Fixed-budget paired fit and prespecified decision-relevant screening."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.factored_value import ROOT,VALUE_ROOT,FactoredNetwork,relative
from major_revision.rollout_value_fit import metrics,select_action


def loss_scales(costs,reference):
    diff=costs-costs[np.arange(len(costs)),reference][:,None]
    return tuple(torch.from_numpy(np.maximum(np.sqrt((a**2).mean((0,1))),1e-3)).float()
                 for a in (costs,diff))


def objective(prediction,truth,reference,scales):
    return (((prediction-truth)/scales[0]).square().mean()
            +((relative(prediction,reference)-relative(truth,reference))/scales[1]).square().mean())


def load_data():
    result={}
    for split in ('train','validation'):
        folder=ROOT/split
        assert (folder/'COMPLETE').exists()
        manifest=json.loads((folder/'manifest.json').read_text())
        for path,digest in manifest['hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        audit=json.loads((folder/'audit.json').read_text())
        assert hashlib.sha256((folder/'features.npz').read_bytes()).hexdigest()==audit['feature_hash']
        with np.load(folder/'features.npz',allow_pickle=False) as new, np.load(VALUE_ROOT/split/'labels.npz',allow_pickle=False) as old:
            np.testing.assert_array_equal(new['keys'],old['keys'])
            np.testing.assert_array_equal(new['reference'],old['reference'])
            np.testing.assert_array_equal(new['features'][:,:1436],old['features'])
            result[split]=dict(features=new['features'].copy(),costs=old['costs'].copy(),
                difference=old['difference'].copy(),reference=old['reference'].copy(),keys=old['keys'].copy())
    return result


def fit(seed):
    data=load_data()
    cfg=make_config('validation')
    folder=ROOT/f'fit_seed{seed}'
    folder.mkdir(exist_ok=False)
    files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('factored_value.py','test_factored_value.py','test_factored_value_fit.py','FACTORED_VALUE_PLAN.md')],
        *[ROOT/split/name for split in ('train','validation') for name in ('features.npz','audit.json','manifest.json')]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(seed=seed,epochs=100,batch_roots=64,
        learning_rate=4e-4,weight_decay=1e-5,clip=5.,threads=1,hashes=hashes,
        role='bounded value fit, viewed validation only, no closed-loop superiority claim'),indent=2))
    train,val=data['train'],data['validation']
    x=torch.from_numpy(train['features']).float()
    reference=torch.from_numpy(train['reference']).long()
    y=torch.from_numpy(train['costs']).float()
    scales=loss_scales(train['costs'],train['reference'])
    vx=torch.from_numpy(val['features']).float()
    reports,history=[],[]
    for mode in ('structured','free'):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        model=FactoredNetwork(cfg,structured=mode=='structured')
        model.initialize_from_training(train['features'],train['costs'])
        parameters=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=1e-5)
        torch.save(dict(online=model.state_dict(),seed=seed,mode=mode,epoch=0,parameters=parameters),folder/f'{mode}_initial.pt')
        for epoch in range(100):
            total=0.
            order=rng.permutation(len(x))
            for start in range(0,len(order),64):
                indices=order[start:start+64]
                prediction=model(x[indices])
                loss=objective(prediction,y[indices],reference[indices],scales)
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
                total+=float(loss.detach())*len(indices)
            history.append(dict(seed=seed,mode=mode,epoch=epoch+1,train_loss=total/len(x)))
            if (epoch+1)%25==0:
                print('factored fit',seed,mode,epoch+1,round(total/len(x),6),flush=True)
        model.eval()
        with torch.no_grad():
            predictions={'train':model(x).numpy(),'validation':model(vx).numpy()}
        for split,pred in predictions.items():
            target=data[split]
            diff=pred-pred[np.arange(len(pred)),target['reference']][:,None]
            for h,label in enumerate(('C4','C16')):
                for j in (-1,0,1,2):
                    indices=np.ones(len(pred),bool) if j==-1 else target['keys'][:,0]==j
                    result=metrics(diff[indices,:,h],target['difference'][indices,:,h],target['reference'][indices])
                    result.update({f'absolute_{name}_mse':float(((pred[indices,:,h,k]-target['costs'][indices,:,h,k])**2).mean())
                                   for k,name in enumerate(('energy','outage'))})
                    reports.append(dict(seed=seed,mode=mode,split=split,horizon=label,scenario=j,**result))
        torch.save(dict(online=model.state_dict(),seed=seed,mode=mode,epoch=100,
            parameters=parameters,absolute_scale=scales[0],relative_scale=scales[1]),folder/f'{mode}_final.pt')
        np.savez_compressed(folder/f'{mode}_predictions.npz',**predictions)
        _write_rows(folder/'metrics.csv',reports)
        _write_rows(folder/'history.csv',history)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (folder/'COMPLETE').write_text('Both fixed endpoints saved; no epoch selection.\n')


def audit():
    data=load_data()
    cfg=make_config('validation')
    val=data['validation']
    raw,summary,gates=[],[],[]
    files=[Path(__file__),Path(__file__).with_name('FACTORED_VALUE_PLAN.md')]
    for seed in (11,22):
        folder=ROOT/f'fit_seed{seed}'
        assert (folder/'COMPLETE').exists()
        manifest=json.loads((folder/'manifest.json').read_text())
        for path,digest in manifest['hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        for mode in ('structured','free'):
            checkpoint=torch.load(folder/f'{mode}_final.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==100 and checkpoint['mode']==mode
            model=FactoredNetwork(cfg,structured=mode=='structured')
            model.load_state_dict(checkpoint['online'],strict=True)
            model.eval()
            with torch.no_grad():
                pred=model(torch.from_numpy(val['features']).float()).numpy()
            saved=np.load(folder/f'{mode}_predictions.npz',allow_pickle=False)['validation']
            np.testing.assert_array_equal(pred,saved)
            assert np.all(np.isfinite(pred)) and np.all(pred>=0.)
            assert np.all(pred[:,:,1,1]<=1+model.tail_weight+1e-6)
            diff=pred-pred[np.arange(len(pred)),val['reference']][:,None]
            fit_metrics=metrics(diff[:,:,1],val['difference'][:,:,1],val['reference'])
            old=np.load(VALUE_ROOT/f'fit_seed{seed}/C16_predictions.npz',allow_pickle=False)['validation']
            old_metrics=metrics(old,val['difference'][:,:,1],val['reference'])
            selected=[]
            for index,(j,tr,t) in enumerate(val['keys']):
                action=select_action(diff[index,:,1],int(val['reference'][index]),2.)
                p=diff[index,action,1]
                real=val['difference'][index,action,1]
                row=dict(seed=seed,mode=mode,scenario=int(j),trajectory=int(tr),step=int(t),
                    action_count=action+1,reference_count=int(val['reference'][index])+1,
                    predicted_energy=float(p[0]),predicted_outage=float(p[1]),
                    actual_energy=float(real[0]),actual_outage=float(real[1]),
                    predicted_cost=float(p[0]+2*p[1]),actual_cost=float(real[0]+2*real[1]))
                selected.append(row)
            raw.extend(selected)
            for j in range(3):
                group=[r for r in selected if r['scenario']==j]
                summary.append(dict(seed=seed,mode=mode,scenario=j,roots=len(group),
                    changed=sum(r['action_count']!=r['reference_count'] for r in group),
                    **{key:float(np.mean([r[key] for r in group])) for key in
                       ('predicted_energy','predicted_outage','actual_energy','actual_outage','predicted_cost','actual_cost')}))
            nominal=next(r for r in summary if r['seed']==seed and r['mode']==mode and r['scenario']==1)
            gates.append(dict(seed=seed,mode=mode,
                energy_mse=fit_metrics['energy_mse'],old_energy_mse=old_metrics['energy_mse'],
                outage_mse=fit_metrics['outage_mse'],old_outage_mse=old_metrics['outage_mse'],
                fit_pass=fit_metrics['energy_mse']<=old_metrics['energy_mse'] and fit_metrics['outage_mse']<=old_metrics['outage_mse'],
                nominal_mean_energy=nominal['actual_energy'],nominal_mean_outage=nominal['actual_outage'],
                decision_pass=nominal['actual_energy']<0 and nominal['actual_outage']<=0))
            files.extend((folder/f'{mode}_final.pt',folder/f'{mode}_predictions.npz'))
        files.extend((folder/'metrics.csv',folder/'history.csv',folder/'manifest.json'))
    assert len(raw)==1152 and len(summary)==12
    allow=all(r['fit_pass'] and r['decision_pass'] for r in gates if r['mode']=='structured')
    output=ROOT/'fit_audit'
    output.mkdir(exist_ok=False)
    _write_rows(output/'selected_raw.csv',raw)
    _write_rows(output/'summary.csv',summary)
    (output/'gate.json').write_text(json.dumps(dict(permit_closed_loop=allow,rows=gates,
        interpretation='Viewed validation screen, not a significance test or policy superiority.'),indent=2))
    (output/'audit.json').write_text(json.dumps(dict(input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        weights_reloaded=True,predictions_exact=True,records=len(raw),unique_roots=288,
        checkpoint_selection=False,price=2.,gate=allow),indent=2))
    (output/'COMPLETE').write_text('All variants and both seeds retained.\n')
    if allow:
        (ROOT/'PERMIT_CLOSED_LOOP').write_text('Fit/ranking gate passed; freeze fair closed-loop protocol before use.\n')
    print(json.dumps(dict(gates=gates,summary=summary,permit_closed_loop=allow),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(11,22))
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    if args.audit==(args.seed is not None):
        parser.error('Choose either --seed or --audit')
    torch.set_num_threads(1)
    audit() if args.audit else fit(args.seed)
