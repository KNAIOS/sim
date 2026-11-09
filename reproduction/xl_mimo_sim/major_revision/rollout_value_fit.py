"""Fixed-budget differential policy-value regression; no online oracle access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import action_size,observation_size
from major_revision.gram_control import representation_size
from major_revision.rollout_value_data import ROOT
from major_revision.rollout_value_audit import validate_arrays


class DifferenceNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.common=nn.Sequential(nn.Linear(representation_size(cfg)+2,64),nn.LayerNorm(64),nn.SiLU())
        self.action=nn.Sequential(nn.Linear(action_size(cfg),32),nn.LayerNorm(32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,reference):
        common_end=representation_size(self.cfg)
        legacy_end=observation_size(self.cfg)
        common=self.common(torch.cat((x[:,:common_end],x[:,legacy_end:]),dim=-1))
        common=common[:,None].expand(-1,self.cfg.subarrays,-1)
        actions=x[:,common_end:legacy_end].reshape(-1,self.cfg.subarrays,action_size(self.cfg))
        values=self.head(torch.cat((common,self.action(actions)),dim=-1))
        anchor=values[torch.arange(len(x),device=x.device),reference]
        return values-anchor[:,None]


def select_action(difference,reference,price):
    """Fallback to the actual reference on zero/no predicted improvement."""
    values=np.asarray(difference)
    if values.ndim!=2 or values.shape[1]!=2 or not np.all(np.isfinite(values)):
        raise ValueError('Expected a finite action-by-two cost difference matrix')
    cost=values[:,0]+price*values[:,1]
    action=int(np.argmin(cost))
    return int(reference) if cost[action]>=-1e-6 else action


def metrics(prediction,target,reference):
    prediction=np.asarray(prediction,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    near=np.abs(np.arange(target.shape[1])[None]-reference[:,None])<=1
    near[np.arange(len(reference)),reference]=False
    result={}
    for h,name in enumerate(('energy','outage')):
        squared=(prediction[:,:,h]-target[:,:,h])**2
        zero=target[:,:,h]**2
        result.update({name+'_mse':float(squared.mean()),name+'_zero_mse':float(zero.mean()),
                       name+'_near_mse':float(squared[near].mean()),name+'_near_zero_mse':float(zero[near].mean())})
    assert np.all(prediction[np.arange(len(reference)),reference]==0.)
    return result


def fit(seed):
    cfg=make_config('validation')
    assert (ROOT/'data_audit/COMPLETE').exists()
    data_audit=json.loads((ROOT/'data_audit/audit.json').read_text())
    for path,digest in data_audit['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    train=np.load(ROOT/'train/labels.npz',allow_pickle=False)
    val=np.load(ROOT/'validation/labels.npz',allow_pickle=False)
    validate_arrays(train,576)
    validate_arrays(val,288)
    output=ROOT/f'fit_seed{seed}'
    output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('ROLLOUT_VALUE_PLAN.md'),
        Path(__file__).with_name('rollout_value_data.py'),Path(__file__).with_name('rollout_value_audit.py'),
        Path(__file__).with_name('test_rollout_value_fit.py'),ROOT/'data_audit/audit.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=dict(seed=seed,epochs=100,batch_roots=64,lr=4e-4,weight_decay=1e-5,
        gradient_clip=5.,input_dimension=1436,common_dimension=364,action_dimension=67,
        output_costs=['energy','outage'],source_hashes=hashes,threads=1,
        role='fixed-budget offline value approximation, not yet closed-loop RL validation')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    x=torch.from_numpy(train['features']).float()
    reference=torch.from_numpy(train['reference']).long()
    vx=torch.from_numpy(val['features']).float()
    vr=torch.from_numpy(val['reference']).long()
    reports,history=[],[]
    for horizon,label in enumerate(('C4','C16')):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        model=DifferenceNetwork(cfg)
        target=torch.from_numpy(train['difference'][:,:,horizon,:]).float()
        val_target=val['difference'][:,:,horizon,:]
        scale=torch.clamp(torch.sqrt(torch.mean(target**2,dim=(0,1))),min=1e-6)
        optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=1e-5)
        parameters=sum(p.numel() for p in model.parameters())
        torch.save(dict(online=model.state_dict(),seed=seed,horizon=label,epoch=0,
                        scale=scale,parameters=parameters),output/f'{label}_initial.pt')
        for epoch in range(100):
            order=rng.permutation(len(x))
            total=0.
            for start in range(0,len(order),64):
                indices=order[start:start+64]
                prediction=model(x[indices],reference[indices])
                loss=torch.mean(((prediction-target[indices])/scale)**2)
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
                total+=float(loss.detach())*len(indices)
            history.append(dict(seed=seed,horizon=label,epoch=epoch+1,scaled_train_loss=total/len(x)))
            if (epoch+1)%25==0:
                print('rollout fit',seed,label,epoch+1,'training loss',round(total/len(x),5),flush=True)
        # Fixed epoch only: no validation checkpoint selection.
        with torch.no_grad():
            tp=model(x,reference).numpy()
            vp=model(vx,vr).numpy()
        for split,predicted,truth,refs,keys in (
            ('train',tp,train['difference'][:,:,horizon,:],train['reference'],train['keys']),
            ('validation',vp,val_target,val['reference'],val['keys'])):
            reports.append(dict(seed=seed,horizon=label,split=split,scenario=-1,
                **metrics(predicted,truth,refs)))
            for j in range(3):
                selected=keys[:,0]==j
                reports.append(dict(seed=seed,horizon=label,split=split,scenario=j,
                    **metrics(predicted[selected],truth[selected],refs[selected])))
        torch.save(dict(online=model.state_dict(),seed=seed,horizon=label,epoch=100,
                        scale=scale,parameters=parameters),output/f'{label}_final.pt')
        np.savez_compressed(output/f'{label}_predictions.npz',train=tp,validation=vp)
        _write_rows(output/'metrics.csv',reports)
        _write_rows(output/'history.csv',history)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (output/'COMPLETE').write_text('C4 and C16 fixed regression complete; inspect gate before closed-loop use.\n')
    print(json.dumps([r for r in reports if r['scenario']==-1],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(11,22),required=True)
    torch.set_num_threads(1)
    fit(parser.parse_args().seed)
