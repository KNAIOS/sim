"""Reconstruct paired-value regression scores and the prespecified stage-2 gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.ready_report import read
from major_revision.rollout_value_data import ROOT
from major_revision.rollout_value_fit import DifferenceNetwork,metrics


def fit_gate(rows):
    relevant=[r for r in rows if r['horizon']=='C16' and r['split']=='validation' and r['scenario']==-1]
    assert {r['seed'] for r in relevant}=={11,22} and len(relevant)==2
    checks=[]
    for row in relevant:
        for name in ('energy','outage'):
            assert row[name+'_zero_mse']>0
            checks.append(dict(seed=row['seed'],metric=name,mse=row[name+'_mse'],
                zero_mse=row[name+'_zero_mse'],ratio=row[name+'_mse']/row[name+'_zero_mse'],
                passed=bool(row[name+'_mse']<=.9*row[name+'_zero_mse'])))
    return dict(permit_closed_loop=all(c['passed'] for c in checks),checks=checks)


def main():
    cfg=make_config('validation')
    data={split:np.load(ROOT/split/'labels.npz',allow_pickle=False) for split in ('train','validation')}
    audit=json.loads((ROOT/'data_audit/audit.json').read_text())
    for path,digest in audit['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    rows,hashes=[],{}
    for seed in (11,22):
        folder=ROOT/f'fit_seed{seed}'
        assert (folder/'COMPLETE').exists()
        manifest=json.loads((folder/'manifest.json').read_text())
        for path,digest in manifest['source_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        stored=read(folder/'metrics.csv')
        history=read(folder/'history.csv')
        for h,label in enumerate(('C4','C16')):
            selected=[r for r in history if r['horizon']==label]
            assert len(selected)==100 and {int(r['epoch']) for r in selected}==set(range(1,101))
            assert all(np.isfinite(float(r['scaled_train_loss'])) for r in selected)
            initial=torch.load(folder/f'{label}_initial.pt',map_location='cpu',weights_only=True)
            final=torch.load(folder/f'{label}_final.pt',map_location='cpu',weights_only=True)
            assert initial['epoch']==0 and final['epoch']==100
            assert initial['parameters']==final['parameters']==32066
            target=torch.from_numpy(data['train']['difference'][:,:,h,:]).float()
            scale=torch.clamp(torch.sqrt((target**2).mean((0,1))),min=1e-6)
            torch.testing.assert_close(final['scale'],scale,atol=0.,rtol=0.)
            model=DifferenceNetwork(cfg)
            model.load_state_dict(initial['online'])
            with torch.no_grad():
                zeros=model(torch.from_numpy(data['validation']['features']).float(),
                    torch.from_numpy(data['validation']['reference']).long())
            assert torch.count_nonzero(zeros)==0
            model.load_state_dict(final['online'])
            predictions=np.load(folder/f'{label}_predictions.npz',allow_pickle=False)
            for split,values in data.items():
                with torch.no_grad():
                    prediction=model(torch.from_numpy(values['features']).float(),
                        torch.from_numpy(values['reference']).long()).numpy()
                np.testing.assert_allclose(prediction,predictions[split],atol=1e-7,rtol=1e-7)
                truth=values['difference'][:,:,h,:]
                for j in (-1,0,1,2):
                    mask=np.ones(len(truth),dtype=bool) if j==-1 else values['keys'][:,0]==j
                    result=metrics(prediction[mask],truth[mask],values['reference'][mask])
                    old=next(r for r in stored if r['horizon']==label and r['split']==split and int(r['scenario'])==j)
                    for key,value in result.items():
                        assert abs(value-float(old[key]))<1e-10,(label,split,j,key)
                    rows.append(dict(seed=seed,horizon=label,split=split,scenario=j,**result))
        for path in folder.iterdir():
            if path.is_file():
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    gate=fit_gate(rows)
    output=ROOT/'fit_audit'
    output.mkdir(exist_ok=False)
    _write_rows(output/'metrics.csv',rows)
    (output/'gate.json').write_text(json.dumps(gate,indent=2))
    (output/'audit.json').write_text(json.dumps(dict(input_hashes=hashes,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_replay=True,training_epochs=400,parameters_per_network=32066,
        role='predictive fit only; not a closed-loop energy advantage'),indent=2))
    (output/'COMPLETE').write_text('Regression gate reconstructed; no performance claim.\n')
    if gate['permit_closed_loop']:
        (ROOT/'PERMIT_CLOSED_LOOP').write_text('Both C16 seeds passed the frozen regression gate.\n')
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
