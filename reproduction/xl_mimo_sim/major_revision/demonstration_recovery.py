"""Recover completed local training after a strict metadata-loading failure.

No unsafe unrestricted unpickling and no retraining. Only the specifically
identified NumPy scalar/dtype constructors are allowed in a temporary context,
after verifying the run's source/input provenance. Original checkpoints remain.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainDQN,full_domain_episode
from major_revision.nested_search import ROOT
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read


def portable(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {k:portable(v) for k,v in value.items()}
    if isinstance(value,tuple):
        return tuple(portable(v) for v in value)
    if isinstance(value,list):
        return [portable(v) for v in value]
    if isinstance(value,(str,int,float,bool,type(None),torch.Tensor)):
        return value
    raise TypeError(f'Unexpected local checkpoint metadata type: {type(value)}')


def convert_local(source,destination):
    # The caller verifies these are this run's own trusted checkpoints.
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(source))
    if not unsafe <= {'numpy.dtype','numpy._core.multiarray.scalar'}:
        raise ValueError(f'Unexpected checkpoint constructors: {unsafe}')
    permitted=[np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]
    with torch.serialization.safe_globals(permitted):
        original=torch.load(source,map_location='cpu',weights_only=True)
    normalized=portable(original)
    torch.save(normalized,destination)
    loaded=torch.load(destination,map_location='cpu',weights_only=True)
    for name,tensor in original['online'].items():
        assert torch.equal(tensor,loaded['online'][name])
    return dict(original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                portable_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                tensor_identity=True)


def main():
    torch.set_num_threads(1)
    cfg=make_config('validation')
    for seed in (11,22):
        folder=ROOT/f'imitation_seed{seed}'
        manifest=json.loads((folder/'manifest.json').read_text())
        for key in ('source_hashes','input_hashes'):
            for path,digest in manifest[key].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        selections=read(folder/'selection.csv')
        validation=read(folder/'validation.csv')
        classification=read(folder/'classification.csv')
        assert len(selections)==3 and len(validation)==108 and len(classification)==40
        assert {(int(r['stage']),int(r['epoch']),r['split']) for r in classification}=={
            (s,e,split) for s,epochs in enumerate((100,50,50))
            for e in range(10,epochs+1,10) for split in ('train','validation')}
        assert not (folder/'development_raw.csv').exists(),'Do not overwrite development outcomes'
        converted=folder/'portable_checkpoints'
        converted.mkdir(exist_ok=False)
        receipts={}
        for name in ('stage0.pt','stage1.pt','stage2.pt','selected.pt'):
            receipts[name]=convert_local(folder/name,converted/name)
        (converted/'CONVERSION.json').write_text(json.dumps(receipts,indent=2))
        models=[]
        for name,filename in (('Teacher-only imitation','stage0.pt'),('Selected imitation','selected.pt')):
            saved=torch.load(converted/filename,map_location='cpu',weights_only=True)
            model=FullDomainDQN(cfg,seed)
            model.online.load_state_dict(saved['online'])
            models.append((name,saved['stage'],model))
        rows=[]
        for j,sc in enumerate(SCENARIOS):
            for i in range(16):
                tr=trajectory(cfg,993000000+j*10000+i,sc)
                for name,stage,model in models:
                    result=full_domain_episode(cfg,tr,model)
                    result['method']=name
                    rows.append(dict(seed=seed,scenario=j,trajectory=i,stage=stage,**result))
                _write_rows(folder/'development_raw.csv',rows)
            print('recovered imitation evaluation',seed,j,'complete',flush=True)
        assert len(rows)==96
        receipt=dict(reason='Original trainer finished all stages then rejected NumPy scalar metadata in strict load',
            training_repeated=False,selection_changed=False,development_records=96,
            strict_weights_only=True,conversion_receipt=str(converted/'CONVERSION.json'),
            recovery_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (folder/'RECOVERY.json').write_text(json.dumps(receipt,indent=2))
        (folder/'COMPLETE').write_text('All fixed training stages retained; evaluation recovered without retraining or reselection.\n')


if __name__=='__main__':
    main()
