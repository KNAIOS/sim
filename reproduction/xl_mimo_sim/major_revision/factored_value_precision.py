"""Float64 recheck of serialized outage fractions; no model/data mutation."""
import hashlib
import json
import numpy as np
from future_vr.config import make_config
from major_revision.factored_value import ROOT,FactoredNetwork


def main():
    cfg=make_config('validation')
    rows=[]
    for split in ('train','validation'):
        path=ROOT/split/'features.npz'
        with np.load(path,allow_pickle=False) as data:
            fractions=data['features'][:,1436:].reshape(-1,16,4)[:,:,-1].astype(np.float64)
            grid=np.rint(fractions*cfg.users)/cfg.users
            error=float(np.max(np.abs(fractions-grid)))
            assert error<3e-8
        rows.append(dict(split=split,float64_fraction_rounding_max=error,
            source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    models={str(mode):sum(p.numel() for p in FactoredNetwork(cfg,mode).parameters()) for mode in (True,False)}
    assert models['True']==models['False']
    folder=ROOT/'precision_audit'
    folder.mkdir(exist_ok=False)
    report=dict(rows=rows,parameters=models,
        note='Original scalar audit subtracts a Python float from a NumPy float32 scalar; NumPy weak scalar promotion can round the reference. Float64 audit above measures serialization rounding, not a new simulation or changed label.')
    (folder/'audit.json').write_text(json.dumps(report,indent=2))
    (folder/'COMPLETE').write_text('Only rounding/parameter metadata checked; frozen inputs untouched.\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
