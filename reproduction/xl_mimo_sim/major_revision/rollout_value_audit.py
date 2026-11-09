"""Audit fixed rollout targets before permitting any value-function fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.rollout_value_data import ROOT


def validate_arrays(data, roots):
    x,reference,costs,difference,keys=(data[k] for k in ('features','reference','costs','difference','keys'))
    assert x.shape==(roots,1436)
    assert costs.shape==difference.shape==(roots,16,2,2)
    assert keys.shape==(roots,3) and len({tuple(k) for k in keys})==roots
    assert reference.shape==(roots,) and np.all((reference>=0)&(reference<16))
    assert all(np.all(np.isfinite(a)) for a in (x,costs,difference))
    np.testing.assert_array_equal(difference,costs-costs[np.arange(roots),reference][:,None])
    np.testing.assert_array_equal(difference[np.arange(roots),reference],0.)
    assert np.all(costs>=0.)
    assert np.all(costs[:,:,0,1]<=1.)
    assert np.all(costs[:,:,1,1]<=sum(.98**np.arange(4))+1e-12)
    assert np.all(costs[:,:,1,:]>=costs[:,:,0,:]-1e-12)


def main():
    output=ROOT/'data_audit'
    output.mkdir(exist_ok=False)
    inputs={}
    summaries=[]
    for split,roots in (('train',576),('validation',288)):
        folder=ROOT/split
        assert (folder/'COMPLETE').exists(),split
        manifest=json.loads((folder/'manifest.json').read_text())
        audit=json.loads((folder/'audit.json').read_text())
        for path,digest in manifest['source_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256((folder/'labels.npz').read_bytes()).hexdigest()==audit['data_hash']
        assert audit['root_states']==roots and audit['branches']==16*roots and audit['replay_error']<1e-10
        with np.load(folder/'labels.npz',allow_pickle=False) as data:
            validate_arrays(data,roots)
            diff=data['difference']
            for j in range(3):
                subset=diff[data['keys'][:,0]==j]
                for h,label in enumerate(('C4','C16')):
                    d=subset[:,:,h,:]
                    # Ex-post opportunities are diagnostics, never policy performance.
                    favorable=(d[:,:,0]<-1e-9)&(d[:,:,1]<=1e-12)
                    summaries.append(dict(split=split,scenario=j,horizon=label,
                        root_states=len(d),roots_with_ex_post_favorable_action=int(favorable.any(1).sum()),
                        favorable_actions=int(favorable.sum()),
                        zero_energy_mse=float(np.mean(d[:,:,0]**2)),
                        zero_outage_mse=float(np.mean(d[:,:,1]**2))))
        for name in ('manifest.json','audit.json','labels.npz','episodes.csv'):
            path=folder/name
            inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    _write_rows(output/'summary.csv',summaries)
    (output/'audit.json').write_text(json.dumps(dict(roots=864,branches=13824,
        input_hashes=inputs,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        passed=True,role='target correctness only, not policy quality or superiority'),indent=2))
    (output/'COMPLETE').write_text('All targets audited. Stage-2 fixed regression permitted, not policy promotion.\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
