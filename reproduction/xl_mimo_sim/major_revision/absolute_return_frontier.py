"""Post-hoc expected episode-mixture frontier; not simulated deployments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.absolute_return_screen import ROOT,METHODS,COUNT
from major_revision.absolute_return_audit import rows,mean


def mixture_frontier(points,target):
    """Enumerate pure and two-point optima of a finite energy-risk LP."""
    candidates=[]
    for i,p in enumerate(points):
        if p['outage']<=target+1e-12:
            candidates.append(dict(a=i,b=i,weight_a=1.,energy_j=p['energy_j'],outage=p['outage']))
        for j,q in enumerate(points[:i]):
            if (p['outage']-target)*(q['outage']-target)>=0:
                continue
            w=(target-q['outage'])/(p['outage']-q['outage'])
            assert 0<w<1
            candidates.append(dict(a=i,b=j,weight_a=w,energy_j=w*p['energy_j']+(1-w)*q['energy_j'],outage=target))
    return min(candidates,key=lambda c:(c['energy_j'],c['outage'],c['a'],c['b'])) if candidates else None


def main():
    assert (ROOT/'audit/COMPLETE').exists()
    gate=json.loads((ROOT/'audit/gate.json').read_text())
    assert not gate['permit_confirmation']
    previous=json.loads((ROOT/'audit/audit.json').read_text())
    for path,digest in previous['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    files=[Path(__file__),Path(__file__).with_name('ABSOLUTE_RETURN_FRONTIER_PLAN.md'),
        Path(__file__).with_name('test_absolute_return_frontier.py'),
        ROOT/'baselines/raw.csv',ROOT/'learned/raw.csv',ROOT/'audit/audit.json',ROOT/'audit/gate.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    baseline=rows(ROOT/'baselines/raw.csv')
    learned=rows(ROOT/'learned/raw.csv')
    output=ROOT/'frontier_audit'
    output.mkdir(exist_ok=False)
    result,paired=[],[]
    error=0.
    for sc in range(3):
        points=[]
        keys=sorted({(r['method'],r['minimum'],r['margin']) for r in baseline if r['scenario']==sc and r['method'] in (*METHODS,'full')})
        assert len(keys)==49
        for method,minimum,margin in keys:
            group=[r for r in baseline if r['scenario']==sc and (r['method'],r['minimum'],r['margin'])==(method,minimum,margin)]
            assert len(group)==COUNT
            points.append(dict(method=method,minimum=minimum,margin=margin,**mean(group)))
        targets=[]
        for seed in (11,22):
            group=[r for r in learned if r['scenario']==sc and r['seed']==seed and r['horizon']=='C16']
            targets.append(dict(seed=seed,target=mean(group)['outage'],learned_energy_j=mean(group)['energy_j']))
        targets.append(dict(seed=0,target=.05,learned_energy_j=0.))
        for target in targets:
            best=mixture_frontier(points,target['target'])
            if best is None:
                result.append(dict(scenario=sc,seed=target['seed'],target_outage=target['target'],feasible=False,
                    learned_energy_j=target['learned_energy_j']))
                continue
            a,b=points[best['a']],points[best['b']]
            w=best['weight_a']
            pure=[p for p in points if p['outage']<=target['target']+1e-12]
            pure_best=min(pure,key=lambda p:p['energy_j'])
            means_check=[]
            for i in range(COUNT):
                def get(point):
                    return next(r for r in baseline if r['scenario']==sc and r['trajectory']==i and
                        r['method']==point['method'] and r['minimum']==point['minimum'] and r['margin']==point['margin'])
                ra,rb=get(a),get(b)
                energy=w*ra['energy_j']+(1-w)*rb['energy_j']
                outage=w*ra['outage']+(1-w)*rb['outage']
                paired.append(dict(scenario=sc,seed=target['seed'],trajectory=i,
                    expected_mixture_energy_j=energy,expected_mixture_outage=outage))
                means_check.append([energy,outage])
            error=max(error,float(np.max(np.abs(np.mean(means_check,axis=0)-[best['energy_j'],best['outage']]))))
            result.append(dict(scenario=sc,seed=target['seed'],target_outage=target['target'],feasible=True,
                a_method=a['method'],a_minimum=a['minimum'],a_margin=a['margin'],
                b_method=b['method'],b_minimum=b['minimum'],b_margin=b['margin'],weight_a=w,
                expected_mixture_energy_j=best['energy_j'],expected_mixture_outage=best['outage'],
                best_pure_energy_j=pure_best['energy_j'],learned_energy_j=target['learned_energy_j'],
                learned_minus_mixture_j=(target['learned_energy_j']-best['energy_j']) if target['seed'] else ''))
    assert len(result)==9 and error<1e-10
    from major_revision.absolute_return_audit import rectangular
    _write_rows(output/'summary.csv',rectangular(result))
    _write_rows(output/'expected_paired.csv',paired)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (output/'audit.json').write_text(json.dumps(dict(input_hashes=hashes,expected_value_recalculation_error=error,
        settings_per_scenario=49,targets=9,new_simulations=0,
        interpretation='Post-hoc validation-calibrated mixture expectations, no independent policy superiority.'),indent=2))
    (output/'COMPLETE').write_text('Original failed screen gate preserved.\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
