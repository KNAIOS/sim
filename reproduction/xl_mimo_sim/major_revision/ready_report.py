"""Matched development analysis for the readiness-rate feature ablation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows

ROOT = Path('results/major_revision_20260920_ready_ablation')
REFERENCE = Path('results/major_revision_20260920_constraint_screen')


def read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def run():
    assert (REFERENCE/'COMPLETE').exists()
    rows = [r for r in read(REFERENCE/'development_all.csv') if r['method']!='Rate-DQN price8']
    inputs = [REFERENCE/'development_all.csv']
    for seed in (11,22):
        folder = ROOT/f'seed{seed}'
        assert (folder/'COMPLETE').exists()
        inputs.append(folder/'development_raw.csv')
        rows.extend(read(inputs[-1]))
    keys = [(r['scenario'],r['trajectory'],r['seed'],r['method']) for r in rows]
    assert len(rows)==len(set(keys))==336
    residual = max(abs(float(r['energy_j'])-float(r['operating_j'])-
        float(r['pilot_j'])-float(r['switching_j'])) for r in rows)
    assert residual < 1e-7
    rng = np.random.default_rng(974900000)
    summary,pairs = [],[]
    for scenario in range(3):
        group = [r for r in rows if int(r['scenario'])==scenario]
        matrices = {}
        for method in sorted({r['method'] for r in group}):
            subset = [r for r in group if r['method']==method]
            seeds = sorted({int(r['seed']) for r in subset})
            lookup = {(int(r['seed']),int(r['trajectory'])):r for r in subset}
            assert len(subset)==16*len(seeds)
            si = rng.integers(len(seeds),size=(4000,len(seeds)))
            ti = rng.integers(16,size=(4000,16))
            result = dict(scenario=scenario,method=method,training_seeds=len(seeds),trajectories=16)
            for metric in ('energy_j','outage'):
                data = np.array([[float(lookup[s,i][metric]) for i in range(16)] for s in seeds])
                matrices[method,metric] = data
                boot = data[si[:,:,None],ti[:,None,:]].mean((1,2))
                result.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                               metric+'_high':float(np.quantile(boot,.975))})
            summary.append(result)
        for baseline in ('Rate-DQN price4','PRS','Switch-aware PRS'):
            result = dict(scenario=scenario,method='Ready-rate DQN',baseline=baseline)
            si = rng.integers(2,size=(4000,2))
            ti = rng.integers(16,size=(4000,16))
            for metric in ('energy_j','outage'):
                data = matrices['Ready-rate DQN',metric]-matrices[baseline,metric]
                boot = data[si[:,:,None],ti[:,None,:]].mean((1,2))
                result.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                               metric+'_high':float(np.quantile(boot,.975))})
            result['mean_feasible_energy_signal'] = bool(
                matrices['Ready-rate DQN','outage'].mean()<=.05 and
                matrices[baseline,'outage'].mean()<=.05 and result['energy_j']<0)
            pairs.append(result)
    _write_rows(ROOT/'development_all.csv',rows)
    _write_rows(ROOT/'summary.csv',summary)
    _write_rows(ROOT/'paired.csv',pairs)
    audit = dict(rows=len(rows),unique_keys=len(set(keys)),energy_identity_max_error_j=residual,
        bootstrap_resamples=4000,bootstrap_seed=974900000,
        feasible_energy_signals=[p for p in pairs if p['mean_feasible_energy_signal']],
        scope='two training seeds; already-viewed development trajectories; not final confirmation',
        raw_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Readiness-feature development ablation aggregated and checked.\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    run()
