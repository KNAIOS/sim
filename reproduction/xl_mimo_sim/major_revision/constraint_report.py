"""Summarize the complete bounded screen, including failures and paired intervals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows

ROOT = Path('results/major_revision_20260920_constraint_screen')


def summarize():
    rows = []
    for part in ('learned','prs','cost'):
        folder = ROOT/part
        assert (folder/'COMPLETE').exists(), folder
        with (folder/'development_raw.csv').open(encoding='utf-8-sig') as stream:
            rows.extend(csv.DictReader(stream))
    keys = [(r['scenario'],r['trajectory'],r['seed'],r['method']) for r in rows]
    assert len(rows) == len(set(keys)) == 336
    residual = max(abs(float(r['energy_j'])-float(r['operating_j'])-
        float(r['pilot_j'])-float(r['switching_j'])) for r in rows)
    assert residual < 1e-7
    rng = np.random.default_rng(973900000)
    methods = sorted({r['method'] for r in rows})
    summary, pairs = [], []
    for scenario in range(3):
        group = [r for r in rows if int(r['scenario']) == scenario]
        matrices = {}
        for method in methods:
            subset = [r for r in group if r['method'] == method]
            seeds = sorted({int(r['seed']) for r in subset})
            lookup = {(int(r['seed']),int(r['trajectory'])):r for r in subset}
            assert len(subset) == 16*len(seeds)
            si = rng.integers(len(seeds), size=(4000,len(seeds)))
            ti = rng.integers(16, size=(4000,16))
            result = dict(scenario=scenario, method=method, training_seeds=len(seeds), trajectories=16)
            for metric in ('energy_j','outage'):
                data = np.array([[float(lookup[s,i][metric]) for i in range(16)] for s in seeds])
                matrices[method,metric] = data
                boot = data[si[:,:,None],ti[:,None,:]].mean((1,2))
                result.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                               metric+'_high':float(np.quantile(boot,.975))})
            summary.append(result)
        for price in (4,8):
            method = f'Rate-DQN price{price}'
            for baseline in ('PRS','Switch-aware PRS'):
                result = dict(scenario=scenario,method=method,baseline=baseline)
                si = rng.integers(2,size=(4000,2))
                ti = rng.integers(16,size=(4000,16))
                for metric in ('energy_j','outage'):
                    data = matrices[method,metric]-matrices[baseline,metric]
                    boot = data[si[:,:,None],ti[:,None,:]].mean((1,2))
                    result.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                                   metric+'_high':float(np.quantile(boot,.975))})
                result['mean_feasible_energy_signal'] = bool(
                    matrices[method,'outage'].mean() <= .05 and
                    matrices[baseline,'outage'].mean() <= .05 and result['energy_j'] < 0)
                pairs.append(result)
    _write_rows(ROOT/'development_all.csv',rows)
    _write_rows(ROOT/'summary.csv',summary)
    _write_rows(ROOT/'paired.csv',pairs)
    audit = dict(rows=len(rows),unique_keys=len(set(keys)),energy_identity_max_error_j=residual,
        bootstrap_resamples=4000,bootstrap_seed=973900000,
        mean_feasible_signals=[p for p in pairs if p['mean_feasible_energy_signal']],
        caveat='development screen, two trained seeds only; no guarantee or independent confirmation',
        raw_sha256={part:hashlib.sha256((ROOT/part/'development_raw.csv').read_bytes()).hexdigest()
                    for part in ('learned','prs','cost')})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Bounded development screen aggregated and audited.\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    summarize()
