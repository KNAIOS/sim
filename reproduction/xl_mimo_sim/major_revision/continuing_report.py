"""Paired development comparison for the explicitly continuing-control candidate."""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.ready_report import read

ROOT = Path('results/major_revision_20260920_time_limit')
REFERENCE = Path('results/major_revision_20260920_ready_ablation')
METHOD = 'Continuing-ready DQN'


def run():
    assert (REFERENCE/'COMPLETE').exists()
    inputs = [REFERENCE/'development_all.csv']
    rows = read(inputs[0])
    for seed in (11,22):
        folder = ROOT/f'seed{seed}'
        assert (folder/'COMPLETE').exists()
        inputs.append(folder/'development_raw.csv')
        rows.extend(read(inputs[-1]))
    keys = [(r['scenario'],r['trajectory'],r['seed'],r['method']) for r in rows]
    assert len(rows) == len(set(keys)) == 432
    residual = max(abs(float(r['energy_j'])-float(r['operating_j'])-
        float(r['pilot_j'])-float(r['switching_j'])) for r in rows)
    assert residual < 1e-7
    rng = np.random.default_rng(975900000)
    summary,pairs,decomposition = [],[],[]
    for scenario in range(3):
        group = [r for r in rows if int(r['scenario']) == scenario]
        matrices = {}
        for method in sorted({r['method'] for r in group}):
            subset = [r for r in group if r['method'] == method]
            decomposition.append(dict(scenario=scenario,method=method,**{
                metric:float(np.mean([float(r[metric]) for r in subset]))
                for metric in ('energy_j','operating_j','switching_j','pilot_j',
                               'transitions','mean_count','short_sleep_reactivations')}))
            seeds = sorted({int(r['seed']) for r in subset})
            lookup = {(int(r['seed']),int(r['trajectory'])):r for r in subset}
            assert len(subset) == 16*len(seeds)
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
        for baseline in ('Ready-rate DQN','Rate-DQN price4','PRS','Switch-aware PRS'):
            result = dict(scenario=scenario,method=METHOD,baseline=baseline)
            si = rng.integers(2,size=(4000,2))
            ti = rng.integers(16,size=(4000,16))
            for metric in ('energy_j','outage'):
                data = matrices[METHOD,metric]-matrices[baseline,metric]
                boot = data[si[:,:,None],ti[:,None,:]].mean((1,2))
                result.update({metric:float(data.mean()),metric+'_low':float(np.quantile(boot,.025)),
                               metric+'_high':float(np.quantile(boot,.975))})
            result['mean_feasible_energy_signal'] = bool(
                matrices[METHOD,'outage'].mean() <= .05 and
                matrices[baseline,'outage'].mean() <= .05 and result['energy_j'] < 0)
            pairs.append(result)
    _write_rows(ROOT/'development_all.csv',rows)
    _write_rows(ROOT/'summary.csv',summary)
    _write_rows(ROOT/'paired.csv',pairs)
    _write_rows(ROOT/'energy_decomposition.csv',decomposition)
    audit = dict(rows=len(rows),unique_keys=len(set(keys)),energy_identity_max_error_j=residual,
        bootstrap_resamples=4000,bootstrap_seed=975900000,
        feasible_energy_signals=[p for p in pairs if p['mean_feasible_energy_signal']],
        search_energy_signals=[p for p in pairs if p['mean_feasible_energy_signal']
                               and p['baseline'] in ('PRS','Switch-aware PRS')],
        scope='two seeds on previously viewed development traces; not independent confirmation',
        raw_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Time-limit ablation aggregated and checked.\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    run()
