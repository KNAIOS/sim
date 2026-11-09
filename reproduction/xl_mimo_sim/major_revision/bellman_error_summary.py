"""Float64-only statistics fix; frozen collectors/models/labels stay unchanged.

The original summary stopped at its strict identity check due to float32
subtraction order. This copy changes only analysis precision, not any policy,
selection, sample, normalization or outcome.
"""
import json
import numpy as np
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.bellman_branch_data import digest
from major_revision.bellman_error_audit import ROOT, verified_data, error_components


def summarize():
    result, errors, interventions, predictions, cluster_rows = [], [], [], [], []
    source = {}
    shared = []
    identity_error = 0.
    cfg = make_config('validation')
    for seed in (11, 22):
        data, training, audit = verified_data(seed)
        # Statistics only: preserve every stored float32 value exactly in float64.
        # Actions remain the original frozen integer selections, not recomputed.
        data = {k: v.astype(np.float64) if np.issubdtype(v.dtype, np.floating) else v
                for k, v in data.items()}
        training = {k: v.astype(np.float64) if np.issubdtype(v.dtype, np.floating) else v
                    for k, v in training.items()}
        source[str(ROOT/f'seed{seed}/roots.npz')] = digest(ROOT/f'seed{seed}/roots.npz')
        source[str(ROOT/f'seed{seed}/training_reference.npz')] = digest(ROOT/f'seed{seed}/training_reference.npz')
        shared.append(data)
        for dataset, d in (('train', training), ('visited', data)):
            for rollin in (0, 1):
                for j in range(3):
                    idx = np.flatnonzero((d['keys'][:, 0] == rollin) & (d['keys'][:, 1] == j))
                    assert len(idx) == 12
                    selections = {'all': None, 'learned': d['selected_action'][idx] if dataset == 'train' else d['actions'][idx, 0]}
                    if dataset == 'visited':
                        selections['search'] = d['actions'][idx, 1]
                    for which, action in selections.items():
                        pred, real = d['predicted4'][idx, :, 1], d['actual4'][idx, :, 1]
                        if action is not None:
                            pred = pred[np.arange(len(idx)), action]
                            real = real[np.arange(len(idx)), action]
                        error = pred-real
                        predictions.append(dict(seed=seed, dataset=dataset, rollin=rollin, scenario=j,
                            selection=which, roots=12, scalar_labels=int(error.size),
                            predicted=float(np.mean(pred)), actual=float(np.mean(real)),
                            bias=float(np.mean(error)), mae=float(np.mean(np.abs(error))),
                            rmse=float(np.sqrt(np.mean(error**2)))))
        for n, key in enumerate(data['keys']):
            rollin, j, i, t = map(int, key)
            price = float(audit['prices'][str(j)])
            a, b, oracle = data['actions'][n]
            pred4, pred64 = data['predicted4'][n], data['predicted64'][n]
            real4, real64 = data['actual4'][n], data['actual64'][n]
            parts = (error_components(pred4[a], pred64[a], real4[a], real64[0], price)
                     -error_components(pred4[b], pred64[b], real4[b], real64[1], price))
            pred_delta = (pred64[a]-pred64[b]) @ np.array([1., price])
            true_delta = (real64[0]-real64[1]) @ np.array([1., price])
            identity_error = max(identity_error, abs(float(parts.sum()-(pred_delta-true_delta))))
            assert identity_error < 1e-9
            row = dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, time=t, price=price,
                       action=int(a), search_action=int(b), oracle_action=int(oracle),
                       predicted_delta=float(pred_delta), actual_delta=float(true_delta),
                       immediate_energy_error=float(parts[0]), immediate_risk_error=float(parts[1]),
                       tail_energy_error=float(parts[2]), tail_risk_error=float(parts[3]),
                       actual_energy_delta=float(real64[0, 0]-real64[1, 0]),
                       actual_outage_cost_delta=float(real64[0, 1]-real64[1, 1]))
            errors.append(row)
            interventions.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, time=t,
                price=price, changed=int(oracle != a),
                current_risk_change=float(real4[oracle, 1]-real4[a, 1]),
                energy_change=float(real64[2, 0]-real64[0, 0]),
                outage_cost_change=float(real64[2, 1]-real64[0, 1]),
                scalar_change=float((real64[2]-real64[0]) @ np.array([1., price]))))
        np.testing.assert_allclose(data['actual64'],
            (data['blocks']*cfg.rl_gamma**np.arange(16)[None, None, :, None]).sum(2), atol=1e-12, rtol=0)
        for rollin in (0, 1):
            for j in range(3):
                group = [r for r in errors if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, j)]
                intr = [r for r in interventions if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, j)]
                changed = [r for r in group if r['action'] != r['search_action']]
                row = dict(seed=seed, rollin=rollin, scenario=j, roots=len(group),
                    different_from_search=len(changed),
                    wrong_sign=sum(r['predicted_delta'] < -1e-7 and r['actual_delta'] > 1e-7 for r in group),
                    predicted_delta=float(np.mean([r['predicted_delta'] for r in group])),
                    actual_delta=float(np.mean([r['actual_delta'] for r in group])),
                    oracle_changes=sum(r['changed'] for r in intr),
                    oracle_better=sum(r['changed'] and r['scalar_change'] < -1e-7 for r in intr),
                    oracle_worse=sum(r['changed'] and r['scalar_change'] > 1e-7 for r in intr),
                    oracle_scalar_change=float(np.mean([r['scalar_change'] for r in intr])))
                for component in ('immediate_energy_error', 'immediate_risk_error', 'tail_energy_error', 'tail_risk_error'):
                    row[component] = float(np.mean([r[component] for r in group]))
                    row[component+'_mae'] = float(np.mean([abs(r[component]) for r in group]))
                result.append(row)
                for i in range(4):
                    roots = [r for r in group if r['trajectory'] == i]
                    assert len(roots) == 3
                    cluster_rows.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i,
                        predicted_delta=float(np.mean([r['predicted_delta'] for r in roots])),
                        actual_delta=float(np.mean([r['actual_delta'] for r in roots]))))
    for field in ('features', 'keys', 'actual4'):
        left = shared[0][field][shared[0]['keys'][:, 0] == 1]
        right = shared[1][field][shared[1]['keys'][:, 0] == 1]
        np.testing.assert_array_equal(left, right)
    folder = ROOT/'summary'
    folder.mkdir(parents=True, exist_ok=False)
    for filename, values in (('risk_prediction.csv', predictions), ('relative_errors.csv', errors),
                             ('oracle_interventions.csv', interventions), ('summary.csv', result),
                             ('trajectory_clusters.csv', cluster_rows)):
        _write_rows(folder/filename, values)
    report = dict(roots=144, distinct_exogenous_trajectories=12, error_sum_residual=identity_error,
                  duplicate_search_states_exact=True, return_reconstruction_exact=True,
                  input_hashes=source, oracle_is_deployable=False, training_updates=0,
                  independent_confirmation=False, failed_candidate_gate_unchanged=True)
    (folder/'audit.json').write_text(json.dumps(report, indent=2))
    (folder/'COMPLETE').write_text('Error diagnosis summarized; no promotion or new policy claim.\n')
    print(json.dumps(dict(audit=report, summary=result), indent=2))



if __name__ == '__main__':
    summarize()

