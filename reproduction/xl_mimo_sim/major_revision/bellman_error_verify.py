"""Independent prediction reload, return, oracle and decomposition checks."""
import csv
import json

import numpy as np
import torch

from future_vr.config import make_config
from major_revision.bellman_branch_data import ROOT as FIT_ROOT, digest
from major_revision.bellman_error_audit import ROOT, TIMES
from major_revision.bellman_value import BellmanValue


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main():
    folder = ROOT/'summary'
    assert (folder/'COMPLETE').exists()
    summary_audit = json.loads((folder/'audit.json').read_text())
    for path, expected in summary_audit['input_hashes'].items():
        assert digest(path) == expected, path
    with (folder/'relative_errors.csv').open(encoding='utf-8-sig', newline='') as f:
        relative = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    with (folder/'oracle_interventions.csv').open(encoding='utf-8-sig', newline='') as f:
        oracle_rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    cfg = make_config('validation')
    errors = dict(prediction_reload=0., training_reload=0., return_reconstruction=0.,
                  decomposition_reconstruction=0., oracle_label_reconstruction=0.)
    all_data = []
    for seed in (11, 22):
        source = ROOT/f'seed{seed}'
        data = load_npz(source/'roots.npz')
        all_data.append(data)
        labels = load_npz(source/'training_reference.npz')
        meta = json.loads((source/'audit.json').read_text())
        assert data['features'].shape == (72, 1500)
        assert data['actual4'].shape == (72, 16, 2)
        assert data['actual64'].shape == (72, 3, 2)
        assert data['blocks'].shape == (72, 3, 16, 2)
        assert set(data['keys'][:, 3]) == set(TIMES)
        model = BellmanValue(cfg).eval()
        model.load_state_dict(torch.load(FIT_ROOT/f'fit_seed{seed}/final.pt', weights_only=True)['online'], strict=True)
        for n, (rollin, scenario, trajectory, time) in enumerate(data['keys']):
            price = float(meta['prices'][str(scenario)])
            x = torch.from_numpy(data['features'][n:n+1])
            with torch.no_grad():
                p4 = model.immediate(x)[0].numpy()
                p64 = model(x, price, 16)[0].numpy()
            errors['prediction_reload'] = max(errors['prediction_reload'],
                float(np.max(abs(p4-data['predicted4'][n]))), float(np.max(abs(p64-data['predicted64'][n]))))
            assert np.array_equal(p4, data['predicted4'][n]) and np.array_equal(p64, data['predicted64'][n])
            a, b, oracle = data['actions'][n]
            # Verify the stored choice in its original operational precision.
            assert a == np.argmin(p64[:, 0]+price*p64[:, 1])
            scalar_changed = p64[:, 0]+price*(p64[:, 1]-p4[:, 1]+data['actual4'][n, :, 1])
            assert oracle == np.argmin(scalar_changed)
            # Re-express those unchanged prediction values in double precision
            # for algebraic statistics, exactly as the corrected summary does.
            p4, p64 = p4.astype(np.float64), p64.astype(np.float64)
            np.testing.assert_allclose(data['blocks'][n, :, 0], data['actual4'][n, [a, b, oracle]], rtol=0, atol=1e-10)
            total = np.sum(data['blocks'][n]*cfg.rl_gamma**np.arange(16)[None, :, None], axis=1)
            errors['return_reconstruction'] = max(errors['return_reconstruction'],
                float(np.max(abs(total-data['actual64'][n]))))
            long_error = (p64[a]-p64[b])-(total[0]-total[1])
            now_error = (p4[a]-p4[b])-(data['actual4'][n, a]-data['actual4'][n, b])
            later_error = long_error-now_error
            expected = dict(immediate_energy_error=now_error[0], immediate_risk_error=price*now_error[1],
                            tail_energy_error=later_error[0], tail_risk_error=price*later_error[1],
                            predicted_delta=(p64[a]-p64[b]) @ [1., price],
                            actual_delta=(total[0]-total[1]) @ [1., price])
            row = next(r for r in relative if (r['seed'], r['rollin'], r['scenario'], r['trajectory'], r['time']) ==
                       (seed, rollin, scenario, trajectory, time))
            for key, value in expected.items():
                errors['decomposition_reconstruction'] = max(errors['decomposition_reconstruction'], abs(value-row[key]))
            ir = next(r for r in oracle_rows if (r['seed'], r['rollin'], r['scenario'], r['trajectory'], r['time']) ==
                      (seed, rollin, scenario, trajectory, time))
            difference = total[2]-total[0]
            expected_oracle = dict(energy_change=difference[0], outage_cost_change=difference[1],
                                   scalar_change=difference @ [1., price],
                                   current_risk_change=data['actual4'][n, oracle, 1]-data['actual4'][n, a, 1])
            assert ir['changed'] == int(oracle != a)
            for key, value in expected_oracle.items():
                errors['oracle_label_reconstruction'] = max(errors['oracle_label_reconstruction'], abs(value-ir[key]))
        original = load_npz(FIT_ROOT/f'collect_seed{seed}/transitions.npz')
        mask = np.isin(original['keys'][:, 3], TIMES)
        np.testing.assert_array_equal(original['keys'][mask], labels['keys'])
        np.testing.assert_array_equal(original['costs'][mask], labels['actual4'])
        with torch.no_grad():
            pred = model.immediate(torch.from_numpy(original['features'][mask])).numpy()
        errors['training_reload'] = max(errors['training_reload'], float(np.max(abs(pred-labels['predicted4']))))
        assert np.array_equal(pred, labels['predicted4'])
    assert errors['prediction_reload'] == 0 and errors['training_reload'] == 0
    assert errors['decomposition_reconstruction'] < 1e-10
    assert errors['return_reconstruction'] < 1e-12 and errors['oracle_label_reconstruction'] < 1e-12
    left, right = all_data
    lm, rm = left['keys'][:, 0] == 1, right['keys'][:, 0] == 1
    np.testing.assert_array_equal(left['actions'][lm, 1], right['actions'][rm, 1])
    out = ROOT/'verified'
    out.mkdir(parents=True, exist_ok=False)
    report = dict(errors=errors, roots=144, immutable_predictions_exact=True,
                  training_labels_unchanged=True, oracle_uses_no_long_actual_return=True,
                  search_actions_across_seeds_equal=True, training_updates=0,
                  verifier_hash=digest(__file__), inputs=summary_audit['input_hashes'])
    (out/'audit.json').write_text(json.dumps(report, indent=2))
    (out/'COMPLETE').write_text('Independent audit passed; hindsight is not a controller.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
