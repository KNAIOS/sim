"""Post-selection diagnostic on existing labels; no further policy fitting."""
import json

import numpy as np
import torch

from joint_power.aggregation.collect import ROOT, load
from joint_power.aggregation.control import input_checks, models
from joint_power.aggregation.train import ARMS, tensors
from joint_power.rollout.probe import FULL_SLOT, select
from joint_power.study import digest, write_rows


def main():
    torch.set_num_threads(1)
    input_checks()
    folder = ROOT/'control_validation'
    hashes = json.loads((folder/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    selected = json.loads((folder/'selection.json').read_text())
    prices = {(r['method'], r['seed']): r['parameter'] for r in selected}
    loaded, _ = models()
    rows, state_rows = [], []
    for (method, seed), model in loaded.items():
        price = prices[method, seed]
        for split, name in (('training', ARMS[method]), ('prediction-validation', 'validation')):
            data = load(ROOT/f'{name}.npz')
            x, ref, remaining, y = tensors(data)
            with torch.no_grad():
                prediction = model(x, ref, remaining).numpy()
                _, predicted_counts, _ = model.components(x, ref, remaining)
            actual = y.numpy()
            reference = ref.numpy()
            action = select(prediction, reference, price)
            at = np.arange(len(x))
            predicted_delta = prediction[at, action]*[FULL_SLOT, 3.]
            actual_delta = actual[at, action]-actual[at, reference]
            selected_counts = predicted_counts.numpy()[at, action]
            active = data['x'][:, -1, -112:].reshape(-1, 16, 7)[:, :, 1].sum(-1)
            for i, identity in enumerate(data['identity']):
                scenario, trajectory, slot = map(int, identity)
                state_rows.append(dict(method=method, seed=seed, split=split, scenario=scenario,
                    trajectory=trajectory, root_index=i, slot=slot, active_before=float(active[i]),
                    action=int(action[i]), reference=int(reference[i]), parameter=price,
                    predicted_energy_delta_j=float(predicted_delta[i, 0]), actual_energy_delta_j=float(actual_delta[i, 0]),
                    predicted_failed_delta=float(predicted_delta[i, 1]), actual_failed_delta=float(actual_delta[i, 1]),
                    predicted_selected_failed=float(selected_counts[i]), actual_selected_failed=float(actual[i, action[i], 1])))
            for s in range(3):
                for stratum, mask in (('all', np.ones(len(x), bool)), ('below4', active < 4), ('at_least4', active >= 4)):
                    use = mask & (data['identity'][:, 0] == s)
                    if not use.any():
                        continue
                    rows.append(dict(method=method, seed=seed, split=split, scenario=s, stratum=stratum,
                        states=int(use.sum()), parameter=price,
                        predicted_delta_j=float(predicted_delta[use, 0].mean()),
                        actual_delta_j=float(actual_delta[use, 0].mean()),
                        selected_energy_mae_j=float(np.abs(predicted_delta[use, 0]-actual_delta[use, 0]).mean()),
                        predicted_failed_delta=float(predicted_delta[use, 1].mean()),
                        actual_failed_delta=float(actual_delta[use, 1].mean()),
                        selected_failed_mae=float(np.abs(predicted_delta[use, 1]-actual_delta[use, 1]).mean()),
                        fraction_unrealized_negative_risk_credit=float(np.mean(
                            (predicted_delta[use, 1] < -1e-3) & (actual_delta[use, 1] >= 0)))))
    output = ROOT/'prediction_audit'
    output.mkdir(exist_ok=False)
    write_rows(output/'summary.csv', rows)
    write_rows(output/'states.csv', state_rows)
    (output/'scope.json').write_text(json.dumps(dict(source_sha256=digest(__file__),
        post_hoc_diagnostic=True, policy_changed=False,
        interpretation='Single first action then PRS, not repeated learned control; realized counts '
        'are noisy labels, not true conditional expectations; no calibration guarantee',
        uses_training_labels_only_for_diagnosis=True), indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps([r for r in rows if r['split'] == 'prediction-validation' and
                      r['scenario'] == 1 and r['stratum'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
