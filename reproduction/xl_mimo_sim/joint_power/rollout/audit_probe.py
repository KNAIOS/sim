"""Post-hoc prediction-error diagnosis; never changes the frozen probe choice."""
import json

import numpy as np
import torch

from joint_power.rollout.analyze import intact, load
from joint_power.rollout.collect import ROOT
from joint_power.rollout.probe import DifferenceValue, FULL_SLOT, select
from joint_power.study import digest, write_rows


def main():
    torch.set_num_threads(1)
    intact()
    hashes = json.loads((ROOT/'probe/output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    output = ROOT/'probe_audit'
    output.mkdir(exist_ok=False)
    rows = []
    for split in ('train', 'validation'):
        data = load(split)
        x, ref = torch.from_numpy(data['x']), torch.from_numpy(data['reference'])
        full = data['y'][:, :, -1]
        actual = full-full[np.arange(len(full)), data['reference'], None]
        for seed in (11, 22):
            for epoch in (20, 40, 60, 80):
                saved = torch.load(ROOT/f'probe/seed{seed}_epoch{epoch:03d}.pt', weights_only=True)
                model = DifferenceValue(saved['state']['mean'], saved['state']['std'])
                model.load_state_dict(saved['state'])
                with torch.no_grad():
                    prediction = model(x, ref).numpy()
                for mode in ('original', 'no_negative_risk_credit'):
                    objective = prediction.copy()
                    if mode != 'original':
                        objective[..., 1] = np.maximum(objective[..., 1], 0)
                    for price in (4., 8., 16.):
                        action = select(objective, data['reference'], price)
                        chosen = actual[np.arange(len(x)), action]
                        estimated = prediction[np.arange(len(x)), action]*[FULL_SLOT, 3.]
                        for s in range(3):
                            use = data['identity'][:, 0] == s
                            real, predicted = chosen[use], estimated[use]
                            rows.append(dict(split=split, seed=seed, epoch=epoch, mode=mode,
                                price=price, scenario=s, states=int(use.sum()),
                                changed_fraction=float(np.mean(action[use] != data['reference'][use])),
                                mean_energy_delta_j=float(real[:, 0].mean()),
                                mean_failures_delta=float(real[:, 1].mean()),
                                mean_predicted_energy_delta_j=float(predicted[:, 0].mean()),
                                mean_predicted_failures_delta=float(predicted[:, 1].mean()),
                                risk_reduction_not_realized_fraction=float(np.mean(
                                    (predicted[:, 1] < -1e-6) & (real[:, 1] >= 0))),
                                energy_saving_not_realized_fraction=float(np.mean(
                                    (predicted[:, 0] < -1e-6) & (real[:, 0] > 1e-9)))))
    write_rows(output/'prediction_errors.csv', rows)
    selected = json.loads((ROOT/'probe/selection.json').read_text())
    mc = [r for r in selected if r['method'] == 'MC']
    # Inspect every saved checkpoint; keep original epoch0 choices unchanged.
    report = dict(original_selected=mc, causal_probe_passed=all(r['epoch'] > 0 for r in mc),
        closed_loop_evaluation_launched=False,
        reason='Both prespecified probes selected their unchanged baseline; do not run redundant '
               'development episodes or relabel a diagnostic intervention as a policy win.',
        exploratory_variant='Clipping negative predicted risk credit is a post-hoc diagnostic only; '
                            'not a validated policy and no safety guarantee.',
        source_sha256=digest(__file__), validation_states_per_scenario=32,
        independent_validation_trajectories_per_scenario=4)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
