"""Single-intervention hindsight diagnosis with whole-trajectory statistics."""
import json

import numpy as np

from joint_power.rollout.collect import ROOT
from joint_power.study import digest, write_rows


def load(split):
    with np.load(ROOT/f'{split}.npz') as saved:
        return {k: saved[k] for k in saved.files}


def intact():
    for name in ('manifest.json', 'collection_hashes.json'):
        data = json.loads((ROOT/name).read_text())
        hashes = data.get('input_hashes', data)
        assert all(digest(p) == h for p, h in hashes.items()), name


def choices(data, horizon, prefix_safe=False):
    y, ref = data['y'], data['reference']
    base = y[np.arange(len(y)), ref]
    if prefix_safe:
        feasible = np.all(y[..., 1] <= base[:, None, :, 1]+1e-9, axis=-1)
    else:
        feasible = y[:, :, horizon, 1] <= base[:, None, horizon, 1]+1e-9
    cost = np.where(feasible, y[:, :, horizon, 0], np.inf)
    best = cost.argmin(1)
    # Prefer the exact reference when energy ties, not an arbitrary other mask.
    return np.where(cost[np.arange(len(y)), best] < base[:, horizon, 0]-1e-9, best, ref)


def main():
    intact()
    assert (ROOT/'collection_complete.json').exists()
    output = ROOT/'diagnosis'
    output.mkdir(exist_ok=False)
    rows, summary = [], []
    rng = np.random.default_rng(2026092216)
    for split in ('train', 'validation'):
        data = load(split)
        y, ref = data['y'], data['reference']
        base = y[np.arange(len(y)), ref]
        methods = {'Hindsight4': choices(data, 0), 'Hindsight16': choices(data, 1),
                   'HindsightFull': choices(data, 2),
                   'HindsightFull-prefix-safe': choices(data, 2, True),
                   'Causal raw search': data['raw_action'], 'PRS': ref}
        for method, actions in methods.items():
            delta = y[np.arange(len(y)), actions]-base
            for i, identity in enumerate(data['identity']):
                s, trajectory, slot = map(int, identity)
                rows.append(dict(split=split, scenario=s, trajectory=trajectory, slot=slot,
                    method=method, action=int(actions[i]), reference=int(ref[i]),
                    energy_delta_j=float(delta[i, -1, 0]), failures_delta=int(delta[i, -1, 1]),
                    prefix4_failures_delta=int(delta[i, 0, 1]),
                    prefix16_failures_delta=int(delta[i, 1, 1]),
                    local_saving_j=float(-delta[i, {'Hindsight4': 0, 'Hindsight16': 1}.get(method, 2), 0]),
                    baseline_remaining_j=float(base[i, -1, 0]),
                    remaining_slots=140-slot))
            for s in range(3):
                group = [r for r in rows if r['split'] == split and r['scenario'] == s and r['method'] == method]
                ids = sorted({r['trajectory'] for r in group})
                trajectory_means = np.array([np.mean([r['energy_delta_j'] for r in group if r['trajectory'] == i])
                                             for i in ids])
                boot = trajectory_means[rng.integers(0, len(ids), (4000, len(ids)))].mean(1)
                summary.append(dict(split=split, scenario=s, method=method, states=len(group),
                    trajectories=len(ids), energy_delta_j=float(trajectory_means.mean()),
                    energy_ci_low=float(np.quantile(boot, .025)), energy_ci_high=float(np.quantile(boot, .975)),
                    failures_delta=float(np.mean([r['failures_delta'] for r in group])),
                    fraction_extra_failures=float(np.mean([r['failures_delta'] > 0 for r in group])),
                    fraction_gain_at_least_point2_no_extra_failure=float(np.mean([
                        r['energy_delta_j'] <= -.2 and r['failures_delta'] <= 0 for r in group])),
                    local_gain_but_full_energy_loss=float(np.mean([
                        r['local_saving_j'] > 1e-9 and r['energy_delta_j'] > 1e-9 for r in group]))))
    selected = next(r for r in summary if r['split'] == 'validation' and r['scenario'] == 1
                    and r['method'] == 'HindsightFull')
    gate = (selected['energy_delta_j'] <= -.2 and
            selected['fraction_gain_at_least_point2_no_extra_failure'] >= .25)
    write_rows(output/'states.csv', rows)
    write_rows(output/'summary.csv', summary)
    report = dict(passed_for_probe=bool(gate), nominal_validation=selected,
        source_sha256=digest(__file__), scope='Overlapping single interventions, not additive episode savings '
        'or deployable clairvoyant controller; no policy-improvement guarantee')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
