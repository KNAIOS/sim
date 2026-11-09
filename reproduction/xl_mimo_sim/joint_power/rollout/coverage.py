"""Verify extended choices and measure a concrete training-state coverage gap."""
import csv
import json

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.rollout.absolute import ROOT
from joint_power.rollout.analyze import load
from joint_power.rollout.control import episode, input_checks, load_models
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    torch.set_num_threads(1)
    input_checks()
    source = ROOT/'price_extension'
    hashes = json.loads((source/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    old = read(ROOT/'validation/selected_episodes.csv')
    rows = read(source/'selected_episodes.csv')
    key = lambda r: (r['scenario'], r['trajectory'], r['method'], r['seed'], r['parameter'])
    old_lookup = {key(r): r for r in old}
    untouched = [r for r in rows if r['method'] != 'MC absolute']
    assert len(untouched) == 84
    for row in untouched:
        reference = old_lookup[key(row)]
        for metric in ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active'):
            assert float(row[metric]) == float(reference[metric])
    train = load('train')
    post = train['x'][:, -1, -112:].reshape(-1, 16, 7)
    active = post[:, :, 1].sum(-1)
    assert np.allclose(active, np.rint(active))
    training_min, training_max = int(active.min()), int(active.max())
    models, _ = load_models()
    cfg, predictor = make_config('validation'), load_predictor()
    observations, max_error, energy_error = [], 0., 0.
    for row in rows:
        if row['method'] != 'MC absolute':
            continue
        s, i, seed = int(row['scenario']), int(row['trajectory']), int(row['seed'])
        trace = make_trace(cfg, 3909220000+s*10000+i, SCENARIOS[s])
        metrics, records = episode(trace, row['method'], float(row['parameter']), predictor,
                                   models[row['method'], seed])
        for k, v in metrics.items():
            assert abs(v-float(row[k])) < 1e-8
        errors = independent(records, trace)
        max_error = max(max_error, errors['rate_mbps'])
        energy_error = max(energy_error, max(v for k, v in errors.items() if k != 'rate_mbps'))
        for record in records[::4]:
            count = record['observed_count']
            observations.append(dict(scenario=s, trajectory=i, seed=seed, slot=record['slot'],
                active_before_action=count, below_training_min=count < training_min,
                above_training_max=count > training_max))
    assert max_error < 1e-5 and energy_error < 1e-8
    summary = []
    for s in range(3):
        group = [r for r in observations if r['scenario'] == s]
        summary.append(dict(scenario=s, decisions=len(group),
            fraction_below_training_min=float(np.mean([r['below_training_min'] for r in group])),
            mean_active_before_action=float(np.mean([r['active_before_action'] for r in group]))))
    output = ROOT/'coverage_audit'
    output.mkdir(exist_ok=False)
    write_rows(output/'states.csv', observations)
    write_rows(output/'summary.csv', summary)
    report = dict(training_states=len(active), training_min_active=training_min,
        training_max_active=training_max, training_states_below4=int(np.sum(active < 4)),
        summary=summary, replayed_episodes=24, replayed_slots=3360,
        unchanged_previously_verified_episodes=84, max_rate_error_mbps=max_error,
        max_energy_component_error_j=energy_error, source_sha256=digest(__file__),
        scope='A measured count-support gap, not a proof that it causes the entire performance deficit',
        post_hoc_validation_only=True, development_launched=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
