"""Integrity, matched sample sizes, independent physics and paired diagnostics."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.collect import ROOT, STREAMS, load
from joint_power.aggregation.control import METRICS, input_checks, models, read_rows, summaries
from joint_power.calibrated.histogram import load_predictor
from joint_power.rollout.analyze import load as old_load
from joint_power.rollout.control import episode
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows


def dataset_audit():
    old = old_load('train')
    arms = {name: load(ROOT/f'{name}.npz') for name in ('prs_added', 'visitation_added')}
    for data in arms.values():
        assert len(data['x']) == 384
        for key in old:
            np.testing.assert_array_equal(data[key][:192], old[key])
    train_ids, valid_ids, summary = set(), set(), []
    for split, (_, number) in STREAMS.items():
        for behavior in ('prs', 'learned'):
            data = load(ROOT/f'{split}_{behavior}.npz')
            ids = set(map(int, data['trajectory_seed']))
            assert len(ids) == 3*number
            (train_ids if split == 'train' else valid_ids).update(ids)
            active = data['x'][:, -1, -112:].reshape(-1, 16, 7)[:, :, 1].sum(-1)
            np.testing.assert_array_equal(active, data['active_before'])
            capacity = 3*(140-data['identity'][:, 2])
            assert (data['y'][..., 1] >= 0).all()
            assert (data['y'][:, :, -1, 1] <= capacity[:, None]).all()
            for s in range(3):
                use = data['identity'][:, 0] == s
                summary.append(dict(split=split, behavior=behavior, scenario=s, states=int(use.sum()),
                    below4=int(np.sum(active[use] < 4)), minimum_active=float(active[use].min()),
                    mean_active=float(active[use].mean())))
    assert len(train_ids) == 24 and len(valid_ids) == 12 and train_ids.isdisjoint(valid_ids)
    for name, behavior in (('prs_added', 'prs'), ('visitation_added', 'learned')):
        new = load(ROOT/f'train_{behavior}.npz')
        for key in arms[name]:
            np.testing.assert_array_equal(arms[name][key][192:], new[key])
    validation = load(ROOT/'validation.npz')
    for behavior, interval in (('prs', slice(0, 96)), ('learned', slice(96, 192))):
        source = load(ROOT/f'validation_{behavior}.npz')
        for key in validation:
            np.testing.assert_array_equal(validation[key][interval], source[key])
    return summary


def differences(rows):
    rng = np.random.default_rng(2026092221)
    output = []
    for s in range(3):
        for reference in ('PRS-added', 'Previous absolute', 'PRS original'):
            for metric in ('energy_j', 'outage'):
                delta = np.empty((2, 4))
                for n, seed in enumerate((11, 22)):
                    for i in range(4):
                        first = next(r for r in rows if (r['method'], r['seed'], r['scenario'], r['trajectory'])
                                     == ('Visitation-added', seed, s, i))
                        ref_seed = 0 if reference == 'PRS original' else seed
                        second = next(r for r in rows if (r['method'], r['seed'], r['scenario'], r['trajectory'])
                                      == (reference, ref_seed, s, i))
                        delta[n, i] = first[metric]-second[metric]
                seed_ids = rng.integers(0, 2, (4000, 2))
                trace_ids = rng.integers(0, 4, (4000, 4))
                boot = delta[seed_ids[:, :, None], trace_ids[:, None, :]].mean((1, 2))
                output.append(dict(scenario=s, reference=reference, metric=metric,
                    difference=float(delta.mean()), ci_low=float(np.quantile(boot, .025)),
                    ci_high=float(np.quantile(boot, .975)), independent_trajectories=4, training_seeds=2))
    return output


def main():
    torch.set_num_threads(1)
    input_checks()
    folder = ROOT/'control_validation'
    hashes = json.loads((folder/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    selected = read_rows(folder/'selected_episodes.csv')
    candidates = read_rows(folder/'candidate_episodes.csv')
    assert len(candidates) == 432 and len(selected) == 156
    key = lambda r: (r['method'], r['seed'], r['scenario'], r['trajectory'], r['parameter'])
    assert len({key(r) for r in candidates}) == len(candidates)
    assert len({key(r) for r in selected}) == len(selected)
    coverage = dataset_audit()
    loaded, _ = models()
    cfg, predictor = make_config('validation'), load_predictor()
    traces = {(s, i): make_trace(cfg, 3909220000+s*10000+i, spec)
              for s, spec in enumerate(SCENARIOS) for i in range(4)}
    maximum, slots, visits = defaultdict(float), [], []
    replayed = 0
    for row in selected:
        if (row['method'], row['seed']) not in loaded:
            continue
        s, i, seed = row['scenario'], row['trajectory'], row['seed']
        trace = traces[s, i]
        model = loaded[row['method'], seed]
        metrics, records = episode(trace, 'MC absolute', row['parameter'], predictor, model)
        for key, value in metrics.items():
            assert abs(value-row[key]) < 1e-8
        for key, value in independent(records, trace).items():
            maximum[key] = max(maximum[key], value)
        identity = {key: row[key] for key in ('scenario', 'trajectory', 'method', 'seed', 'parameter')}
        slots.extend(dict(**identity, **record) for record in records)
        visits.extend(dict(**identity, slot=r['slot'], active_before=r['observed_count']) for r in records[::4])
        replayed += 1
        if replayed % 12 == 0:
            print(f'aggregation physical replay {replayed}/48', flush=True)
    assert replayed == 48 and len(slots) == 6720
    assert maximum['rate_mbps'] < 1e-5
    assert max(value for key, value in maximum.items() if key != 'rate_mbps') < 1e-8
    saved = read_rows(folder/'summary.csv')
    recomputed = summaries(selected)
    for actual, expected in zip(saved, recomputed):
        assert (actual['scenario'], actual['method']) == (expected['scenario'], expected['method'])
        for metric in METRICS:
            assert abs(actual[metric]-expected[metric]) < 1e-8
    comparisons = differences(selected)
    output = ROOT/'audit'
    output.mkdir(exist_ok=False)
    write_rows(output/'training_coverage.csv', coverage)
    write_rows(output/'selected_slots.csv', slots)
    write_rows(output/'selected_visits.csv', visits)
    write_rows(output/'paired_differences.csv', comparisons)
    report = dict(new_selected_episodes=48, independently_checked_slots=6720,
        maximum_errors=dict(maximum), new_candidate_validation_episodes=432,
        reused_previously_verified_episodes=108, training_states_each=384,
        common_prediction_validation_states=192, training_coverage=coverage,
        paired_differences=comparisons, repeated_control_validation=True,
        uncertainty='Descriptive two-way seed/whole-trajectory bootstrap, not post-selection significance',
        gate=json.loads((folder/'gate.json').read_text()), source_sha256=digest(__file__))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
