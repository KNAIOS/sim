"""Audit written ledgers and independently replay cached first/last drops."""
import json
from collections import defaultdict

import numpy as np

from joint_power.focus_confirmation.study import ROOT, PREVIOUS, NUMBER, SPECS, analyze, configuration
from joint_power.standard_transfer.audit import read, typed
from joint_power.standard_transfer.verify import independent
from joint_power.study import digest


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    hashes = manifest['input_hashes'] | json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(v['path']) == v['sha256'] for v in manifest['checkpoints'].values())
    rows = typed(read(ROOT/'episodes.csv'))
    assert len(rows) == 120
    old_channels = {json.loads(p.read_text())['channel_sha256'] for p in PREVIOUS.glob('*/channel_metadata.json')}
    old_seeds = {json.loads(p.read_text())['seed'] for p in PREVIOUS.glob('*/channel_metadata.json')}
    new_channels, errors = set(), defaultdict(float)
    slot_count = decision_count = replay_count = 0
    for index in range(NUMBER):
        folder = ROOT/f'NLOS_{index}'
        metadata = json.loads((folder/'channel_metadata.json').read_text())
        assert metadata['seed'] == manifest['channel_stream']+index
        assert metadata['demand_seed'] == manifest['demand_stream']+index
        assert metadata['seed'] not in old_seeds and metadata['channel_sha256'] not in old_channels
        assert metadata['channel_sha256'] not in new_channels
        new_channels.add(metadata['channel_sha256'])
        local = typed(read(folder/'episodes.csv'))
        grouped = defaultdict(list)
        for row in read(folder/'slots.csv'):
            ident = (row['method'], int(row['seed']))
            strings = ('state', 'method', 'allocator_mode', 'full_target_status')
            integers = ('trajectory', 'seed', 'slot', 'command_bits', 'active_bits',
                        'active_count', 'waking_count', 'observed_count', 'numerical_fallback', 'allocator_iterations')
            values = {k: v if k in strings else int(v) if k in integers else float(v) for k, v in row.items()}
            grouped[ident].append(values)
            slot_count += 1
        assert set(grouped) == set(SPECS)
        for row in local:
            records = grouped[row['method'], row['seed']]
            assert [r['slot'] for r in records] == list(range(140))
            assert all(4 <= r['command_bits'].bit_count() <= 16 for r in records)
            for metric in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
                assert abs(sum(r[metric] for r in records)-row[metric]) < 1e-8
            assert abs(np.mean([r['outage'] for r in records])-row['outage']) < 1e-12
            assert row == next(r for r in rows if all(r[k] == row[k]
                for k in ('state', 'trajectory', 'load', 'method', 'seed')))
        for row in read(folder/'decisions.csv'):
            records = grouped[row['method'], int(row['seed'])]
            t = int(row['slot'])
            failed = sum(round(3*r['outage']) for r in records[t:t+4])
            assert int(row['block_failed']) == failed
            assert int(row['executed_floor']) <= failed
            assert int(row['cumulative_failed']) == sum(round(3*r['outage']) for r in records[:t+4])
            assert all(int(row['executed_bits']) == r['command_bits'] for r in records[t:t+4])
            decision_count += 1
        if index in (0, NUMBER-1):
            with np.load(folder/'example.npz') as payload:
                physical, requirements = payload['channel'], payload['requirements_mbps']
            for records in grouped.values():
                for key, error in independent(configuration(), records, physical, requirements).items():
                    errors[key] = max(errors[key], error)
                replay_count += len(records)
    summaries, paired, report = analyze(rows)
    for name, expected in (('summary', summaries), ('paired', paired)):
        actual = read(ROOT/f'{name}.csv')
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            for key, value in b.items():
                if isinstance(value, str):
                    assert a[key] == value
                else:
                    np.testing.assert_allclose(float(a[key]), value, rtol=1e-12, atol=1e-12)
    saved = json.loads((ROOT/'report.json').read_text())
    assert all(saved[key] == value for key, value in report.items())
    assert slot_count == 16800 and decision_count == 4200 and replay_count == 840
    assert errors['rate_mbps'] < 1e-5
    output = ROOT/'audited'
    output.mkdir(exist_ok=False)
    result = dict(episodes=120, independent_new_channels=len(new_channels),
        overlap_with_previous_channels=0, slot_aggregates_checked=slot_count,
        decisions_checked=decision_count, cached_slots_replayed=replay_count,
        replay_errors=dict(errors), frozen_hashes_checked=len(hashes),
        all_saved_statistics_recomputed=True, no_result_changes=True,
        independent_confirmation_passed=report['independent_confirmation_passed'])
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
