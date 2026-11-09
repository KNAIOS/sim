"""Read-only result audit and example replay; no channel generation or retuning."""
import csv
import json
from collections import defaultdict
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.standard_transfer.study import ROOT, STATES, LOADS, analyze, intact
from joint_power.standard_transfer.verify import independent
from joint_power.study import digest


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def typed(rows):
    return [{k: v if k in ('state', 'method') else (int(v) if k in ('trajectory', 'seed') else float(v))
             for k, v in row.items()} for row in rows]


def main():
    intact()
    inputs = json.loads((ROOT/'manifest.json').read_text())['input_hashes']
    outputs = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in (inputs | outputs).items())
    rows = typed(read(ROOT/'episodes.csv'))
    assert len(rows) == 720
    slot_count, decision_count, replay_count = 0, 0, 0
    errors = defaultdict(float)
    cfg = replace(make_config('validation'), array_aperture_m=15*.5*299792458/14e9)
    for state in STATES:
        for index in range(20):
            folder = ROOT/f'{state}_{index}'
            local = typed(read(folder/'episodes.csv'))
            assert len(local) == 18
            grouped = defaultdict(list)
            for row in read(folder/'slots.csv'):
                ident = (float(row['load']), row['method'], int(row['seed']))
                values = {k: v if k in ('state', 'method', 'allocator_mode', 'full_target_status')
                          else (int(v) if k in ('trajectory', 'seed', 'slot', 'command_bits', 'active_bits',
                              'active_count', 'waking_count', 'observed_count', 'numerical_fallback', 'allocator_iterations')
                              else float(v)) for k, v in row.items()}
                grouped[ident].append(values)
                slot_count += 1
            for row in local:
                records = grouped[row['load'], row['method'], row['seed']]
                assert len(records) == 140 and [r['slot'] for r in records] == list(range(140))
                minimum = 16 if row['method'] == 'Full array' else int(row['method'][-1])
                assert all(r['command_bits'].bit_count() >= minimum for r in records)
                for metric in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
                    assert abs(sum(r[metric] for r in records)-row[metric]) < 1e-8
                assert abs(np.mean([r['outage'] for r in records])-row['outage']) < 1e-12
                global_row = next(r for r in rows if all(r[k] == row[k]
                    for k in ('state', 'trajectory', 'load', 'method', 'seed')))
                assert row == global_row
            for row in read(folder/'decisions.csv'):
                records = grouped[float(row['load']), row['method'], int(row['seed'])]
                t = int(row['slot'])
                failed = sum(round(3*r['outage']) for r in records[t:t+4])
                assert int(row['block_failed']) == failed
                assert int(row['executed_floor']) <= failed
                assert int(row['cumulative_failed']) == sum(round(3*r['outage']) for r in records[:t+4])
                assert all(int(row['executed_bits']) == r['command_bits'] for r in records[t:t+4])
                decision_count += 1
            if index == 0:
                with np.load(folder/'example.npz') as payload:
                    physical, requirements = payload['channel'], payload['requirements_mbps']
                for (load, _, _), records in grouped.items():
                    for key, error in independent(cfg, records, physical, requirements*load).items():
                        errors[key] = max(errors[key], error)
                    replay_count += len(records)
    table_counts = {}
    for name, expected in zip(('summary', 'paired'), analyze(rows)):
        actual = read(ROOT/f'{name}.csv')
        assert len(actual) == len(expected)
        table_counts[name] = len(actual)
        for a, b in zip(actual, expected):
            for key, value in b.items():
                if isinstance(value, str):
                    assert a[key] == value
                else:
                    np.testing.assert_allclose(float(a[key]), value, rtol=1e-12, atol=1e-12)
    assert slot_count == 100800 and decision_count == 25200 and replay_count == 5040
    assert errors['rate_mbps'] < 1e-5
    out = ROOT/'audited'
    out.mkdir(exist_ok=False)
    report = dict(episodes=720, slot_aggregates_checked=slot_count, decisions_checked=decision_count,
        cached_example_slots_independently_replayed=replay_count, replay_errors=dict(errors),
        table_rows=table_counts, frozen_hashes_checked=len(inputs)+len(outputs),
        audit_source_sha256=digest(__file__), no_result_changes=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
