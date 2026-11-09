"""Audit saved episode/slot consistency without rerunning or changing results."""
import csv
import json
from collections import defaultdict

import numpy as np

from joint_power.sensitivity.study import ROOT, comparisons, intact, summarize
from joint_power.study import digest


IDENTITY = ('scenario', 'trajectory', 'setting', 'method', 'seed')


def main():
    intact()
    hashes = json.loads((ROOT/'output_hashes.json').read_text())
    inputs = json.loads((ROOT/'manifest.json').read_text())['input_hashes']
    assert all(digest(p) == h for p, h in (hashes | inputs).items())
    with (ROOT/'episodes.csv').open(newline='') as stream:
        episodes = [{k: (v if k == 'method' else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(stream)]
    for row in episodes:
        for key in ('scenario', 'trajectory', 'setting', 'seed', 'trace_seed'):
            row[key] = int(row[key])
    saved = {tuple(str(r[k]) for k in IDENTITY): r for r in episodes}
    assert len(saved) == len(episodes) == 984
    totals, counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    with (ROOT/'slots.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in IDENTITY)
            assert int(row['slot']) == counts[key]
            counts[key] += 1
            minimum = 16 if row['method'] == 'Full array' else int(row['method'][-1])
            assert int(row['command_bits']).bit_count() >= minimum
            for metric in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'outage', 'active_count'):
                totals[key][metric] += float(row[metric])
    for key, row in saved.items():
        assert counts[key] == 140
        for metric in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
            assert abs(totals[key][metric]-row[metric]) < 1e-8
        assert abs(totals[key]['outage']/140-row['outage']) < 1e-10
        assert abs(totals[key]['active_count']/140-row['mean_active']) < 1e-10
    table_lengths = {}
    for filename, expected in (('summary', summarize(episodes)), ('paired', comparisons(episodes))):
        with (ROOT/f'{filename}.csv').open(newline='') as stream:
            actual = list(csv.DictReader(stream))
        assert len(actual) == len(expected)
        table_lengths[filename] = len(actual)
        for a, b in zip(actual, expected):
            for k, v in b.items():
                if isinstance(v, str):
                    assert a[k] == v
                else:
                    np.testing.assert_allclose(float(a[k]), v, rtol=1e-12, atol=1e-12)
    # Preserve the first audit, whose descriptive row count was mistyped as 123.
    # The CSV and its actual row-by-row equality check were correct (93 rows).
    output = ROOT/'verified_v2'
    output.mkdir(exist_ok=False)
    report = dict(episodes=984, slots=sum(counts.values()), independent_traces=24,
        summary_rows=table_lengths['summary'], paired_rows=table_lengths['paired'],
        all_slot_aggregates_verified=True, default_baseline_implementations_matched=4,
        fixture_not_performance_data=True,
        audit_source_sha256=digest(__file__), preserved_hashes=len(hashes)+len(inputs),
        R1_closed=False, R3_closed=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
