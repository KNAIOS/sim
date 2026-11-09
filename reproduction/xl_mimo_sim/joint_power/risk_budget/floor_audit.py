"""Audit frozen predictions against known current-slot/hardware lower bounds."""
import csv
import json

import numpy as np

from joint_power.risk_budget.study import ROOT, input_checks
from joint_power.study import digest, write_rows


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    input_checks()
    groups = [json.loads((ROOT/p/'output_hashes.json').read_text()) for p in ('audit', 'performance_difference')]
    assert all(digest(p) == h for group in groups for p, h in group.items())
    states = read(ROOT/'performance_difference/states.csv')
    records = read(ROOT/'audit/selected_slots.csv')
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory'], int(r['slot'])): r for r in records}
    rows = []
    for row in states:
        key = tuple(row[k] for k in ('method', 'seed', 'scenario', 'trajectory'))
        t = int(row['slot'])
        block = [lookup[(*key, t+j)] for j in range(4)]
        instant = round(3*float(block[0]['outage']))
        hardware_floor = instant+sum(3 for r in block[1:] if int(r['active_count']) == 0)
        predicted = float(row['selected_count_prediction'])
        assert hardware_floor <= int(row['chosen_remaining_failed'])
        rows.append(dict(method=row['method'], seed=int(row['seed']), scenario=int(row['scenario']),
            trajectory=int(row['trajectory']), slot=t, predicted_remaining_failed=predicted,
            current_slot_failed=instant, causal_hardware_floor=hardware_floor,
            below_current_slot=predicted+1e-8 < instant,
            below_causal_floor=predicted+1e-8 < hardware_floor,
            deficit=max(0., hardware_floor-predicted)))
    summary = []
    for s in range(3):
        group = [r for r in rows if r['scenario'] == s]
        summary.append(dict(scenario=s, states=len(group),
            below_current_slot=sum(r['below_current_slot'] for r in group),
            below_causal_floor=sum(r['below_causal_floor'] for r in group),
            positive_floor=sum(r['causal_hardware_floor'] > 0 for r in group),
            max_deficit=max(r['deficit'] for r in group)))
    output = ROOT/'floor_audit'
    output.mkdir(exist_ok=False)
    write_rows(output/'states.csv', rows)
    write_rows(output/'summary.csv', summary)
    report = dict(summary=summary, policy_modified=False, source_sha256=digest(__file__),
        scope='Existing selected predictions only. Current-slot service is deterministic given '
        'the simulator causal CSI/action; later zero-active availability is determined by hardware. '
        'This proves a lower-bound inconsistency, not that fixing it suffices for a reliable energy win.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for group in groups for p, h in group.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
