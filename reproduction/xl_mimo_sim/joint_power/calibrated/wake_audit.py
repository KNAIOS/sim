"""Count wake-ups closed before any data-serving slot, without assuming regret."""
import csv
import json
from collections import defaultdict

import numpy as np

from joint_power.study import digest, write_rows
from joint_power.calibrated.histogram import ROOT


def main():
    hashes = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    groups = defaultdict(list)
    with (ROOT/'evaluation/slots.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = int(row['scenario']), int(row['trajectory']), row['method'], int(row['seed'])
            groups[key].append(row)
    episodes = []
    for key, rows in groups.items():
        active, countdown = np.ones(16, bool), np.zeros(16, int)
        waiting = np.zeros(16, bool)
        wake_count = unused_close = served = 0
        for row in sorted(rows, key=lambda r: int(r['slot'])):
            command = np.array([bool(int(row['command_bits']) & 1 << q) for q in range(16)])
            off = ~command & (active | (countdown > 0))
            on = command & ~active & (countdown == 0)
            unused_close += int((off & waiting).sum())
            waiting[off] = False
            assert not waiting[on].any()
            waiting[on] = True
            wake_count += int(on.sum())
            active[off], countdown[off], countdown[on] = False, 0, 4
            assert int(row['active_bits']) == sum(int(b) << q for q, b in enumerate(active))
            served += int((waiting & active).sum())
            waiting[active] = False
            ready = countdown == 1
            countdown[countdown > 0] -= 1
            active[ready] = True
        assert wake_count == unused_close+served+int(waiting.sum())
        episodes.append(dict(scenario=key[0], trajectory=key[1], method=key[2], seed=key[3],
            wakes=wake_count, closed_before_data_slot=unused_close,
            reached_data_slot=served, horizon_censored=int(waiting.sum())))
    summary = []
    for s in range(3):
        for method in dict.fromkeys(r['method'] for r in episodes):
            rows = [r for r in episodes if r['scenario'] == s and r['method'] == method]
            summary.append(dict(scenario=s, method=method,
                **{k: float(np.mean([r[k] for r in rows])) for k in
                   ('wakes', 'closed_before_data_slot', 'reached_data_slot', 'horizon_censored')}))
    folder = ROOT/'wake_audit'
    folder.mkdir(exist_ok=False)
    write_rows(folder/'episodes.csv', episodes)
    write_rows(folder/'summary.csv', summary)
    (folder/'scope.json').write_text(json.dumps(dict(source_sha256=digest(__file__),
        caveat='A wake without a data-serving slot may still acquire CSI; not proof it has zero future value'), indent=2)+'\n')
    print(json.dumps([r for r in summary if r['scenario'] == 1], indent=2))


if __name__ == '__main__':
    main()
