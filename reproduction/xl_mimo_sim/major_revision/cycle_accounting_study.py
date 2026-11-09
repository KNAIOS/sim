"""All-cycle energy/service census of144 archived command schedules."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.cycle_accounting import (
    processing_per_module, cycle_difference, break_even_slots, replay, completed_cycles, bridge,
)
from major_revision.hold_action_audit import ROOT as SOURCE, MODES, STREAM
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction_study import csv_rows, all_files


ROOT = Path('results/major_revision_20260922_cycle_accounting')


def initialize(part, extras=()):
    folder = ROOT/part
    folder.mkdir(parents=True, exist_ok=False)
    files = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
        Path('major_revision/CYCLE_ACCOUNTING_PLAN.md'), *extras]
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(), threads=1,
        hashes=hashes, config=make_config('validation').as_dict(), viewed_data=True,
        independent_confirmation=False, causal_policy=False, training_updates=0), indent=2))
    return folder, hashes


def collect(shard):
    source = SOURCE/f'shard{shard}'
    verify(source)
    verify(SOURCE/'verified')
    folder, hashes = initialize(f'shard{shard}', all_files(source)+all_files(SOURCE/'verified'))
    cfg, rows, cycles, censored = make_config('validation'), [], [], []
    raw = csv_rows(source/'policies.csv')
    error, energy_error = 0., 0.
    for j, scenario in enumerate(SCENARIOS):
        for i in range(4*shard, 4*shard+4):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            for mode in MODES:
                for extended in (0, 1):
                    name = f'{mode}_{extended}_s{j}_i{i}'
                    with np.load(source/f'trace_{name}.npz', allow_pickle=False) as z:
                        trace = replay(cfg, tr, z['commands'])
                        error = max(error, float(np.max(np.abs(trace['energy']-z['energy']))),
                            float(np.max(np.abs(trace['outage']-z['outage']))))
                    saved = next(r for r in raw if int(r['scenario']) == j and int(r['trajectory']) == i
                        and r['mode'] == mode and int(r['extended']) == extended)
                    error = max(error, abs(trace['energy'].sum()-float(saved['energy_j'])),
                        abs(trace['outage'].mean()-float(saved['outage'])))
                    complete, open_cycles = completed_cycles(cfg, trace)
                    identifiers = dict(scenario=j, trajectory=i, mode=mode, extended=extended)
                    for cycle in complete:
                        assert cycle['sleep_slots'] >= 4 and cycle['sleep_slots'] % 4 == 0
                        values = bridge(cfg, tr, trace, cycle)
                        energy_error = max(energy_error, abs(values['energy_difference_j']-values['analytic_difference_j']))
                        cycles.append(dict(**identifiers, **cycle, **values))
                    censored += [dict(**identifiers, **c) for c in open_cycles]
                    rows.append(dict(**identifiers, energy_j=float(trace['energy'].sum()),
                        outage=float(trace['outage'].mean()), complete_cycles=len(complete),
                        censored_cycles=len(open_cycles), short_cycles=sum(c['sleep_slots'] <= cfg.horizon for c in complete)))
                    np.savez_compressed(folder/f'timeline_{name}.npz', **trace)
            print('cycle accounting', j, i, flush=True)
    assert error < 1e-9 and energy_error < 1e-9
    _write_rows(folder/'policies.csv', rows)
    _write_rows(folder/'cycles.csv', cycles)
    _write_rows(folder/'censored.csv', censored)
    (folder/'audit.json').write_text(json.dumps(dict(episodes=len(rows), complete_cycles=len(cycles),
        censored_cycles=len(censored), archive_error=error, identity_error=energy_error,
        processing_w_per_module=processing_per_module(cfg), minimum_cycle_difference_j=cycle_difference(cfg, 4),
        break_even_sleep_slots=break_even_slots(cfg), all_realized_cycles_sleep_save=all(r['energy_difference_j'] > 0 for r in cycles)), indent=2))
    finish(folder, hashes)
    print('cycle accounting complete', shard, len(cycles), energy_error, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    collect(parser.parse_args().shard)
