"""Independent scalar hardware replay for every completed diagnostic cycle."""
import json
import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.cycle_accounting import cycle_difference, break_even_slots, scalar_bridge, completed_cycles
from major_revision.cycle_accounting_study import ROOT, SOURCE, MODES, STREAM, initialize
from major_revision.information_probe_study import finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction_study import csv_rows, all_files


def main():
    sources = [ROOT/f'shard{s}' for s in (0, 1)]
    for source in sources:
        verify(source)
    folder, hashes = initialize('verified', [p for s in sources for p in all_files(s)])
    policies = [r for s in sources for r in csv_rows(s/'policies.csv')]
    cycles = [r for s in sources for r in csv_rows(s/'cycles.csv')]
    censored = [r for s in sources for r in csv_rows(s/'censored.csv')]
    cfg, error, count, open_count, originals = make_config('validation'), 0., 0, 0, 0
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            source = ROOT/f'shard{i//4}'
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            for mode in MODES:
                for extended in (0, 1):
                    with np.load(source/f'timeline_{mode}_{extended}_s{j}_i{i}.npz', allow_pickle=False) as z:
                        trace = {k: z[k].copy() for k in z.files}
                    complete, open_cycles = completed_cycles(cfg, trace)
                    for cycle in complete:
                        saved = next(r for r in cycles if int(r['scenario']) == j and int(r['trajectory']) == i
                            and r['mode'] == mode and int(r['extended']) == extended and
                            int(r['module']) == cycle['module'] and int(r['off']) == cycle['off'])
                        measured = scalar_bridge(cfg, tr, trace, cycle)
                        formula = cycle_difference(cfg, cycle['sleep_slots'])
                        for k, value in measured.items():
                            error = max(error, abs(value-float(saved[k])))
                        error = max(error, abs(measured['energy_difference_j']-formula),
                            abs(formula-float(saved['analytic_difference_j'])))
                        count += 1
                    open_count += len(open_cycles)
                    originals += 1
            print('cycle scalar replay', j, i, flush=True)
    assert originals == len(policies) == 144 and count == len(cycles) and open_count == len(censored)
    assert error < 1e-9
    summary = []
    for j in range(3):
        for mode in MODES:
            for extended in (0, 1):
                rows = [r for r in cycles if int(r['scenario']) == j and r['mode'] == mode and int(r['extended']) == extended]
                short = [r for r in rows if int(r['sleep_slots']) <= cfg.horizon]
                for label, data in (('all', rows), ('short', short)):
                    energies = np.array([float(r['energy_difference_j']) for r in data])
                    failures = np.array([int(r['failed_user_slot_difference']) for r in data])
                    summary.append(dict(scenario=j, mode=mode, extended=extended, subset=label,
                        cycles=len(data), min_extra_keep_energy_j=float(energies.min()) if len(data) else 0.,
                        mean_extra_keep_energy_j=float(energies.mean()) if len(data) else 0.,
                        max_extra_keep_energy_j=float(energies.max()) if len(data) else 0.,
                        keeping_costs_more=int(np.sum(energies > 1e-10)),
                        keeping_reduces_failures=int(np.sum(failures < 0)),
                        keeping_increases_failures=int(np.sum(failures > 0)),
                        keeping_equal_failures=int(np.sum(failures == 0))))
    all_positive = all(float(r['energy_difference_j']) > 0 for r in cycles)
    gate = dict(all_completed_cycles_cheaper_than_keeping=all_positive,
        minimum_realizable_cycle_already_saves=cycle_difference(cfg, 4) > 0,
        anti_cycle_energy_only_justification_supported=False if all_positive else None,
        rl_superiority_established=False, independent_confirmation=False, scientific_completion=False)
    _write_rows(folder/'summary.csv', summary)
    _write_rows(folder/'identity.csv', [dict(sleep_slots=l, wake_slots=cfg.wake_delay_slots,
        keeping_minus_cycling_j=cycle_difference(cfg, l)) for l in (4, 8, 16, 32)])
    (folder/'verification.json').write_text(json.dumps(dict(archived_policy_schedules=originals,
        completed_cycle_scalar_replays=count, censored_cycles=open_count, maximum_error=error,
        break_even_sleep_slots=break_even_slots(cfg), viewed_external_trajectories=24,
        causal_policy=False, training_updates=0, fixed_other_commands=True), indent=2))
    (folder/'gate.json').write_text(json.dumps(gate, indent=2))
    finish(folder, hashes)
    print(json.dumps(dict(verification=dict(cycles=count, censored=open_count, error=error), gate=gate), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
