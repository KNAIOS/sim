"""Replay every restored-hold policy and branch; preserve viewed-data boundaries."""
import json
import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.full_mask_audit import mask_id
from major_revision.hold_action_audit import (
    ROOT, MODES, STREAM, augmented_data, choose, branch, policy, initialize,
)
from major_revision.information_probe_study import finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction_study import csv_rows, all_files
from major_revision.risk_rl_study import setup


def read_optional(path):
    return csv_rows(path) if path.exists() else []


def main():
    settings, model = setup()
    sources = [ROOT/f'shard{s}' for s in (0, 1)]
    for source in sources:
        verify(source)
    folder, hashes = initialize('verified', [p for s in sources for p in all_files(s)])
    raw = [r for s in sources for r in csv_rows(s/'policies.csv')]
    roots = [r for s in sources for r in csv_rows(s/'roots.csv')]
    branches = [r for s in sources for r in read_optional(s/'branches.csv')]
    assert len(raw) == 144 and len(roots) == 2520
    assert len(branches) == sum(int(r['changed']) for r in roots)
    cfg, error, root_count, branch_count, episode_count = make_config('validation'), 0., 0, 0, 0
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            source = ROOT/f'shard{i//4}'
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            for mode in MODES:
                parameter, minimum = settings[mode, j]
                for extended in (False, True):
                    result, trace, _ = policy(cfg, tr, mode, parameter, minimum, model, extended)
                    row = next(r for r in raw if int(r['scenario']) == j and int(r['trajectory']) == i
                        and r['mode'] == mode and int(r['extended']) == int(extended))
                    for k, value in result.items():
                        if k != 'method':
                            error = max(error, abs(value-float(row[k])))
                    with np.load(source/f'trace_{mode}_{int(extended)}_s{j}_i{i}.npz', allow_pickle=False) as z:
                        for k in trace:
                            np.testing.assert_array_equal(trace[k], z[k])
                    error = max(error, abs(trace['energy'].sum()-result['energy_j']),
                        abs(trace['outage'].mean()-result['outage']),
                        abs(result['energy_j']-result['switching_j']-result['pilot_j']-result['operating_j']))
                    episode_count += 1
                run = FullDomainRollout(cfg, tr)
                while run.t < cfg.episode_steps:
                    if run.t % 4:
                        run.step()
                        continue
                    data = augmented_data(run)
                    a = choose(data, run.req, mode, parameter, minimum, model, False)
                    b = choose(data, run.req, mode, parameter, minimum, model, True)
                    with np.load(source/f'{mode}_s{j}_i{i}/root_t{run.t}.npz', allow_pickle=False) as z:
                        for k in data:
                            np.testing.assert_array_equal(data[k], z[k])
                        np.testing.assert_array_equal(z['choices'], [a, b])
                        # Select again independently from stored causal arrays.
                        costs = z['energy'].sum(1)
                        feasible = z['risk'] <= float(z['threshold'])
                        key = np.lexsort((np.arange(len(costs)), costs,
                            np.where(feasible, costs, z['fallback']), ~feasible))
                        eligible = z['masks'].sum(1) >= minimum
                        independent = next(int(k) for k in key if eligible[k])
                        assert independent == b
                    r = next(r for r in roots if int(r['scenario']) == j and int(r['trajectory']) == i
                        and r['mode'] == mode and int(r['t']) == run.t)
                    assert int(r['current']) == mask_id(run.hw.command)
                    assert int(r['prefix']) == mask_id(data['masks'][a])
                    assert int(r['extended']) == mask_id(data['masks'][b])
                    changed = not np.array_equal(data['masks'][a], data['masks'][b])
                    assert int(r['changed']) == int(changed)
                    if changed:
                        replay = branch(run, data['masks'][b], mode, parameter, minimum, model)
                        stored = next(r for r in branches if int(r['scenario']) == j and
                            int(r['trajectory']) == i and r['mode'] == mode and int(r['t']) == run.t)
                        reference = next(r for r in raw if int(r['scenario']) == j and int(r['trajectory']) == i
                            and r['mode'] == mode and int(r['extended']) == 0)
                        for k, v in replay.items():
                            if k != 'method':
                                error = max(error, abs(v-float(stored[k])))
                        error = max(error, abs(replay['energy_j']-float(reference['energy_j'])-float(stored['energy_difference_j'])),
                            abs(100*(replay['outage']-float(reference['outage']))-float(stored['outage_difference_pp'])))
                        branch_count += 1
                    root_count += 1
                    run.step(data['masks'][a])
            print('hold action replay', j, i, flush=True)
    assert error < 1e-9 and root_count == 2520 and branch_count == len(branches)
    summary, paired, mechanism = [], [], []
    draw = np.random.default_rng(1220900001).integers(0, 8, (10000, 8))
    for j in range(3):
        for mode in MODES:
            values = []
            for extended in (0, 1):
                data = sorted([r for r in raw if int(r['scenario']) == j and r['mode'] == mode
                    and int(r['extended']) == extended], key=lambda r: int(r['trajectory']))
                assert len(data) == 8
                values.append(data)
                summary.append(dict(scenario=j, mode=mode, extended=extended, trajectories=8,
                    **{k: float(np.mean([float(r[k]) for r in data])) for k in (
                        'energy_j', 'outage', 'switching_j', 'operating_j', 'pilot_j',
                        'transitions', 'short_sleep_reactivations', 'mean_count')}))
            e = np.array([float(b['energy_j'])-float(a['energy_j']) for a, b in zip(*values)])
            o = np.array([100*(float(b['outage'])-float(a['outage'])) for a, b in zip(*values)])
            ci, oi = np.quantile(e[draw].mean(1), [.025, .975]), np.quantile(o[draw].mean(1), [.025, .975])
            paired.append(dict(scenario=j, mode=mode, energy_difference_j=float(e.mean()),
                saving_pct=float(-100*e.mean()/np.mean([float(r['energy_j']) for r in values[0]])),
                outage_difference_pp=float(o.mean()), energy_ci95_low=float(ci[0]), energy_ci95_high=float(ci[1]),
                outage_ci95_low=float(oi[0]), outage_ci95_high=float(oi[1]),
                joint_better_trajectories=int(np.sum((e < -1e-9) & (o <= 1e-10)))))
            data = [r for r in roots if int(r['scenario']) == j and r['mode'] == mode]
            br = [r for r in branches if int(r['scenario']) == j and r['mode'] == mode]
            mechanism.append(dict(scenario=j, mode=mode, roots=len(data),
                absent_roots=sum(int(r['absent_from_prefix']) for r in data),
                absent_eligible_feasible=sum(int(r['absent_from_prefix'])*int(r['hold_eligible'])*int(r['hold_feasible']) for r in data),
                changed_choices=sum(int(r['changed']) for r in data),
                simultaneous_replacement_roots=sum(int(r['replacement_pairs']) > 0 for r in data),
                replacement_pairs=sum(int(r['replacement_pairs']) for r in data),
                branched_joint_better=sum(float(r['energy_difference_j']) < -1e-9 and float(r['outage_difference_pp']) <= 1e-10 for r in br),
                branch_mean_energy_difference_j=float(np.mean([float(r['energy_difference_j']) for r in br])) if br else 0.,
                branch_mean_outage_difference_pp=float(np.mean([float(r['outage_difference_pp']) for r in br])) if br else 0.))
    nominal = next(r for r in summary if r['scenario'] == 1 and r['mode'] == 'calibrated' and r['extended'] == 1)
    difference = next(r for r in paired if r['scenario'] == 1 and r['mode'] == 'calibrated')
    passed = (nominal['outage'] <= .05 and difference['outage_difference_pp'] <= 1e-10
              and difference['energy_difference_j'] < -1e-9)
    gate = dict(nominal_mechanism_screen_pass=passed, permit_new_method_design=passed,
        rl_superiority_established=False, independent_confirmation=False, scientific_completion=False)
    _write_rows(folder/'summary.csv', summary)
    _write_rows(folder/'paired.csv', paired)
    _write_rows(folder/'mechanism.csv', mechanism)
    (folder/'gate.json').write_text(json.dumps(gate, indent=2))
    (folder/'verification.json').write_text(json.dumps(dict(complete_policy_replays=episode_count,
        complete_branch_replays=branch_count, reproduced_causal_roots=root_count,
        external_trajectories=24, maximum_error=error, no_training=True,
        viewed_data=True, independent_confirmation=False), indent=2))
    finish(folder, hashes)
    print(json.dumps(dict(paired=paired, mechanism=mechanism, gate=gate), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
