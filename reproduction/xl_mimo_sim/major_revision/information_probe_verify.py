"""Independent observation assimilation/replay and complete probe result accounting."""
import csv
import json
import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe_study import (ROOT, STREAM, TIMES, MARGINS, MINIMUM,
                                                   initialize, finish, verify)
from major_revision.pilot import SCENARIOS, trajectory


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return [{k: v if k in ('variant', 'method') else float(v) for k, v in row.items()}
                for row in csv.DictReader(file)]


def replay(root, reference, q, withheld, margin, minimum):
    """Use original rollout and manually assimilate each observation before stepping."""
    run = root.clone()
    t0 = run.t
    first = reference.copy()
    if q >= 0:
        first[q] = True
    first_energy = 0.
    while run.t < run.cfg.episode_steps:
        if not run.prepared:
            paid = run.hw.active.copy()
            used = paid.copy()
            if withheld and q >= 0 and t0+run.cfg.wake_delay_slots <= run.t < t0+run.cfg.wake_delay_slots+4:
                used[q] = False
            run.observed = paid
            run.age = run.age+1
            run.age[used] = 0.
            run.tracker.update(run.tr.gain_db[run.t], used)
            for module in np.flatnonzero(used):
                sl = slice(module*run.cfg.antennas_per_subarray, (module+1)*run.cfg.antennas_per_subarray)
                run.tracker.channel_estimate[:, sl] = run.tr.channel[run.t][:, sl]
            run.req = run.tr.requirements_mbps[run.t]
            run.prepared = True
        command = None
        if run.t % 4 == 0:
            command = first if run.t == t0 else baseline_command(run, 'cost', margin, minimum)
        before = run.energy
        run.step(command)
        if run.t <= t0+4:
            first_energy += run.energy-before
    result = run.metrics()
    result.pop('method')
    result.update(first_block_energy_j=first_energy,
                  first_block_outage=float(np.mean(run.outages[t0:t0+4])),
                  suffix_energy_j=run.energy-root.energy)
    return result


def main():
    raw, extras = [], []
    for shard in (0, 1):
        folder = ROOT/f'shard{shard}'
        verify(folder)
        raw.extend(read_rows(folder/'raw.csv'))
        extras.extend(p for p in folder.rglob('*') if p.is_file())
    cfg = make_config('validation')
    indexed = {(int(r['scenario']), int(r['trajectory']), int(r['t']),
                int(r['q']), r['variant']): r for r in raw}
    assert len(indexed) == len(raw)
    errors = dict(replay=0., decomposition=0., delta=0., baseline=0.)
    replay_count = roots = 0
    policies, opportunities = [], []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(4):
            run = FullDomainRollout(cfg, trajectory(cfg, STREAM+j*10000+i, scenario))
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                reference = baseline_command(run, 'cost', MARGINS[j], MINIMUM[j])
                if run.t in TIMES:
                    roots += 1
                    _, score = run.base_observation()
                    eligible = [q for q in range(cfg.subarrays) if not run.hw.active[q]
                                and run.hw.countdown[q] == 0 and not reference[q]]
                    choices = dict(score=min(eligible, key=lambda q: (-score[q], q)) if eligible else -1,
                        age=min(eligible, key=lambda q: (-run.age[q], -score[q], q)) if eligible else -1)
                    path = ROOT/f'shard{i//2}/root_s{j}_i{i}_t{run.t}.npz'
                    with np.load(path, allow_pickle=False) as z:
                        for key, value in dict(reference=reference, eligible=eligible, score=score,
                            age=run.age, active=run.hw.active, countdown=run.hw.countdown,
                            command=run.hw.command, requirements=run.req,
                            channel_estimate=run.tracker.channel_estimate, gain_db=run.tracker.gain_db,
                            velocity=run.tracker.velocity, uncertainty=run.tracker.uncertainty,
                            probability=run.tracker.probability).items():
                            np.testing.assert_array_equal(z[key], value)
                        for name, q in choices.items():
                            assert int(z[name+'_choice']) == q
                    base = indexed[j, i, run.t, -1, 'normal']
                    group = []
                    for q in [-1, *eligible]:
                        for variant in (('normal',) if q == -1 else ('normal', 'withheld')):
                            row = indexed[j, i, run.t, q, variant]
                            actual = replay(run, reference, q, variant == 'withheld', MARGINS[j], MINIMUM[j])
                            for key, value in actual.items():
                                errors['replay'] = max(errors['replay'], abs(row[key]-value))
                            errors['decomposition'] = max(errors['decomposition'],
                                abs(row['energy_j']-row['switching_j']-row['pilot_j']-row['operating_j']))
                            for metric, delta in (('energy_j', 'energy_difference_j'), ('outage', 'outage_difference')):
                                errors['delta'] = max(errors['delta'], abs(row[delta]-(row[metric]-base[metric])))
                            if variant == 'normal' and q >= 0:
                                blind = indexed[j, i, run.t, q, 'withheld']
                                info_e = row['energy_j']-blind['energy_j']
                                info_o = row['outage']-blind['outage']
                                group.append(dict(scenario=j, trajectory=i, t=run.t, q=q,
                                    energy_difference_j=row['energy_difference_j'],
                                    outage_difference_pp=100*row['outage_difference'],
                                    information_energy_difference_j=info_e,
                                    information_outage_difference_pp=100*info_o,
                                    beneficial=row['energy_difference_j'] < -1e-9 and row['outage_difference'] <= 1e-12,
                                    info_beneficial=info_e < -1e-9 and info_o <= 1e-12))
                            replay_count += 1
                    opportunities.extend(group)
                    for name, q in choices.items():
                        row = indexed[j, i, run.t, q, 'normal']
                        blind = indexed[j, i, run.t, q, 'withheld'] if q >= 0 else base
                        policies.append(dict(scenario=j, trajectory=i, t=run.t, rule=name, q=q,
                            energy_difference_j=row['energy_j']-base['energy_j'],
                            outage_difference_pp=100*(row['outage']-base['outage']),
                            information_energy_difference_j=row['energy_j']-blind['energy_j'],
                            information_outage_difference_pp=100*(row['outage']-blind['outage'])))
                    print('probe verified', j, i, run.t, 'replays', replay_count, flush=True)
                run.step(reference)
            stored = read_rows(ROOT/f'shard{i//2}/baseline_episodes.csv')
            old = next(r for r in stored if r['scenario'] == j and r['trajectory'] == i)
            for key, value in run.metrics().items():
                if key != 'method':
                    errors['baseline'] = max(errors['baseline'], abs(value-old[key]))
    assert roots == 36 and replay_count == len(raw) and max(errors.values()) < 1e-9, errors
    summaries = []
    for j in range(3):
        for name in ('score', 'age'):
            rows = [r for r in policies if r['scenario'] == j and r['rule'] == name]
            assert len(rows) == 12
            values = {key: float(np.mean([r[key] for r in rows])) for key in (
                'energy_difference_j', 'outage_difference_pp', 'information_energy_difference_j',
                'information_outage_difference_pp')}
            summaries.append(dict(scenario=j, rule=name, roots=12, **values))
    useful_trajectories = sorted({r['trajectory'] for r in opportunities
                                 if r['scenario'] == 1 and r['beneficial'] and r['info_beneficial']})
    rule_pass = [r['rule'] for r in summaries if r['scenario'] == 1 and
                 r['energy_difference_j'] < -1e-9 and r['outage_difference_pp'] <= 1e-10]
    gate = dict(nominal_causal_rules_pass=rule_pass,
                nominal_joint_benefit_trajectories=useful_trajectories,
                permit_full_policy_screen=bool(rule_pass and len(useful_trajectories) >= 2),
                permit_rl_training=False, scientific_completion=False)
    folder, hashes = initialize('verified', extras)
    _write_rows(folder/'all_branches.csv', raw)
    _write_rows(folder/'causal_rules.csv', policies)
    _write_rows(folder/'rule_summary.csv', summaries)
    _write_rows(folder/'opportunities.csv', opportunities)
    (folder/'gate.json').write_text(json.dumps(gate, indent=2))
    (folder/'audit.json').write_text(json.dumps(dict(errors=errors, roots=roots,
        external_trajectories=12, all_branch_replays=replay_count, raw_records=len(raw),
        training_updates=0, independent_confirmation=False, scientific_completion=False), indent=2))
    finish(folder, hashes)
    print(json.dumps(dict(summary=summaries, gate=gate, errors=errors), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
