"""Independent saved-power PHY and statistics checks, without allocation solving."""
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from major_revision.audit_rollout import AuditRollout
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.power_policy_study import ROOT, PREVIOUS, SEEDS, STREAM, command, controllers, read


def physical_rates(cfg, run, channel, available, powers):
    h = channel[:, np.repeat(available, cfg.antennas_per_subarray)]
    if h.shape[1] < cfg.users:
        return np.zeros(cfg.users)
    inverse = np.linalg.inv(h@h.conj().T + cfg.users*run.system.noise_w/cfg.tx_power_w*np.eye(cfg.users))
    beam = h.conj().T@inverse
    beam /= np.maximum(np.linalg.norm(beam, axis=0, keepdims=True), 1e-15)
    received = np.abs(h@(beam*np.sqrt(powers)[None]))**2
    signal = np.diag(received).copy()
    np.fill_diagonal(received, 0.)
    return cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+signal/(received.sum(1)+run.system.noise_w))


def main():
    torch.set_num_threads(1)
    verify(ROOT)
    verify(PREVIOUS/'verified')
    out = ROOT/'verified'
    out.mkdir(exist_ok=False)
    paths = [Path(__file__), *[ROOT/name for name in
        ('manifest.json', 'output_hashes.json', 'slots.csv', 'episodes.csv', 'summary.csv', 'paired_gaps.csv', 'per_seed.csv')]]
    hashes = {str(p): digest(p) for p in paths}
    (out/'manifest.json').write_text(json.dumps(dict(hashes=hashes, allocator_not_called=True), indent=2))
    rows = read(ROOT/'slots.csv')
    records = read(ROOT/'episodes.csv')
    slots = {(int(r['scenario']), int(r['trajectory']), int(r['controller_seed']), int(r['t'])): r for r in rows}
    episodes = {(int(r['scenario']), int(r['trajectory']), r['method'], int(r['controller_seed'])): r for r in records}
    assert len(rows) == len(slots) == 20160 and len(episodes) == len(records) == 192
    cfg = make_config('validation')
    models = controllers(cfg)
    rate_error = energy_error = 0.
    changed_failures = checked = 0
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            for seed in (*SEEDS, 0):
                run, adjusted, saving = AuditRollout(cfg, tr), 0., 0.
                while run.t < cfg.episode_steps:
                    row = slots[j, i, seed, run.t]
                    run.prepare()
                    mask = command(run, models.get(seed)) if run.t % 4 == 0 else run.hw.command.copy()
                    active = run.hw.active & mask
                    assert int(row['command_bits']) == sum(int(flag)<<k for k, flag in enumerate(mask))
                    assert int(row['active_bits']) == sum(int(flag)<<k for k, flag in enumerate(active))
                    p = np.array([float(row[f'power_user{k}']) for k in range(cfg.users)])
                    assert np.isfinite(p).all() and np.all(p >= 0) and p.sum() <= cfg.tx_power_w+1e-10
                    before = physical_rates(cfg, run, tr.channel[run.t], active, np.full(cfg.users, cfg.tx_power_w/cfg.users))
                    after = physical_rates(cfg, run, tr.channel[run.t], active, p)
                    for label, values in [('rate_before', before), ('rate_after', after), ('requirement', run.req)]:
                        recorded = np.array([float(row[f'{label}_user{k}']) for k in range(cfg.users)])
                        rate_error = max(rate_error, float(np.max(abs(recorded-values))))
                    changed_failures += int(np.sum((before < run.req) != (after < run.req)))
                    assert np.all(after >= np.minimum(before, run.req)-1e-8)
                    assert float(np.mean(after < run.req)) == float(row['counterfactual_outage'])
                    delta = cfg.slot_s*(cfg.tx_power_w-p.sum())/cfg.pa_efficiency
                    run.step(mask if run.t % 4 == 0 else None)
                    energy_error = max(energy_error, abs(delta-float(row['saving_j'])),
                        abs(run.slot_energy[-1]-float(row['original_energy_j'])),
                        abs(run.slot_energy[-1]-delta-float(row['counterfactual_energy_j'])))
                    adjusted += run.slot_energy[-1]-delta
                    saving += delta
                    checked += 1
                method = 'RA-DQN' if seed else 'Manuscript PRS'
                episode = episodes[j, i, method, seed]
                for key, value in [('original_energy_j', run.energy), ('counterfactual_energy_j', adjusted),
                                   ('saving_j', saving), ('original_outage', np.mean(run.outages))]:
                    energy_error = max(energy_error, abs(value-float(episode[key])))
            print('independent shared-power verification', j, i, flush=True)
    # All summary and per-seed means independently recomputed from episode rows.
    summary_error = 0.
    fields = ('original_energy_j', 'counterfactual_energy_j', 'saving_j', 'original_outage')
    for name in ('summary.csv', 'per_seed.csv'):
        for row in read(ROOT/name):
            group = [r for r in records if r['scenario']==row['scenario'] and r['method']==row['method']
                     and ('controller_seed' not in row or row['controller_seed']==r['controller_seed'])]
            for key in fields:
                summary_error = max(summary_error, abs(np.mean([float(r[key]) for r in group])-float(row[key])))
    rng = np.random.default_rng(20260923)
    sd, td = rng.integers(0, 5, (10000, 5)), rng.integers(0, 8, (10000, 8))
    pair_error = 0.
    for row in read(ROOT/'paired_gaps.csv'):
        j, baseline, metric = int(row['scenario']), row['baseline'], row['metric']
        values = np.zeros((5, 8))
        for a, seed in enumerate(SEEDS):
            for i in range(8):
                rl, base = episodes[j,i,'RA-DQN',seed], episodes[j,i,baseline,0]
                before = float(rl['original_energy_j'])-float(base['original_energy_j'])
                after = float(rl['counterfactual_energy_j'])-float(base['counterfactual_energy_j'])
                values[a,i] = {'original_energy_gap_j': before, 'adjusted_energy_gap_j': after,
                    'gap_change_j': after-before,
                    'outage_gap_pp': 100*(float(rl['original_outage'])-float(base['original_outage']))}[metric]
        means = values[sd[:,:,None], td[:,None,:]].mean((1,2))
        expected = (values.mean(), *np.quantile(means, [.025,.975]))
        pair_error = max(pair_error, *[abs(float(row[k])-v) for k,v in
            zip(('mean','ci95_low','ci95_high'), expected)])
    assert rate_error < 1e-6 and energy_error < 1e-9 and summary_error < 1e-9 and pair_error < 1e-9
    assert checked == 20160 and changed_failures == 0
    result = dict(replays=144, slot_checks=checked, user_slot_failures_checked=3*checked,
        changed_failures=changed_failures, maximum_rate_error_mbps=rate_error,
        maximum_energy_error_j=energy_error, maximum_summary_error=summary_error,
        maximum_paired_statistic_error=pair_error, allocator_called=False,
        independent_new_test_set=False, new_joint_policy=False, scientific_completion=False)
    (out/'verification.json').write_text(json.dumps(result, indent=2))
    finish(out, hashes)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
