"""Independent explicit-beam PHY replay of saved power allocations (no solver)."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.power_headroom_study import ROOT
from major_revision.risk_correction import select_action
from major_revision.risk_correction_study import causal_data
from major_revision.risk_rl_study import setup


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    verify(ROOT)
    folder = ROOT/'verified'
    folder.mkdir(exist_ok=False)
    paths = [Path(__file__), ROOT/'slots.csv', ROOT/'episodes.csv', ROOT/'summary.csv',
             ROOT/'manifest.json', ROOT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in paths}
    (folder/'manifest.json').write_text(json.dumps(dict(hashes=hashes, no_allocation_solver=True), indent=2))
    slots, episodes, summaries = (read(ROOT/f) for f in ('slots.csv', 'episodes.csv', 'summary.csv'))
    slot_map = {(int(r['scenario']), int(r['trajectory']), r['method'], int(r['t'])): r for r in slots}
    episode_map = {(int(r['scenario']), int(r['trajectory']), r['method']): r for r in episodes}
    assert len(slot_map) == len(slots) == 10080 and len(episode_map) == len(episodes) == 72
    settings, model = setup()
    cfg = make_config('validation')
    maximum_rate_error, maximum_energy_error, changed_failures, checked = 0., 0., 0, 0
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, 1210000000+j*10000+i, scenario)
            for mode in ('old', 'calibrated', 'full'):
                run, adjusted, saving = FullDomainRollout(cfg, tr), 0., 0.
                while run.t < cfg.episode_steps:
                    run.prepare()
                    command = run.hw.command.copy()
                    if run.t % 4 == 0:
                        if mode == 'full':
                            command = np.ones(cfg.subarrays, bool)
                        else:
                            data = causal_data(run)
                            a = select_action(data, run.req, mode, *settings[mode, j], model)
                            command = data['masks'][a]
                    row = slot_map[j, i, mode, run.t]
                    p = np.array([float(row[f'power_user{k}']) for k in range(cfg.users)])
                    active = run.hw.active & command
                    h = tr.channel[run.t][:, np.repeat(active, cfg.antennas_per_subarray)]
                    if h.shape[1] < cfg.users:
                        new_rates = np.zeros(cfg.users)
                    else:
                        inverse = np.linalg.inv(h@h.conj().T + cfg.users*run.system.noise_w/cfg.tx_power_w*np.eye(cfg.users))
                        w = h.conj().T@inverse
                        w /= np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-15)
                        # Explicit transmit amplitudes, independently of the saved gain-matrix routine.
                        received = np.abs(h@(w*np.sqrt(p)[None]))**2
                        useful = np.diag(received).copy()
                        np.fill_diagonal(received, 0.)
                        sinr = useful/(received.sum(1)+run.system.noise_w)
                        new_rates = cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+sinr)
                    old_rates = run.system.achievable_rates(tr.channel[run.t], active)
                    stored = np.array([float(row[f'rate_after_user{k}']) for k in range(cfg.users)])
                    maximum_rate_error = max(maximum_rate_error, float(np.max(abs(new_rates-stored))))
                    changed_failures += int(np.sum((old_rates < run.req) != (new_rates < run.req)))
                    assert np.all(p >= 0) and p.sum() <= cfg.tx_power_w+1e-10
                    assert np.all(new_rates >= np.minimum(old_rates, run.req)-1e-8)
                    delta = cfg.slot_s*(cfg.tx_power_w-p.sum())/cfg.pa_efficiency
                    run.step(command if run.t % 4 == 0 else None)
                    adjusted += run.slot_energy[-1]-delta
                    saving += delta
                    maximum_energy_error = max(maximum_energy_error,
                        abs(delta-float(row['saving_j'])), abs(run.slot_energy[-1]-float(row['original_energy_j'])))
                    checked += 1
                expected = episode_map[j, i, mode]
                maximum_energy_error = max(maximum_energy_error,
                    abs(adjusted-float(expected['counterfactual_energy_j'])),
                    abs(saving-float(expected['saving_j'])))
            print('power replay verified', j, i, flush=True)
    summary_error = 0.
    for row in summaries:
        group = [r for r in episodes if r['scenario'] == row['scenario'] and r['method'] == row['method']]
        for field in ('original_energy_j', 'counterfactual_energy_j', 'saving_j',
                      'original_outage', 'counterfactual_outage'):
            summary_error = max(summary_error, abs(np.mean([float(r[field]) for r in group])-float(row[field])))
    assert maximum_rate_error < 1e-6 and maximum_energy_error < 1e-9
    assert summary_error < 1e-9 and changed_failures == 0 and checked == 10080
    result = dict(slot_physics_checks=checked, full_replays=72,
        maximum_rate_error_mbps=maximum_rate_error, maximum_energy_error_j=maximum_energy_error,
        summary_error=summary_error, changed_user_slot_failure_indicators=changed_failures,
        no_power_optimization_or_training_in_verifier=True, independent_confirmation=False,
        learning_superiority=False, scientific_completion=False)
    (folder/'verification.json').write_text(json.dumps(result, indent=2))
    finish(folder, hashes)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
