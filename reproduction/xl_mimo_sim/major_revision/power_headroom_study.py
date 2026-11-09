"""Replay 24 viewed traces with an optional fixed-beam power accounting layer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.power_headroom import allocation, beam_gains
from major_revision.risk_correction import select_action
from major_revision.risk_correction_study import causal_data
from major_revision.risk_rl_study import setup, ROOT as PREVIOUS


ROOT = Path('results/major_revision_20260922_power_headroom')


def main():
    cfg = make_config('validation')
    settings, model = setup()
    verify(PREVIOUS/'audit')
    ROOT.mkdir(parents=True, exist_ok=False)
    sources = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
               Path('major_revision/POWER_HEADROOM_PLAN.md')]
    hashes = {str(p): digest(p) for p in sources}
    (ROOT/'manifest.json').write_text(json.dumps(dict(hashes=hashes, stream=1210000000,
        already_viewed=True, trajectories=24, config=cfg.as_dict(), methods=['old', 'calibrated', 'full'],
        learning_updates=0, frozen_schedule_only=True, new_model_authorization_pending=True), indent=2))
    rows, slot_rows, errors, traces = [], [], [], []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, 1210000000+j*10000+i, scenario)
            source_hash = hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            traces.append(dict(scenario=j, trajectory=i, trace_hash=source_hash))
            for mode in ('old', 'calibrated', 'full'):
                run, actions, records = FullDomainRollout(cfg, tr), [], []
                while run.t < cfg.episode_steps:
                    run.prepare()
                    command = run.hw.command.copy()
                    if run.t % 4 == 0:
                        if mode == 'full':
                            command = np.ones(cfg.subarrays, bool)
                        else:
                            data = causal_data(run)
                            action = select_action(data, run.req, mode, *settings[mode, j], model)
                            command = data['masks'][action]
                            actions.append(action)
                    # d>0: no newly commanded branch can serve in this slot.
                    assert cfg.wake_delay_slots > 0
                    available = run.hw.active & command
                    gain = beam_gains(cfg, run.tracker.channel_estimate, available)
                    result = allocation(cfg, gain, run.req)
                    old_rates = run.system.achievable_rates(tr.channel[run.t], available)
                    errors.append(float(np.max(np.abs(old_rates-result['before']))))
                    saving = cfg.slot_s*(cfg.tx_power_w-result['power'].sum())/cfg.pa_efficiency
                    saved_t = run.t
                    run.step(command if run.t % 4 == 0 else None)
                    assert np.isclose(run.outages[-1], np.mean(result['after'] < run.req))
                    record = dict(scenario=j, trajectory=i, method=mode, t=saved_t,
                        original_energy_j=run.slot_energy[-1], saving_j=float(saving),
                        counterfactual_energy_j=run.slot_energy[-1]-float(saving),
                        original_outage=run.outages[-1],
                        counterfactual_outage=float(np.mean(result['after'] < run.req)),
                        tx_power_w=float(result['power'].sum()), fallback=int(result['fallback']),
                        **{f'{key}_user{k}': float(values[k]) for key, values in
                           [('power', result['power']), ('rate_before', result['before']),
                            ('rate_after', result['after']), ('requirement', run.req)] for k in range(cfg.users)})
                    slot_rows.append(record)
                    records.append(record)
                if mode != 'full':
                    with np.load(PREVIOUS/f'audit/trace_{mode}_0_s{j}_i{i}.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(actions, z['actions'])
                        np.testing.assert_array_equal(run.slot_energy, z['energy'])
                        np.testing.assert_array_equal(run.outages, z['outage'])
                energy = run.energy
                saving = sum(r['saving_j'] for r in records)
                assert -1e-9 <= saving <= 21+1e-9
                rows.append(dict(scenario=j, trajectory=i, method=mode,
                    original_energy_j=energy, counterfactual_energy_j=energy-saving,
                    saving_j=saving, saving_pct=100*saving/energy,
                    original_outage=float(np.mean(run.outages)),
                    counterfactual_outage=float(np.mean([r['counterfactual_outage'] for r in records])),
                    numerical_fallbacks=sum(r['fallback'] for r in records)))
            assert source_hash == hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            _write_rows(ROOT/'episodes.csv', rows)
            print('power headroom', j, i, 'episodes', len(rows), flush=True)
    _write_rows(ROOT/'slots.csv', slot_rows)
    _write_rows(ROOT/'trajectories.csv', traces)
    rng = np.random.default_rng(20260922)
    draw = rng.integers(0, 8, (10000, 8))
    summary = []
    for j in range(3):
        for method in ('old', 'calibrated', 'full'):
            group = [r for r in rows if r['scenario'] == j and r['method'] == method]
            saving = np.array([r['saving_j'] for r in group])
            ci = np.quantile(saving[draw].mean(1), [.025, .975])
            means = {k: float(np.mean([r[k] for r in group])) for k in
                     ('original_energy_j', 'counterfactual_energy_j', 'saving_j',
                      'original_outage', 'counterfactual_outage', 'numerical_fallbacks')}
            summary.append(dict(scenario=j, method=method, trajectories=8, **means,
                saving_pct=100*means['saving_j']/means['original_energy_j'],
                saving_ci95_low_j=float(ci[0]), saving_ci95_high_j=float(ci[1])))
    _write_rows(ROOT/'summary.csv', summary)
    assert len(rows) == 72 and len(slot_rows) == 10080 and max(errors) < 1e-6
    assert all(r['original_outage'] == r['counterfactual_outage'] for r in slot_rows)
    (ROOT/'audit.json').write_text(json.dumps(dict(episodes=72, slot_records=10080,
        original_saved_policy_replays=48, maximum_original_rate_error_mbps=max(errors),
        exact_slot_outage_identity=True, absolute_pa_saving_ceiling_j=21,
        numerical_fallbacks=sum(r['fallback'] for r in slot_rows), rl_superiority=False,
        independent_confirmation=False, scientific_completion=False,
        added_computational_energy_measured=False), indent=2))
    finish(ROOT, hashes)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
