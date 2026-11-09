"""Equal-ability power accounting for all five original frozen RL policies."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows, simulate_episode
from future_vr.learned_study import _load_controller
from future_vr.rate_search import rate_search_command
from major_revision.audit_rollout import AuditRollout
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, state, trajectory
from major_revision.power_headroom import allocation, beam_gains

ROOT = Path('results/major_revision_20260922_power_policy')
PREVIOUS = Path('results/major_revision_20260922_power_headroom')
CHECKPOINTS = Path('results/revision_20260905/checkpoints')
SEEDS = (11, 22, 33, 44, 55)
STREAM = 1210000000


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def controllers(cfg):
    return {seed: _load_controller(cfg, 'RA-FVV', seed, CHECKPOINTS/f'RA-FVV_{seed}.pt')
            for seed in SEEDS}


def command(run, controller):
    run.prepare()
    observation, score = state(run.cfg, run.tracker, run.hw, run.req,
        run.previous_req, run.age, run.last_outage, False)
    if controller is None:
        return rate_search_command(run.cfg, run.tracker, run.hw, run.req, score)
    _, count = controller.choose(observation, 0.)
    return select_top(score, count)


def summarize(rows):
    rng = np.random.default_rng(20260923)
    seed_draw = rng.integers(0, 5, (10000, 5))
    trace_draw = rng.integers(0, 8, (10000, 8))
    metrics = ('original_energy_j', 'counterfactual_energy_j', 'saving_j', 'original_outage')
    summaries, comparisons, per_seed = [], [], []
    for j in range(3):
        matrices = {}
        for method in ('RA-DQN', 'Manuscript PRS', 'Calibrated search', 'Full array'):
            seeds = SEEDS if method == 'RA-DQN' else (0,)
            group = {(int(r['controller_seed']), int(r['trajectory'])): r for r in rows
                     if int(r['scenario']) == j and r['method'] == method}
            assert len(group) == len(seeds)*8
            data = {key: np.array([[float(group[seed, i][key]) for i in range(8)]
                                  for seed in seeds]) for key in metrics}
            matrices[method] = data
            summary = dict(scenario=j, method=method, trajectories=8, controller_seeds=len(seeds),
                           **{key: float(value.mean()) for key, value in data.items()})
            summary['saving_pct'] = 100*summary['saving_j']/summary['original_energy_j']
            summaries.append(summary)
            for k, seed in enumerate(seeds):
                per_seed.append(dict(scenario=j, method=method, controller_seed=seed,
                    **{key: float(value[k].mean()) for key, value in data.items()}))
        rl = matrices['RA-DQN']
        for method in ('Manuscript PRS', 'Calibrated search'):
            baseline = matrices[method]
            for label, values in [
                ('original_energy_gap_j', rl['original_energy_j']-baseline['original_energy_j']),
                ('adjusted_energy_gap_j', rl['counterfactual_energy_j']-baseline['counterfactual_energy_j']),
                ('gap_change_j', baseline['saving_j']-rl['saving_j']),
                ('outage_gap_pp', 100*(rl['original_outage']-baseline['original_outage']))]:
                boot = values[seed_draw[:, :, None], trace_draw[:, None, :]].mean((1, 2))
                lo, hi = np.quantile(boot, [.025, .975])
                comparisons.append(dict(scenario=j, baseline=method, metric=label,
                    mean=float(values.mean()), ci95_low=float(lo), ci95_high=float(hi)))
    return summaries, comparisons, per_seed


def main():
    torch.set_num_threads(1)
    cfg = make_config('validation')
    verify(PREVIOUS)
    verify(PREVIOUS/'verified')
    ROOT.mkdir(parents=True, exist_ok=False)
    paths = [*Path('future_vr').glob('*.py'), *Path('major_revision').glob('*.py'),
        Path('major_revision/POWER_POLICY_PLAN.md'), *[CHECKPOINTS/f'RA-FVV_{s}.pt' for s in SEEDS],
        PREVIOUS/'episodes.csv', PREVIOUS/'trajectories.csv', PREVIOUS/'verified/verification.json']
    hashes = {str(p): digest(p) for p in paths}
    (ROOT/'manifest.json').write_text(json.dumps(dict(hashes=hashes, config=cfg.as_dict(),
        stream=STREAM, already_viewed=True, trajectories=24, learning_seeds=SEEDS,
        training_updates=0, new_policy_replays=144, source_policy='original manuscript72-input',
        power_integrated_into_decisions=False, scope_approval_pending=True), indent=2))
    models = controllers(cfg)
    old_traces = {(int(r['scenario']), int(r['trajectory'])): r['trace_hash']
                  for r in read(PREVIOUS/'trajectories.csv')}
    episodes, slots = [], []
    reference_error = 0.
    original_rate_error = 0.
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            trace_seed = STREAM+j*10000+i
            tr = trajectory(cfg, trace_seed, scenario)
            trace_hash = hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            assert trace_hash == old_traces[j, i]
            for seed in (*SEEDS, 0):
                model = models.get(seed)
                method = 'RA-DQN' if seed else 'Manuscript PRS'
                run = AuditRollout(cfg, tr)
                local_rows, masks = [], []
                while run.t < cfg.episode_steps:
                    run.prepare()
                    mask = command(run, model) if run.t % 4 == 0 else run.hw.command.copy()
                    active = run.hw.active & mask
                    result = allocation(cfg, beam_gains(cfg, run.tracker.channel_estimate, active), run.req)
                    physical_before = run.system.achievable_rates(tr.channel[run.t], active)
                    original_rate_error = max(original_rate_error,
                        float(np.max(abs(physical_before-result['before']))))
                    delta = cfg.slot_s*(cfg.tx_power_w-result['power'].sum())/cfg.pa_efficiency
                    row = dict(scenario=j, trajectory=i, controller_seed=seed, method=method, t=run.t,
                        command_bits=sum(int(flag)<<k for k, flag in enumerate(mask)),
                        active_bits=sum(int(flag)<<k for k, flag in enumerate(active)),
                        saving_j=float(delta), tx_power_w=float(result['power'].sum()),
                        fallback=int(result['fallback']),
                        **{f'{key}_user{k}': float(values[k]) for key, values in
                            [('power', result['power']), ('rate_before', result['before']),
                             ('rate_after', result['after']), ('requirement', run.req)] for k in range(cfg.users)})
                    run.step(mask if run.t % 4 == 0 else None)
                    row.update(original_energy_j=run.slot_energy[-1],
                        counterfactual_energy_j=run.slot_energy[-1]-float(delta),
                        original_outage=run.outages[-1],
                        counterfactual_outage=float(np.mean(result['after'] < run.req)))
                    assert row['original_outage'] == row['counterfactual_outage']
                    local_rows.append(row)
                    masks.append(mask.copy())
                # Original unmodified evaluation is the schedule/accounting reference.
                _, reference = simulate_episode(cfg, tr, 'RA-FVV' if seed else 'Predictive rate search',
                    model, 1., trace_seed, control_period=4)
                np.testing.assert_array_equal(run.outages, reference['outage'])
                np.testing.assert_array_equal(np.sum(masks, axis=1), reference['action_count'])
                reference_error = max(reference_error, float(np.max(abs(np.asarray(run.slot_energy)-reference['energy_j']))))
                saving = sum(r['saving_j'] for r in local_rows)
                assert 0 <= saving <= 21+1e-9
                episodes.append(dict(scenario=j, trajectory=i, method=method, controller_seed=seed,
                    original_energy_j=run.energy, counterfactual_energy_j=run.energy-saving,
                    saving_j=saving, original_outage=float(np.mean(run.outages)),
                    counterfactual_outage=float(np.mean([r['counterfactual_outage'] for r in local_rows])),
                    numerical_fallbacks=sum(r['fallback'] for r in local_rows), provenance='new fixed-policy diagnostic'))
                slots.extend(local_rows)
            assert trace_hash == hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            _write_rows(ROOT/'episodes.csv', episodes)
            print('shared-power original-policy audit', j, i, 'episodes', len(episodes), flush=True)
    assert len(episodes) == 144 and len(slots) == 20160
    assert reference_error == 0 and original_rate_error < 1e-6
    for row in read(PREVIOUS/'episodes.csv'):
        if row['method'] not in ('calibrated', 'full'):
            continue
        episodes.append(dict(scenario=int(row['scenario']), trajectory=int(row['trajectory']),
            method='Calibrated search' if row['method']=='calibrated' else 'Full array', controller_seed=0,
            **{key: float(row[key]) for key in ('original_energy_j', 'counterfactual_energy_j',
                'saving_j', 'original_outage', 'counterfactual_outage', 'numerical_fallbacks')},
            provenance='archived verified fixed-policy diagnostic'))
    assert len(episodes) == 192
    _write_rows(ROOT/'episodes.csv', episodes)
    _write_rows(ROOT/'slots.csv', slots)
    for name, rows in zip(('summary.csv', 'paired_gaps.csv', 'per_seed.csv'), summarize(episodes)):
        _write_rows(ROOT/name, rows)
    (ROOT/'audit.json').write_text(json.dumps(dict(new_policy_replays=144, archived_episodes=48,
        slot_records=20160, reference_energy_error_j=reference_error,
        original_rate_error_mbps=original_rate_error,
        fallback_slots=sum(r['fallback'] for r in slots), scientific_completion=False,
        independent_confirmation=False, new_learning=False), indent=2))
    finish(ROOT, hashes)


if __name__ == '__main__':
    main()
