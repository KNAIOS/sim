"""Post-hoc recovery/forecast attribution; oracle data never choose actions."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from future_vr.system import XLSystem
from major_revision.action_value import observation, hold_predictions, baseline_command
from major_revision.action_value_report import numeric_rows
from major_revision.action_value_study import ROOT as OLD_ROOT
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import subarray_grams, rates_from_gram
from major_revision.horizon_prior import FixedHorizon
from major_revision.horizon_prior_screen import ROOT as H8_ROOT
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260921_recovery_cause')


def forecast_grams_now(cfg, tracker):
    grams = subarray_grams(cfg, tracker.channel_estimate)
    measured = np.diagonal(grams, axis1=-2, axis2=-1).real / cfg.antennas_per_subarray
    future, _, _ = tracker.predict(cfg.horizon)
    gain = np.concatenate((tracker.gain_db[None], future[:-1]))
    scale = np.sqrt(10.**(gain / 10.) / np.maximum(measured[None], 1e-20))
    return grams[None] * scale[..., :, None] * scale[..., None, :]


def available_masks(cfg, hardware, masks, horizon):
    """Independent hardware replay; the original object is never modified."""
    available = []
    for mask in masks:
        trial = copy.deepcopy(hardware)
        sample = []
        for _ in range(horizon):
            active, _, _ = trial.apply(mask, 1.)
            sample.append(active)
            trial.tick()
        available.append(sample)
    return np.asarray(available)


def evaluate_grams(cfg, grams, available):
    total = np.einsum('ahq,hqkl->ahkl', available, grams)
    return rates_from_gram(cfg, total, available.sum(-1) * cfg.antennas_per_subarray)


def full_upper_bound(cfg, grams):
    norms = np.diagonal(grams.sum(axis=1), axis1=-2, axis2=-1).real
    snr = cfg.tx_power_w / cfg.users * norms / XLSystem(cfg).noise_w
    return cfg.bandwidth_mhz * (1. - cfg.pilot_fraction) * np.log2(1. + snr)


def diagnose(run, action, score):
    """Action is fixed by the caller BEFORE this function reads true future CSI."""
    cfg, start = run.cfg, run.t
    prediction = hold_predictions(cfg, run.tracker, run.hw, score)
    masks = prediction['masks']
    available = available_masks(cfg, run.hw, masks, cfg.horizon)
    grams = forecast_grams_now(cfg, run.tracker)
    predicted = evaluate_grams(cfg, grams, available)
    np.testing.assert_allclose(predicted, prediction['rates'], rtol=1e-9, atol=1e-7)
    immediate = np.broadcast_to(masks[:, None, :], available.shape)
    instant_rates = evaluate_grams(cfg, grams, immediate)
    bound = full_upper_bound(cfg, grams)
    assert np.all(instant_rates <= bound[None] + 1e-6)
    outage = (predicted < run.req[None, None]).mean((1, 2))
    instant_outage = (instant_rates < run.req[None, None]).mean((1, 2))
    bound_outage = float((bound < run.req[None]).mean())
    any_feasible = bool(np.any(outage <= cfg.outage_target))
    instant_feasible = bool(np.any(instant_outage <= cfg.outage_target))
    category = ('predicted_feasible' if any_feasible else
        'recovery_limited_prediction' if instant_feasible else
        'predicted_upper_bound_infeasible' if bound_outage > cfg.outage_target else
        'unresolved_prediction')
    # Only the executed four-slot block; no hypothetical future control changes.
    actual_grams = np.stack([subarray_grams(cfg, channel) for channel in run.tr.channel[start:start+4]])
    assert len(actual_grams) == 4
    real_rates = evaluate_grams(cfg, actual_grams, available[:, :4])[action]
    full_available = np.ones((1, 4, cfg.subarrays), dtype=bool)
    full_rates = evaluate_grams(cfg, actual_grams, full_available)[0]
    real_bound = full_upper_bound(cfg, actual_grams)
    assert np.all(full_rates <= real_bound + 1e-6)
    req = run.tr.requirements_mbps[start:start+4]
    actual_fail = real_rates < req
    predicted_fail = predicted[action, :4] < run.req[None]
    fixed_req_fail = real_rates < run.req[None]
    false_safe = ~predicted_fail & actual_fail
    full_rescue = actual_fail & (full_rates >= req)
    unavoidable = actual_fail & (real_bound < req)
    assert not np.any(full_rescue & unavoidable)
    row = dict(step=start, category=category, action_count=action+1,
        active_count=int(run.hw.active.sum()), recovering_count=int((run.hw.countdown>0).sum()),
        mean_observation_age=float(run.age.mean()),
        min_predicted_outage=float(outage.min()), min_instant_predicted_outage=float(instant_outage.min()),
        upper_bound_predicted_outage=bound_outage, selected_predicted_outage=float(outage[action]),
        user_slots=12, actual_failures=int(actual_fail.sum()), predicted_failures=int(predicted_fail.sum()),
        false_safe=int(false_safe.sum()), false_unsafe=int((predicted_fail & ~actual_fail).sum()),
        channel_only_false_safe=int((~predicted_fail & fixed_req_fail).sum()),
        demand_added_failures=int((~fixed_req_fail & actual_fail).sum()),
        demand_removed_failures=int((fixed_req_fail & ~actual_fail).sum()),
        full_ready_rescued_failures=int(full_rescue.sum()), upper_bound_unavoidable_failures=int(unavoidable.sum()),
        unresolved_actual_failures=int((actual_fail & ~full_rescue & ~unavoidable).sum()))
    return row, float(actual_fail.mean())


def main():
    assert (H8_ROOT/'objective_audit/COMPLETE').exists()
    screen_manifest = json.loads((H8_ROOT/'screen/manifest.json').read_text())
    for key in ('source_hashes', 'input_hashes'):
        for path, digest in screen_manifest[key].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    records = {'H8': numeric_rows(H8_ROOT/'screen/raw.csv'),
               'cost': numeric_rows(OLD_ROOT/'baselines/development.csv')}
    settings = numeric_rows(OLD_ROOT/'baselines/selected.csv')
    cfg = make_config('validation')
    files = [Path(__file__), Path(__file__).with_name('RECOVERY_CAUSE_PLAN.md'),
             Path(__file__).with_name('test_recovery_cause.py'), H8_ROOT/'screen/manifest.json',
             H8_ROOT/'screen/raw.csv', OLD_ROOT/'baselines/development.csv', OLD_ROOT/'baselines/selected.csv']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    ROOT.mkdir(exist_ok=False)
    (ROOT/'manifest.json').write_text(json.dumps(dict(config=cfg.as_dict(),
        hashes=hashes, role='post-hoc mechanism diagnostic on viewed development',
        stream=998000000, trajectories_per_scenario=4, policies=['H8','cost'], threads=1), indent=2))
    states, episodes = [], []
    max_error = 0.
    model = FixedHorizon(cfg)
    for j, sc in enumerate(SCENARIOS):
        setting = next(r for r in settings if r['method']=='cost' and r['scenario']==j)
        for i in range(4):
            tr = trajectory(cfg, 998000000+j*10000+i, sc)
            for policy in ('H8', 'cost'):
                run = FullDomainRollout(cfg, tr)
                while run.t < cfg.episode_steps:
                    state, score = observation(run)
                    if policy == 'H8':
                        action, count = model.choose(state)
                        command = select_top(score, count)
                    else:
                        command = baseline_command(run, 'cost', setting['margin'], setting['minimum'])
                        action = int(command.sum())-1
                        np.testing.assert_array_equal(command, select_top(score, action+1))
                    # No true future information was read to choose this command.
                    row, expected_outage = diagnose(run, action, score)
                    states.append(dict(policy=policy, scenario=j, trajectory=i, **row))
                    run.step(command)
                    for _ in range(3):
                        run.step()
                    assert abs(np.mean(run.outages[-4:])-expected_outage) < 1e-12
                result = run.metrics()
                prior = next(r for r in records[policy] if r['scenario']==j and r['trajectory']==i
                    and (r.get('split')=='development' if policy=='H8' else r['method']=='cost'))
                for key in ('energy_j','outage','switching_j','pilot_j','operating_j'):
                    error = abs(result[key]-prior[key])
                    assert error < 1e-9
                    max_error = max(max_error, error)
                episodes.append(dict(policy=policy, scenario=j, trajectory=i, **result))
        print('recovery cause', j, 'complete', flush=True)
    categories = ('predicted_feasible','recovery_limited_prediction','predicted_upper_bound_infeasible','unresolved_prediction')
    summary, per_trajectory = [], []
    measures = ('user_slots','actual_failures','predicted_failures','false_safe','false_unsafe',
        'channel_only_false_safe','demand_added_failures','demand_removed_failures',
        'full_ready_rescued_failures','upper_bound_unavoidable_failures','unresolved_actual_failures')
    for policy in ('H8','cost'):
        for j in range(3):
            for index in (-1,0,1,2,3):
                group = [r for r in states if r['policy']==policy and r['scenario']==j
                         and (index==-1 or r['trajectory']==index)]
                row = dict(policy=policy, scenario=j, trajectory=index, states=len(group),
                    **{key:sum(r[key] for r in group) for key in measures},
                    **{cat:sum(r['category']==cat for r in group) for cat in categories})
                (summary if index==-1 else per_trajectory).append(row)
    assert len(states)==840 and len(episodes)==24
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
    for name, data in (('states',states),('episodes',episodes),('summary',summary),('per_trajectory',per_trajectory)):
        _write_rows(ROOT/f'{name}.csv', data)
    (ROOT/'audit.json').write_text(json.dumps(dict(states=840, episodes=24, replay_error=max_error,
        hashes_match=True, causal_choices=True, future_csi_evaluation_only=True), indent=2))
    (ROOT/'COMPLETE').write_text('Mechanism audit complete. No policy change or training.\n')
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
