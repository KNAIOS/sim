"""Bounded risk fitting, disjoint calibration, fair search tuning, and audit."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from future_vr.system import HardwareState, XLSystem
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction import (
    FEATURE_NAMES, RateRiskModel, predictions, true_hold_outcomes, select_action,
)


ROOT = Path('results/major_revision_20260922_risk_correction')
PARTITIONS = {'fit_data': (1160000000, 8), 'calibration_data': (1170000000, 4),
              'audit_data': (1180000000, 8)}
TUNE_STREAM = 1190000000
MODES = ('old', 'anchored', 'calibrated')
MARGINS = (.90, 1., 1.04, 1.08, 1.16, 1.24)
THRESHOLDS = (.01, .025, .05, .075, .10, .15)
TIMES = tuple(range(0, 129, 8))


def initialize(part, extras=()):
    folder = ROOT/part
    folder.mkdir(parents=True, exist_ok=False)
    paths = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path('major_revision/RISK_CORRECTION_PLAN.md'), *extras]
    hashes = {str(p): digest(p) for p in paths}
    manifest = dict(pid=os.getpid(), part=part, threads=1, partitions=PARTITIONS,
        tune_stream=TUNE_STREAM, times=TIMES, margins=MARGINS, thresholds=THRESHOLDS,
        features=FEATURE_NAMES, config=make_config('validation').as_dict(),
        hashes=hashes, scientific_completion=False, rl_updates=0)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return folder, hashes


def all_files(folder):
    return [p for p in folder.rglob('*') if p.is_file()]


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def causal_data(run):
    _, score = run.base_observation()
    return predictions(run.cfg, run.tracker, run.hw, run.req, run.previous_req, run.age, score)


def check_physical_hold(cfg, tr, t, hw, data, actual_rates):
    """Independent scalar PHY/hardware checks for 1/8/16 modules at every root."""
    import copy
    rate_error = energy_error = 0.
    for action in (0, 7, 15):
        hardware = copy.deepcopy(hw)
        for h in range(cfg.horizon):
            observed = int(hardware.active.sum())
            active, switch, _ = hardware.apply(data['masks'][action], 1.)
            result = XLSystem(cfg).evaluate(tr.channel[t+h], active, hardware.countdown > 0,
                switch, False, tr.requirements_mbps[t+h], observed_count=observed)
            rate_error = max(rate_error, float(np.max(np.abs(result.rates_mbps-actual_rates[action, h]))))
            energy_error = max(energy_error, abs(result.energy_j-data['energy'][action, h]))
            hardware.tick()
    return rate_error, energy_error


def collect(part):
    if part == 'audit_data':
        verify(ROOT/'model')
        for mode in MODES:
            verify(ROOT/f'tune_{mode}')
    folder, hashes = initialize(part)
    cfg = make_config('validation')
    stream, count = PARTITIONS[part]
    errors = dict(physical_rate=0., physical_energy=0., current_rate=0., old_current_rate_max=0.)
    records = []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(count):
            tr = trajectory(cfg, stream+j*10000+i, scenario)
            input_hash = {k: __import__('hashlib').sha256(getattr(tr, k).tobytes()).hexdigest()
                          for k in ('channel', 'gain_db', 'requirements_mbps')}
            run = FullDomainRollout(cfg, tr)
            rng = np.random.default_rng(stream+900000+j*1000+i)
            stored = {k: [] for k in ('x', 'y', 'rates', 'old_rates', 'available', 'requirements', 'truth_rates')}
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                reference = baseline_command(run, 'cost', (1., 1.08, 1.04)[j], (1, 1, 4)[j])
                if run.t in TIMES:
                    data = causal_data(run)
                    # Snapshot the model inputs before obtaining any future outcome.
                    np.savez_compressed(folder/f'causal_s{j}_i{i}_t{run.t}.npz',
                        x=data['features'], rates=data['rates'], old_rates=data['old_rates'],
                        energy=data['energy'], masks=data['masks'], available=data['available'],
                        requirements=run.req)
                    rates, y = true_hold_outcomes(cfg, tr.channel[run.t:run.t+cfg.horizon],
                        tr.requirements_mbps[run.t:run.t+cfg.horizon], data['available'])
                    re, ee = check_physical_hold(cfg, tr, run.t, run.hw, data, rates)
                    errors['physical_rate'] = max(errors['physical_rate'], re)
                    errors['physical_energy'] = max(errors['physical_energy'], ee)
                    errors['current_rate'] = max(errors['current_rate'],
                        float(np.max(np.abs(data['rates'][:, 0]-rates[:, 0]))))
                    errors['old_current_rate_max'] = max(errors['old_current_rate_max'],
                        float(np.max(np.abs(data['old_rates'][:, 0]-rates[:, 0]))))
                    for key, value in dict(x=data['features'], y=y, rates=data['rates'],
                        old_rates=data['old_rates'], available=data['available'],
                        requirements=run.req.copy(), truth_rates=rates).items():
                        stored[key].append(value)
                command = reference
                if part == 'fit_data' and i % 4 == 3 and (run.t//4) % 2 == 1:
                    _, score = run.base_observation()
                    from future_vr.controllers import select_top
                    command = select_top(score, int(rng.integers(1, cfg.subarrays+1)))
                run.step(command)
            assert len(stored['x']) == len(TIMES)
            np.savez_compressed(folder/f'data_s{j}_i{i}.npz', **{k: np.stack(v) for k, v in stored.items()})
            for k, expected in input_hash.items():
                assert __import__('hashlib').sha256(getattr(tr, k).tobytes()).hexdigest() == expected
            records.append(dict(scenario=j, trajectory=i, **run.metrics()))
            print('risk data', part, j, i, flush=True)
    assert errors['physical_rate'] < 1e-6 and errors['current_rate'] < 1e-6
    assert errors['physical_energy'] < 1e-10
    _write_rows(folder/'history_episodes.csv', records)
    (folder/'audit.json').write_text(json.dumps(dict(errors=errors, trajectories=count*3,
        roots=count*3*len(TIMES), user_slot_labels=count*3*len(TIMES)*16*8*3,
        scalar_phy_checks=count*3*len(TIMES)*3*8, unchanged_physics=True), indent=2))
    finish(folder, hashes)
    print('risk data complete', part, errors, flush=True)


def load_future(part):
    verify(ROOT/part)
    xs, ys = [], []
    for p in sorted((ROOT/part).glob('data_*.npz')):
        with np.load(p, allow_pickle=False) as z:
            x, y = z['x'][:, :, 1:], z['y'][:, :, 1:]
            valid = np.broadcast_to(z['available'][:, :, 1:].any(-1)[..., None], y.shape)
            xs.append(x[valid])
            ys.append(y[valid])
    return np.concatenate(xs), np.concatenate(ys)


def fit_model():
    x, y = load_future('fit_data')
    cx, cy = load_future('calibration_data')
    folder, hashes = initialize('model', all_files(ROOT/'fit_data')+all_files(ROOT/'calibration_data'))
    model = RateRiskModel.fit(x, y)
    model.save(folder/'uncalibrated.npz')
    model.calibrate(cx, cy)
    model.save(folder/'calibrated.npz')
    assert np.isfinite(model.probability(cx)).all()
    np.testing.assert_array_equal(model.probability(cx), RateRiskModel.load(folder/'calibrated.npz').probability(cx))
    audit = dict(fit_samples=len(x), calibration_samples=len(cx), temperature=model.temperature,
        fit_trajectories=24, calibration_trajectories=12, training_only_normalization=True,
        l2_slope_penalty=1e-4, predictors=1, rl_updates=0)
    (folder/'audit.json').write_text(json.dumps(audit, indent=2))
    finish(folder, hashes)
    print('risk model frozen', audit, flush=True)


def episode(cfg, tr, mode, parameter, minimum, model=None, trace=False):
    run, actions = FullDomainRollout(cfg, tr), []
    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
        else:
            data = causal_data(run)
            action = select_action(data, run.req, mode, parameter, minimum, model)
            actions.append(action)
            run.step(data['masks'][action])
    result = run.metrics()
    result['method'] = mode
    if trace:
        return result, np.asarray(actions), np.asarray(run.slot_energy), np.asarray(run.outages)
    return result


def best_setting(means):
    feasible = [r for r in means if r['outage'] <= .05]
    return min(feasible, key=lambda r: (r['energy_j'], r['minimum'], r['parameter'])) if feasible else min(
        means, key=lambda r: (r['outage'], r['energy_j'], r['minimum'], r['parameter']))


def tune(mode):
    verify(ROOT/'model')
    folder, hashes = initialize(f'tune_{mode}', all_files(ROOT/'model'))
    model = RateRiskModel.load(ROOT/'model/calibrated.npz')
    cfg, raw, means, selected = make_config('validation'), [], [], []
    parameters = THRESHOLDS if mode == 'calibrated' else MARGINS
    for j, scenario in enumerate(SCENARIOS):
        trials = [trajectory(cfg, TUNE_STREAM+j*10000+i, scenario) for i in range(4)]
        group_means = []
        for minimum in (1, 4):
            for parameter in parameters:
                group = [dict(scenario=j, trajectory=i, parameter=parameter, minimum=minimum,
                    **episode(cfg, tr, mode, parameter, minimum, model)) for i, tr in enumerate(trials)]
                raw.extend(group)
                mean = dict(scenario=j, method=mode, parameter=parameter, minimum=minimum,
                    **{k: float(np.mean([r[k] for r in group])) for k in ('energy_j', 'outage')})
                means.append(mean)
                group_means.append(mean)
                _write_rows(folder/'raw.csv', raw)
        selected.append(best_setting(group_means))
        print('risk tuned', mode, selected[-1], flush=True)
    _write_rows(folder/'summary.csv', means)
    _write_rows(folder/'selected.csv', selected)
    finish(folder, hashes)


def evaluate_policies():
    verify(ROOT/'model')
    extras, settings = all_files(ROOT/'model'), {}
    for mode in MODES:
        verify(ROOT/f'tune_{mode}')
        extras += all_files(ROOT/f'tune_{mode}')
        for r in csv_rows(ROOT/f'tune_{mode}/selected.csv'):
            settings[mode, int(r['scenario'])] = (float(r['parameter']), int(r['minimum']))
    folder, hashes = initialize('policy_audit', extras)
    cfg, raw = make_config('validation'), []
    model = RateRiskModel.load(ROOT/'model/calibrated.npz')
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, 1180000000+j*10000+i, scenario)
            for mode in MODES:
                parameter, minimum = settings[mode, j]
                result, actions, energy, outage = episode(cfg, tr, mode, parameter, minimum, model, True)
                raw.append(dict(scenario=j, trajectory=i, parameter=parameter, minimum=minimum, **result))
                np.savez_compressed(folder/f'trace_{mode}_s{j}_i{i}.npz',
                                    actions=actions, energy=energy, outage=outage)
            _write_rows(folder/'raw.csv', raw)
            print('risk policy audit', j, i, flush=True)
    finish(folder, hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=(*PARTITIONS, 'fit', 'tune', 'evaluate'))
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage in PARTITIONS:
        collect(args.stage)
    elif args.stage == 'fit':
        fit_model()
    elif args.stage == 'tune':
        if args.mode is None:
            parser.error('tune requires --mode')
        tune(args.mode)
    else:
        evaluate_policies()
