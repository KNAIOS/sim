"""Causal previous-command restoration; viewed-trajectory mechanism audit only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from future_vr.system import XLSystem
from major_revision.full_domain import FullDomainRollout
from major_revision.full_mask_audit import causal_forecast, mask_id
from major_revision.gram_control import subarray_grams, rates_from_gram
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction import hardware_path, risk_features, select_action
from major_revision.risk_correction_study import causal_data, csv_rows, all_files
from major_revision.risk_rl_study import setup, ROOT as PREVIOUS_ROOT


ROOT = Path('results/major_revision_20260922_hold_action')
MODES = ('old', 'anchored', 'calibrated')
STREAM = 1210000000  # Already viewed: never relabel as untouched test data.


def augmented_data(run):
    """Append one literal keep-command action without changing prefix rows."""
    data = causal_data(run)
    cfg, hw = run.cfg, run.hw
    mask = hw.command[None].copy()
    if not mask.any():
        raise ValueError('Audit requires a nonempty previous command')
    old_forecast, _ = causal_forecast(cfg, run.tracker)
    measured = subarray_grams(cfg, run.tracker.channel_estimate)
    forecast = old_forecast.copy()
    forecast[0, hw.active] = measured[hw.active]
    available, energy = hardware_path(cfg, hw, mask)
    def rate(values):
        gram = np.einsum('ahq,hqkl->ahkl', available, values)
        return rates_from_gram(cfg, gram, available.sum(2)*cfg.antennas_per_subarray)
    rates = rate(forecast)
    added = dict(masks=mask, available=available, energy=energy,
        rates=rates, old_rates=rate(old_forecast),
        features=risk_features(cfg, run.tracker, hw, run.req,
            run.previous_req, run.age, available, rates, measured))
    return {k: np.concatenate((data[k], added[k]), axis=0) for k in data}


def scores(data, requirements, mode, parameter, model):
    energy = data['energy'].sum(1)
    if mode in ('old', 'anchored'):
        rates = data['old_rates'] if mode == 'old' else data['rates']
        req = requirements[None, None]*parameter
        risk = (rates < req).mean((1, 2))
        fallback = (np.maximum(req-rates, 0.)/req).mean((1, 2))
        threshold = .05
    elif mode == 'calibrated':
        risk = model.predict(data, requirements).mean((1, 2))
        fallback, threshold = risk, parameter
    else:
        raise ValueError(mode)
    return energy, risk, fallback, threshold


def choose(data, requirements, mode, parameter, minimum, model, extended):
    energy, risk, fallback, threshold = scores(data, requirements, mode, parameter, model)
    limit = len(energy) if extended else len(energy)-1
    eligible = [i for i in range(limit) if data['masks'][i].sum() >= minimum]
    def key(i):
        feasible = risk[i] <= threshold
        return (not feasible, energy[i] if feasible else fallback[i], energy[i], i)
    return min(eligible, key=key)


def physical_check(run, data):
    """Additional action only; separate scalar PHY and hardware implementation."""
    if run.t+run.cfg.horizon > len(run.tr.channel):
        return None
    cfg, hardware, system = run.cfg, copy.deepcopy(run.hw), XLSystem(run.cfg)
    error, current_error = 0., 0.
    for h in range(cfg.horizon):
        observed = int(hardware.active.sum())
        active, switch, _ = hardware.apply(data['masks'][-1], 1.)
        result = system.evaluate(run.tr.channel[run.t+h], active, hardware.countdown > 0,
            switch, False, run.tr.requirements_mbps[run.t+h], observed_count=observed)
        error = max(error, abs(result.energy_j-data['energy'][-1, h]))
        np.testing.assert_array_equal(active, data['available'][-1, h])
        if h == 0:
            current_error = float(np.max(np.abs(result.rates_mbps-data['rates'][-1, 0])))
        hardware.tick()
    assert error < 1e-10 and current_error < 1e-6
    return error, current_error


def branch(root, command, mode, parameter, minimum, model):
    """One changed command block then the same original prefix continuation."""
    run, first = root.clone(), True
    while run.t < run.cfg.episode_steps:
        if run.t % 4:
            run.step()
            continue
        if first:
            action_mask = command
        else:
            data = causal_data(run)
            action_mask = data['masks'][select_action(data, run.req, mode, parameter, minimum, model)]
        run.step(action_mask)
        first = False
    return run.metrics()


def policy(cfg, tr, mode, parameter, minimum, model, extended, census=None):
    run = FullDomainRollout(cfg, tr)
    commands, roots, branches = [], [], []
    max_error, max_current, checks = 0., 0., 0
    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
            continue
        data = augmented_data(run)
        a = choose(data, run.req, mode, parameter, minimum, model, False)
        b = choose(data, run.req, mode, parameter, minimum, model, True)
        selected = b if extended else a
        if census is not None:
            energy, risk, fallback, threshold = scores(data, run.req, mode, parameter, model)
            current = run.hw.command.copy()
            changed = not np.array_equal(data['masks'][a], data['masks'][b])
            on = int(np.sum(data['masks'][a] & ~current))
            off = int(np.sum(~data['masks'][a] & current))
            row = dict(t=run.t, current=mask_id(current), prefix=mask_id(data['masks'][a]),
                extended=mask_id(data['masks'][b]), changed=int(changed), on=on, off=off,
                replacement_pairs=min(on, off), current_count=int(current.sum()),
                prefix_count=int(data['masks'][a].sum()),
                absent_from_prefix=int(not np.any(np.all(data['masks'][:-1] == current, axis=1))),
                hold_eligible=int(current.sum() >= minimum), hold_feasible=int(risk[-1] <= threshold),
                prefix_feasible=int(risk[a] <= threshold),
                hold_energy_j=float(energy[-1]), prefix_energy_j=float(energy[a]),
                hold_risk=float(risk[-1]), prefix_risk=float(risk[a]),
                chosen_forecast_energy_difference_j=float(energy[b]-energy[a]))
            roots.append(row)
            # Causal arrays persisted BEFORE using physical future labels.
            np.savez_compressed(census/f'root_t{run.t}.npz', **data,
                requirements=run.req.copy(), current=current, choices=np.array([a, b]),
                risk=risk, fallback=fallback, threshold=threshold)
            checked = physical_check(run, data)
            if checked is not None:
                max_error, max_current = max(max_error, checked[0]), max(max_current, checked[1])
                checks += 1
            if changed:
                before = run.metrics()
                value = branch(run, data['masks'][b], mode, parameter, minimum, model)
                assert run.metrics() == before
                np.testing.assert_array_equal(run.hw.command, current)
                branches.append(dict(t=run.t, command=mask_id(data['masks'][b]), **value))
        commands.append(data['masks'][selected].copy())
        run.step(data['masks'][selected])
    trace = dict(commands=np.array(commands), energy=np.array(run.slot_energy), outage=np.array(run.outages))
    diagnostics = dict(roots=roots, branches=branches, physical_energy_error=max_error,
                       current_rate_error=max_current, scalar_roots=checks)
    return run.metrics(), trace, diagnostics


def initialize(part, extras=()):
    folder = ROOT/part
    folder.mkdir(parents=True, exist_ok=False)
    paths = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path('major_revision/HOLD_ACTION_PLAN.md'), *extras]
    hashes = {str(p): digest(p) for p in paths}
    (folder/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(), threads=1,
        hashes=hashes, config=make_config('validation').as_dict(), stream=STREAM,
        viewed_data=True, independent_confirmation=False, training_updates=0), indent=2))
    return folder, hashes


def collect(shard):
    settings, model = setup()
    for part in ('audit', 'verified'):
        verify(PREVIOUS_ROOT/part)
    extras = all_files(PREVIOUS_ROOT/'verified')+[PREVIOUS_ROOT/'audit/raw.csv']
    from major_revision.risk_correction_study import ROOT as RISK_ROOT
    extras += all_files(RISK_ROOT/'model')+[RISK_ROOT/f'tune_{m}/selected.csv' for m in MODES]
    folder, hashes = initialize(f'shard{shard}', extras)
    cfg, results, roots, branches = make_config('validation'), [], [], []
    original = csv_rows(PREVIOUS_ROOT/'audit/raw.csv')
    errors = dict(archived_metrics=0., archived_energy=0., archived_outage=0., physical_energy=0., current_rate=0.)
    scalar_roots, external_hashes = 0, []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(4*shard, 4*shard+4):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            fingerprint = hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
            external_hashes.append(dict(scenario=j, trajectory=i, hash=fingerprint))
            for mode in MODES:
                parameter, minimum = settings[mode, j]
                census = folder/f'{mode}_s{j}_i{i}'
                census.mkdir()
                for extended in (False, True):
                    result, trace, diagnostic = policy(cfg, tr, mode, parameter, minimum,
                        model, extended, None if extended else census)
                    row = dict(scenario=j, trajectory=i, mode=mode, extended=int(extended), **result)
                    results.append(row)
                    np.savez_compressed(folder/f'trace_{mode}_{int(extended)}_s{j}_i{i}.npz', **trace)
                    if not extended:
                        saved = next(r for r in original if int(r['scenario']) == j and
                            int(r['trajectory']) == i and r['method'] == mode and int(r['seed']) == 0)
                        for k, value in result.items():
                            if k != 'method':
                                errors['archived_metrics'] = max(errors['archived_metrics'], abs(value-float(saved[k])))
                        with np.load(PREVIOUS_ROOT/f'audit/trace_{mode}_0_s{j}_i{i}.npz', allow_pickle=False) as z:
                            errors['archived_energy'] = max(errors['archived_energy'], float(np.max(np.abs(trace['energy']-z['energy']))))
                            errors['archived_outage'] = max(errors['archived_outage'], float(np.max(np.abs(trace['outage']-z['outage']))))
                            np.testing.assert_array_equal(trace['commands'].sum(1), z['actions']+1)
                        roots += [dict(scenario=j, trajectory=i, mode=mode, **r) for r in diagnostic['roots']]
                        for r in diagnostic['branches']:
                            branches.append(dict(scenario=j, trajectory=i, mode=mode, **r,
                                energy_difference_j=r['energy_j']-result['energy_j'],
                                outage_difference_pp=100*(r['outage']-result['outage'])))
                        scalar_roots += diagnostic['scalar_roots']
                        errors['physical_energy'] = max(errors['physical_energy'], diagnostic['physical_energy_error'])
                        errors['current_rate'] = max(errors['current_rate'], diagnostic['current_rate_error'])
                print('hold action audit', j, i, mode, flush=True)
            assert hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest() == fingerprint
    assert all(value < 1e-6 for value in errors.values())
    _write_rows(folder/'policies.csv', results)
    _write_rows(folder/'roots.csv', roots)
    _write_rows(folder/'branches.csv', branches)
    (folder/'audit.json').write_text(json.dumps(dict(errors=errors, scalar_roots=scalar_roots,
        policy_episodes=len(results), roots=len(roots), changed_branches=len(branches),
        external_trajectories=external_hashes, training_updates=0), indent=2))
    finish(folder, hashes)
    print('hold action shard complete', shard, len(branches), errors, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    collect(parser.parse_args().shard)
