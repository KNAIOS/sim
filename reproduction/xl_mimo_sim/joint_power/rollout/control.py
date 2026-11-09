"""Freeze prices/baseline thresholds on actual causal closed-loop validation."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.calibrated.study import selection_key
from joint_power.model import Environment, search_action
from joint_power.postdecision.encoding import decision
from joint_power.rollout.absolute import AbsoluteValue, ROOT
from joint_power.rollout.analyze import intact
from joint_power.rollout.collect import ROOT as DATA, base_command
from joint_power.rollout.probe import DifferenceValue, select
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from major_revision.budget_search import search


def input_checks():
    intact()
    for root, name in ((ROOT, 'training_hashes.json'), (DATA/'probe', 'output_hashes.json')):
        hashes = json.loads((root/name).read_text())
        assert all(digest(p) == h for p, h in hashes.items()), str(root/name)


def load_models():
    with (DATA/'probe/validation.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    models, checkpoints = {}, {}
    for seed in (11, 22):
        group = [r for r in rows if r['method'] == f'MC seed{seed}']
        epoch = int(min(group, key=lambda r: (float(r['huber']), int(r['epoch'])))['epoch'])
        for name, cls, path in (
            ('MC signed', DifferenceValue, DATA/f'probe/seed{seed}_epoch{epoch:03d}.pt'),
            ('MC absolute', AbsoluteValue, ROOT/f'seed{seed}_selected.pt')):
            saved = torch.load(path, weights_only=True)
            model = cls(saved['state']['mean'], saved['state']['std'])
            model.load_state_dict(saved['state'])
            models[name, seed] = model.eval()
            checkpoints[f'{name} {seed}'] = dict(path=str(path), sha256=digest(path), epoch=saved['epoch'])
    return models, checkpoints


def episode(trace, method, parameter, predictor, model=None):
    cfg = make_config('validation')
    env = Environment(cfg, trace)
    while env.t < cfg.episode_steps:
        env.prepare()
        if method == 'Full power':
            command = np.ones(16, bool)
        elif method in ('PRS tuned', 'PRS original'):
            command = search(replace(cfg, outage_target=parameter), env.tracker, env.hw, env.req)
        else:
            c = env.candidates()
            if method.startswith('Raw'):
                command = c.masks[search_action(c, stride=3 if method == 'Raw coarse' else 1, target=parameter)]
            else:
                state = decision(env, c, predictor)
                reference = int(base_command(env).sum())-1
                np.testing.assert_array_equal(state.masks[reference], base_command(env))
                args = (torch.from_numpy(state.x[None]), torch.tensor([reference]))
                if method == 'MC absolute':
                    args += (torch.tensor([cfg.episode_steps-env.t], dtype=torch.float32),)
                with torch.no_grad():
                    predicted = model(*args).numpy()
                action = int(select(predicted, np.array([reference]), parameter)[0])
                command = c.masks[action]
        env.block(command)
    return env.metrics(), env.records


def gate(rows, threshold):
    def mean(s, method, field):
        return float(np.mean([r[field] for r in rows if r['scenario'] == s and r['method'] == method]))
    per_seed = {str(seed): [float(np.mean([r['outage'] for r in rows if r['method'] == 'MC absolute'
                   and r['seed'] == seed and r['scenario'] == s])) for s in range(3)] for seed in (11, 22)}
    energies = {str(seed): float(np.mean([r['energy_j'] for r in rows if r['method'] == 'MC absolute'
                    and r['seed'] == seed and r['scenario'] == 1])) for seed in (11, 22)}
    prs = min(mean(1, m, 'energy_j') for m in ('PRS original', 'PRS tuned'))
    gain = 1-mean(1, 'MC absolute', 'energy_j')/prs
    return dict(passed=bool(max(max(v) for v in per_seed.values()) <= .05 and gain >= threshold
                           and max(energies.values()) <= prs), nominal_saving_vs_stronger_prs=gain,
                per_seed_outage=per_seed, nominal_energy_by_seed=energies,
                nominal_stronger_prs_energy=prs, required_saving=threshold)


def main(phase):
    torch.set_num_threads(1)
    input_checks()
    models, checkpoints = load_models()
    predictor, cfg = load_predictor(), make_config('validation')
    if phase == 'development':
        assert json.loads((ROOT/'validation/gate.json').read_text())['passed']
        selected_hashes = json.loads((ROOT/'validation/output_hashes.json').read_text())
        assert all(digest(p) == h for p, h in selected_hashes.items())
    folder = ROOT/phase
    folder.mkdir(exist_ok=False)
    source = {str(Path(__file__)): digest(__file__)}
    (folder/'manifest.json').write_text(json.dumps(dict(source=source, checkpoints=checkpoints,
        stream=3909220000 if phase == 'validation' else 4009220000), indent=2)+'\n')
    fixed = [('Full power', 0, .05), ('PRS original', 0, .05)]
    if phase == 'validation':
        specifications = fixed+[(name, seed, price) for name, seed in models for price in (4., 8., 16., 32.)]
        specifications += [(name, 0, p) for name in ('PRS tuned', 'Raw full', 'Raw coarse') for p in (0., .025, .05, .1)]
    else:
        selected = json.loads((ROOT/'validation/selection.json').read_text())
        specifications = fixed+[(r['method'], r['seed'], r['parameter']) for r in selected]
    rows, slots = [], []
    stream = 3909220000 if phase == 'validation' else 4009220000
    for s, spec in enumerate(SCENARIOS):
        for i in range(4):
            trace = make_trace(cfg, stream+s*10000+i, spec)
            for method, seed, parameter in specifications:
                metrics, records = episode(trace, method, parameter, predictor, models.get((method, seed)))
                identity = dict(scenario=s, trajectory=i, method=method, seed=seed, parameter=parameter)
                rows.append(dict(**identity, **metrics))
                if phase == 'development':
                    slots.extend(dict(**identity, **r) for r in records)
            print(f'absolute control {phase} scenario={s} trajectory={i}', flush=True)
    write_rows(folder/'all_episodes.csv', rows)
    if phase == 'validation':
        selected, selected_rows = [], []
        for method, seed in dict.fromkeys((m, s) for m, s, _ in specifications if (m, s, .05) not in fixed):
            options = []
            for parameter in sorted({p for m, se, p in specifications if (m, se) == (method, seed)}):
                group = [r for r in rows if (r['method'], r['seed'], r['parameter']) == (method, seed, parameter)]
                options.append((selection_key(group), parameter, group))
            key, parameter, group = min(options, key=lambda v: v[:2])
            selected.append(dict(method=method, seed=seed, parameter=parameter, validation_key=key))
            selected_rows.extend(group)
        selected_rows.extend(r for r in rows if (r['method'], r['seed'], r['parameter']) in fixed)
        (folder/'selection.json').write_text(json.dumps(selected, indent=2)+'\n')
    else:
        selected_rows = rows
        write_rows(folder/'slots.csv', slots)
    write_rows(folder/'selected_episodes.csv', selected_rows)
    summary = []
    for s in range(3):
        for method in dict.fromkeys(r['method'] for r in selected_rows):
            group = [r for r in selected_rows if r['method'] == method and r['scenario'] == s]
            summary.append(dict(scenario=s, method=method, **{k: float(np.mean([r[k] for r in group])) for k in
                ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')}))
    write_rows(folder/'summary.csv', summary)
    result = gate(selected_rows, .01 if phase == 'validation' else .02)
    result['noninitial_checkpoints'] = all(v['epoch'] > 0 for k, v in checkpoints.items() if k.startswith('MC absolute'))
    result['passed'] &= result['noninitial_checkpoints']
    (folder/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in source.items())
    (folder/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in folder.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, gate=result), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('validation', 'development'))
    main(parser.parse_args().phase)
