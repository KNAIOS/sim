"""Frozen two-seed control screen; calibration is shared with strong search."""
import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.model import Environment, search_action
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from joint_power.calibrated.calibration import ROOT, load_predictor
from joint_power.calibrated.control import Learner, PRICES, calibrated_search, decision
from major_revision.budget_search import search as legacy_search

TRAIN, VALID, DEVELOP = 2709220000, 2809220000, 2909220000
SEEDS = (11, 22)
THRESHOLDS = (0., .02, .05, .10)


def intact():
    for name in ('manifest.json', 'calibration_hashes.json', 'control_manifest.json'):
        path = ROOT/name
        if not path.exists():
            continue
        hashes = json.loads(path.read_text())
        hashes = hashes.get('input_hashes', hashes)
        assert all(digest(p) == h for p, h in hashes.items()), name


def episode(trace, method, predictor, learner=None, parameter=2., epsilon=0., train=False):
    cfg = make_config('validation')
    env = Environment(cfg, trace, variable_power=method != 'Full fixed')
    learning_method = method in ('Calibrated full', 'Calibrated coarse', 'Myopic', 'RL')
    c = env.candidates()
    state = decision(c, predictor, cfg.episode_steps) if learning_method else None
    losses = []
    while env.t < cfg.episode_steps:
        if method.startswith('Full'):
            command = np.ones(cfg.subarrays, bool)
        elif method == 'PRS power':
            command = legacy_search(cfg, env.tracker, env.hw, env.req)
        elif method.startswith('Raw'):
            command = c.masks[search_action(c, stride=3 if method == 'Raw coarse' else 1)]
        else:
            if method == 'RL':
                action = learner.choose(state, parameter, epsilon)
            elif method == 'Myopic':
                action = int((state.immediate[:, 0]+parameter*state.immediate[:, 1]).argmin())
            else:
                action = calibrated_search(state, parameter, stride=3 if method == 'Calibrated coarse' else 1)
            command = state.masks[action]
        costs = env.block(command)
        done = env.t == cfg.episode_steps
        if not done:
            c = env.candidates()
        if learning_method:
            following = state if done else decision(c, predictor, cfg.episode_steps-env.t)
            if train:
                losses.append(learner.remember(state, action, costs, following, done))
            state = following
    if train:
        assert not learner.pending
    return env.metrics() | {'loss': float(np.mean(losses)) if losses else 0.}, env.records


def selection_key(rows):
    worst = max(np.mean([r['outage'] for r in rows if r['scenario'] == s]) for s in range(3))
    energy = np.mean([r['energy_j'] for r in rows])
    return max(0., float(worst)-.05), float(energy), float(worst)


def validation(method, predictor, learner=None):
    parameters = THRESHOLDS if method.startswith('Calibrated') else PRICES
    rows, options = [], []
    cfg = make_config('validation')
    for parameter in parameters:
        group = []
        for scenario, spec in enumerate(SCENARIOS):
            for i in range(4):
                trace = make_trace(cfg, VALID+scenario*10000+i, spec)
                metrics, _ = episode(trace, method, predictor, learner, parameter)
                group.append(dict(method=method, parameter=parameter, scenario=scenario, trajectory=i, **metrics))
        options.append((selection_key(group), parameter))
        rows.extend(group)
    key, parameter = min(options)
    return key, parameter, rows


def train_seed(seed):
    torch.set_num_threads(1)
    predictor = load_predictor()
    cfg = make_config('validation')
    learner = Learner(seed)
    rng = np.random.default_rng(seed)
    folder = ROOT/f'seed{seed}'
    folder.mkdir(exist_ok=False)
    key, parameter, rows = validation('RL', predictor, learner)
    best_key, best_parameter, best_episode = key, parameter, 0
    best = copy.deepcopy(learner.online.state_dict())
    validation_rows = [dict(episode=0, **r) for r in rows]
    torch.save(dict(state=best, parameter=parameter, episode=0), folder/'initial.pt')
    history = []
    for ep in range(160):
        scenario = int(rng.integers(3))
        price = float(rng.choice(PRICES))
        trace = make_trace(cfg, TRAIN+seed*100000+ep, SCENARIOS[scenario])
        epsilon = .03+.27*np.exp(-4*ep/159)
        metrics, _ = episode(trace, 'RL', predictor, learner, price, epsilon, train=True)
        history.append(dict(episode=ep+1, scenario=scenario, parameter=price,
                            epsilon=epsilon, updates=learner.updates, **metrics))
        if (ep+1) % 40 == 0:
            key, parameter, rows = validation('RL', predictor, learner)
            validation_rows.extend(dict(episode=ep+1, **r) for r in rows)
            if key < best_key:
                best_key, best_parameter, best_episode = key, parameter, ep+1
                best = copy.deepcopy(learner.online.state_dict())
            write_rows(folder/'history.csv', history)
            write_rows(folder/'validation.csv', validation_rows)
            print(f'phase2 seed={seed} ep={ep+1}, validation={key}, price={parameter}, selected={best_episode}', flush=True)
    torch.save(dict(state=best, parameter=best_parameter, episode=best_episode), folder/'selected.pt')
    torch.save(dict(state=learner.online.state_dict(), episode=160), folder/'last.pt')
    metadata = dict(seed=seed, selected_episode=best_episode, selected_price=best_parameter,
        validation_key=best_key, updates=learner.updates,
        parameters=sum(p.numel() for p in learner.online.parameters()))
    (folder/'selection.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def train(worker=None):
    torch.set_num_threads(1)
    intact()
    calibration = json.loads((ROOT/'calibration/report.json').read_text())
    if not calibration['passed']:
        raise RuntimeError('Calibration gate failed; do not launch downstream RL')
    assert not (ROOT/'control_manifest.json').exists()
    files = [*Path('joint_power/calibrated').glob('*.py'), Path('joint_power/calibrated/PLAN.md'),
             ROOT/'calibration/selected.pt']
    (ROOT/'control_manifest.json').write_text(json.dumps({str(p): digest(p) for p in files}, indent=2)+'\n')
    predictor = load_predictor()
    choices, baseline_rows = {}, []
    for method in ('Calibrated full', 'Calibrated coarse', 'Myopic'):
        key, parameter, rows = validation(method, predictor)
        choices[method] = dict(parameter=parameter, validation_key=key)
        baseline_rows.extend(rows)
        print(f'baseline selection {method}: {choices[method]}', flush=True)
    write_rows(ROOT/'baseline_validation.csv', baseline_rows)
    (ROOT/'baseline_selection.json').write_text(json.dumps(choices, indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker or train_seed, SEEDS))
    intact()
    (ROOT/'training_complete.json').write_text(json.dumps(results, indent=2)+'\n')


def screen(rows):
    mean = lambda scenario, method, metric: float(np.mean([r[metric] for r in rows if
        r['scenario'] == scenario and r['method'] == method]))
    per_seed = {str(seed): [float(np.mean([r['outage'] for r in rows if r['seed'] == seed and
        r['method'] == 'RL' and r['scenario'] == s])) for s in range(3)] for seed in SEEDS}
    baselines = {m: mean(1, m, 'energy_j') for m in ('Calibrated full', 'PRS power')}
    saving = {m: 1-mean(1, 'RL', 'energy_j')/e for m, e in baselines.items()}
    energies = {str(seed): float(np.mean([r['energy_j'] for r in rows if r['scenario'] == 1 and
        r['method'] == 'RL' and r['seed'] == seed])) for seed in SEEDS}
    selected = [json.loads((ROOT/f'seed{s}/selection.json').read_text())['selected_episode'] for s in SEEDS]
    reliable = all(mean(s, m, 'outage') <= .05 for m in baselines for s in range(3))
    passed = (max(max(v) for v in per_seed.values()) <= .05 and min(saving.values()) >= .02
        and max(energies.values()) <= min(baselines.values()) and min(selected) > 0 and reliable)
    return dict(passed=bool(passed), per_seed_outage=per_seed, nominal_saving=saving,
        nominal_energy_by_seed=energies, selected_episodes=selected,
        primary_baselines_below_outage_limit=bool(reliable), independent_confirmation=False,
        manuscript_replacement_authorized_by_result=False)


def evaluate():
    torch.set_num_threads(1)
    intact()
    assert (ROOT/'training_complete.json').exists()
    folder = ROOT/'evaluation'
    folder.mkdir(exist_ok=False)
    predictor, cfg = load_predictor(), make_config('validation')
    choices = json.loads((ROOT/'baseline_selection.json').read_text())
    learners = {}
    policies = [(m, 0, choices.get(m, {}).get('parameter', 2.)) for m in
                ('Full fixed', 'Full power', 'PRS power', 'Raw full', 'Raw coarse',
                 'Calibrated full', 'Calibrated coarse', 'Myopic')]
    for seed in SEEDS:
        saved = torch.load(ROOT/f'seed{seed}/selected.pt', weights_only=True)
        learner = Learner(seed)
        learner.online.load_state_dict(saved['state'])
        learners[seed] = learner
        policies.append(('RL', seed, saved['parameter']))
    episodes, slots = [], []
    for scenario, spec in enumerate(SCENARIOS):
        for i in range(4):
            trace_seed = DEVELOP+scenario*10000+i
            trace = make_trace(cfg, trace_seed, spec)
            for method, seed, parameter in policies:
                metrics, records = episode(trace, method, predictor, learners.get(seed), parameter)
                identity = dict(scenario=scenario, trajectory=i, trajectory_seed=trace_seed,
                                method=method, seed=seed, parameter=parameter)
                episodes.append(dict(**identity, **metrics))
                slots.extend(dict(**identity, **r) for r in records)
            print(f'phase2 development scenario={scenario} trace={i}', flush=True)
    write_rows(folder/'episodes.csv', episodes)
    write_rows(folder/'slots.csv', slots)
    summary = []
    for s in range(3):
        for method in dict.fromkeys(m for m, _, _ in policies):
            group = [r for r in episodes if r['scenario'] == s and r['method'] == method]
            summary.append(dict(scenario=s, method=method, trajectories=4,
                training_seeds=2 if method == 'RL' else 0,
                **{k: float(np.mean([r[k] for r in group])) for k in
                   ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')}))
    write_rows(folder/'summary.csv', summary)
    gate = screen(episodes)
    (folder/'screen.json').write_text(json.dumps(gate, indent=2)+'\n')
    intact()
    hashes = {str(p): digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name != 'output_hashes.json'}
    (ROOT/'output_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, screen=gate), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'evaluate'))
    args = parser.parse_args()
    train() if args.phase == 'train' else evaluate()
