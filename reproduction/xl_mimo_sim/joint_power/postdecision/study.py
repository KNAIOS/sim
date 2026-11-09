"""Four frozen training runs: post-decision one-step versus action branching."""
import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from joint_power.calibrated.control import PRICES, Learner as PreviousLearner
from joint_power.calibrated.histogram import ROOT as PREVIOUS, load_predictor
from joint_power.calibrated.study import episode as baseline_episode, selection_key
from joint_power.postdecision.learning import Learner, episode

ROOT = Path('results/joint_power_20260922_phase3')
TRAIN, VALID, DEVELOP = 3209220000, 3309220000, 3409220000
VARIANTS = ('PDS-one-step', 'PDS-branch')
SEEDS = (11, 22)


def intact():
    for root, names in ((PREVIOUS, ('output_hashes.json',)), (ROOT, ('manifest.json',))):
        for name in names:
            data = json.loads((root/name).read_text())
            hashes = data.get('input_hashes', data)
            assert all(digest(p) == h for p, h in hashes.items()), str(root/name)


def validation(predictor, learner):
    cfg, rows, options = make_config('validation'), [], []
    for price in PRICES:
        group = []
        for scenario, spec in enumerate(SCENARIOS):
            for i in range(4):
                trace = make_trace(cfg, VALID+scenario*10000+i, spec)
                metrics, _ = episode(trace, predictor, learner, price)
                group.append(dict(parameter=price, scenario=scenario, trajectory=i, **metrics))
        options.append((selection_key(group), price))
        rows.extend(group)
    key, price = min(options)
    return key, price, rows


def train_one(task):
    variant, seed = task
    torch.set_num_threads(1)
    cfg, predictor, learner = make_config('validation'), load_predictor(), Learner(seed)
    rng = np.random.default_rng(seed)
    folder = ROOT/f'{variant}_seed{seed}'
    folder.mkdir(exist_ok=False)
    best_key, best_price, rows = validation(predictor, learner)
    best = copy.deepcopy(learner.online.state_dict())
    best_episode = 0
    validation_rows = [dict(episode=0, **r) for r in rows]
    torch.save(dict(state=best, parameter=best_price, episode=0), folder/'epoch000.pt')
    history = []
    for ep in range(160):
        scenario, price = int(rng.integers(3)), float(rng.choice(PRICES))
        trace = make_trace(cfg, TRAIN+seed*100000+ep, SCENARIOS[scenario])
        epsilon = .03+.27*np.exp(-4*ep/159)
        metrics, _ = episode(trace, predictor, learner, price, training=True,
                             branching=variant == 'PDS-branch', epsilon=epsilon)
        history.append(dict(episode=ep+1, scenario=scenario, parameter=price,
            epsilon=epsilon, updates=learner.updates, transitions=learner.transitions_seen, **metrics))
        if (ep+1) % 10 == 0:
            print(f'{variant} seed={seed} ep={ep+1}, train energy={metrics["energy_j"]:.2f}, '
                  f'outage={metrics["outage"]:.4f}, updates={learner.updates}', flush=True)
        if (ep+1) % 40 == 0:
            key, price, rows = validation(predictor, learner)
            validation_rows.extend(dict(episode=ep+1, **r) for r in rows)
            torch.save(dict(state=learner.online.state_dict(), parameter=price, episode=ep+1),
                       folder/f'epoch{ep+1:03d}.pt')
            if key < best_key:
                best_key, best_price, best_episode = key, price, ep+1
                best = copy.deepcopy(learner.online.state_dict())
            write_rows(folder/'history.csv', history)
            write_rows(folder/'validation.csv', validation_rows)
            print(f'{variant} seed={seed} VALID ep={ep+1}, key={key}, price={price}, selected={best_episode}', flush=True)
    torch.save(dict(state=best, parameter=best_price, episode=best_episode), folder/'selected.pt')
    metadata = dict(variant=variant, seed=seed, selected_episode=best_episode, selected_price=best_price,
        validation_key=best_key, updates=learner.updates, transitions=learner.transitions_seen,
        parameters=sum(p.numel() for p in learner.online.parameters()))
    (folder/'selection.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def select_baselines():
    predictor, cfg = load_predictor(), make_config('validation')
    grid = sorted({0., .02, .05, .10} | {float(v) for v in predictor.table if 0 <= v <= .10})
    all_rows, choices = [], {}
    for method in ('Calibrated full', 'Calibrated coarse', 'Myopic'):
        options = []
        for parameter in (PRICES if method == 'Myopic' else grid):
            rows = []
            for scenario, spec in enumerate(SCENARIOS):
                for i in range(4):
                    trace = make_trace(cfg, VALID+scenario*10000+i, spec)
                    metrics, _ = baseline_episode(trace, method, predictor, parameter=parameter)
                    rows.append(dict(method=method, parameter=parameter, scenario=scenario, trajectory=i, **metrics))
            options.append((selection_key(rows), parameter))
            all_rows.extend(rows)
        key, parameter = min(options)
        choices[method] = dict(parameter=parameter, validation_key=key)
        print(f'phase3 baseline {method}: {choices[method]}', flush=True)
    write_rows(ROOT/'baseline_validation.csv', all_rows)
    (ROOT/'baseline_selection.json').write_text(json.dumps(choices, indent=2)+'\n')


def train():
    torch.set_num_threads(1)
    assert Path('results/joint_power_20260922_postdecision_diagnosis/report.json').exists()
    ROOT.mkdir(parents=True, exist_ok=False)
    files = [*Path('joint_power/postdecision').glob('*.py'), Path('joint_power/postdecision/PLAN.md'),
        *Path('joint_power/calibrated').glob('*.py'), Path('joint_power/model.py'),
        *Path('future_vr').glob('*.py'), Path('major_revision/budget_search.py'),
        PREVIOUS/'calibration/selected.pt', PREVIOUS/'output_hashes.json']
    files += [Path('../ICECAI2026_4pages_20260922')/name for name in
              ('icecai2026_energy_rl.tex', 'output/pdf/clean.pdf', 'output/pdf/icecai2026_energy_rl.pdf')]
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        config=make_config('validation').as_dict(), host_environment_changed=False,
        streams=dict(train=TRAIN, validation=VALID, development=DEVELOP)), indent=2)+'\n')
    intact()
    # Two workers maximum; no system-wide thread or process configuration.
    tasks = [(variant, seed) for seed in SEEDS for variant in VARIANTS]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(train_one, tasks))
    select_baselines()
    intact()
    (ROOT/'training_complete.json').write_text(json.dumps(results, indent=2)+'\n')


def screen(episodes):
    selected = [json.loads((ROOT/f'PDS-branch_seed{s}/selection.json').read_text())['selected_episode'] for s in SEEDS]
    means = lambda method, s, metric: float(np.mean([r[metric] for r in episodes if r['method'] == method and r['scenario'] == s]))
    outage = {str(seed): [float(np.mean([r['outage'] for r in episodes if r['method'] == 'PDS-branch'
        and r['seed'] == seed and r['scenario'] == s])) for s in range(3)] for seed in SEEDS}
    energies = {str(seed): float(np.mean([r['energy_j'] for r in episodes if r['method'] == 'PDS-branch'
        and r['seed'] == seed and r['scenario'] == 1])) for seed in SEEDS}
    reference = means('PRS power', 1, 'energy_j')
    saving = 1-means('PDS-branch', 1, 'energy_j')/reference
    passed = (max(max(v) for v in outage.values()) <= .05 and saving >= .02 and
              max(energies.values()) <= reference and min(selected) > 0)
    return dict(passed=bool(passed), per_seed_outage=outage, nominal_energy_by_seed=energies,
        nominal_prs_energy=reference, nominal_saving_vs_prs=saving, selected_episodes=selected,
        independent_confirmation=False, multi_step_calibrated_search_confirmation_missing=True,
        manuscript_replacement_authorized_by_result=False)


def evaluate():
    torch.set_num_threads(1)
    intact()
    assert (ROOT/'training_complete.json').exists()
    folder = ROOT/'evaluation'
    folder.mkdir(exist_ok=False)
    predictor, cfg = load_predictor(), make_config('validation')
    choices = json.loads((ROOT/'baseline_selection.json').read_text())
    policies = [(m, 0, choices.get(m, {}).get('parameter', 2.), None) for m in
        ('Full fixed', 'Full power', 'PRS power', 'Raw full', 'Raw coarse',
         'Calibrated full', 'Calibrated coarse', 'Myopic')]
    for seed in SEEDS:
        checkpoint = torch.load(PREVIOUS/f'seed{seed}/selected.pt', weights_only=True)
        learner = PreviousLearner(seed)
        learner.online.load_state_dict(checkpoint['state'])
        policies.append(('Previous RL', seed, checkpoint['parameter'], learner))
        for variant in VARIANTS:
            checkpoint = torch.load(ROOT/f'{variant}_seed{seed}/selected.pt', weights_only=True)
            learner = Learner(seed)
            learner.online.load_state_dict(checkpoint['state'])
            policies.append((variant, seed, checkpoint['parameter'], learner))
    episodes, slots = [], []
    for scenario, spec in enumerate(SCENARIOS):
        for i in range(4):
            trace_seed = DEVELOP+scenario*10000+i
            trace = make_trace(cfg, trace_seed, spec)
            for method, seed, price, learner in policies:
                if method in VARIANTS:
                    metrics, records = episode(trace, predictor, learner, price)
                else:
                    metrics, records = baseline_episode(trace, 'RL' if method == 'Previous RL' else method,
                        predictor, learner, parameter=price)
                identity = dict(scenario=scenario, trajectory=i, trajectory_seed=trace_seed,
                    method=method, seed=seed, parameter=price)
                episodes.append(dict(**identity, **metrics))
                slots.extend(dict(**identity, **r) for r in records)
            print(f'phase3 development scenario={scenario} trace={i}', flush=True)
    write_rows(folder/'episodes.csv', episodes)
    write_rows(folder/'slots.csv', slots)
    summary = []
    for s in range(3):
        for method in dict.fromkeys(m for m, _, _, _ in policies):
            rows = [r for r in episodes if r['method'] == method and r['scenario'] == s]
            summary.append(dict(scenario=s, method=method, trajectories=4,
                training_seeds=2 if method in (*VARIANTS, 'Previous RL') else 0,
                **{k: float(np.mean([r[k] for r in rows])) for k in
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
