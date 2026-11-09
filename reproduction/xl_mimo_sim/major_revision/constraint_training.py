"""Bounded fixed-price/longer-budget diagnostic with feasibility-first selection."""
import argparse
import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS, episode, trajectory
from major_revision.rate_context import candidate_rate_context


@contextmanager
def preserved_exploration_rng(controller):
    saved = copy.deepcopy(controller.rng.bit_generator.state)
    try:
        yield
    finally:
        controller.rng.bit_generator.state = saved


def selection_key(metrics, target=.05):
    grouped = [[r for r in metrics if r['scenario'] == j] for j in range(3)]
    if any(not group for group in grouped):
        raise ValueError('Every scenario must be represented')
    worst = max(float(np.mean([r['outage'] for r in group])) for group in grouped)
    energy = float(np.mean([r['energy_j'] for r in metrics]))
    return max(0., worst-target), energy, worst


def run(args):
    torch.set_num_threads(1)
    cfg = replace(make_config('validation'), train_episodes=args.episodes)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).with_name('pilot.py'),
             Path(__file__).with_name('rate_context.py'), *Path('future_vr').glob('*.py')]
    manifest = dict(price=args.price, seed=args.seed, episodes=args.episodes,
        state_dimension=178, checkpoint_every=50, validation_per_scenario=12,
        scenarios=SCENARIOS, config=cfg.as_dict(), train_stream=971000000,
        validation_stream=972000000, development_stream=973000000,
        selection='minimum worst-scenario mean-outage violation, then mean energy',
        epsilon='0.04 + 0.76 * exp(-4 * zero_based_episode / 99)',
        status='bounded development diagnostic, not independent superiority evidence',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    validation = [(j, i, trajectory(cfg, 972000000+j*10000+i, scenario))
        for j, scenario in enumerate(SCENARIOS) for i in range(12)]
    model = ConstrainedDQN(cfg, 178, args.seed)
    model.dual = args.price
    rng = np.random.default_rng(args.seed)
    history, validation_rows, selections = [], [], []
    best_key = (float('inf'), float('inf'))
    selected_episode = None
    for ep in range(args.episodes):
        scenario = int(rng.integers(3))
        tr = trajectory(cfg, 971000000+args.seed*10000+ep, SCENARIOS[scenario])
        epsilon = .04 + .76*np.exp(-4*ep/99)
        metrics = episode(cfg, tr, 'Enriched-DQN', model, True, epsilon,
                          state_features=candidate_rate_context)
        history.append(dict(seed=args.seed, price=args.price, episode=ep+1,
                            scenario=scenario, epsilon=epsilon, **metrics))
        if (ep+1) % 10 == 0:
            _write_rows(out/'training.csv', history)
            print('train', args.price, args.seed, ep+1, flush=True)
        if (ep+1) % 50 == 0 or ep+1 == args.episodes:
            current = []
            with preserved_exploration_rng(model):
                for j, i, tr in validation:
                    result = episode(cfg, tr, 'Enriched-DQN', model,
                                     state_features=candidate_rate_context)
                    current.append(dict(scenario=j, trajectory=i, **result))
            violation, energy, worst = selection_key(current)
            key = (violation, energy)
            checkpoint = dict(online=copy.deepcopy(model.online.state_dict()),
                target=copy.deepcopy(model.target.state_dict()),
                state_dimension=178, price=args.price, seed=args.seed,
                episode=ep+1, validation_key=key, worst_validation_outage=worst,
                validation_feasible=violation == 0.)
            torch.save(checkpoint, out/f'checkpoint_{ep+1:03d}.pt')
            if key < best_key:
                best_key, selected_episode = key, ep+1
                torch.save(checkpoint, out/'selected.pt')
            selections.append(dict(episode=ep+1, violation=violation,
                energy_j=energy, worst_outage=worst, selected_episode=selected_episode))
            validation_rows.extend(dict(seed=args.seed, price=args.price,
                episode=ep+1, **result) for result in current)
            _write_rows(out/'validation.csv', validation_rows)
            _write_rows(out/'selection.csv', selections)
            print('validation', args.price, args.seed, ep+1,
                  'worst outage', round(worst, 5), 'energy', round(energy, 3),
                  'selected', selected_episode, flush=True)
    _write_rows(out/'training.csv', history)
    (out/'COMPLETE').write_text('Bounded training completed; validation is not final evidence.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, choices=(11, 22), required=True)
    parser.add_argument('--price', type=float, choices=(4., 8.), required=True)
    parser.add_argument('--episodes', type=int, default=300)
    run(parser.parse_args())
