"""Matched time-limit ablation: bootstrap actual successors of sampled windows.

This is an exploratory continuing-control candidate, not a silent reinterpretation
of the terminal-window manuscript. Physical evaluation and all splits stay fixed.
"""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.constraint_training import preserved_exploration_rng, selection_key
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.ready_episode import ready_episode


def run(args):
    torch.set_num_threads(1)
    cfg = replace(make_config('validation'), train_episodes=300)
    assert cfg.episode_steps % 4 == 0, 'Truncate only at a complete decision block'
    extra_cfg = replace(cfg, episode_steps=cfg.episode_steps+1)
    reference = Path(f'results/major_revision_20260920_ready_ablation/seed{args.seed}')
    assert (reference/'COMPLETE').exists()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = ConstrainedDQN(cfg, 217, args.seed)
    model.dual = 4.
    sources = [Path(__file__).with_name(name) for name in (
        'continuing_training.py', 'ready_episode.py', 'ready_rate_context.py',
        'rate_context.py', 'pilot.py', 'constraint_training.py')]
    sources += list(Path('future_vr').glob('*.py'))
    manifest = dict(seed=args.seed, price=4., episodes=300, state_dimension=217,
        parameters=sum(p.numel() for p in model.online.parameters()),
        hidden_width=64, checkpoint_every=50, validation_per_scenario=12,
        development_per_scenario=16, scenarios=SCENARIOS, config=cfg.as_dict(),
        reference=str(reference), train_stream=971000000, validation_stream=972000000,
        development_stream=973000000,
        selection='minimum worst-scenario mean-outage violation, then mean energy',
        epsilon='0.04 + 0.76 * exp(-4 * zero_based_episode / 99)',
        objective='discounted continuing control; sampled windows are truncations',
        final_target='actual causal observation at t=140; done=False; no extra action',
        evaluation='unchanged 140-slot episodes with identical physical accounting',
        status='development ablation on already-viewed traces; not independent confirmation',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    validation = [(j, i, trajectory(cfg, 972000000+j*10000+i, scenario))
        for j, scenario in enumerate(SCENARIOS) for i in range(12)]
    rng = np.random.default_rng(args.seed)
    history, validation_rows, selections = [], [], []
    best_key = (float('inf'), float('inf'))
    selected_episode = None
    for ep in range(300):
        j = int(rng.integers(3))
        tr = trajectory(extra_cfg, 971000000+args.seed*10000+ep, SCENARIOS[j])
        epsilon = .04 + .76*np.exp(-4*ep/99)
        result = ready_episode(cfg, tr, model, True, epsilon, bootstrap_truncation=True)
        history.append(dict(seed=args.seed, price=4., episode=ep+1,
                            scenario=j, epsilon=epsilon, **result))
        if (ep+1) % 10 == 0:
            _write_rows(out/'training.csv', history)
            print('continuing train', args.seed, ep+1, flush=True)
        if (ep+1) % 50 == 0:
            current = []
            with preserved_exploration_rng(model):
                for j, i, tr in validation:
                    current.append(dict(scenario=j, trajectory=i,
                                        **ready_episode(cfg, tr, model)))
            violation, energy, worst = selection_key(current)
            key = (violation, energy)
            checkpoint = dict(online=copy.deepcopy(model.online.state_dict()),
                target=copy.deepcopy(model.target.state_dict()), state_dimension=217,
                price=4., seed=args.seed, episode=ep+1, validation_key=key,
                worst_validation_outage=worst, validation_feasible=violation == 0.,
                bootstrap_truncation=True)
            torch.save(checkpoint, out/f'checkpoint_{ep+1:03d}.pt')
            if key < best_key:
                best_key, selected_episode = key, ep+1
                torch.save(checkpoint, out/'selected.pt')
            selections.append(dict(episode=ep+1, violation=violation, energy_j=energy,
                                   worst_outage=worst, selected_episode=selected_episode))
            validation_rows.extend(dict(seed=args.seed, price=4., episode=ep+1, **r)
                                   for r in current)
            _write_rows(out/'selection.csv', selections)
            _write_rows(out/'validation.csv', validation_rows)
            print('continuing validation', args.seed, ep+1, 'worst', round(worst, 5),
                  'energy', round(energy, 3), 'selected', selected_episode, flush=True)
    saved = torch.load(out/'selected.pt', map_location='cpu', weights_only=True)
    model.online.load_state_dict(saved['online'])
    rows = []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(16):
            tr = trajectory(cfg, 973000000+j*10000+i, scenario)
            result = ready_episode(cfg, tr, model)
            result['method'] = 'Continuing-ready DQN'
            rows.append(dict(scenario=j, trajectory=i, seed=args.seed, margin=1., **result))
            _write_rows(out/'development_raw.csv', rows)
        print('continuing development', args.seed, j, 'complete', flush=True)
    (out/'COMPLETE').write_text('Time-limit development ablation completed; not final evidence.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(11,22), required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
