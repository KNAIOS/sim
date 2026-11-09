"""Paired hold/re-close outcomes on fresh TRAINING-diagnostic trajectories."""
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from joint_power.calibrated.calibration import branch
from joint_power.calibrated.control import Learner, decision
from joint_power.calibrated.histogram import ROOT as PREVIOUS, load_predictor

ROOT = Path('results/joint_power_20260922_postdecision_diagnosis')


def selected_action(env, predictor, learner, price):
    c = env.candidates()
    state = decision(c, predictor, env.cfg.episode_steps-env.t)
    return c.masks[learner.choose(state, price)]


def rollout(env, first_command, predictor, learner, price):
    alternative = branch(env)
    outcomes = []
    discounted = np.zeros(2)
    for index in range(4):
        if alternative.t >= alternative.cfg.episode_steps:
            break
        command = first_command if index == 0 else selected_action(alternative, predictor, learner, price)
        costs = alternative.block(command)
        discounted += .98**index*np.asarray(costs)
        metrics = alternative.metrics()
        outcomes.append(dict(horizon=4*(index+1), discounted_energy=float(discounted[0]),
            discounted_outage=float(discounted[1]), weighted_cost=float(discounted@[1., price]),
            energy_j=metrics['energy_j'], outage=metrics['outage'], switching_j=metrics['switching_j']))
    return outcomes


def main():
    torch.set_num_threads(1)
    previous_hashes = json.loads((PREVIOUS/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in previous_hashes.items())
    ROOT.mkdir(parents=True, exist_ok=False)
    files = [*Path('joint_power/postdecision').glob('*.py'), Path('joint_power/postdecision/PLAN.md'),
             PREVIOUS/'calibration/selected.pt', *[PREVIOUS/f'seed{s}/selected.pt' for s in (11, 22)],
             Path('joint_power/model.py')]
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        stream=3109220000, scope='Training diagnosis, not independent evaluation'), indent=2)+'\n')
    predictor, cfg = load_predictor(), make_config('validation')
    rows, episode_rows = [], []
    for seed in (11, 22):
        checkpoint = torch.load(PREVIOUS/f'seed{seed}/selected.pt', weights_only=True)
        learner = Learner(seed)
        learner.online.load_state_dict(checkpoint['state'])
        price = checkpoint['parameter']
        for scenario, spec in enumerate(SCENARIOS):
            for i in range(4):
                env = Environment(cfg, make_trace(cfg, 3109220000+scenario*10000+i, spec))
                pending = np.zeros(cfg.subarrays, bool)
                events = 0
                while env.t < cfg.episode_steps:
                    command = selected_action(env, predictor, learner, price)
                    abandoned = pending & env.hw.active & ~command
                    if abandoned.any():
                        planned = rollout(env, command, predictor, learner, price)
                        held = rollout(env, env.hw.command.copy(), predictor, learner, price)
                        events += 1
                        for first, second in zip(planned, held):
                            if first['horizon'] not in (4, 8, 16):
                                continue
                            rows.append(dict(seed=seed, scenario=scenario, trajectory=i, slot=env.t,
                                abandoned_modules=int(abandoned.sum()), horizon=first['horizon'],
                                **{f'planned_{k}': v for k, v in first.items() if k != 'horizon'},
                                **{f'hold_{k}': v for k, v in second.items() if k != 'horizon'},
                                **{f'difference_{k}': second[k]-first[k] for k in
                                   ('energy_j', 'outage', 'weighted_cost', 'switching_j')}))
                    off = ~command & (env.hw.active | (env.hw.countdown > 0))
                    on = command & ~env.hw.active & (env.hw.countdown == 0)
                    pending[off], pending[on] = False, True
                    before = len(env.records)
                    env.block(command)
                    for record in env.records[before:]:
                        served = np.array([bool(record['active_bits'] & 1 << q) for q in range(cfg.subarrays)])
                        pending[served] = False
                episode_rows.append(dict(seed=seed, scenario=scenario, trajectory=i,
                    events=events, **env.metrics()))
                print(f'postdecision diagnostic seed={seed}, scenario={scenario}, trace={i}, events={events}', flush=True)
    write_rows(ROOT/'paired_events.csv', rows)
    write_rows(ROOT/'base_episodes.csv', episode_rows)
    summary = []
    for horizon in (4, 8, 16):
        selected = [r for r in rows if r['horizon'] == horizon]
        summary.append(dict(horizon=horizon, events=len(selected),
            mean_energy_difference=float(np.mean([r['difference_energy_j'] for r in selected])),
            mean_outage_difference=float(np.mean([r['difference_outage'] for r in selected])),
            mean_weighted_cost_difference=float(np.mean([r['difference_weighted_cost'] for r in selected])),
            hold_lower_energy_no_higher_outage=float(np.mean([r['difference_energy_j'] < -1e-9 and
                r['difference_outage'] <= 1e-12 for r in selected])),
            hold_higher_outage=float(np.mean([r['difference_outage'] > 1e-12 for r in selected]))))
    write_rows(ROOT/'summary.csv', summary)
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(p) == h for p, h in previous_hashes.items())
    report = dict(summary=summary, independent_trajectories=12, base_policy_rollouts=24,
        weights_updated=False, prior_data_preserved=True,
        conclusion='Conditional finite-horizon hold outcomes; no guaranteed deployed-policy improvement')
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.rglob('*')
        if p.is_file() and p.name != 'output_hashes.json'}, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
