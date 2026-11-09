"""Causal energy minimization on an estimated remaining-count feasible set."""
from dataclasses import replace

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.rollout.collect import base_command
from major_revision.budget_search import search


def choose(energy, count, budget, reference):
    energy, count = np.asarray(energy), np.asarray(count)
    if energy.ndim != 1 or energy.shape != count.shape or not np.isfinite(energy).all() or not np.isfinite(count).all():
        raise ValueError('Expected finite one-dimensional energy/count arrays')
    if (count < 0).any() or budget < 0 or not 0 <= reference < len(energy):
        raise ValueError('Invalid nonnegative count/budget or reference')
    feasible = np.flatnonzero(count <= budget+1e-10)
    if len(feasible):
        action = min(feasible, key=lambda i: (energy[i], count[i], i != reference, i))
    else:
        action = min(range(len(energy)), key=lambda i: (count[i], energy[i], i != reference, i))
    return int(action), int(len(feasible))


def action_at(env, predictor, model, rule, alpha, past):
    if not 0 <= alpha <= 1 or past < 0:
        raise ValueError('Invalid budget factor or failure count')
    env.prepare()
    cfg = env.cfg
    remaining = cfg.episode_steps-env.t
    residual = max(0., cfg.outage_target*cfg.users*cfg.episode_steps-past)
    bound = alpha*residual
    metadata = dict(slot=env.t, past_failed=past, residual_budget=residual, applied_bound=bound)
    if rule == 'PRS budget':
        threshold = min(1., bound/(cfg.users*remaining))
        command = search(replace(cfg, outage_target=threshold), env.tracker, env.hw, env.req)
        return command, dict(**metadata, feasible_candidates=-1, selected_count_prediction=-1.,
                             reference_count_prediction=-1., predicted_relative_energy=0.)
    candidates = env.candidates()
    baseline = base_command(env)
    reference = int(baseline.sum())-1
    np.testing.assert_array_equal(candidates.masks[reference], baseline)
    if rule == 'Raw budget':
        energy = candidates.energy
        count = candidates.outage*cfg.users*remaining
    else:
        if rule not in ('Value', 'Myopic'):
            raise ValueError('Unknown budget rule')
        state = decision(env, candidates, predictor)
        with torch.no_grad():
            learned_energy, predicted_count, _ = model.components(
                torch.from_numpy(state.x[None]), torch.tensor([reference]),
                torch.tensor([remaining], dtype=torch.float32))
        energy = learned_energy.numpy()[0] if rule == 'Value' else candidates.features[:, -13]
        count = predicted_count.numpy()[0]
    action, feasible = choose(energy, count, bound, reference)
    return candidates.masks[action], dict(**metadata, feasible_candidates=feasible,
        selected_count_prediction=float(count[action]), reference_count_prediction=float(count[reference]),
        predicted_relative_energy=float(energy[action]-energy[reference]))


def episode(trace, rule, alpha, predictor, model=None, cfg=None):
    cfg = make_config('validation') if cfg is None else cfg
    env, past, decisions = Environment(cfg, trace), 0, []
    while env.t < cfg.episode_steps:
        command, metadata = action_at(env, predictor, model, rule, alpha, past)
        start = len(env.records)
        env.block(command)
        failures = sum(round(cfg.users*r['outage']) for r in env.records[start:])
        past += failures
        decisions.append(dict(**metadata, block_failed=failures, cumulative_failed=past))
    assert past == sum(round(cfg.users*r['outage']) for r in env.records)
    return env.metrics(), env.records, decisions
