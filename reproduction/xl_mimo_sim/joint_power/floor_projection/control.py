"""Minimal absolute-risk correction; no weight or search-baseline changes."""
import numpy as np
import torch

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.risk_budget.control import choose
from joint_power.risk_budget.physical_floor import failure_floor
from joint_power.rollout.collect import base_command


def project_counts(count, lower):
    count, lower = np.asarray(count), np.asarray(lower)
    if count.ndim != 1 or count.shape != lower.shape:
        raise ValueError('Matching one-dimensional counts required')
    if not np.isfinite(count).all() or not np.isfinite(lower).all() or (count < 0).any() or (lower < 0).any():
        raise ValueError('Finite nonnegative counts required')
    return np.maximum(count, lower)


def action_at(env, predictor, model, rule, alpha, past):
    if rule not in ('Value', 'Myopic', 'Raw budget') or not 0 <= alpha <= 1 or past < 0:
        raise ValueError('Invalid rule or budget')
    env.prepare()
    cfg, remaining = env.cfg, env.cfg.episode_steps-env.t
    residual = max(0., cfg.outage_target*cfg.users*cfg.episode_steps-past)
    bound = alpha*residual
    candidates = env.candidates()
    baseline = base_command(env)
    reference = int(baseline.sum())-1
    np.testing.assert_array_equal(candidates.masks[reference], baseline)
    if rule == 'Raw budget':
        energy, count = candidates.energy, candidates.outage*cfg.users*remaining
    else:
        state = decision(env, candidates, predictor)
        with torch.no_grad():
            learned, predicted, _ = model.components(torch.from_numpy(state.x[None]),
                torch.tensor([reference]), torch.tensor([remaining], dtype=torch.float32))
        energy = learned.numpy()[0] if rule == 'Value' else candidates.features[:, -13]
        count = predicted.numpy()[0]
    lower = failure_floor(env, candidates.masks)
    projected = project_counts(count, lower)
    original_action, _ = choose(energy, count, bound, reference)
    action, feasible = choose(energy, projected, bound, reference)
    return candidates.masks[action], dict(slot=env.t, past_failed=past,
        residual_budget=residual, applied_bound=bound, feasible_candidates=feasible,
        selected_raw_count=float(count[action]), selected_projected_count=float(projected[action]),
        selected_floor=int(lower[action]), corrected_candidates=int(np.sum(count < lower)),
        action=action, uncorrected_action=original_action,
        changed_command=int(not np.array_equal(candidates.masks[action], candidates.masks[original_action])),
        predicted_relative_energy=float(energy[action]-energy[reference]))


def episode(trace, rule, alpha, predictor, model=None, cfg=None):
    cfg = make_config('validation') if cfg is None else cfg
    env, past, ledger = Environment(cfg, trace), 0, []
    while env.t < cfg.episode_steps:
        command, metadata = action_at(env, predictor, model, rule, alpha, past)
        start = len(env.records)
        env.block(command)
        failures = sum(round(cfg.users*r['outage']) for r in env.records[start:])
        assert metadata['selected_floor'] <= failures
        assert metadata['selected_projected_count']+1e-10 >= metadata['selected_floor']
        past += failures
        ledger.append(dict(**metadata, block_failed=failures, cumulative_failed=past))
    return env.metrics(), env.records, ledger
