"""Admit an estimated improvement or execute the unchanged reference policy."""
import numpy as np

from future_vr.config import make_config
from joint_power.floor_projection.control import action_at as proposal_at
from joint_power.model import Environment
from joint_power.risk_budget.physical_floor import failure_floor
from joint_power.rollout.collect import base_command


def admission(feasible, relative_energy, rule):
    if rule not in ('Empty', 'Energy') or feasible < 0 or not np.isfinite(relative_energy):
        raise ValueError('Invalid admission inputs')
    if feasible == 0:
        return False, 'empty'
    if rule == 'Energy' and relative_energy >= -1e-8:
        return False, 'no_predicted_saving'
    return True, 'accepted'


def action_at(env, predictor, model, proposer, rule, alpha, past):
    proposal, info = proposal_at(env, predictor, model, proposer, alpha, past)
    accepted, reason = admission(info['feasible_candidates'], info['predicted_relative_energy'], rule)
    reference = base_command(env)
    command = proposal if accepted else reference
    physical = int(failure_floor(env, command[None])[0])
    bits = lambda mask: sum(int(b) << q for q, b in enumerate(mask))
    ledger = dict(slot=env.t, past_failed=past, residual_budget=info['residual_budget'],
        applied_bound=info['applied_bound'], proposal_feasible_candidates=info['feasible_candidates'],
        proposal_relative_energy=info['predicted_relative_energy'],
        proposal_predicted_count=info['selected_projected_count'],
        proposal_bits=bits(proposal), reference_bits=bits(reference), executed_bits=bits(command),
        accepted=int(accepted), reason=reason, executed_floor=physical,
        changed_proposal=int(not np.array_equal(proposal, command)),
        differs_from_reference=int(not np.array_equal(command, reference)))
    return command, ledger


def episode(trace, proposer, rule, alpha, predictor, model=None, cfg=None):
    cfg = make_config('validation') if cfg is None else cfg
    env, past, ledger = Environment(cfg, trace), 0, []
    while env.t < cfg.episode_steps:
        command, info = action_at(env, predictor, model, proposer, rule, alpha, past)
        start = len(env.records)
        env.block(command)
        failures = sum(round(cfg.users*r['outage']) for r in env.records[start:])
        assert info['executed_floor'] <= failures
        past += failures
        ledger.append(dict(**info, block_failed=failures, cumulative_failed=past))
    return env.metrics(), env.records, ledger
