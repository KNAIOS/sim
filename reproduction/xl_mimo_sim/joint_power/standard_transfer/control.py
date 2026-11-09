"""Frozen geometric policy with physically consistent wideband current floor."""
from dataclasses import replace

import numpy as np
import torch

from future_vr.controllers import select_top
from future_vr.experiment import _prediction
from joint_power.anchor_confirmation.domain_prs import command_at
from joint_power.baseline_anchor.control import admission
from joint_power.floor_projection.control import project_counts
from joint_power.matched_four.control import allowed, restricted_choice
from joint_power.postdecision.encoding import decision
from joint_power.rollout.collect import base_command
from joint_power.standard_transfer.environment import WidebandEnvironment, failure_floor
from major_revision.budget_search import predicted_rates

REFERENCES = ('PRS4', 'budgetPRS4', 'PRS1', 'budgetPRS1', 'WBbudgetPRS4', 'WBbudgetPRS1', 'Full array')


def learned_action(env, predictor, model, past, minimum=4, alpha=.1):
    env.prepare()
    cfg, remaining = env.cfg, env.cfg.episode_steps-env.t
    residual = max(0., cfg.outage_target*cfg.users*cfg.episode_steps-past)
    candidates, baseline = env.candidates(), base_command(env)
    reference = int(baseline.sum())-1
    np.testing.assert_array_equal(candidates.masks[reference], baseline)
    ids = allowed(candidates.masks, minimum)
    state = decision(env, candidates, predictor)
    with torch.no_grad():
        learned, predicted, _ = model.components(torch.from_numpy(state.x[None]),
            torch.tensor([reference]), torch.tensor([remaining], dtype=torch.float32))
    energy, count = learned.numpy()[0], predicted.numpy()[0]
    lower = failure_floor(env, candidates.masks)
    projected = project_counts(count, lower)
    action, feasible = restricted_choice(energy, projected, alpha*residual, reference, ids)
    relative = float(energy[action]-energy[reference])
    accepted, reason = admission(feasible, relative, 'Energy')
    executed = action if accepted else reference
    command = candidates.masks[executed]
    bits = lambda mask: sum(int(b) << q for q, b in enumerate(mask))
    return command, dict(slot=env.t, minimum=minimum, domain_size=len(ids), past_failed=past,
        residual_budget=residual, applied_bound=alpha*residual, proposal_feasible_candidates=feasible,
        proposal_relative_energy=relative, proposal_predicted_count=float(projected[action]),
        proposal_bits=bits(candidates.masks[action]), reference_bits=bits(baseline),
        executed_bits=bits(command), command_count=int(command.sum()),
        accepted=int(accepted), reason=reason, executed_floor=int(lower[executed]),
        differs_from_reference=int(not np.array_equal(command, baseline)))


def corrected_search(env, threshold, minimum):
    """Current wideband plus seven original causal centre-frequency forecasts."""
    env.prepare()
    cfg = env.cfg
    score, _ = _prediction(cfg, env.tracker, env.hw, env.req)
    counts = np.arange(minimum, cfg.subarrays+1)
    masks = np.array([select_top(score, int(m)) for m in counts])
    future = predicted_rates(cfg, env.tracker, env.hw, score, counts)
    current = np.array([env.current_phy(env.hw.active & command)[1] for command in masks])
    projected = np.concatenate((current[:, None], future[:, :cfg.horizon-1]), axis=1)
    outage = (projected < env.req).mean((1, 2))
    shortfall = (np.maximum(env.req-projected, 0.)/env.req).mean((1, 2))
    chosen = min(range(len(counts)), key=lambda i: (outage[i] > threshold,
        counts[i] if outage[i] <= threshold else shortfall[i], counts[i]))
    return masks[chosen]


def episode(cfg, trace, physical, method, predictor=None, model=None):
    if method not in (*REFERENCES, 'Value4'):
        raise ValueError('Unknown transfer method')
    env, past, ledger = WidebandEnvironment(cfg, trace, physical), 0, []
    while env.t < cfg.episode_steps:
        env.prepare()
        if method == 'Value4':
            command, info = learned_action(env, predictor, model, past)
        else:
            residual = max(0., cfg.outage_target*cfg.users*cfg.episode_steps-past)
            threshold = min(1., residual/(cfg.users*(cfg.episode_steps-env.t)))
            if method == 'Full array':
                command = np.ones(cfg.subarrays, bool)
            elif method.startswith('WB'):
                command = corrected_search(env, threshold, int(method[-1]))
            else:
                command = command_at(replace(cfg, outage_target=threshold if method.startswith('budget')
                    else cfg.outage_target), env.tracker, env.hw, env.req, minimum=int(method[-1]))
            info = dict(slot=env.t, minimum=16 if method == 'Full array' else int(method[-1]),
                domain_size=-1, past_failed=past, residual_budget=residual, applied_bound=-1.,
                proposal_feasible_candidates=-1, proposal_relative_energy=0., proposal_predicted_count=-1.,
                proposal_bits=-1, reference_bits=-1,
                executed_bits=sum(int(b) << q for q, b in enumerate(command)), command_count=int(command.sum()),
                accepted=-1, reason='reference', executed_floor=int(failure_floor(env, command[None])[0]),
                differs_from_reference=-1)
        assert int(command.sum()) >= info['minimum']
        start = len(env.records)
        env.block(command)
        failed = sum(round(cfg.users*r['outage']) for r in env.records[start:])
        assert info['executed_floor'] <= failed
        past += failed
        ledger.append(dict(**info, block_failed=failed, cumulative_failed=past))
    assert past == sum(round(cfg.users*r['outage']) for r in env.records)
    return env.metrics(), env.records, ledger
