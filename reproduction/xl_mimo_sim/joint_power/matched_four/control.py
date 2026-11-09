"""Apply a shared command-count domain before selection, without retraining."""
import numpy as np
import torch

from future_vr.config import make_config
from joint_power.baseline_anchor.control import admission
from joint_power.floor_projection.control import project_counts
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.risk_budget.control import choose
from joint_power.risk_budget.physical_floor import failure_floor
from joint_power.rollout.collect import base_command


def allowed(masks, minimum):
    if masks.ndim != 2 or not 1 <= minimum <= masks.shape[1]:
        raise ValueError('Invalid command domain')
    ids = np.flatnonzero(masks.sum(1) >= minimum)
    if not len(ids):
        raise ValueError('Empty command domain')
    return ids


def restricted_choice(energy, count, bound, reference, ids):
    position = np.flatnonzero(ids == reference)
    if len(position) != 1:
        raise ValueError('Reference must belong to the allowed command domain')
    local, feasible = choose(energy[ids], count[ids], bound, int(position[0]))
    return int(ids[local]), feasible


def raw_action(candidates, minimum, target=.05):
    ids = allowed(candidates.masks, minimum)
    feasible = [a for a in ids if candidates.outage[a] <= target]
    return int(min(feasible, key=lambda a: (candidates.energy[a], a)) if feasible else min(ids,
        key=lambda a: (candidates.shortfall[a], candidates.energy[a], a)))


def action_at(env, predictor, model, proposer, alpha, past, minimum):
    if proposer not in ('Value', 'Myopic', 'Raw budget') or not 0 <= alpha <= 1 or past < 0:
        raise ValueError('Invalid proposer or risk allowance')
    env.prepare()
    cfg, remaining = env.cfg, env.cfg.episode_steps-env.t
    residual = max(0., cfg.outage_target*cfg.users*cfg.episode_steps-past)
    bound = alpha*residual
    candidates, baseline = env.candidates(), base_command(env)
    reference = int(baseline.sum())-1
    np.testing.assert_array_equal(candidates.masks[reference], baseline)
    ids = allowed(candidates.masks, minimum)
    if proposer == 'Raw budget':
        energy, count = candidates.energy, candidates.outage*cfg.users*remaining
    else:
        state = decision(env, candidates, predictor)
        with torch.no_grad():
            learned, predicted, _ = model.components(torch.from_numpy(state.x[None]),
                torch.tensor([reference]), torch.tensor([remaining], dtype=torch.float32))
        energy = learned.numpy()[0] if proposer == 'Value' else candidates.features[:, -13]
        count = predicted.numpy()[0]
    lower = failure_floor(env, candidates.masks)
    projected = project_counts(count, lower)
    a, feasible = restricted_choice(energy, projected, bound, reference, ids)
    relative = float(energy[a]-energy[reference])
    accepted, reason = admission(feasible, relative, 'Energy')
    executed = a if accepted else reference
    command = candidates.masks[executed]
    assert int(command.sum()) >= minimum
    bits = lambda mask: sum(int(b) << q for q, b in enumerate(mask))
    return command, dict(slot=env.t, minimum=minimum, domain_size=len(ids), past_failed=past,
        residual_budget=residual, applied_bound=bound, proposal_feasible_candidates=feasible,
        proposal_relative_energy=relative, proposal_predicted_count=float(projected[a]),
        proposal_bits=bits(candidates.masks[a]), reference_bits=bits(baseline),
        executed_bits=bits(command), command_count=int(command.sum()),
        accepted=int(accepted), reason=reason, executed_floor=int(lower[executed]),
        differs_from_reference=int(not np.array_equal(command, baseline)))


def episode(trace, proposer, alpha, predictor, model, minimum, cfg=None):
    cfg = make_config('validation') if cfg is None else cfg
    env, past, ledger = Environment(cfg, trace), 0, []
    while env.t < cfg.episode_steps:
        command, info = action_at(env, predictor, model, proposer, alpha, past, minimum)
        start = len(env.records)
        env.block(command)
        failed = sum(round(cfg.users*r['outage']) for r in env.records[start:])
        assert info['executed_floor'] <= failed
        past += failed
        ledger.append(dict(**info, block_failed=failed, cumulative_failed=past))
    return env.metrics(), env.records, ledger
