"""Evaluate only preselected candidates; no noise or extra baseline penalty.

This isolated kernel does not change any frozen phase1/2/3 experiment. It
implements the same uncalibrated eight-slot objective on a smaller domain.
Candidate work counts are not measured wall time or equal-compute evidence.
"""
from dataclasses import dataclass

import numpy as np

from future_vr.controllers import select_top
from future_vr.experiment import _prediction
from joint_power.model import allocate, effective_gains, slot_energy
from major_revision.budget_search import forecasts


@dataclass
class BudgetedCosts:
    ids: np.ndarray
    masks: np.ndarray
    energy: np.ndarray
    outage: np.ndarray
    shortfall: np.ndarray
    candidate_slots: int

    def choose(self, target=.05):
        feasible = np.flatnonzero(self.outage <= target)
        if len(feasible):
            return int(min(feasible, key=lambda i: (self.energy[i], self.ids[i])))
        return min(range(len(self.ids)), key=lambda i:
                   (self.shortfall[i], self.energy[i], self.ids[i]))


def evaluate(env, stride=1):
    """Choose the count grid BEFORE batched RZF/power solves.

    Include full-array and hold actions for every grid. Forecast, CSI access,
    reliability threshold, physical power allocator and fallback are unchanged.
    """
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('stride must be a positive integer')
    env.prepare()
    cfg = env.cfg
    ids = np.array(sorted(set(range(0, cfg.subarrays, stride)) |
                          {cfg.subarrays-1, cfg.subarrays}))
    score, _ = _prediction(cfg, env.tracker, env.hw, env.req)
    masks = np.array([select_top(score, i+1) if i < cfg.subarrays
                      else env.hw.command.copy() for i in ids])
    active = np.broadcast_to(env.hw.active, masks.shape).copy()
    countdown = np.broadcast_to(env.hw.countdown, masks.shape).copy()
    off = ~masks & (active | (countdown > 0))
    on = masks & ~active & (countdown == 0)
    switching = cfg.sleep_energy_j*off.sum(-1)+cfg.wake_energy_j*on.sum(-1)
    observed = active.sum(-1).copy()
    active[off], countdown[off], countdown[on] = False, 0, cfg.wake_delay_slots
    available, waking, observations = [], [], []
    for _ in range(cfg.horizon):
        available.append(active.copy())
        waking.append((countdown > 0).sum(-1))
        observations.append(observed.copy())
        ready = countdown == 1
        countdown[countdown > 0] -= 1
        active[ready] = True
        observed = active.sum(-1)
    available = np.stack(available, axis=1)
    predicted = np.concatenate((env.tracker.channel_estimate[None],
                                forecasts(cfg, env.tracker)[:-1]))
    blocks = predicted.reshape(cfg.horizon, cfg.users, cfg.subarrays,
                               cfg.antennas_per_subarray).transpose(0, 2, 1, 3)
    module_gram = blocks @ blocks.conj().swapaxes(-2, -1)
    gram = np.einsum('ahq,hqij->ahij', available, module_gram)
    gains = effective_gains(cfg, gram)
    power, achieved, _ = allocate(cfg, gains, env.req)
    transition = np.zeros(available.shape[:2])
    transition[:, 0] = switching
    energy = slot_energy(cfg, available.sum(-1), np.array(waking).T,
                         power.sum(-1), transition, np.array(observations).T)[0]
    return BudgetedCosts(ids, masks, energy.mean(-1),
        (achieved < env.req).mean((1, 2)),
        (np.maximum(env.req-achieved, 0)/env.req).mean((1, 2)),
        int(gram.shape[0]*gram.shape[1]))
