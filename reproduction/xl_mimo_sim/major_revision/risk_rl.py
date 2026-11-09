"""Matched masked Double DQN on original versus calibrated causal forecasts."""
import copy
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from future_vr.controllers import ConstrainedDQN, Transition
from major_revision.action_value import full_slot_energy
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import gram_belief_features, representation_size
from major_revision.risk_correction import select_action
from major_revision.risk_correction_study import causal_data


PRICE = 2.


def action_dimension(cfg):
    return cfg.subarrays+2*cfg.horizon+2*cfg.horizon*cfg.users+1


def state(run, mode, parameter, minimum, risk_model):
    data = causal_data(run)
    cfg, req = run.cfg, run.req
    rates = data['old_rates'] if mode == 'old' else data['rates']
    if mode == 'old':
        risk = (rates < req[None, None]).astype(float)
        allowed = (rates < req[None, None]*parameter).mean((1, 2)) <= .05
    elif mode == 'calibrated':
        risk = risk_model.predict(data, req)
        allowed = risk.mean((1, 2)) <= parameter
    else:
        raise ValueError('Only matched old and calibrated variants are supported')
    allowed[:minimum-1] = False
    if not np.any(allowed):
        allowed[select_action(data, req, mode, parameter, minimum, risk_model)] = True
    common = gram_belief_features(cfg, run.tracker, run.hw, run.req,
        run.previous_req, run.age, run.last_outage)
    candidates = np.concatenate((data['masks'].astype(float), data['available'].mean(2),
        data['energy']/full_slot_energy(cfg),
        np.clip(rates/req[None, None], 0, 4).reshape(cfg.subarrays, -1),
        risk.reshape(cfg.subarrays, -1), allowed[:, None]), axis=1)
    assert candidates.shape == (cfg.subarrays, action_dimension(cfg))
    return np.concatenate((common, candidates.ravel())).astype(np.float32), data, allowed


class RiskQNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.common = nn.Sequential(nn.Linear(representation_size(cfg), 64), nn.LayerNorm(64), nn.SiLU())
        self.candidate = nn.Sequential(nn.Linear(action_dimension(cfg), 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        cfg = self.cfg
        rows = x[:, representation_size(cfg):].reshape(-1, cfg.subarrays, action_dimension(cfg))
        start = cfg.subarrays+cfg.horizon
        prior = -rows[:, :, start:start+cfg.horizon].mean(-1)
        common = self.common(x[:, :representation_size(cfg)])[:, None].expand(-1, cfg.subarrays, -1)
        residual = self.head(torch.cat((common, self.candidate(rows)), dim=-1)).squeeze(-1)
        return (prior+residual).masked_fill(rows[:, :, -1] < .5, -1e9)


class RiskDQN(ConstrainedDQN):
    def __init__(self, cfg, seed):
        dimension = representation_size(cfg)+cfg.subarrays*action_dimension(cfg)
        super().__init__(cfg, dimension, seed)
        self.min_count, self.actions = 1, cfg.subarrays
        torch.manual_seed(seed)
        self.online = RiskQNetwork(cfg)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=cfg.rl_lr, weight_decay=1e-5)
        self.dual = PRICE

    def choose(self, observation, epsilon):
        allowed = observation[representation_size(self.cfg):].reshape(self.cfg.subarrays, -1)[:, -1] > .5
        if self.rng.random() < epsilon:
            action = int(self.rng.choice(np.flatnonzero(allowed)))
        else:
            with torch.no_grad():
                action = int(self.online(torch.as_tensor(observation)[None]).argmax(1).item())
        assert allowed[action]
        return action, action+1


def episode(cfg, tr, mode, parameter, minimum, risk_model, controller, training=False,
            epsilon=0., trace=False):
    if training and len(tr.channel) <= cfg.episode_steps:
        raise ValueError('A real successor is required for continuing-task training')
    env_cfg = replace(cfg, episode_steps=cfg.episode_steps+1) if training else cfg
    run = FullDomainRollout(env_cfg, tr)
    pending, actions, losses = None, [], []
    normalizer = full_slot_energy(cfg)
    def remember(successor):
        old, action, start, energy = pending
        n = run.t-start
        controller.remember(Transition(old, action, (run.energy-energy)/(n*normalizer),
            float(np.mean(run.outages[start:run.t])), successor, False))
        losses.append(controller.learn())
    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
            continue
        observation, data, _ = state(run, mode, parameter, minimum, risk_model)
        if training and pending is not None:
            remember(observation)
        action, _ = controller.choose(observation, epsilon if training else 0.)
        pending = (observation, action, run.t, run.energy)
        actions.append(action)
        run.step(data['masks'][action])
    if training:
        successor, _, _ = state(run, mode, parameter, minimum, risk_model)
        remember(successor)
    result = run.metrics()
    result['method'] = f'{mode}_rl'
    result['loss'] = float(np.mean(losses)) if losses else 0.
    if trace:
        return result, np.array(actions), np.array(run.slot_energy), np.array(run.outages)
    return result
