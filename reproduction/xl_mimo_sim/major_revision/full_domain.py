"""Count-domain adapters. Frozen physical and historical sources stay intact."""
import copy
from dataclasses import replace
import numpy as np
import torch
from future_vr.controllers import ConstrainedDQN, QNetwork, Transition, select_top
from future_vr.system import XLSystem
from major_revision.audit_rollout import AuditRollout
from major_revision.pilot import state
from major_revision.spatial_actions import causal_forecasts, predicted_metrics


def observation_size(cfg, minimum=1):
    enriched = 4*cfg.subarrays + 2*cfg.users + 2
    enriched += cfg.subarrays*cfg.users + cfg.subarrays + cfg.users
    return enriched + 2*(cfg.subarrays-minimum+1)*cfg.users


class FullDomainDQN(ConstrainedDQN):
    def __init__(self, cfg, seed, minimum=1):
        super().__init__(cfg, observation_size(cfg, minimum), seed)
        if not 1 <= minimum <= cfg.subarrays:
            raise ValueError('Invalid minimum count')
        self.min_count = minimum
        self.actions = cfg.subarrays-minimum+1
        torch.manual_seed(seed)
        self.online = QNetwork(observation_size(cfg, minimum), self.actions, cfg.rl_hidden)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=cfg.rl_lr,
                                          weight_decay=1e-5)


class FullDomainRollout(AuditRollout):
    def __init__(self, cfg, tr, minimum=1):
        super().__init__(cfg, tr)
        if not 1 <= minimum <= cfg.subarrays:
            raise ValueError('Invalid minimum count')
        self.minimum = minimum

    def base_observation(self):
        self.prepare()
        if self.t % 4:
            raise ValueError('Not a decision epoch')
        return state(self.cfg, self.tracker, self.hw, self.req, self.previous_req,
                     self.age, self.last_outage, True)

    def observation(self):
        base, score = self.base_observation()
        fully_ready, actually_ready = [], []
        for count in range(self.minimum, self.cfg.subarrays+1):
            command = select_top(score, count)
            fully_ready.append(self.system.achievable_rates(self.tracker.channel_estimate, command))
            hardware = copy.deepcopy(self.hw)
            available, _, _ = hardware.apply(command, 1.)
            actually_ready.append(self.system.achievable_rates(self.tracker.channel_estimate, available))
        features = [base]
        for rates in (fully_ready, actually_ready):
            features.append(np.clip(np.asarray(rates)/np.maximum(self.req[None], 1e-9), 0, 4).ravel())
        return np.concatenate(features).astype(np.float32), score

    def command(self, margin=1., mode='prs'):
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('Prediction margin must be finite and positive')
        _, score = self.base_observation()
        candidates = [select_top(score, n) for n in range(self.minimum, self.cfg.subarrays+1)]
        if mode == 'full':
            return candidates[-1]
        if mode == 'prs':
            forecasts = causal_forecasts(self.cfg, self.tracker)
            measured = [predicted_metrics(self.cfg, self.hw, self.req*margin, forecasts, mask)
                        for mask in candidates]
            def key(i):
                infeasible = measured[i]['outage'] > self.cfg.outage_target
                return (infeasible, measured[i]['shortfall'] if infeasible else i, i)
            return candidates[min(range(len(candidates)), key=key)]
        if mode != 'cost':
            raise ValueError('Unknown controller')
        candidates.append(self.hw.command.copy())
        forecasts = causal_forecasts(self.cfg, self.tracker)
        forecasts = np.concatenate((self.tracker.channel_estimate[None], forecasts[:-1]))
        keys = []
        for mask in candidates:
            hardware = copy.deepcopy(self.hw)
            energy = outage = shortfall = 0.
            for h in range(self.cfg.horizon):
                observed = int(hardware.active.sum())
                active, transition, _ = hardware.apply(mask, 1.)
                result = self.system.evaluate(forecasts[h], active, hardware.countdown > 0,
                    transition, False, self.req*margin, observed_count=observed)
                energy += result.energy_j
                outage += result.outage/self.cfg.horizon
                shortfall += result.shortfall/self.cfg.horizon
                hardware.tick()
            keys.append((outage > self.cfg.outage_target,
                         energy if outage <= self.cfg.outage_target else shortfall, energy))
        return candidates[min(range(len(candidates)), key=lambda i: keys[i])]


def full_domain_episode(cfg, tr, controller=None, training=False, epsilon=0.,
                        margin=1., mode='prs', minimum=1):
    if training and (controller is None or len(tr.channel) <= cfg.episode_steps):
        raise ValueError('Training needs a model and a real successor observation')
    env_cfg = replace(cfg, episode_steps=cfg.episode_steps+1) if training else cfg
    run = FullDomainRollout(env_cfg, tr, minimum)
    full_energy = run.system.evaluate(tr.channel[0], np.ones(cfg.subarrays, bool),
        np.zeros(cfg.subarrays, bool), 0., False, tr.requirements_mbps[0],
        observed_count=cfg.subarrays).energy_j
    pending = None

    def remember(successor):
        old, action, start, energy = pending
        n = run.t-start
        controller.remember(Transition(old, action, (run.energy-energy)/(n*full_energy),
            float(np.mean(run.outages[start:run.t])), successor, False))
        controller.learn()

    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
            continue
        if controller is None:
            command = run.command(margin, mode)
        else:
            observation, score = run.observation()
            if training and pending is not None:
                remember(observation)
            action, count = controller.choose(observation, epsilon if training else 0.)
            if not minimum <= count <= cfg.subarrays:
                raise ValueError('Controller count lies outside common domain')
            command = select_top(score, count)
            pending = (observation, action, run.t, run.energy)
        run.step(command)
    if training:
        successor, _ = run.observation()
        remember(successor)
    result = run.metrics()
    result['method'] = 'Full-domain DQN' if controller is not None else mode
    result['below_four_fraction'] = float(np.mean(np.asarray(run.counts) < 4))
    return result
