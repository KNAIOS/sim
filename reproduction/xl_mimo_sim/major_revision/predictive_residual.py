"""Three-action Double DQN around a causal predictive rate-search anchor."""
import copy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from future_vr.controllers import ConstrainedDQN, QNetwork, Transition, select_top
from future_vr.rate_search import rate_search_command
from major_revision.audit_rollout import AuditRollout


class ResidualNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.network = QNetwork(218,3,cfg.rl_hidden)
        final = self.network.layers[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor([-.01,0.,-.01]))

    def forward(self,x):
        value = self.network(x)
        anchor = torch.round(x[:,-1]*self.cfg.subarrays)
        allowed = torch.stack((anchor>self.cfg.users+1,
                               torch.ones_like(anchor,dtype=torch.bool),
                               anchor<self.cfg.subarrays),dim=-1)
        return value.masked_fill(~allowed,-1e9)


class ResidualDQN(ConstrainedDQN):
    def __init__(self,cfg,seed):
        super().__init__(cfg,218,seed)
        torch.manual_seed(seed)
        self.actions = 3
        self.online = ResidualNetwork(cfg)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.AdamW(self.online.parameters(),lr=cfg.rl_lr,weight_decay=1e-5)
        self.dual = 4.

    def choose(self,observation,epsilon):
        anchor = int(round(float(observation[-1])*self.cfg.subarrays))
        legal = [a for a in range(3) if self.min_count <= anchor+a-1 <= self.cfg.subarrays]
        if self.rng.random() < epsilon:
            action = int(self.rng.choice(legal))
        else:
            with torch.no_grad():
                action = int(self.online(torch.as_tensor(observation)[None]).argmax(1).item())
        return action,anchor+action-1


class FixedResidual:
    def __init__(self,cfg,delta):
        self.cfg,self.delta = cfg,delta
    def choose(self,observation,epsilon=0.):
        anchor = int(round(float(observation[-1])*self.cfg.subarrays))
        count = int(np.clip(anchor+self.delta,self.cfg.users+1,self.cfg.subarrays))
        return count-anchor+1,count


def anchored_observation(run,margin):
    observation,score = run.observation()
    command = rate_search_command(run.cfg,run.tracker,run.hw,run.req*margin,score)
    state = np.concatenate((observation,[command.sum()/run.cfg.subarrays])).astype(np.float32)
    return state,score


def residual_episode(cfg,tr,controller,margin=1.,training=False,epsilon=0.):
    if training and len(tr.channel) <= cfg.episode_steps:
        raise ValueError('Continuing training requires one real successor observation')
    # The extra training observation is prepared, never executed or charged.
    env_cfg = replace(cfg,episode_steps=cfg.episode_steps+1) if training else cfg
    run = AuditRollout(env_cfg,tr)
    full = run.system.evaluate(tr.channel[0],np.ones(cfg.subarrays,bool),
        np.zeros(cfg.subarrays,bool),0.,False,tr.requirements_mbps[0],
        observed_count=cfg.subarrays).energy_j
    pending = None
    while run.t < cfg.episode_steps:
        if run.t % 4 == 0:
            observation,score = anchored_observation(run,margin)
            if training and pending is not None:
                old,action,start,energy = pending
                n = run.t-start
                controller.remember(Transition(old,action,(run.energy-energy)/(n*full),
                    float(np.mean(run.outages[start:run.t])),observation,False))
                controller.learn()
            action,count = controller.choose(observation,epsilon if training else 0.)
            pending = (observation,action,run.t,run.energy)
            run.step(select_top(score,count))
        else:
            run.step()
    if training:
        old,action,start,energy = pending
        n = run.t-start
        successor,_ = anchored_observation(run,margin)
        controller.remember(Transition(old,action,(run.energy-energy)/(n*full),
            float(np.mean(run.outages[start:run.t])),successor,False))
        controller.learn()
    result = run.metrics()
    result['method'] = 'Predictive-residual DQN'
    return result
