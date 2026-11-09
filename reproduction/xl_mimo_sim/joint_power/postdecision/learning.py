"""One-step updates; optional all-action simulator branching during training."""
import copy
from collections import deque

import numpy as np
import torch

from joint_power.calibrated.calibration import branch
from joint_power.calibrated.control import Learner as BaseLearner, ValueNetwork
from joint_power.postdecision.encoding import decision


class Learner(BaseLearner):
    def __init__(self, seed):
        super().__init__(seed)
        torch.manual_seed(seed)
        self.online = ValueNetwork(width=343)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=.0004, weight_decay=1e-5)
        self.memory = deque(maxlen=20000)
        self.transitions_seen = 0

    def learn_transitions(self, transitions):
        self.memory.extend(transitions)
        self.transitions_seen += len(transitions)
        if len(self.memory) < 64:
            return 0.
        return float(np.mean([self.update() for _ in range(4)]))


def counterfactuals(env, state, predictor):
    """Each command has its own physical and observation-state successor."""
    transitions = []
    for action, command in enumerate(state.masks):
        alternative = branch(env)
        costs = alternative.block(command)
        done = alternative.t == alternative.cfg.episode_steps
        following = state if done else decision(alternative, alternative.candidates(), predictor)
        transitions.append((state, action, np.asarray(costs), following, 0. if done else .98))
    return transitions


def episode(trace, predictor, learner, price, *, training=False, branching=False, epsilon=0.):
    from future_vr.config import make_config
    from joint_power.model import Environment
    cfg = make_config('validation')
    env = Environment(cfg, trace)
    state = decision(env, env.candidates(), predictor)
    losses = []
    while env.t < cfg.episode_steps:
        action = learner.choose(state, price, epsilon)
        transitions = counterfactuals(env, state, predictor) if training and branching else None
        costs = env.block(state.masks[action])
        done = env.t == cfg.episode_steps
        following = state if done else decision(env, env.candidates(), predictor)
        if training:
            if transitions is None:
                transitions = [(state, action, np.asarray(costs), following, 0. if done else .98)]
            else:
                # A paired simulator replay must reproduce the actually selected action.
                np.testing.assert_allclose(transitions[action][2], costs, atol=1e-10)
                np.testing.assert_allclose(transitions[action][3].x, following.x, atol=1e-6)
                np.testing.assert_allclose(transitions[action][3].immediate, following.immediate, atol=1e-7)
            losses.append(learner.learn_transitions(transitions))
        state = following
    return env.metrics() | {'loss': float(np.mean(losses)) if losses else 0.}, env.records
