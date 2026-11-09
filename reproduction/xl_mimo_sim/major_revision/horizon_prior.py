"""Two-decision-block physical prior; no change to learned residual or physics."""
import copy
import numpy as np
import torch
from major_revision.action_value import ActionValueNetwork, ActionValueDQN, action_size, BLOCK, PRICE
from major_revision.gram_control import representation_size


def block_costs(cfg, state):
    """Return four-slot costs from serialized causal hold predictions (Torch)."""
    if cfg.horizon != 2 * BLOCK:
        raise ValueError('This protocol fixes exactly two four-slot blocks')
    rows = state[..., representation_size(cfg):].reshape(
        *state.shape[:-1], cfg.subarrays, action_size(cfg))
    start = cfg.subarrays
    energy = rows[..., start:start + cfg.horizon].reshape(
        *rows.shape[:-1], 2, BLOCK).mean(-1)
    start += cfg.horizon
    ratios = rows[..., start:start + cfg.horizon * cfg.users].reshape(
        *rows.shape[:-1], 2, BLOCK, cfg.users)
    return energy + PRICE * (ratios < 1.).to(state.dtype).mean((-1, -2))


def horizon_values(cfg, state):
    costs = block_costs(cfg, state)
    return -costs[..., 0] - cfg.rl_gamma * costs[..., 1]


class HorizonNetwork(ActionValueNetwork):
    def forward(self, state):
        # Parent returns -c0 + residual; subtract the discounted second cost.
        return super().forward(state) - self.cfg.rl_gamma * block_costs(self.cfg, state)[..., 1]


class HorizonDQN(ActionValueDQN):
    def __init__(self, cfg, seed):
        super().__init__(cfg, seed, 'shared')
        torch.manual_seed(seed)
        self.online = HorizonNetwork(cfg, 'shared')
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=cfg.rl_lr, weight_decay=1e-5)


class FixedHorizon:
    """No learned weights: isolated test of the initialization, not an RL result."""
    def __init__(self, cfg):
        self.cfg = cfg

    def choose(self, state, epsilon=0.):
        if epsilon != 0.:
            raise ValueError('Deterministic prior has no exploration')
        with torch.no_grad():
            values = horizon_values(self.cfg, torch.as_tensor(np.asarray(state)))
        action = int(values.argmax())
        return action, action + 1
