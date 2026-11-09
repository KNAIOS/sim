"""Known block energy, frozen immediate risk, and price-conditioned FQI value."""
import torch
from torch import nn

from major_revision.action_value import action_size, observation_size
from major_revision.gram_control import representation_size

PRICES = (.5, 1., 2., 4., 8., 16.)
DEPTH = 16


def split_features(cfg, x):
    common_end, legacy_end = representation_size(cfg), observation_size(cfg)
    common = torch.cat((x[:, :common_end], x[:, legacy_end:legacy_end+2]), -1)
    actions = x[:, common_end:legacy_end].reshape(-1, cfg.subarrays, action_size(cfg))
    current = x[:, legacy_end+2:].reshape(-1, cfg.subarrays, cfg.users+1)
    return common, torch.cat((actions, current), -1), actions, current


class ActionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.common = nn.Sequential(nn.Linear(representation_size(cfg)+2, 64),
                                    nn.LayerNorm(64), nn.SiLU())
        self.action = nn.Sequential(nn.Linear(action_size(cfg)+cfg.users+1, 32),
                                    nn.LayerNorm(32), nn.SiLU())

    def forward(self, x):
        common, action, _, _ = split_features(self.cfg, x)
        shared = self.common(common)[:, None].expand(-1, self.cfg.subarrays, -1)
        return torch.cat((shared, self.action(action)), -1)


class ImmediateRisk(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ActionEncoder(cfg)
        self.head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


class Continuation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ActionEncoder(cfg)
        self.head = nn.Sequential(nn.Linear(97, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, price):
        z = self.encoder(x)
        price = torch.as_tensor(price, dtype=x.dtype, device=x.device)
        if price.ndim == 0:
            price = price.expand(len(x))
        if price.shape != (len(x),) or torch.any(price <= 0):
            raise ValueError('One finite positive price per root is required')
        if not torch.isfinite(price).all():
            raise ValueError('Non-finite price')
        p = (torch.log2(price)/4)[:, None, None].expand(-1, z.shape[1], 1)
        raw = self.head(torch.cat((z, p), -1))
        return torch.stack((nn.functional.softplus(raw[..., 0]),
                            torch.sigmoid(raw[..., 1])), -1)


def tail_weight(gamma, depth):
    if not 1 <= depth <= DEPTH:
        raise ValueError('Depth outside frozen fitted-value range')
    return sum(gamma**j for j in range(1, depth))


class BellmanValue(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.risk = ImmediateRisk(cfg)
        self.tail = Continuation(cfg)

    def immediate(self, x):
        _, _, actions, current = split_features(self.cfg, x)
        energy = actions[:, :, self.cfg.subarrays:self.cfg.subarrays+4].mean(-1)
        outage = current[:, :, -1]/4 + .75*torch.sigmoid(self.risk(x))
        return torch.stack((energy, outage), -1)

    def forward(self, x, price, depth=DEPTH):
        weight = tail_weight(self.cfg.rl_gamma, depth)
        result = self.immediate(x)
        return result if weight == 0 else result + weight*self.tail(x, price)


def greedy_cost(costs, prices):
    """One chosen action and BOTH costs from the same scalarized decision."""
    prices = torch.as_tensor(prices, device=costs.device, dtype=costs.dtype)
    if prices.ndim == 0:
        prices = prices.expand(len(costs))
    if prices.shape != (len(costs),):
        raise ValueError('Invalid price batch')
    action = (costs[..., 0]+prices[:, None]*costs[..., 1]).argmin(1)
    selected = costs[torch.arange(len(costs), device=costs.device), action]
    return action, selected


def continuation_target(successor_costs, prices, gamma, depth):
    """Normalized future-cost target; the present block is not counted twice."""
    if depth <= 1:
        raise ValueError('Depth one has no continuation fitting target')
    _, selected = greedy_cost(successor_costs, prices)
    return gamma*selected/tail_weight(gamma, depth)
