from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn

from .config import Config


def select_top(score: np.ndarray, count: int) -> np.ndarray:
    count = int(np.clip(count, 0, score.size))
    mask = np.zeros(score.size, dtype=bool)
    if count:
        mask[np.argsort(score, kind="stable")[-count:]] = True
    return mask


def readiness_aligned_value(
    cfg: Config,
    gain_db: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
    readiness: np.ndarray,
    requirements: np.ndarray,
    *,
    align_readiness: bool = True,
    future: bool = True,
) -> np.ndarray:
    """Conservative future visibility value evaluated only after RF readiness.

    Prediction index zero represents slot ``t+1``.  Hence a chain with a
    readiness of ``d`` slots first contributes at index ``d-1``; an already
    active chain (``d=0``) also starts at index zero.
    """
    if not future:
        gain_db = gain_db[:1]
        probability = probability[:1]
        uncertainty = uncertainty[:1]
    horizon, q_count, users = gain_db.shape
    linear = 10.0 ** ((gain_db - gain_db.max(axis=1, keepdims=True)) / 10.0)
    demand_weight = requirements / max(float(np.mean(requirements)), 1e-9)
    conservative_probability = np.maximum(probability - cfg.uncertainty_penalty * uncertainty, 0.0)
    discount = cfg.visibility_discount ** np.arange(horizon)
    usable = (np.arange(1, horizon + 1)[:, None] >= readiness[None, :]
              if align_readiness else np.ones((horizon, q_count), dtype=bool))
    utility = (conservative_probability * linear * demand_weight[None, None, :]
               * discount[:, None, None] * usable[:, :, None])
    score = utility.sum(axis=(0, 2))
    scale = max(float(np.max(score)), 1e-9)
    score /= scale
    return score


def oracle_value(cfg: Config, future_gain_db: np.ndarray, future_visibility: np.ndarray) -> np.ndarray:
    linear = 10.0 ** ((future_gain_db - future_gain_db.max(axis=1, keepdims=True)) / 10.0)
    discount = 0.93 ** np.arange(future_gain_db.shape[0])[:, None, None]
    score = np.sum(discount * future_visibility * linear, axis=(0, 2))
    return score / max(float(score.max()), 1e-9)


def retention_count(gain_db: np.ndarray, score: np.ndarray, users: int, target: float = 0.88) -> int:
    """Smallest ranked set retaining a target fraction of every user's array gain."""
    order = np.argsort(score)[::-1]
    linear = 10.0 ** ((gain_db - gain_db.max(axis=0, keepdims=True)) / 10.0)
    total = np.maximum(linear.sum(axis=0), 1e-12)
    minimum = min(score.size, users + 1)
    for count in range(minimum, score.size + 1):
        if np.all(linear[order[:count]].sum(axis=0) / total >= target):
            return count
    return score.size


def symmetric_guard_mask(cfg: Config, probability: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    mask = np.zeros(cfg.subarrays, dtype=bool)
    for k in range(cfg.users):
        support = np.flatnonzero(probability[:, k] >= 0.35)
        if not support.size:
            support = np.array([int(np.argmax(probability[:, k]))])
        guard = int(np.ceil(abs(velocity[k]) * max(cfg.wake_delay_slots, 1) + 1.0))
        lo = max(0, int(support.min()) - guard)
        hi = min(cfg.subarrays, int(support.max()) + guard + 1)
        mask[lo:hi] = True
    return mask


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, actions: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    energy_cost: float
    outage_cost: float
    next_state: np.ndarray
    done: bool


class ConstrainedDQN:
    def __init__(self, cfg: Config, state_dim: int, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.cfg = cfg
        self.min_count = cfg.users + 1
        self.actions = cfg.subarrays - self.min_count + 1
        self.online = QNetwork(state_dim, self.actions, cfg.rl_hidden)
        self.target = QNetwork(state_dim, self.actions, cfg.rl_hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=cfg.rl_lr, weight_decay=1e-5)
        self.memory: deque[Transition] = deque(maxlen=cfg.replay_size)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.dual = cfg.dual_initial

    def choose(self, state: np.ndarray, epsilon: float) -> tuple[int, int]:
        if self.rng.random() < epsilon:
            action = int(self.rng.integers(self.actions))
        else:
            with torch.no_grad():
                action = int(self.online(torch.as_tensor(state, dtype=torch.float32)[None]).argmax(1).item())
        return action, self.min_count + action

    def remember(self, transition: Transition) -> None:
        self.memory.append(transition)

    def learn(self, batch_size: int = 64) -> float:
        if len(self.memory) < batch_size:
            return 0.0
        sample = random.sample(self.memory, batch_size)
        state = torch.as_tensor(np.stack([x.state for x in sample]), dtype=torch.float32)
        action = torch.as_tensor([x.action for x in sample], dtype=torch.int64)
        energy_cost = torch.as_tensor([x.energy_cost for x in sample], dtype=torch.float32)
        outage_cost = torch.as_tensor([x.outage_cost for x in sample], dtype=torch.float32)
        reward = -energy_cost - self.dual * outage_cost
        next_state = torch.as_tensor(np.stack([x.next_state for x in sample]), dtype=torch.float32)
        done = torch.as_tensor([x.done for x in sample], dtype=torch.float32)
        value = self.online(state).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            next_action = self.online(next_state).argmax(1)
            target = reward + self.cfg.rl_gamma * (1.0 - done) * self.target(next_state).gather(
                1, next_action[:, None]
            ).squeeze(1)
        loss = nn.functional.smooth_l1_loss(value, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def update_dual(self, outage: float) -> None:
        self.dual = float(np.clip(self.dual + self.cfg.dual_lr * (outage - self.cfg.outage_target), 0.0, 80.0))


def build_state(
    cfg: Config,
    score: np.ndarray,
    active: np.ndarray,
    commanded: np.ndarray,
    readiness: np.ndarray,
    requirements: np.ndarray,
    velocity: np.ndarray,
    last_outage: float,
) -> np.ndarray:
    return np.concatenate(
        (
            score,
            active.astype(float),
            commanded.astype(float),
            readiness / max(cfg.horizon, 1),
            requirements / max(cfg.demand_levels_mbps),
            np.clip(velocity, -1.0, 1.0),
            np.array([last_outage, cfg.wake_delay_slots / max(cfg.horizon, 1)]),
        )
    ).astype(np.float32)
