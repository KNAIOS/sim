from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import Config


def ranked_mask(score: np.ndarray, count: int) -> np.ndarray:
    count = int(np.clip(count, 1, len(score)))
    indices = np.argsort(score)[-count:]
    mask = np.zeros(len(score), dtype=bool)
    mask[indices] = True
    return mask


def utility_score(mean: np.ndarray, sigma: np.ndarray | None = None, z: float = 0.0) -> np.ndarray:
    conservative = mean if sigma is None else mean - z * sigma
    linear = 10.0 ** (conservative / 10.0)
    # Geometric mean prevents a subarray useful to only one user from dominating.
    return np.exp(np.mean(np.log(np.maximum(linear, 1e-30)), axis=1))


class QNetwork(nn.Module):
    def __init__(self, state_size: int, actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Replay:
    def __init__(self, capacity: int, seed: int):
        self.data: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, item: tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        self.data.append(item)

    def sample(self, size: int) -> tuple[torch.Tensor, ...]:
        index = self.rng.choice(len(self.data), size=size, replace=False)
        items = [self.data[int(i)] for i in index]
        s, a, r, ns, d = zip(*items)
        return (
            torch.as_tensor(np.asarray(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(np.asarray(ns), dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32),
        )


@dataclass
class DQNController:
    cfg: Config
    state_size: int
    seed: int

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.actions = self.cfg.subarrays - self.cfg.users + 1
        self.online = QNetwork(self.state_size, self.actions, self.cfg.rl_hidden)
        self.target = QNetwork(self.state_size, self.actions, self.cfg.rl_hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.rl_lr)
        self.replay = Replay(self.cfg.rl_replay, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.steps = 0

    def choose(self, state: np.ndarray, explore: bool) -> int:
        fraction = min(1.0, self.steps / max(self.cfg.train_episodes * self.cfg.episode_steps * 0.7, 1.0))
        epsilon = self.cfg.rl_epsilon_start + fraction * (self.cfg.rl_epsilon_end - self.cfg.rl_epsilon_start)
        if explore and self.rng.random() < epsilon:
            return int(self.rng.integers(self.actions))
        with torch.no_grad():
            q = self.online(torch.as_tensor(state[None], dtype=torch.float32))
        return int(q.argmax(dim=1).item())

    def active_count(self, action: int) -> int:
        return self.cfg.users + int(action)

    def learn(self, transition: tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        self.replay.add(transition)
        self.steps += 1
        if len(self.replay.data) < 128 or self.steps % 4:
            return
        s, a, r, ns, d = self.replay.sample(128)
        value = self.online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            next_action = self.online(ns).argmax(dim=1, keepdim=True)
            target = r + self.cfg.rl_gamma * (1.0 - d) * self.target(ns).gather(1, next_action).squeeze(1)
        loss = nn.functional.smooth_l1_loss(value, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        if self.steps % 400 == 0:
            self.target.load_state_dict(self.online.state_dict())


def make_state(cfg: Config, mean: np.ndarray, sigma: np.ndarray, previous: np.ndarray, switch_scale: float, speed: float, requirements: np.ndarray) -> np.ndarray:
    # Preserve absolute channel strength.  The former per-snapshot centering
    # removed the information needed to decide how many subarrays are required.
    gain_state = np.clip((mean + 90.0) / 15.0, -2.0, 2.0)
    uncertainty = np.clip(sigma / 3.0, 0.0, 2.0)
    demand = np.asarray(requirements, dtype=float) / 220.0
    return np.concatenate((gain_state.reshape(-1), uncertainty.reshape(-1), previous.astype(float), demand, [switch_scale / 3.0, speed / 4.0])).astype(np.float32)
