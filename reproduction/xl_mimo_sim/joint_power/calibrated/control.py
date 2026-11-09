"""Price-conditioned multi-step DDQN with an action-independent value tail."""
import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

PRICES = (1., 2., 4., 8.)
GAMMA = .98


@dataclass
class Decision:
    masks: np.ndarray
    x: np.ndarray
    immediate: np.ndarray
    tail: float


def decision(candidates, predictor, remaining_steps):
    features = torch.from_numpy(candidates.features)
    with torch.no_grad():
        immediate = predictor(features)
        x = torch.cat((predictor.standardize(features), immediate), dim=-1)
    remaining_blocks = (remaining_steps+3)//4
    tail = sum(GAMMA**n for n in range(1, remaining_blocks))
    return Decision(candidates.masks, x.numpy(), immediate.numpy(), tail)


class ValueNetwork(nn.Module):
    def __init__(self, width=231, context_width=216):
        super().__init__()
        self.context_width = context_width
        def mlp(inputs):
            return nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        self.value = mlp(context_width+1)
        self.advantage = mlp(width+1)
        for head in (self.value, self.advantage):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.constant_(self.value[-1].bias, .6)

    def forward(self, x, immediate, tail, price):
        context = torch.cat((x[..., 0, :self.context_width], price[..., None]/8), dim=-1)
        value = self.value(context).squeeze(-1)
        price_actions = (price[..., None, None]/8).expand(*x.shape[:-1], 1)
        advantage = self.advantage(torch.cat((x, price_actions), dim=-1)).squeeze(-1)
        advantage = advantage-advantage.mean(-1, keepdim=True)
        instant = immediate[..., 0]+price[..., None]*immediate[..., 1]
        return instant+(tail*value)[..., None]+advantage


def pack(items):
    return (torch.from_numpy(np.stack([s.x for s in items])),
            torch.from_numpy(np.stack([s.immediate for s in items])),
            torch.tensor([s.tail for s in items], dtype=torch.float32))


class Learner:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.online = ValueNetwork()
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=.0004, weight_decay=.00001)
        self.memory = deque(maxlen=5000)
        self.pending = deque()
        self.updates = 0

    def choose(self, state, price, epsilon=0.):
        with torch.no_grad():
            values = self.online(torch.from_numpy(state.x), torch.from_numpy(state.immediate),
                                 torch.tensor(state.tail), torch.tensor(price))
        action = int(values.argmin())
        if self.rng.random() < epsilon:
            if self.rng.random() < .25:
                return self.rng.randrange(len(state.masks))
            count = int(state.masks[action].sum())
            nearby = [i for i, mask in enumerate(state.masks) if abs(int(mask.sum())-count) <= 2]
            nearby = sorted(set(nearby+[len(state.masks)-1]))
            return self.rng.choice(nearby)
        return action

    def remember(self, state, action, costs, following, done):
        self.pending.append((state, action, np.asarray(costs), following, done))
        while len(self.pending) >= 3 or (done and self.pending):
            horizon = min(3, len(self.pending))
            segment = list(self.pending)[:horizon]
            returns = sum(GAMMA**i * item[2] for i, item in enumerate(segment))
            discount = 0. if segment[-1][4] else GAMMA**horizon
            self.memory.append((segment[0][0], segment[0][1], returns,
                                segment[-1][3], discount))
            self.pending.popleft()
        losses = []
        if len(self.memory) >= 64:
            for _ in range(2):
                losses.append(self.update())
        return float(np.mean(losses)) if losses else 0.

    def update(self):
        batch = self.rng.sample(list(self.memory), 32)
        states, actions, costs, following, discounts = zip(*batch)
        price = torch.tensor([self.rng.choice(PRICES) for _ in batch])
        x, immediate, tail = pack(states)
        nx, ni, nt = pack(following)
        values = self.online(x, immediate, tail, price)
        estimate = values.gather(1, torch.tensor(actions)[:, None]).squeeze(1)
        costs = torch.from_numpy(np.asarray(costs, np.float32))
        discounts = torch.tensor(discounts, dtype=torch.float32)
        with torch.no_grad():
            next_actions = self.online(nx, ni, nt, price).argmin(1)
            continuation = self.target(nx, ni, nt, price).gather(1, next_actions[:, None]).squeeze(1)
            target = costs[:, 0]+price*costs[:, 1]+discounts*continuation
        loss = nn.functional.smooth_l1_loss(estimate, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; retain failed run')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())


def calibrated_search(state, threshold, stride=1):
    ids = sorted(set(range(0, len(state.masks)-1, stride)) |
                 {len(state.masks)-2, len(state.masks)-1})
    feasible = [i for i in ids if state.immediate[i, 1] <= threshold]
    if feasible:
        return min(feasible, key=lambda i: (state.immediate[i, 0], i))
    return min(ids, key=lambda i: (state.immediate[i, 1], state.immediate[i, 0], i))
