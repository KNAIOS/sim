"""Matched reference-return critics; predictions are not reliability guarantees."""
import torch
from torch import nn

from joint_power.rollout.probe import FULL_SLOT


class ReturnCritic(nn.Module):
    def __init__(self, mean, std, frequency, joint=False):
        super().__init__()
        self.joint = bool(joint)
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).clone())
        if not bool(torch.all(self.std > 0)):
            raise ValueError('Normalization scale must be positive')
        # Construct the common trunk before the differently sized output heads.
        self.trunk = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(),
                                   nn.Linear(64, 64), nn.SiLU())
        self.head = nn.Linear(64, 2 if self.joint else 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        rate = torch.as_tensor(frequency).clamp(1e-4, 1-1e-4)
        with torch.no_grad():
            self.head.bias[0] = torch.logit(rate)

    def forward(self, x, reference, remaining):
        z = ((x-self.mean)/self.std).clamp(-10, 10)
        raw = self.head(self.trunk(z))
        logits = raw[..., 0]
        count = 3*remaining[:, None]*torch.sigmoid(logits)
        energy = None
        if self.joint:
            energy = raw[..., 1]-raw[..., 1].gather(1, reference[:, None])
        return count, logits, energy


def frequency(outcome, remaining):
    """Total labeled failed user-slots divided by total labeled capacity."""
    count = outcome[..., 1]
    capacity = (3*remaining[:, None]).expand_as(count)
    if bool(torch.any(count < 0)) or bool(torch.any(count > capacity)):
        raise ValueError('Failure counts outside physical capacity')
    return float((count.sum()/capacity.sum()).clamp(1e-4, 1-1e-4))


def losses(model, x, reference, remaining, outcome):
    _, logits, energy = model(x, reference, remaining)
    capacity = 3*remaining[:, None]
    risk = (capacity*nn.functional.binary_cross_entropy_with_logits(
        logits, outcome[..., 1]/capacity, reduction='none')).mean()
    auxiliary = logits.new_zeros(())
    if energy is not None:
        target = (outcome[..., 0]-outcome[..., 0].gather(1, reference[:, None]))/FULL_SLOT
        auxiliary = nn.functional.smooth_l1_loss(energy, target)
    return risk, auxiliary


def batches(roots, generator):
    # Dedicated RNG prevents output-head size from changing the training order.
    return torch.randperm(roots, generator=generator).split(16)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload['state']
    model = ReturnCritic(state['mean'], state['std'], payload['frequency'],
                         joint=payload['arm'] == 'joint')
    model.load_state_dict(state)
    return model.eval(), payload
