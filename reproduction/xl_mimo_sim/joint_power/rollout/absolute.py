"""Bounded absolute failure-count critic; no safety/calibration guarantee."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from joint_power.rollout.analyze import intact, load
from joint_power.rollout.collect import ROOT as DATA
from joint_power.rollout.probe import DifferenceValue, FULL_SLOT
from joint_power.study import digest, write_rows

ROOT = Path('results/joint_power_20260922_rollout_absolute')


class AbsoluteValue(DifferenceValue):
    def __init__(self, mean, std):
        super().__init__(mean, std)
        nn.init.constant_(self.net[-1].bias[1], float(np.log(.01/.99)))

    def components(self, x, reference, remaining):
        raw = self.net(((x-self.mean)/self.std).clamp(-10, 10))
        ref_energy = raw[..., 0].gather(1, reference[:, None])
        energy = raw[..., 0]-ref_energy
        count = 3*remaining[:, None]*torch.sigmoid(raw[..., 1])
        return energy, count, raw[..., 1]

    def forward(self, x, reference, remaining):
        energy, count, _ = self.components(x, reference, remaining)
        difference = count-count.gather(1, reference[:, None])
        return torch.stack((energy, difference/3), -1)


def loss(model, x, reference, remaining, outcome):
    energy, count, logits = model.components(x, reference, remaining)
    delta = (outcome[..., 0]-outcome[..., 0].gather(1, reference[:, None]))/FULL_SLOT
    capacity = 3*remaining[:, None]
    rate = outcome[..., 1]/capacity
    risk_loss = (capacity*nn.functional.binary_cross_entropy_with_logits(logits, rate, reduction='none')).mean()
    return nn.functional.smooth_l1_loss(energy, delta)+risk_loss


def main():
    torch.set_num_threads(1)
    intact()
    ROOT.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/rollout/CONTROL_PLAN.md'),
             DATA/'train.npz', DATA/'validation.npz', DATA/'probe/output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    train, valid = load('train'), load('validation')
    def tensors(data):
        return (torch.from_numpy(data['x']), torch.from_numpy(data['reference']),
                torch.tensor(140-data['identity'][:, 2], dtype=torch.float32),
                torch.tensor(data['y'][:, :, -1], dtype=torch.float32))
    x, ref, remaining, y = tensors(train)
    vx, vr, vt, vy = tensors(valid)
    mean, std = x.mean((0, 1)), x.std((0, 1), unbiased=False).clamp_min(.05)
    history, selected_rows = [], []
    for seed in (11, 22):
        torch.manual_seed(seed)
        model = AbsoluteValue(mean, std)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        best, best_epoch, best_loss = None, None, float('inf')
        for epoch in range(81):
            if epoch:
                model.train()
                for ids in torch.randperm(len(x)).split(16):
                    value = loss(model, x[ids], ref[ids], remaining[ids], y[ids])
                    if not torch.isfinite(value):
                        raise RuntimeError('Nonfinite absolute count critic loss')
                    optimizer.zero_grad()
                    value.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    value = float(loss(model, vx, vr, vt, vy))
                torch.save(dict(state=model.state_dict(), epoch=epoch), ROOT/f'seed{seed}_epoch{epoch:03d}.pt')
                if value < best_loss:
                    best, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epoch, value
                history.append(dict(seed=seed, epoch=epoch, validation_prediction_loss=value))
                print(f'absolute count seed={seed} epoch={epoch} loss={value:.5f} selected={best_epoch}', flush=True)
        torch.save(dict(state=best, epoch=best_epoch), ROOT/f'seed{seed}_selected.pt')
        selected_rows.append(dict(seed=seed, epoch=best_epoch, validation_prediction_loss=best_loss))
    write_rows(ROOT/'validation_prediction.csv', history)
    (ROOT/'training_complete.json').write_text(json.dumps(selected_rows, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'training_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
