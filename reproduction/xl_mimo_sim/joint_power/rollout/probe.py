"""One Monte Carlo policy-improvement probe, never uses future at inference."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from future_vr.config import make_config
from joint_power.model import slot_energy
from joint_power.rollout.analyze import intact, load
from joint_power.rollout.collect import ROOT
from joint_power.study import digest, write_rows

PRICES = (4., 8., 16.)
FULL_SLOT = float(slot_energy(make_config('validation'), 16, 0, 1.2, 0, 16)[0])


class DifferenceValue(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, reference):
        values = self.net(((features-self.mean)/self.std).clamp(-10, 10))
        anchor = values.gather(1, reference[:, None, None].expand(-1, 1, 2))
        return values-anchor


def select(prediction, reference, price):
    costs = prediction[..., 0]+price*prediction[..., 1]
    action = costs.argmin(-1)
    return np.where(costs[np.arange(len(costs)), action] < -1e-8, action, reference)


def labels(data):
    y, reference = data['y'][:, :, -1], data['reference']
    target = y-y[np.arange(len(y)), reference, None]
    return (target/[FULL_SLOT, 3.]).astype(np.float32)


def evaluate_probe(prediction, data, epoch, method):
    target = labels(data)
    huber = float(nn.functional.smooth_l1_loss(torch.from_numpy(prediction), torch.from_numpy(target)))
    options, rows = [], []
    for price in PRICES:
        action = select(prediction, data['reference'], price)
        delta = target[np.arange(len(target)), action]*[FULL_SLOT, 3.]
        means = []
        for s in range(3):
            use = data['identity'][:, 0] == s
            means.append(delta[use].mean(0))
            rows.append(dict(method=method, epoch=epoch, price=price, scenario=s,
                energy_delta_j=float(delta[use, 0].mean()), failures_delta=float(delta[use, 1].mean()),
                changed_fraction=float(np.mean(action[use] != data['reference'][use])), huber=huber))
        key = (max(0., max(m[1] for m in means)), float(delta[:, 0].mean()), huber)
        options.append((key, price))
    return min(options), rows


def ridge_prediction(train, valid):
    mean = train['x'].mean((0, 1))
    std = np.maximum(train['x'].std((0, 1)), .05)
    def transform(data):
        x = np.clip((data['x']-mean)/std, -10, 10).astype(float)
        return x-x[np.arange(len(x)), data['reference'], None]
    x, y = transform(train), labels(train).astype(float)
    a, b = x.reshape(-1, x.shape[-1]), y.reshape(-1, 2)
    weights = np.linalg.solve(a.T@a+np.eye(a.shape[1]), a.T@b)
    return (transform(valid)@weights).astype(np.float32), dict(mean=mean, std=std, weights=weights)


def main():
    torch.set_num_threads(1)
    intact()
    gate = json.loads((ROOT/'diagnosis/report.json').read_text())
    if not gate['passed_for_probe']:
        raise RuntimeError('Diagnostic gate failed; do not fit this probe')
    output = ROOT/'probe'
    output.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/rollout/analyze.py'),
             ROOT/'train.npz', ROOT/'validation.npz', ROOT/'diagnosis/report.json']
    hashes = {str(p): digest(p) for p in files}
    (output/'manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    train, valid = load('train'), load('validation')
    x = torch.from_numpy(train['x'])
    refs, target = torch.from_numpy(train['reference']), torch.from_numpy(labels(train))
    vx, vr = torch.from_numpy(valid['x']), torch.from_numpy(valid['reference'])
    mean, std = x.mean((0, 1)), x.std((0, 1), unbiased=False).clamp_min(.05)
    rows, selections = [], []
    ridge, payload = ridge_prediction(train, valid)
    np.savez_compressed(output/'ridge.npz', **payload)
    option, scores = evaluate_probe(ridge, valid, 0, 'Ridge')
    rows.extend(scores)
    selections.append(dict(method='Ridge', key=option[0], price=option[1], epoch=0))
    for seed in (11, 22):
        torch.manual_seed(seed)
        model = DifferenceValue(mean, std)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        best_key, best, selected = None, None, None
        history = []
        for epoch in range(81):
            losses = []
            if epoch:
                model.train()
                for ids in torch.randperm(len(x)).split(16):
                    prediction = model(x[ids], refs[ids])
                    loss = nn.functional.smooth_l1_loss(prediction, target[ids])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite Monte Carlo probe loss')
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch, training_huber=float(np.mean(losses))))
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    prediction = model(vx, vr).numpy()
                (key, price), scores = evaluate_probe(prediction, valid, epoch, f'MC seed{seed}')
                rows.extend(scores)
                torch.save(dict(state=model.state_dict(), price=price, epoch=epoch),
                           output/f'seed{seed}_epoch{epoch:03d}.pt')
                if best_key is None or key < best_key:
                    best_key, best = key, copy.deepcopy(model.state_dict())
                    selected = dict(method='MC', seed=seed, epoch=epoch, price=price, key=key)
                print(f'MC probe seed={seed} epoch={epoch} validation={key} selected={selected["epoch"]}', flush=True)
        torch.save(dict(state=best, **selected), output/f'seed{seed}_selected.pt')
        write_rows(output/f'seed{seed}_history.csv', history)
        selections.append(selected)
    write_rows(output/'validation.csv', rows)
    (output/'selection.json').write_text(json.dumps(selections, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file() and p.name != 'output_hashes.json'}, indent=2)+'\n')
    print(json.dumps(selections, indent=2))


if __name__ == '__main__':
    main()
