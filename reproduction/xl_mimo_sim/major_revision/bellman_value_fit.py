"""Bounded 100-epoch immediate risk and 15x5-epoch Bellman continuation fit."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.bellman_branch_data import ROOT, digest, known_costs
from major_revision.bellman_value import (
    BellmanValue, DEPTH, PRICES, continuation_target,
)


def verify_data(seed):
    report = json.loads((ROOT/'data_audit/audit.json').read_text())
    assert report['passed'] and (ROOT/'data_audit/COMPLETE').exists()
    for path, expected in report['input_hashes'].items():
        assert digest(path) == expected, path
    folder = ROOT/f'collect_seed{seed}'
    for key in ('source_hashes',):
        for path, expected in json.loads((folder/'manifest.json').read_text())[key].items():
            assert digest(path) == expected, path
    return np.load(folder/'transitions.npz', allow_pickle=False)


def initialize_bias(layer, values, energy_first=False):
    values = np.asarray(values, dtype=float)
    bias = np.log(np.clip(values, .001, .999)/(1-np.clip(values, .001, .999)))
    if energy_first:
        bias[0] = np.log(np.expm1(max(values[0], 1e-3)))
    with torch.no_grad():
        layer.bias.copy_(torch.tensor(bias, dtype=layer.bias.dtype))


def prepare_targets(previous, successor, depth, cfg, batch_size=128):
    """All labels use the same immutable preceding-depth network."""
    n, actions, features = successor.shape
    flat = successor.reshape(n*actions, features)
    targets = torch.empty((n*actions, len(PRICES), 2))
    previous.eval()
    with torch.no_grad():
        for p, price in enumerate(PRICES):
            for start in range(0, len(flat), batch_size):
                stop = min(start+batch_size, len(flat))
                value = previous(flat[start:stop], price, depth-1)
                targets[start:stop, p] = continuation_target(value, price, cfg.rl_gamma, depth)
    if not torch.isfinite(targets).all() or targets.min() < -1e-6:
        raise AssertionError('Invalid continuation target')
    if targets[..., 1].max() > 1+1e-6:
        raise AssertionError('Outage exceeds finite-horizon bound')
    return targets.reshape(n, actions, len(PRICES), 2)


def fit(seed):
    data = verify_data(seed)
    cfg = make_config('validation')
    folder = ROOT/f'fit_seed{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).with_name('bellman_value.py'),
             Path(__file__).with_name('test_bellman_value.py'),
             Path(__file__).with_name('BELLMAN_FIT_PLAN.md'),
             ROOT/'data_audit/audit.json', ROOT/f'collect_seed{seed}/transitions.npz']
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(seed=seed, threads=1,
        prices=PRICES, depth=DEPTH, risk_epochs=100, tail_epochs_per_depth=5,
        config=cfg.as_dict(), input_hashes=hashes, validation_used=False,
        role='bounded fit, no performance result'), indent=2))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BellmanValue(cfg)
    x = torch.from_numpy(data['features']).float()
    successor = torch.from_numpy(data['successors']).float()
    costs = torch.from_numpy(data['costs']).float()
    _, current = known_costs(cfg, data['features'])
    remaining_risk = (data['costs'][..., 1]-current/4)/.75
    # Remove only float32 conversion epsilon, not any substantive invalid label.
    assert remaining_risk.min() >= -5e-7 and remaining_risk.max() <= 1+5e-7
    risk_target = torch.from_numpy(np.clip(remaining_risk, 0, 1)).float()
    initialize_bias(model.risk.head[-1], [float(risk_target.mean())])
    initialize_bias(model.tail.head[-1], [float(costs[..., 0].mean()),
                                        float(costs[..., 1].mean())], energy_first=True)
    optimizer = torch.optim.AdamW(model.risk.parameters(), lr=4e-4, weight_decay=1e-5)
    diagnostics = []
    for epoch in range(1, 101):
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(order), 64):
            index = order[start:start+64]
            loss = nn.functional.binary_cross_entropy_with_logits(model.risk(x[index]), risk_target[index])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.risk.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        diagnostics.append(dict(stage='risk', depth=1, epoch=epoch, loss=float(np.mean(losses))))
        if epoch % 20 == 0:
            print('Bellman risk fit', seed, epoch, diagnostics[-1]['loss'], flush=True)
    model.risk.eval()
    model.risk.requires_grad_(False)
    frozen_risk = {k: v.clone() for k, v in model.risk.state_dict().items()}
    torch.save(dict(online=model.state_dict(), seed=seed, depth=1, risk_epochs=100), folder/'myopic.pt')
    pairs = len(x)*len(PRICES)
    price_values = torch.tensor(PRICES)
    for depth in range(2, DEPTH+1):
        previous = copy.deepcopy(model).eval()
        previous.requires_grad_(False)
        targets = prepare_targets(previous, successor, depth, cfg)
        frozen_previous = {k: v.clone() for k, v in previous.state_dict().items()}
        optimizer = torch.optim.AdamW(model.tail.parameters(), lr=4e-4, weight_decay=1e-5)
        for epoch in range(1, 6):
            losses = []
            order = rng.permutation(pairs)
            for start in range(0, pairs, 64):
                index = order[start:start+64]
                roots, prices = index//len(PRICES), index % len(PRICES)
                prediction = model.tail(x[roots], price_values[prices])
                label = targets[roots, :, prices, :]
                loss = nn.functional.mse_loss(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.tail.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            diagnostics.append(dict(stage='tail', depth=depth, epoch=epoch, loss=float(np.mean(losses))))
        for key, value in frozen_previous.items():
            assert torch.equal(value, previous.state_dict()[key])
        for key, value in frozen_risk.items():
            assert torch.equal(value, model.risk.state_dict()[key])
        print('Bellman tail fit', seed, 'depth', depth, '/', DEPTH,
              'loss', diagnostics[-1]['loss'], flush=True)
    model.eval()
    checkpoint = dict(online=model.state_dict(), seed=seed, depth=DEPTH,
                      prices=list(PRICES), risk_epochs=100, tail_epochs=5)
    torch.save(checkpoint, folder/'final.pt')
    reloaded = BellmanValue(cfg)
    reloaded.load_state_dict(torch.load(folder/'final.pt', weights_only=True)['online'], strict=True)
    reloaded.eval()
    with torch.no_grad():
        for price in PRICES:
            for depth in (1, DEPTH):
                torch.testing.assert_close(model(x[:32], price, depth),
                                           reloaded(x[:32], price, depth), rtol=0, atol=0)
    _write_rows(folder/'training.csv', diagnostics)
    audit = dict(seed=seed, roots=len(x), action_transitions=len(x)*cfg.subarrays,
                 risk_epochs=100, tail_depths=list(range(2, DEPTH+1)),
                 tail_epochs_per_depth=5, parameters=sum(p.numel() for p in model.parameters()),
                 risk_frozen=True, target_copies_unchanged=True, reload_predictions_exact=True,
                 validation_episodes=0, scientific_success_not_established=True)
    (folder/'audit.json').write_text(json.dumps(audit, indent=2))
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    (folder/'output_hashes.json').write_text(json.dumps(
        {str(p): digest(p) for p in folder.iterdir() if p.is_file()}, indent=2))
    (folder/'COMPLETE').write_text('Bounded fit complete; no closed-loop performance evidence.\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', choices=(11, 22), type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fit(args.seed)
