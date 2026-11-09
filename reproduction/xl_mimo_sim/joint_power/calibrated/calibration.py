"""Training-only simulator branches for a shared causal short-term predictor."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from future_vr.config import make_config
from joint_power.model import Environment, search_action
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from major_revision.budget_search import search as legacy_search

ROOT = Path('results/joint_power_20260922_phase2')
TRAIN, VALID = 2509220000, 2609220000


def branch(env):
    result = copy.copy(env)
    result.hw = copy.deepcopy(env.hw)
    result.tracker = copy.deepcopy(env.tracker)
    result.age = env.age.copy()
    result.observed = env.observed.copy()
    result.req = env.req.copy()
    result.records = []
    return result


def collect(stream, per_scenario):
    cfg = make_config('validation')
    features, targets, identities = [], [], []
    for scenario, spec in enumerate(SCENARIOS):
        for index in range(per_scenario):
            seed = stream+10000*scenario+index
            env = Environment(cfg, make_trace(cfg, seed, spec))
            rng = np.random.default_rng(seed)
            while env.t < cfg.episode_steps:
                c = env.candidates()
                for action, mask in enumerate(c.masks):
                    alternative = branch(env)
                    outcome = alternative.block(mask)
                    features.append(c.features[action].copy())
                    targets.append(outcome)
                    identities.append((scenario, index, env.t, action))
                family = index % 4
                if family == 0:
                    command = legacy_search(cfg, env.tracker, env.hw, env.req)
                elif family in (1, 2):
                    command = c.masks[search_action(c, stride=1 if family == 1 else 3)]
                else:
                    current = int(env.hw.command.sum())
                    count = int(np.clip(current+rng.integers(-2, 3), 1, cfg.subarrays))
                    if rng.random() < .25:
                        count = int(rng.integers(1, cfg.subarrays+1))
                    command = c.masks[count-1]
                env.block(command)
            print(f'calibration stream={stream}, scenario={scenario}, trace={index}', flush=True)
    return dict(x=np.array(features, np.float32), y=np.array(targets, np.float32),
                identity=np.array(identities, np.int32))


class Predictor(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def standardize(self, x):
        return ((x-self.mean)/self.std).clamp(-10, 10)

    def forward(self, x):
        residual = self.net(self.standardize(x))
        energy = (x[..., -13]+.2*torch.tanh(residual[..., 0])).clamp_min(.001)
        raw_risk = x[..., -11].clamp(.01, .99)
        risk = torch.sigmoid(torch.logit(raw_risk)+residual[..., 1])
        return torch.stack((energy, risk), dim=-1)

    def predict(self, features):
        with torch.no_grad():
            return self(torch.as_tensor(features)).numpy()


def scores(prediction, targets):
    p, y = np.asarray(prediction), np.asarray(targets)
    risk = np.clip(p[:, 1], 1e-7, 1-1e-7)
    bce = float(np.mean(-y[:, 1]*np.log(risk)-(1-y[:, 1])*np.log1p(-risk)))
    mse = float(np.mean((p[:, 0]-y[:, 0])**2))
    return dict(objective=bce+10*mse, bce=bce, energy_mse=mse,
                brier=float(np.mean((p[:, 1]-y[:, 1])**2)),
                mean_prediction=float(p[:, 1].mean()), mean_actual=float(y[:, 1].mean()),
                mean_risk_error=float(p[:, 1].mean()-y[:, 1].mean()))


def bins(prediction, target):
    rows = []
    boundaries = np.array([0, .01, .025, .05, .1, .2, .4, .6, .8, 1.000001])
    index = np.digitize(prediction[:, 1], boundaries)-1
    for i in range(len(boundaries)-1):
        mask = index == i
        if mask.any():
            rows.append(dict(bin_low=float(boundaries[i]), bin_high=float(boundaries[i+1]),
                observations=int(mask.sum()), predicted=float(prediction[mask, 1].mean()),
                actual=float(target[mask, 1].mean())))
    return rows


def load_predictor():
    data = torch.load(ROOT/'calibration/selected.pt', weights_only=True)
    net = Predictor(data['state']['mean'], data['state']['std'])
    net.load_state_dict(data['state'])
    return net.eval()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026092203)
    ROOT.mkdir(parents=True, exist_ok=False)
    files = [*Path('joint_power').rglob('*.py'), *Path('future_vr').glob('*.py'),
             Path('major_revision/budget_search.py'), Path('joint_power/calibrated/PLAN.md'),
             Path('results/joint_power_20260922_phase1/output_hashes.json')]
    files += [Path('../ICECAI2026_4pages_20260922')/name for name in
              ('icecai2026_energy_rl.tex', 'output/pdf/clean.pdf', 'output/pdf/icecai2026_energy_rl.pdf')]
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        config=make_config('validation').as_dict(), host_environment_changed=False), indent=2)+'\n')
    folder = ROOT/'calibration'
    folder.mkdir()
    train, valid = collect(TRAIN, 16), collect(VALID, 4)
    np.savez_compressed(folder/'train.npz', **train)
    np.savez_compressed(folder/'validation.npz', **valid)
    x, y = torch.from_numpy(train['x']), torch.from_numpy(train['y'])
    model = Predictor(x.mean(0), x.std(0, unbiased=False).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    records = []
    best, selected, best_epoch = float('inf'), None, None
    for epoch in range(41):
        if epoch:
            for ids in torch.randperm(len(x)).split(256):
                prediction = model(x[ids])
                loss = nn.functional.binary_cross_entropy(prediction[:, 1], y[ids, 1])
                loss += 10*nn.functional.mse_loss(prediction[:, 0], y[ids, 0])
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if epoch % 10 == 0:
            row = dict(epoch=epoch, **scores(model.predict(valid['x']), valid['y']))
            records.append(row)
            if row['objective'] < best:
                best, selected, best_epoch = row['objective'], copy.deepcopy(model.state_dict()), epoch
            print(f'calibration validation: {row}', flush=True)
    model.load_state_dict(selected)
    prediction = model.predict(valid['x'])
    raw = valid['x'][:, [-13, -11]]
    calibrated_scores, raw_scores = scores(prediction, valid['y']), scores(raw, valid['y'])
    passed = (best_epoch > 0 and calibrated_scores['brier'] < raw_scores['brier']
              and abs(calibrated_scores['mean_risk_error']) <= .05 and np.isfinite(prediction).all())
    report = dict(passed=bool(passed), selected_epoch=best_epoch, calibrated=calibrated_scores,
        analytic=raw_scores, training_trajectories=48, validation_trajectories=12,
        training_branches=len(train['x']), validation_branches=len(valid['x']),
        scope='Trajectory-held-out prediction, not reliability guarantee or RL advantage')
    torch.save(dict(state=selected, selected_epoch=best_epoch), folder/'selected.pt')
    write_rows(folder/'validation_history.csv', records)
    write_rows(folder/'reliability_bins.csv', bins(prediction, valid['y']))
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    assert all(digest(p) == h for p, h in hashes.items())
    output = {str(p): digest(p) for p in folder.rglob('*') if p.is_file()}
    (ROOT/'calibration_hashes.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
