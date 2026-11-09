"""Transparent grouped risk correction; reuse analytic physical energy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from joint_power.calibrated.calibration import Predictor, collect, scores, bins
from joint_power.study import digest, write_rows

ROOT = Path('results/joint_power_20260922_phase2b')
OLD = Path('results/joint_power_20260922_phase2')


def groups(x):
    count = torch.round(x[..., -2]*16).long()
    bucket = (count >= 2).long()+(count >= 4).long()+(count >= 8).long()
    waking = ((x[..., 160:176] > .5) & (x[..., 144:160] < .5)).any(-1) | (x[..., -4] > 0)
    risk_bin = torch.round(x[..., -11]*12).long().clamp(0, 12)
    return (bucket*2+waking.long())*13+risk_bin, risk_bin


class HistogramPredictor(Predictor):
    def __init__(self, mean, std, table):
        super().__init__(mean, std)
        del self.net
        self.register_buffer('table', torch.as_tensor(table, dtype=torch.float32))

    def forward(self, x):
        index, _ = groups(x)
        return torch.stack((x[..., -13], self.table[index]), -1)


def fit(train, strength):
    x, y = torch.from_numpy(train['x']), train['y'][:, 1]
    group, risk = (v.numpy() for v in groups(x))
    global_sum = np.bincount(risk, weights=y, minlength=13)
    global_count = np.bincount(risk, minlength=13)
    global_mean = (global_sum+32*np.clip(np.arange(13)/12, .001, .999))/(global_count+32)
    group_sum = np.bincount(group, weights=y, minlength=104)
    group_count = np.bincount(group, minlength=104)
    table = (group_sum+strength*np.tile(global_mean, 8))/(group_count+strength)
    model = HistogramPredictor(x.mean(0), x.std(0, unbiased=False).clamp_min(.05), table)
    return model, group_count


def load_predictor():
    data = torch.load(ROOT/'calibration/selected.pt', weights_only=True)
    state = data['state']
    return HistogramPredictor(state['mean'], state['std'], state['table']).eval()


def calibrate():
    torch.set_num_threads(1)
    previous = json.loads((OLD/'calibration/report.json').read_text())
    assert not previous['passed']
    old_hashes = json.loads((OLD/'calibration_hashes.json').read_text())
    assert all(digest(p) == h for p, h in old_hashes.items())
    ROOT.mkdir(parents=True, exist_ok=False)
    folder = ROOT/'calibration'
    folder.mkdir()
    files = [*Path('joint_power/calibrated').glob('*.py'),
        Path('joint_power/calibrated/PLAN.md'), Path('joint_power/calibrated/HISTOGRAM_PLAN.md'),
        Path('joint_power/model.py'), OLD/'calibration_hashes.json', OLD/'calibration/train.npz']
    files += list(Path('future_vr').glob('*.py'))+[Path('major_revision/budget_search.py')]
    files += [Path('../ICECAI2026_4pages_20260922')/name for name in
              ('icecai2026_energy_rl.tex', 'output/pdf/clean.pdf', 'output/pdf/icecai2026_energy_rl.pdf')]
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        source_training_data=str(OLD/'calibration/train.npz'),
        host_environment_changed=False), indent=2)+'\n')
    with np.load(OLD/'calibration/train.npz') as data:
        train = {name: data[name] for name in data.files}
    valid = collect(3009220000, 4)
    np.savez_compressed(folder/'validation.npz', **valid)
    records, models = [], {}
    for strength in (32, 128, 512):
        model, counts = fit(train, strength)
        models[strength] = model
        records.append(dict(strength=strength, **scores(model.predict(valid['x']), valid['y'])))
    best = min(records, key=lambda r: (r['brier'], r['bce']))
    model = models[best['strength']]
    prediction = model.predict(valid['x'])
    raw = valid['x'][:, [-13, -11]]
    clipped = raw.copy()
    clipped[:, 1] = np.clip(clipped[:, 1], .01, .99)
    raw_score, clipped_score = scores(raw, valid['y']), scores(clipped, valid['y'])
    passed = (best['brier'] < min(raw_score['brier'], clipped_score['brier'])
              and abs(best['mean_risk_error']) <= .05 and np.isfinite(prediction).all())
    report = dict(passed=bool(passed), selected_strength=best['strength'], calibrated=best,
        analytic=raw_score, clipped_analytic=clipped_score, training_trajectories=48,
        validation_trajectories=12, training_branches=len(train['x']), validation_branches=len(valid['x']),
        scope='Shared histogram correction, not RL or distribution-free reliability evidence')
    write_rows(folder/'validation_history.csv', records)
    write_rows(folder/'reliability_bins.csv', bins(prediction, valid['y']))
    group, _ = (v.numpy() for v in groups(torch.from_numpy(valid['x'])))
    group_rows = []
    for i in range(104):
        use = group == i
        if use.any():
            group_rows.append(dict(group=i, train_intervals=int(counts[i]),
                validation_intervals=int(use.sum()), predicted=float(prediction[use, 1].mean()),
                actual=float(valid['y'][use, 1].mean())))
    write_rows(folder/'group_validation.csv', group_rows)
    torch.save(dict(state=model.state_dict(), strength=best['strength']), folder/'selected.pt')
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    assert all(digest(p) == h for p, h in hashes.items())
    outputs = {str(p): digest(p) for p in folder.rglob('*') if p.is_file()}
    (ROOT/'calibration_hashes.json').write_text(json.dumps(outputs, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def train_seed_histogram(seed):
    # Windows spawn imports modules afresh: bind the folder/loader in EACH worker.
    from joint_power.calibrated import study
    study.ROOT = ROOT
    study.load_predictor = load_predictor
    return study.train_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('calibrate', 'train', 'evaluate'))
    args = parser.parse_args()
    if args.phase == 'calibrate':
        calibrate()
    else:
        # Explicit configuration of the reusable driver; frozen files unchanged.
        from joint_power.calibrated import study
        study.ROOT = ROOT
        study.load_predictor = load_predictor
        study.train(worker=train_seed_histogram) if args.phase == 'train' else study.evaluate()


if __name__ == '__main__':
    main()
