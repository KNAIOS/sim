"""Independent physical replay and trajectory-grouped uncertainty reporting."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from joint_power.calibrated.histogram import ROOT, load_predictor


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def audit_calibration():
    with np.load(ROOT/'calibration/validation.npz') as data:
        x, y, identity = data['x'], data['y'], data['identity']
    prediction = load_predictor().predict(x)
    errors = (prediction[:, 1]-y[:, 1])**2-(x[:, -11]-y[:, 1])**2
    groups = [errors[(identity[:, 0] == s)&(identity[:, 1] == i)].mean()
              for s in range(3) for i in range(4)]
    rng = np.random.default_rng(2026092206)
    sample = rng.integers(0, 4, (4000, 3, 4))
    values = np.asarray(groups).reshape(3, 4)
    boot = values[np.arange(3)[None, :, None], sample].mean((1, 2))
    return dict(validation_trajectories=12, brier_difference=float(np.mean(groups)),
        trajectory_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
        caveat='Validation selection, descriptive only; no universal probability calibration claim')


def identity(row):
    return int(row['scenario']), int(row['trajectory']), row['method'], int(row['seed'])


def main():
    torch.set_num_threads(1)
    for prior in ('joint_power_20260922_phase1', 'joint_power_20260922_phase2'):
        old_root = Path('results')/prior
        for name in ('manifest.json', 'output_hashes.json', 'calibration_hashes.json'):
            path = old_root/name
            if path.exists():
                saved = json.loads(path.read_text())
                hashes = saved.get('input_hashes', saved)
                assert all(digest(p) == h for p, h in hashes.items()), str(path)
    for name in ('manifest.json', 'calibration_hashes.json', 'control_manifest.json', 'output_hashes.json'):
        data = json.loads((ROOT/name).read_text())
        hashes = data.get('input_hashes', data)
        assert all(digest(p) == h for p, h in hashes.items()), name
    cfg = make_config('validation')
    noise = 10**((-174+10*np.log10(40e6)+7-30)/10)
    episodes = {identity(r): r for r in read(ROOT/'evaluation/episodes.csv')}
    groups = defaultdict(list)
    for r in read(ROOT/'evaluation/slots.csv'):
        groups[identity(r)].append(r)
    assert len(groups) == len(episodes) == 120
    traces = {(s, i): make_trace(cfg, 2909220000+s*10000+i, spec)
              for s, spec in enumerate(SCENARIOS) for i in range(4)}
    errors = defaultdict(float)
    for key, records in groups.items():
        records.sort(key=lambda r: int(r['slot']))
        assert [int(r['slot']) for r in records] == list(range(140))
        active, countdown = np.ones(16, bool), np.zeros(16, int)
        trace = traces[key[:2]]
        energy, outage = 0., 0.
        for t, row in enumerate(records):
            observed = int(active.sum())
            bits = int(row['command_bits'])
            command = np.array([bool(bits & 1 << q) for q in range(16)])
            off = ~command & (active | (countdown > 0))
            on = command & ~active & (countdown == 0)
            active[off], countdown[off], countdown[on] = False, 0, 4
            assert int(row['active_bits']) == sum(int(b) << q for q, b in enumerate(active))
            assert int(row['observed_count']) == observed
            na, nw = int(active.sum()), int((countdown > 0).sum())
            assert na == int(row['active_count']) and nw == int(row['waking_count'])
            power = np.array([float(row[f'power_{k}']) for k in range(3)])
            assert np.isfinite(power).all() and (power >= 0).all() and power.sum() <= 1.2+1e-10
            h = trace.channel[t][:, np.repeat(active, 8)]
            beam = h.conj().T @ np.linalg.inv(h@h.conj().T+3*noise/1.2*np.eye(3))
            beam /= np.maximum(np.linalg.norm(beam, axis=0, keepdims=True), 1e-15)
            received = abs(h@(beam*np.sqrt(power)[None]))**2
            desired = np.diag(received).copy()
            np.fill_diagonal(received, 0.)
            rates = 39.4*np.log2(1+desired/(noise+received.sum(1)))
            saved = np.array([float(row[f'rate_{k}']) for k in range(3)])
            errors['rate_mbps'] = max(errors['rate_mbps'], float(abs(rates-saved).max()))
            failure = float(np.mean(rates < trace.requirements_mbps[t]))
            assert failure == float(row['outage']), (key, t)
            processing = (8*na*8*9+8*27/3)*40e6/200/75e9
            circuit = .05*(8+2*na+1.2*nw+.05*(16-na-nw)+processing)
            pa, switching, pilot = .05*power.sum()/.4, .1*on.sum()+.01*off.sum(), .002*observed
            total = circuit+pa+switching+pilot
            for name, value in [('energy_j', total), ('circuit_j', circuit), ('pa_j', pa),
                                ('switching_j', switching), ('pilot_j', pilot)]:
                errors[name] = max(errors[name], abs(float(row[name])-float(value)))
            energy += total
            outage += failure/140
            ready = countdown == 1
            countdown[countdown > 0] -= 1
            active[ready] = True
        errors['episode_energy_j'] = max(errors['episode_energy_j'], abs(energy-float(episodes[key]['energy_j'])))
        assert abs(outage-float(episodes[key]['outage'])) < 1e-12
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for row in read(ROOT/'evaluation/summary.csv'):
        group = [r for key, r in episodes.items() if key[0] == int(row['scenario']) and key[2] == row['method']]
        for metric in ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active'):
            assert abs(float(row[metric])-np.mean([float(r[metric]) for r in group])) < 1e-10
    rng = np.random.default_rng(2026092207)
    seed_draw = rng.integers(0, 2, (4000, 2))
    trace_draw = rng.integers(0, 4, (4000, 4))
    paired = []
    for s in range(3):
        for baseline in ('Calibrated full', 'Calibrated coarse', 'PRS power', 'Myopic'):
            for metric, factor in [('energy_j', 1.), ('outage', 100.)]:
                diff = np.array([[factor*(float(episodes[s, i, 'RL', seed][metric])-
                                         float(episodes[s, i, baseline, 0][metric])) for i in range(4)] for seed in (11, 22)])
                boot = diff[seed_draw[:, :, None], trace_draw[:, None, :]].mean((1, 2))
                low, high = np.quantile(boot, [.025, .975])
                paired.append(dict(scenario=s, baseline=baseline, metric=metric,
                    difference=float(diff.mean()), ci95_low=float(low), ci95_high=float(high)))
    folder = ROOT/'verified'
    folder.mkdir(exist_ok=False)
    write_rows(folder/'paired.csv', paired)
    report = dict(episodes=len(episodes), independent_environment_trajectories=12,
        slots=sum(len(v) for v in groups.values()), errors=dict(errors),
        calibration=audit_calibration(), outcome=json.loads((ROOT/'evaluation/screen.json').read_text()),
        sources_preserved=True, audit_source_sha256=digest(__file__),
        scope='Physical accounting and descriptive statistics, not scientific completion')
    (folder/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
