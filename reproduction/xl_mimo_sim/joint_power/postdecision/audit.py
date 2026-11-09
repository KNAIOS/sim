"""Independent antenna/hardware replay, paired statistics and hold audit."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from future_vr.config import make_config
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from joint_power.postdecision.study import ROOT


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return int(row['scenario']), int(row['trajectory']), row['method'], int(row['seed'])


def hold_statistics():
    folder = Path('results/joint_power_20260922_postdecision_diagnosis')
    rows = read(folder/'paired_events.csv')
    selected = [r for r in rows if int(r['horizon']) == 16]
    totals = np.zeros((2, 3, 4, 4))
    for r in selected:
        seed = (11, 22).index(int(r['seed']))
        s, i = int(r['scenario']), int(r['trajectory'])
        totals[seed, s, i] += [1, float(r['difference_energy_j']),
                              100*float(r['difference_outage']),
                              float(r['difference_energy_j']) < -1e-9 and float(r['difference_outage']) <= 1e-12]
    rng = np.random.default_rng(2026092213)
    seed_draw = rng.integers(0, 2, (4000, 2))
    trace_draw = rng.integers(0, 4, (4000, 3, 4))
    samples = totals[seed_draw[:, :, None, None], np.arange(3)[None, None, :, None], trace_draw[:, None]]
    sums = samples.sum((1, 2, 3))
    means = sums[:, 1:]/sums[:, :1]
    return dict(events=len(selected), independent_trajectories=12,
        ci95_energy_hold_minus_proposed_j=np.quantile(means[:, 0], [.025, .975]).tolist(),
        ci95_outage_hold_minus_proposed_pp=np.quantile(means[:, 1], [.025, .975]).tolist(),
        ci95_fraction_hold_saves_without_more_outage=np.quantile(means[:, 2], [.025, .975]).tolist(),
        scope='Overlapping conditional events clustered by whole shared trajectory, not policy confirmation')


def main():
    for name in ('manifest.json', 'output_hashes.json'):
        data = json.loads((ROOT/name).read_text())
        hashes = data.get('input_hashes', data)
        assert all(digest(p) == h for p, h in hashes.items()), name
    for previous in ('joint_power_20260922_phase1', 'joint_power_20260922_phase2b',
                     'joint_power_20260922_postdecision_diagnosis'):
        hashes = json.loads((Path('results')/previous/'output_hashes.json').read_text())
        assert all(digest(p) == h for p, h in hashes.items()), previous
    cfg = make_config('validation')
    noise = 10**((-174+10*np.log10(40e6)+7-30)/10)
    episodes = {key(r): r for r in read(ROOT/'evaluation/episodes.csv')}
    groups = defaultdict(list)
    for row in read(ROOT/'evaluation/slots.csv'):
        groups[key(row)].append(row)
    assert len(groups) == len(episodes) == 168
    traces = {(s, i): make_trace(cfg, 3409220000+s*10000+i, spec)
              for s, spec in enumerate(SCENARIOS) for i in range(4)}
    errors, wakes = defaultdict(float), []
    for identity, records in groups.items():
        records.sort(key=lambda r: int(r['slot']))
        assert [int(r['slot']) for r in records] == list(range(140))
        trace = traces[identity[:2]]
        active, countdown, pending = np.ones(16, bool), np.zeros(16, int), np.zeros(16, bool)
        energy = outage = 0.
        n_wakes = n_unserved = n_served = 0
        for t, row in enumerate(records):
            observed = int(active.sum())
            mask = np.array([bool(int(row['command_bits']) & 1 << q) for q in range(16)])
            off = ~mask & (active | (countdown > 0))
            on = mask & ~active & (countdown == 0)
            n_unserved += int((off & pending).sum())
            pending[off], pending[on] = False, True
            n_wakes += int(on.sum())
            active[off], countdown[off], countdown[on] = False, 0, 4
            n_served += int((pending & active).sum())
            pending[active] = False
            na, nw = int(active.sum()), int((countdown > 0).sum())
            assert int(row['active_bits']) == sum(int(b) << q for q, b in enumerate(active))
            assert (int(row['active_count']), int(row['waking_count']), int(row['observed_count'])) == (na, nw, observed)
            power = np.array([float(row[f'power_{k}']) for k in range(3)])
            assert np.isfinite(power).all() and (power >= 0).all() and power.sum() <= 1.2+1e-10
            h = trace.channel[t][:, np.repeat(active, 8)]
            beam = h.conj().T@np.linalg.inv(h@h.conj().T+3*noise/1.2*np.eye(3))
            beam /= np.maximum(np.linalg.norm(beam, axis=0, keepdims=True), 1e-15)
            received = abs(h@(beam*np.sqrt(power)[None]))**2
            desired = np.diag(received).copy()
            np.fill_diagonal(received, 0.)
            rate = 39.4*np.log2(1+desired/(noise+received.sum(1)))
            saved = np.array([float(row[f'rate_{k}']) for k in range(3)])
            errors['rate_mbps'] = max(errors['rate_mbps'], float(abs(rate-saved).max()))
            failed = float(np.mean(rate < trace.requirements_mbps[t]))
            assert failed == float(row['outage']), (identity, t)
            processing = (8*na*8*9+8*27/3)*40e6/200/75e9
            circuit = .05*(8+2*na+1.2*nw+.05*(16-na-nw)+processing)
            pa, switching, pilot = .05*power.sum()/.4, .1*on.sum()+.01*off.sum(), .002*observed
            total = circuit+pa+switching+pilot
            for name, value in [('energy_j', total), ('circuit_j', circuit), ('pa_j', pa),
                                ('switching_j', switching), ('pilot_j', pilot)]:
                errors[name] = max(errors[name], abs(float(row[name])-float(value)))
            energy += total
            outage += failed/140
            ready = countdown == 1
            countdown[countdown > 0] -= 1
            active[ready] = True
        assert n_wakes == n_unserved+n_served+int(pending.sum())
        wakes.append(dict(scenario=identity[0], trajectory=identity[1], method=identity[2], seed=identity[3],
            wakes=n_wakes, closed_before_data=n_unserved, served=n_served, horizon_censored=int(pending.sum())))
        errors['episode_energy_j'] = max(errors['episode_energy_j'], abs(energy-float(episodes[identity]['energy_j'])))
        assert abs(outage-float(episodes[identity]['outage'])) < 1e-12
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for r in read(ROOT/'evaluation/summary.csv'):
        group = [v for identity, v in episodes.items() if identity[0] == int(r['scenario']) and identity[2] == r['method']]
        for metric in ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active'):
            assert abs(float(r[metric])-np.mean([float(v[metric]) for v in group])) < 1e-10
    rng = np.random.default_rng(2026092214)
    seeds = rng.integers(0, 2, (4000, 2))
    trajectories = rng.integers(0, 4, (4000, 4))
    paired = []
    for s in range(3):
        for baseline in ('PRS power', 'PDS-one-step', 'Previous RL', 'Calibrated full'):
            for metric, factor in [('energy_j', 1.), ('outage', 100.)]:
                diff = np.array([[factor*(float(episodes[s, i, 'PDS-branch', seed][metric])-
                    float(episodes[s, i, baseline, seed if baseline in ('PDS-one-step', 'Previous RL') else 0][metric]))
                    for i in range(4)] for seed in (11, 22)])
                boot = diff[seeds[:, :, None], trajectories[:, None]].mean((1, 2))
                lo, hi = np.quantile(boot, [.025, .975])
                paired.append(dict(scenario=s, baseline=baseline, metric=metric,
                    mean=float(diff.mean()), ci95_low=float(lo), ci95_high=float(hi)))
    folder = ROOT/'verified'
    folder.mkdir(exist_ok=False)
    write_rows(folder/'paired.csv', paired)
    write_rows(folder/'wake_episodes.csv', wakes)
    report = dict(episodes=len(episodes), independent_trajectories=12,
        slots=sum(len(v) for v in groups.values()), errors=dict(errors), hold_diagnostic=hold_statistics(),
        outcome=json.loads((ROOT/'evaluation/screen.json').read_text()), source_sha256=digest(__file__),
        old_sources_preserved=True, scope='Physical accounting and descriptive evidence, not a scientific success certificate')
    (folder/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
