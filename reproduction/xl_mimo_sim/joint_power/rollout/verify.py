"""Reproduce selected controls and independently check antenna-domain physics."""
import csv
import json
from collections import defaultdict

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.rollout.absolute import ROOT
from joint_power.rollout.control import episode, input_checks, load_models
from joint_power.study import SCENARIOS, digest, make_trace


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def independent(records, trace):
    active, countdown = np.ones(16, bool), np.zeros(16, int)
    noise = 10**((-174+10*np.log10(40e6)+7-30)/10)
    errors = defaultdict(float)
    for t, row in enumerate(records):
        observed = int(active.sum())
        command = np.array([bool(row['command_bits'] & (1 << q)) for q in range(16)])
        off = ~command & (active | (countdown > 0))
        on = command & ~active & (countdown == 0)
        active[off], countdown[off], countdown[on] = False, 0, 4
        assert row['active_bits'] == sum(int(b) << q for q, b in enumerate(active))
        power = np.array([row[f'power_{k}'] for k in range(3)])
        assert (power >= 0).all() and power.sum() <= 1.2+1e-10
        h = trace.channel[t][:, np.repeat(active, 8)]
        beam = h.conj().T@np.linalg.inv(h@h.conj().T+3*noise/1.2*np.eye(3))
        beam /= np.maximum(np.linalg.norm(beam, axis=0, keepdims=True), 1e-15)
        received = abs(h@(beam*np.sqrt(power)[None]))**2
        desired = np.diag(received).copy()
        np.fill_diagonal(received, 0.)
        rate = 39.4*np.log2(1+desired/(noise+received.sum(1)))
        errors['rate_mbps'] = max(errors['rate_mbps'], float(abs(rate-[row[f'rate_{k}'] for k in range(3)]).max()))
        assert float(np.mean(rate < trace.requirements_mbps[t])) == row['outage']
        na, nw = int(active.sum()), int((countdown > 0).sum())
        processing = (8*na*8*9+8*27/3)*40e6/200/75e9
        circuit = .05*(8+2*na+1.2*nw+.05*(16-na-nw)+processing)
        pa, switching, pilot = .05*power.sum()/.4, .1*on.sum()+.01*off.sum(), .002*observed
        for name, value in (('circuit_j', circuit), ('pa_j', pa), ('switching_j', switching),
                            ('pilot_j', pilot), ('energy_j', circuit+pa+switching+pilot)):
            errors[name] = max(errors[name], abs(row[name]-value))
        ready = countdown == 1
        countdown[countdown > 0] -= 1
        active[ready] = True
    return dict(errors)


def main():
    torch.set_num_threads(1)
    input_checks()
    folder = ROOT/'validation'
    hashes = json.loads((folder/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    rows = read(folder/'selected_episodes.csv')
    all_rows = read(folder/'all_episodes.csv')
    assert len(rows) == 108 and len(all_rows) == 360
    identity = lambda r: (r['scenario'], r['trajectory'], r['method'], r['seed'], r['parameter'])
    assert len({identity(r) for r in rows}) == 108
    assert len({identity(r) for r in all_rows}) == 360
    models, _ = load_models()
    cfg, predictor = make_config('validation'), load_predictor()
    traces = {(s, i): make_trace(cfg, 3909220000+s*10000+i, spec)
              for s, spec in enumerate(SCENARIOS) for i in range(4)}
    errors = defaultdict(float)
    for index, row in enumerate(rows):
        s, i, seed = int(row['scenario']), int(row['trajectory']), int(row['seed'])
        metrics, records = episode(traces[s, i], row['method'], float(row['parameter']),
                                   predictor, models.get((row['method'], seed)))
        for k, value in metrics.items():
            assert abs(value-float(row[k])) < 1e-8, (identity(row), k)
        for k, value in independent(records, traces[s, i]).items():
            errors[k] = max(errors[k], value)
        if (index+1) % 18 == 0:
            print(f'control physical replay {index+1}/108', flush=True)
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for row in read(folder/'summary.csv'):
        group = [r for r in rows if (r['scenario'], r['method']) == (row['scenario'], row['method'])]
        for key in ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active'):
            assert abs(float(row[key])-np.mean([float(r[key]) for r in group])) < 1e-8
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    output = ROOT/'verified'
    output.mkdir(exist_ok=False)
    report = dict(selected_episodes=108, selected_slots=15120, candidate_validation_episodes=360,
        independent_validation_trajectories=12, maximum_errors=dict(errors), source_sha256=digest(__file__),
        validation_only=True, development_launched=False,
        gate=json.loads((folder/'gate.json').read_text()), frozen_sources_and_previous_outputs_preserved=True)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
