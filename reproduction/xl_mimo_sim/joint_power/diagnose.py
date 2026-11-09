"""Post-screen accounting and selected-action forecast audit; no policy tuning.

Replays frozen nominal development commands. This is retrospective diagnosis,
not a new independent test or a causal attribution of prediction error.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.study import ROOT, SCENARIOS, DEVELOPMENT, make_trace, write_rows


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    hashes = json.loads((ROOT/'output_hashes.json').read_text())
    for path, expected in manifest['input_hashes'].items() | hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    summary = {r['method']: r for r in read(ROOT/'evaluation/summary.csv')
               if r['scenario'] == '1'}
    rl = summary['Joint RL']
    accounting = []
    for name, row in summary.items():
        accounting.append(dict(method=name,
            episode_cost_surrogate=float(row['energy_j'])/float(summary['Full fixed']['energy_j'])
                                   + 8*float(row['outage']),
            **{f'rl_minus_baseline_{k}': float(rl[k])-float(row[k]) for k in
               ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')},
            rl_minus_baseline_outage_pp=100*(float(rl['outage'])-float(row['outage']))))

    groups = defaultdict(list)
    for row in read(ROOT/'evaluation/slots.csv'):
        if row['scenario'] == '1' and row['method'].startswith('Joint'):
            groups[int(row['trajectory']), row['method'], int(row['seed'])].append(row)
    cfg = make_config('validation')
    blocks = []
    for (trajectory, method, seed), records in groups.items():
        records.sort(key=lambda r: int(r['slot']))
        trace = make_trace(cfg, DEVELOPMENT+10000+trajectory, SCENARIOS[1])
        env = Environment(cfg, trace)
        for start in range(0, cfg.episode_steps, 4):
            candidates = env.candidates()
            bits = int(records[start]['command_bits'])
            assert all(int(r['command_bits']) == bits for r in records[start:start+4])
            mask = np.array([bool(bits & 1 << q) for q in range(cfg.subarrays)])
            matches = np.flatnonzero(np.all(candidates.masks == mask, axis=1))
            assert len(matches), (trajectory, method, seed, start)
            # The frozen 13 action features are eb[2], cb[2], pb[2], nb[2],
            # shortfall, on, off, count, hold. Audit only cb[0]: the executed
            # four-slot interval, NOT the eight-slot hold forecast against a
            # trajectory that changes command after four slots.
            predicted = float(candidates.features[matches[0], -13+2])
            _, actual = env.block(mask)
            saved = records[start:start+4]
            assert abs(actual-np.mean([float(r['outage']) for r in saved])) < 1e-12
            assert abs(sum(r['energy_j'] for r in env.records[-4:])-
                       sum(float(r['energy_j']) for r in saved)) < 1e-8
            blocks.append(dict(method=method, seed=seed, trajectory=trajectory,
                slot=start, commanded_count=int(mask.sum()),
                predicted_outage=predicted, realized_outage=float(actual)))
        print(f'forecast replay: trajectory={trajectory}, {method}, seed={seed}', flush=True)

    calibration = []
    for method in ('Joint full', 'Joint coarse', 'Joint RL'):
        rows = [r for r in blocks if r['method'] == method]
        subsets = {'all': rows,
                   'command_1_to_3': [r for r in rows if r['commanded_count'] <= 3],
                   'command_4_to_16': [r for r in rows if r['commanded_count'] >= 4]}
        for label, selected in subsets.items():
            if not selected:
                continue
            predicted = np.array([r['predicted_outage'] for r in selected])
            actual = np.array([r['realized_outage'] for r in selected])
            calibration.append(dict(method=method, subset=label, decision_intervals=len(selected),
                predicted_mean_outage=float(predicted.mean()),
                realized_mean_outage=float(actual.mean()),
                mean_absolute_error=float(abs(actual-predicted).mean()),
                zero_forecast_intervals=int(np.count_nonzero(predicted == 0)),
                zero_forecast_but_failure_intervals=int(np.count_nonzero((predicted == 0)&(actual > 0)))))
    folder = ROOT/'diagnosis'
    folder.mkdir(exist_ok=False)
    write_rows(folder/'nominal_accounting.csv', accounting)
    write_rows(folder/'nominal_forecast_blocks.csv', blocks)
    write_rows(folder/'nominal_forecast_summary.csv', calibration)
    (folder/'audit.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_inputs_and_outputs_unchanged=True, replayed_episodes=len(groups),
        decision_intervals=len(blocks), policy_modified=False,
        scope='Retrospective selected-action diagnosis, not counterfactual causation or confirmation'
    ), indent=2)+'\n')
    print(json.dumps(calibration, indent=2))


if __name__ == '__main__':
    main()
