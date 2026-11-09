"""Validation-only histogram breakpoint selection; preserve failed screen."""
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from joint_power.calibrated.histogram import ROOT, load_predictor
from joint_power.calibrated.study import episode, selection_key, VALID, DEVELOP


def main():
    torch.set_num_threads(1)
    hashes = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    folder = ROOT/'threshold_audit'
    folder.mkdir(exist_ok=False)
    predictor, cfg = load_predictor(), make_config('validation')
    grid = sorted({0., .02, .05, .10} | {float(v) for v in predictor.table if 0 <= v <= .10})
    plan = Path('joint_power/calibrated/THRESHOLD_AUDIT_PLAN.md')
    (folder/'manifest.json').write_text(json.dumps(dict(grid=grid,
        plan_sha256=digest(plan), source_sha256=digest(__file__),
        predictor_sha256=digest(ROOT/'calibration/selected.pt'),
        scope='Post-screen diagnosis; no independent confirmation'), indent=2)+'\n')
    choices, validation = {}, []
    for method in ('Calibrated full', 'Calibrated coarse'):
        options = []
        for threshold in grid:
            rows = []
            for scenario, spec in enumerate(SCENARIOS):
                for i in range(4):
                    trace = make_trace(cfg, VALID+scenario*10000+i, spec)
                    metrics, _ = episode(trace, method, predictor, parameter=threshold)
                    rows.append(dict(method=method, threshold=threshold, scenario=scenario, trajectory=i, **metrics))
            key = selection_key(rows)
            validation.extend(rows)
            options.append((key, threshold))
            print(f'grid audit {method} threshold={threshold:.6f} validation={key}', flush=True)
        key, threshold = min(options)
        choices[method] = dict(threshold=threshold, validation_key=key)
    write_rows(folder/'validation.csv', validation)
    (folder/'selected.json').write_text(json.dumps(choices, indent=2)+'\n')
    # Selection above is complete before any development replay begins.
    development, slots, summary = [], [], []
    for method, choice in choices.items():
        for scenario, spec in enumerate(SCENARIOS):
            group = []
            for i in range(4):
                seed = DEVELOP+scenario*10000+i
                trace = make_trace(cfg, seed, spec)
                metrics, records = episode(trace, method, predictor, parameter=choice['threshold'])
                identity = dict(method=method, threshold=choice['threshold'], scenario=scenario,
                                trajectory=i, trajectory_seed=seed)
                group.append(dict(**identity, **metrics))
                slots.extend(dict(**identity, **r) for r in records)
            development.extend(group)
            summary.append(dict(method=method, scenario=scenario, threshold=choice['threshold'],
                **{k: float(np.mean([r[k] for r in group])) for k in
                   ('energy_j', 'outage', 'mean_active', 'switching_j')}))
    write_rows(folder/'episodes.csv', development)
    write_rows(folder/'slots.csv', slots)
    write_rows(folder/'summary.csv', summary)
    assert all(digest(p) == h for p, h in hashes.items())
    outputs = {str(p): digest(p) for p in folder.rglob('*') if p.is_file()}
    (folder/'output_hashes.json').write_text(json.dumps(outputs, indent=2)+'\n')
    print(json.dumps(dict(choices=choices, summary=summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
