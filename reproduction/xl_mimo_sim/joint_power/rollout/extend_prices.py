"""Explicit post-hoc lower-price validation; original choices remain frozen."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.calibrated.study import selection_key
from joint_power.rollout.absolute import ROOT
from joint_power.rollout.control import episode, gate, input_checks, load_models
from joint_power.study import SCENARIOS, digest, make_trace, write_rows


def main():
    torch.set_num_threads(1)
    input_checks()
    old_hashes = json.loads((ROOT/'validation/output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in old_hashes.items())
    with (ROOT/'validation/all_episodes.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    for row in original:
        for k in row:
            if k != 'method':
                row[k] = int(row[k]) if k in ('scenario', 'trajectory', 'seed', 'numerical_fallbacks') else float(row[k])
    models, checkpoints = load_models()
    output = ROOT/'price_extension'
    output.mkdir(exist_ok=False)
    inputs = [Path(__file__), Path('joint_power/rollout/PRICE_EXTENSION_PLAN.md'),
              ROOT/'validation/output_hashes.json', ROOT/'training_hashes.json', Path('joint_power/rollout/control.py')]
    hashes = {str(p): digest(p) for p in inputs}
    (output/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        checkpoints=checkpoints, post_hoc_validation_only=True, prices=[0., .25, .5, 1., 2.]), indent=2)+'\n')
    predictor, cfg, extra = load_predictor(), make_config('validation'), []
    for s, spec in enumerate(SCENARIOS):
        for i in range(4):
            trace = make_trace(cfg, 3909220000+s*10000+i, spec)
            for (method, seed), model in models.items():
                for price in (0., .25, .5, 1., 2.):
                    metrics, _ = episode(trace, method, price, predictor, model)
                    extra.append(dict(scenario=s, trajectory=i, method=method, seed=seed, parameter=price, **metrics))
            print(f'price boundary check scenario={s} trajectory={i}', flush=True)
    assert len(extra) == 240
    write_rows(output/'extra_episodes.csv', extra)
    combined = original+extra
    selected, rows = [], []
    for method, seed in dict.fromkeys((r['method'], r['seed']) for r in combined):
        options = []
        for parameter in sorted({r['parameter'] for r in combined if (r['method'], r['seed']) == (method, seed)}):
            group = [r for r in combined if (r['method'], r['seed'], r['parameter']) == (method, seed, parameter)]
            options.append((selection_key(group), parameter, group))
        key, parameter, group = min(options, key=lambda v: v[:2])
        selected.append(dict(method=method, seed=seed, parameter=parameter, validation_key=key))
        rows.extend(group)
    write_rows(output/'selected_episodes.csv', rows)
    (output/'selection.json').write_text(json.dumps(selected, indent=2)+'\n')
    summary = []
    for s in range(3):
        for method in dict.fromkeys(r['method'] for r in rows):
            group = [r for r in rows if r['scenario'] == s and r['method'] == method]
            summary.append(dict(scenario=s, method=method, **{k: float(np.mean([r[k] for r in group]))
                for k in ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')}))
    write_rows(output/'summary.csv', summary)
    result = gate(rows, .01)
    result.update(post_hoc_validation=True, original_gate_preserved=True, development_launched=False)
    (output/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in old_hashes.items())
    assert all(digest(p) == h for p, h in hashes.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(dict(selected=selected, summary=summary, gate=result), indent=2))


if __name__ == '__main__':
    main()
