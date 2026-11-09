"""Prespecified price selection; preserve previous controls on the same traces."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.collect import ROOT, PREVIOUS
from joint_power.aggregation.train import ARMS, intact
from joint_power.calibrated.histogram import load_predictor
from joint_power.calibrated.study import selection_key
from joint_power.rollout.absolute import AbsoluteValue
from joint_power.rollout.control import episode, gate
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

PRICES = (0., .25, .5, 1., 2., 4., 8., 16., 32.)
METRICS = ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')


def read_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in row:
            if key != 'method':
                row[key] = int(row[key]) if key in ('scenario', 'trajectory', 'seed', 'numerical_fallbacks') else float(row[key])
    return rows


def input_checks():
    intact()
    hashes = json.loads((ROOT/'training/output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())


def models():
    loaded, checkpoints = {}, {}
    for arm, stem in ARMS.items():
        for seed in (11, 22):
            path = ROOT/f'training/{stem}_seed{seed}_selected.pt'
            saved = torch.load(path, weights_only=True)
            model = AbsoluteValue(saved['state']['mean'], saved['state']['std'])
            model.load_state_dict(saved['state'])
            loaded[arm, seed] = model.eval()
            checkpoints[f'{arm} {seed}'] = dict(path=str(path), sha256=digest(path), epoch=saved['epoch'])
    return loaded, checkpoints


def select_options(rows):
    selected, selected_rows = [], []
    for method, seed in dict.fromkeys((r['method'], r['seed']) for r in rows):
        options = []
        for price in sorted({r['parameter'] for r in rows if (r['method'], r['seed']) == (method, seed)}):
            group = [r for r in rows if (r['method'], r['seed'], r['parameter']) == (method, seed, price)]
            options.append((selection_key(group), price, group))
        key, price, group = min(options, key=lambda value: value[:2])
        selected.append(dict(method=method, seed=seed, parameter=price, validation_key=key))
        selected_rows.extend(group)
    return selected, selected_rows


def summaries(rows):
    return [dict(scenario=s, method=method, **{key: float(np.mean([r[key] for r in rows
            if r['scenario'] == s and r['method'] == method])) for key in METRICS})
            for s in range(3) for method in dict.fromkeys(r['method'] for r in rows)]


def main():
    torch.set_num_threads(1)
    input_checks()
    folder = ROOT/'control_validation'
    folder.mkdir(exist_ok=False)
    loaded, checkpoints = models()
    files = [Path(__file__), Path('joint_power/aggregation/PLAN.md'), ROOT/'training/output_hashes.json',
             PREVIOUS/'price_extension/output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        stream=3909220000, previously_examined_validation=True, prices=PRICES), indent=2)+'\n')
    cfg, predictor, rows = make_config('validation'), load_predictor(), []
    for s, scenario in enumerate(SCENARIOS):
        for index in range(4):
            trace = make_trace(cfg, 3909220000+s*10000+index, scenario)
            for (method, seed), model in loaded.items():
                for price in PRICES:
                    metrics, _ = episode(trace, 'MC absolute', price, predictor, model)
                    rows.append(dict(scenario=s, trajectory=index, method=method, seed=seed, parameter=price, **metrics))
            print(f'aggregation control scenario={s} trace={index}', flush=True)
    assert len(rows) == 432
    write_rows(folder/'candidate_episodes.csv', rows)
    selections, chosen = select_options(rows)
    reference = read_rows(PREVIOUS/'price_extension/selected_episodes.csv')
    for row in reference:
        row['method'] = {'MC absolute': 'Previous absolute', 'MC signed': 'Previous signed'}.get(row['method'], row['method'])
    selected_rows = chosen+reference
    assert len(chosen) == 48 and len(reference) == 108
    write_rows(folder/'selected_episodes.csv', selected_rows)
    (folder/'selection.json').write_text(json.dumps(selections, indent=2)+'\n')
    summary = summaries(selected_rows)
    write_rows(folder/'summary.csv', summary)
    gate_rows = [dict(r, method='MC absolute' if r['method'] == 'Visitation-added' else r['method']) for r in selected_rows]
    result = gate(gate_rows, .01)
    result['noninitial_checkpoints'] = all(v['epoch'] > 0 for name, v in checkpoints.items() if name.startswith('Visitation-added'))
    result['passed'] &= result['noninitial_checkpoints']
    result.update(previously_examined_validation=True, development_launched=False,
                  reused_frozen_reference_episodes=108, new_candidate_episodes=432)
    (folder/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (folder/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in folder.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(dict(selection=selections, summary=summary, gate=result), indent=2))


if __name__ == '__main__':
    main()
