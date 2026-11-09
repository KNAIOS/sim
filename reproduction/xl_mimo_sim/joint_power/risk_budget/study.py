"""Bounded common-factor control screen, retaining all strong references."""
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.collect import ROOT as PREVIOUS
from joint_power.aggregation.control import input_checks as old_checks, models, read_rows, select_options, summaries
from joint_power.calibrated.histogram import load_predictor
from joint_power.risk_budget.control import episode
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_risk_budget')
FACTORS = (0., .1, .25, .5, 1.)
BASELINES = ('PRS original', 'PRS tuned', 'Raw full', 'Raw coarse', 'Full power', 'Raw budget', 'PRS budget')


def input_checks():
    old_checks()
    for name in ('control_validation', 'audit', 'prediction_audit', 'price_diagnosis'):
        hashes = json.loads((PREVIOUS/name/'output_hashes.json').read_text())
        assert all(digest(p) == h for p, h in hashes.items()), name


def controllers():
    loaded, checkpoints = models()
    result = {(f'{rule} {arm}', seed): (rule, model) for (arm, seed), model in loaded.items()
              for rule in ('Value', 'Myopic')}
    result.update({(name, 0): (name, None) for name in ('Raw budget', 'PRS budget')})
    return result, checkpoints


def gates(rows, threshold=.01):
    def energy(method, scenario=1, seed=None):
        return float(np.mean([r['energy_j'] for r in rows if r['method'] == method and
            r['scenario'] == scenario and (seed is None or r['seed'] == seed)]))
    feasible = []
    for method in BASELINES:
        worst = max(float(np.mean([r['outage'] for r in rows if r['method'] == method and r['scenario'] == s]))
                    for s in range(3))
        if worst <= .05+1e-12:
            feasible.append(dict(method=method, energy_j=energy(method), worst_outage=worst))
    reference = min(feasible, key=lambda r: (r['energy_j'], r['worst_outage']))
    results = []
    for arm in ('PRS-added', 'Visitation-added'):
        method = f'Value {arm}'
        per_seed = {str(seed): [float(np.mean([r['outage'] for r in rows if
                    (r['method'], r['seed'], r['scenario']) == (method, seed, s)]))
                    for s in range(3)] for seed in (11, 22)}
        energies = {str(seed): energy(method, seed=seed) for seed in (11, 22)}
        gain = 1-energy(method)/reference['energy_j']
        results.append(dict(method=method, nominal_energy_j=energy(method), nominal_saving=gain,
            per_seed_nominal_energy=energies, per_seed_outage=per_seed,
            myopic_nominal_energy_j=energy(f'Myopic {arm}'),
            passed=bool(max(max(v) for v in per_seed.values()) <= .05+1e-12 and
                        gain >= threshold and max(energies.values()) <= reference['energy_j'])))
    return dict(passed=any(r['passed'] for r in results), reference=reference, feasible_baselines=feasible,
                families=results, required_saving=threshold, development_launched=False)


def main():
    torch.set_num_threads(1)
    input_checks()
    ROOT.mkdir(exist_ok=False)
    policies, checkpoints = controllers()
    assert all(r['epoch'] > 0 for r in checkpoints.values())
    files = [*Path('joint_power/risk_budget').glob('*.py'), Path('joint_power/risk_budget/PLAN.md'),
             PREVIOUS/'control_validation/output_hashes.json', PREVIOUS/'training/output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        factors=FACTORS, stream=3909220000, previously_examined_validation=True,
        host_environment_changed=False), indent=2)+'\n')
    cfg, predictor, rows = make_config('validation'), load_predictor(), []
    for s, scenario in enumerate(SCENARIOS):
        for index in range(4):
            trace = make_trace(cfg, 3909220000+s*10000+index, scenario)
            for (method, seed), (rule, model) in policies.items():
                for alpha in FACTORS:
                    metrics, _, _ = episode(trace, rule, alpha, predictor, model)
                    rows.append(dict(method=method, seed=seed, scenario=s, trajectory=index, parameter=alpha, **metrics))
            print(f'budget control scenario={s} trace={index}', flush=True)
    assert len(rows) == 600
    write_rows(ROOT/'candidate_episodes.csv', rows)
    selected, chosen = select_options(rows)
    reference = read_rows(PREVIOUS/'control_validation/selected_episodes.csv')
    assert len(chosen) == 120 and len(reference) == 156
    all_selected = chosen+reference
    write_rows(ROOT/'selected_episodes.csv', all_selected)
    (ROOT/'selection.json').write_text(json.dumps(selected, indent=2)+'\n')
    summary, result = summaries(all_selected), gates(all_selected)
    write_rows(ROOT/'summary.csv', summary)
    (ROOT/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(dict(selection=selected, gate=result), indent=2))


if __name__ == '__main__':
    main()
