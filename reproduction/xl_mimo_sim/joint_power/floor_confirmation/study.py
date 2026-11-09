"""Single bounded, locked-policy development check; no outcome selection."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models, summaries
from joint_power.calibrated.histogram import load_predictor
from joint_power.floor_projection.control import episode as corrected
from joint_power.floor_projection.study import ROOT as SCREEN, intact
from joint_power.risk_budget.control import episode as budget
from joint_power.rollout.control import episode as legacy
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_floor_confirmation')
NONLEARNING = ('PRS original', 'PRS budget', 'Raw full', 'Raw coarse', 'Full power', 'Raw budget', 'Floor Raw budget')


def input_checks():
    intact()
    assert json.loads((SCREEN/'gate.json').read_text())['families'][1]['passed']
    for name in ('output_hashes.json', 'manifest.json'):
        hashes = json.loads((SCREEN/name).read_text())
        hashes = hashes['input_hashes'] if name == 'manifest.json' else hashes
        assert all(digest(p) == h for p, h in hashes.items())


def specifications():
    learned = [(method, seed, .1 if method != 'Fixed-price Value' else 4.) for seed in (11, 22)
               for method in ('Floor Value', 'Floor Myopic', 'Value', 'Fixed-price Value')]
    return learned+[(m, 0, p) for m, p in zip(NONLEARNING, (.05, 1., .05, 0., .05, 0., 0.))]


def analyze(rows):
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    def mean(method, s, key, seed=None):
        return float(np.mean([r[key] for r in rows if r['method'] == method and r['scenario'] == s
                              and (seed is None or seed == r['seed'])]))
    feasible = [dict(method=m, energy_j=mean(m, 1, 'energy_j'),
                    worst_outage=max(mean(m, s, 'outage') for s in range(3))) for m in NONLEARNING]
    feasible = [r for r in feasible if r['worst_outage'] <= .05+1e-12]
    ref = min(feasible, key=lambda r: (r['energy_j'], r['worst_outage']))
    per_seed = {str(seed): [mean('Floor Value', s, 'outage', seed) for s in range(3)] for seed in (11, 22)}
    energies = {str(seed): mean('Floor Value', 1, 'energy_j', seed) for seed in (11, 22)}
    gain = 1-mean('Floor Value', 1, 'energy_j')/ref['energy_j']
    result = dict(passed=bool(max(max(v) for v in per_seed.values()) <= .05+1e-12 and gain >= .02
                            and max(energies.values()) <= ref['energy_j']),
        reference=ref, feasible_baselines=feasible, per_seed_outage=per_seed,
        per_seed_nominal_energy=energies, nominal_energy_j=mean('Floor Value', 1, 'energy_j'),
        nominal_saving=gain, required_saving=.02, split='locked development, not final test')
    rng, paired = np.random.default_rng(2026092224), []
    for s in range(3):
        for reference in ('Floor Myopic', 'Value', 'Fixed-price Value', *NONLEARNING):
            for metric in ('energy_j', 'outage'):
                delta = np.array([[lookup['Floor Value', seed, s, i][metric]-lookup[
                    reference, 0 if reference in NONLEARNING else seed, s, i][metric]
                    for i in range(4)] for seed in (11, 22)])
                si, ti = rng.integers(0, 2, (4000, 2)), rng.integers(0, 4, (4000, 4))
                draws = delta[si[:, :, None], ti[:, None, :]].mean((1, 2))
                paired.append(dict(scenario=s, method='Floor Value', reference=reference, metric=metric,
                    difference=float(delta.mean()), ci_low=float(np.quantile(draws, .025)),
                    ci_high=float(np.quantile(draws, .975))))
    return result, paired


def main():
    torch.set_num_threads(1)
    input_checks()
    ROOT.mkdir(exist_ok=False)
    loaded, checkpoints = models()
    files = [*Path('joint_power/floor_confirmation').glob('*.py'), Path('joint_power/floor_confirmation/PLAN.md'),
             SCREEN/'output_hashes.json', SCREEN/'manifest.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        fixed_specifications=specifications(), stream=4009220000, previously_examined=False,
        split='development', selected_family='Visitation-added', no_retuning=True,
        prs_tuned_alias='PRS original: both use0.05', host_environment_changed=False), indent=2)+'\n')
    cfg, predictor = make_config('validation'), load_predictor()
    rows, slots, floor_decisions, ledgers, errors = [], [], [], [], defaultdict(float)
    for s, scenario in enumerate(SCENARIOS):
        for i in range(4):
            trace = make_trace(cfg, 4009220000+s*10000+i, scenario)
            for method, seed, parameter in specifications():
                model = loaded['Visitation-added', seed] if seed else None
                if method.startswith('Floor '):
                    metrics, records, ledger = corrected(trace, method[6:], parameter, predictor, model)
                elif method in ('Value', 'PRS budget', 'Raw budget'):
                    metrics, records, ledger = budget(trace, method, parameter, predictor, model)
                else:
                    name = 'MC absolute' if method == 'Fixed-price Value' else method
                    metrics, records = legacy(trace, name, parameter, predictor, model)
                    ledger = None
                ident = dict(method=method, seed=seed, scenario=s, trajectory=i, parameter=parameter)
                rows.append(dict(**ident, **metrics))
                slots.extend(dict(**ident, **r) for r in records)
                for name, error in independent(records, trace).items():
                    errors[name] = max(errors[name], error)
                failed = sum(round(3*r['outage']) for r in records)
                ledgers.append(dict(**ident, final_failed=failed, pathwise_target_exceeded=int(failed > 21)))
                for key in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
                    assert abs(metrics[key]-sum(r[key] for r in records)) < 1e-9
                assert abs(metrics['outage']-failed/420) < 1e-10
                if method.startswith('Floor '):
                    floor_decisions.extend(dict(**ident, **r) for r in ledger)
                    for j, item in enumerate(ledger):
                        block = records[4*j:4*j+4]
                        physical = round(3*block[0]['outage'])+sum(3 for r in block[1:] if r['active_count'] == 0)
                        assert item['selected_floor'] == physical <= item['block_failed']
                        assert item['selected_projected_count'] >= physical
            print(f'locked development scenario={s} trace={i}, episodes={len(rows)}', flush=True)
    assert len(rows) == 180 and len(slots) == 25200
    assert errors['rate_mbps'] < 1e-5 and max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    result, comparisons = analyze(rows)
    write_rows(ROOT/'episodes.csv', rows)
    write_rows(ROOT/'slots.csv', slots)
    write_rows(ROOT/'floor_decisions.csv', floor_decisions)
    write_rows(ROOT/'episode_ledgers.csv', ledgers)
    write_rows(ROOT/'summary.csv', summaries(rows))
    write_rows(ROOT/'paired_differences.csv', comparisons)
    (ROOT/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    report = dict(episodes=180, verified_slots=25200, new_exogenous_trajectories=12,
        training_seeds=2, max_physical_errors=dict(errors), frozen_parameters=True,
        rate_outage_energy_accounting_verified=True, gate=result,
        limitation='Small locked development sample; not a final test or hard reliability guarantee')
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
