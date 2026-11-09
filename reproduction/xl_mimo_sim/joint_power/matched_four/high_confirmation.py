"""New-trace confirmation of a development-generated high-load hypothesis."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.anchor_confirmation.domain_prs import episode as expanded_episode
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.control import episode, raw_action
from joint_power.matched_four.study import ROOT as DEVELOPMENT, checks
from joint_power.model import Environment
from joint_power.risk_budget.control import episode as budget_episode
from joint_power.rollout.control import episode as old_episode
from joint_power.rollout.verify import independent
from joint_power.study import digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_min4_high_confirmation')
STREAM, NUMBER = 5309220000, 24
NONLEARNING = ('PRS original', 'PRS budget', 'Expanded PRS', 'Expanded budgetPRS',
               'Raw full min4', 'Full array')
PRIMARY = 'Energy Value min4'


def specifications():
    return [(name, seed) for name in (PRIMARY, 'Energy Myopic min4', 'Energy Value min1')
            for seed in (11, 22)]+[(name, 0) for name in NONLEARNING]


def one(i):
    torch.set_num_threads(1)
    cfg, predictor = make_config('validation'), load_predictor()
    loaded, _ = models()
    trace = make_trace(cfg, STREAM+i, (3., 1.15))
    rows, slots, decisions, errors = [], [], [], defaultdict(float)
    for method, seed in specifications():
        ledger = None
        if seed:
            metrics, records, ledger = episode(trace, 'Myopic' if 'Myopic' in method else 'Value',
                .1, predictor, loaded['Visitation-added', seed], 1 if method.endswith('min1') else 4)
        elif method.startswith('Expanded'):
            metrics, records = expanded_episode(trace, method == 'Expanded budgetPRS')
        elif method == 'PRS budget':
            metrics, records, _ = budget_episode(trace, method, 1., predictor)
        elif method == 'Raw full min4':
            env = Environment(cfg, trace)
            while env.t < cfg.episode_steps:
                c = env.candidates()
                env.block(c.masks[raw_action(c, 4)])
            metrics, records = env.metrics(), env.records
        else:
            metrics, records = old_episode(trace, 'Full power' if method == 'Full array' else method,
                                           .05, predictor)
        ident = dict(scenario=2, trajectory=i, method=method, seed=seed)
        for key, value in independent(records, trace).items():
            errors[key] = max(errors[key], value)
        minimum = 1 if method.endswith('min1') or method.startswith('Expanded') else 4
        counts = [r['command_bits'].bit_count() for r in records]
        assert min(counts) >= minimum
        failed = sum(round(3*r['outage']) for r in records)
        rows.append(dict(**ident, **metrics, min_command=min(counts),
            mean_command=float(np.mean(counts)), pathwise_exceeded=int(failed > 21),
            reference_deviations=sum(r['differs_from_reference'] for r in ledger) if ledger else -1))
        slots.extend(dict(**ident, **r) for r in records)
        if ledger is not None:
            assert ledger[-1]['cumulative_failed'] == failed
            decisions.extend(dict(**ident, **r) for r in ledger)
    print(f'locked high-load trace={i}: {len(rows)} episodes verified', flush=True)
    return rows, slots, decisions, dict(errors)


def analyze(rows):
    methods = list(dict.fromkeys(r['method'] for r in rows))
    fields = ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active', 'mean_command')
    summary = [dict(method=m, **{k: float(np.mean([r[k] for r in rows if r['method'] == m]))
                for k in fields}) for m in methods]
    lookup = {(r['method'], r['seed'], r['trajectory']): r for r in rows}
    rng, paired = np.random.default_rng(2026092233), []
    for other in methods:
        if other == PRIMARY:
            continue
        for metric in ('energy_j', 'outage'):
            delta = np.array([[lookup[PRIMARY, seed, i][metric]-
                lookup[other, 0 if other in NONLEARNING else seed, i][metric]
                for i in range(NUMBER)] for seed in (11, 22)])
            ids, seeds = rng.integers(0, NUMBER, (6000, NUMBER)), rng.integers(0, 2, (6000, 2))
            draws = delta[seeds[:, :, None], ids[:, None, :]].mean((1, 2))
            paired.append(dict(reference=other, metric=metric, difference=float(delta.mean()),
                ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975))))
    reference = min((r for r in summary if r['method'] in NONLEARNING and r['outage'] <= .05),
                    key=lambda r: r['energy_j'])
    primary = next(r for r in summary if r['method'] == PRIMARY)
    per_seed = {str(seed): {key: float(np.mean([r[key] for r in rows if r['method'] == PRIMARY and
                r['seed'] == seed])) for key in ('energy_j', 'outage')} for seed in (11, 22)}
    gain = 1-primary['energy_j']/reference['energy_j']
    interval = next(r for r in paired if r['reference'] == reference['method'] and r['metric'] == 'energy_j')
    gate = dict(passed=bool(gain >= .02 and interval['ci_high'] < 0 and all(
        r['outage'] <= .05 and r['energy_j'] <= reference['energy_j'] for r in per_seed.values())),
        nominal_development_failure_unchanged=True, condition=(3., 1.15),
        primary=primary, reference=reference, saving=gain, energy_interval=interval, per_seed=per_seed,
        conditional_hypothesis_only=True, R1_closed=False)
    return summary, paired, gate


def intact():
    checks()
    data = json.loads((DEVELOPMENT/'manifest.json').read_text())['input_hashes']
    data.update(json.loads((DEVELOPMENT/'output_hashes.json').read_text()))
    assert all(digest(p) == h for p, h in data.items())


def main():
    torch.set_num_threads(1)
    intact()
    ROOT.mkdir(exist_ok=False)
    _, checkpoints = models()
    files = [Path(__file__), Path('joint_power/matched_four/HIGH_CONFIRMATION_PLAN.md'),
             Path('joint_power/anchor_confirmation/domain_prs.py'), DEVELOPMENT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        stream=STREAM, unique_traces=NUMBER, specifications=specifications(),
        generated_after_development_hypothesis=True, no_retuning=True,
        host_environment_changed=False), indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        output = list(pool.map(one, range(NUMBER)))
    rows = [r for group, _, _, _ in output for r in group]
    slots = [r for _, group, _, _ in output for r in group]
    decisions = [r for _, _, group, _ in output for r in group]
    assert len(rows) == 288 and len(slots) == 40320
    summary, paired, gate = analyze(rows)
    errors = {k: max(e[k] for _, _, _, e in output) for k in output[0][3]}
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for name, items in (('episodes', rows), ('slots', slots), ('decisions', decisions),
                         ('summary', summary), ('paired', paired)):
        write_rows(ROOT/f'{name}.csv', items)
    (ROOT/'gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    (ROOT/'report.json').write_text(json.dumps(dict(gate=gate, episodes=len(rows), physical_slots=len(slots),
        max_physical_errors=errors, all_baselines_preserved=True), indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(r['path']) == r['sha256'] for r in checkpoints.values())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
