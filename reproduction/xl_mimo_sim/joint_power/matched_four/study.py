"""Fixed-policy action-domain attribution with the original PRS preserved."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models, summaries
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.control import episode, raw_action
from joint_power.model import Environment
from joint_power.risk_budget.control import episode as budget_episode
from joint_power.rollout.control import episode as old_episode
from joint_power.rollout.verify import independent
from joint_power.strong_headroom.study import ROOT as PREVIOUS, intact as previous_intact
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_matched_four')
STREAM, NUMBER = 5209220000, 4
NONLEARNING = ('PRS original', 'PRS budget', 'Raw full min4', 'Energy Raw min4', 'Full array')


def specifications():
    return [(name, seed) for name in ('Energy Value min4', 'Energy Myopic min4', 'Energy Value min1')
            for seed in (11, 22)]+[(name, 0) for name in NONLEARNING]


def one(task):
    s, i = task
    torch.set_num_threads(1)
    cfg, predictor = make_config('validation'), load_predictor()
    loaded, _ = models()
    trace = make_trace(cfg, STREAM+10000*s+i, SCENARIOS[s])
    rows, slots, decisions, errors = [], [], [], defaultdict(float)
    for method, seed in specifications():
        ledger = None
        if seed:
            proposer = 'Myopic' if 'Myopic' in method else 'Value'
            minimum = 1 if method.endswith('min1') else 4
            metrics, records, ledger = episode(trace, proposer, .1, predictor,
                loaded['Visitation-added', seed], minimum)
        elif method == 'Energy Raw min4':
            metrics, records, ledger = episode(trace, 'Raw budget', 0., predictor, None, 4)
        elif method == 'Raw full min4':
            env = Environment(cfg, trace)
            while env.t < cfg.episode_steps:
                candidates = env.candidates()
                env.block(candidates.masks[raw_action(candidates, 4)])
            metrics, records = env.metrics(), env.records
        elif method == 'PRS budget':
            metrics, records, _ = budget_episode(trace, method, 1., predictor)
        else:
            metrics, records = old_episode(trace, 'Full power' if method == 'Full array' else method,
                                           .05, predictor)
        ident = dict(scenario=s, trajectory=i, method=method, seed=seed)
        for name, error in independent(records, trace).items():
            errors[name] = max(errors[name], error)
        counts = [r['command_bits'].bit_count() for r in records]
        assert min(counts) >= (1 if method.endswith('min1') else 4)
        failed = sum(round(cfg.users*r['outage']) for r in records)
        rows.append(dict(**ident, **metrics, min_command=min(counts), mean_command=float(np.mean(counts)),
            pathwise_exceeded=int(failed > 21),
            reference_deviations=sum(r['differs_from_reference'] for r in ledger) if ledger is not None else -1))
        slots.extend(dict(**ident, **r) for r in records)
        if ledger is not None:
            assert ledger[-1]['cumulative_failed'] == failed
            decisions.extend(dict(**ident, **r) for r in ledger)
    print(f'matched min4 scenario={s} trace={i}: {len(rows)} episodes verified', flush=True)
    return rows, slots, decisions, dict(errors)


def checks():
    previous_intact()
    for folder in (PREVIOUS, PREVIOUS/'target_budget_v2'):
        hashes = json.loads((folder/'manifest.json').read_text())['input_hashes']
        hashes.update(json.loads((folder/'output_hashes.json').read_text()))
        assert all(digest(p) == h for p, h in hashes.items())


def analyze(rows):
    def mean(method, s, key, seed=None):
        return float(np.mean([r[key] for r in rows if r['method'] == method and r['scenario'] == s
                              and (seed is None or r['seed'] == seed)]))
    feasible = [m for m in NONLEARNING if max(mean(m, s, 'outage') for s in range(3)) <= .05]
    reference = min(feasible, key=lambda m: mean(m, 1, 'energy_j'))
    primary, ref_energy = 'Energy Value min4', mean(reference, 1, 'energy_j')
    gain = 1-mean(primary, 1, 'energy_j')/ref_energy
    seeds = {str(seed): dict(nominal_energy_j=mean(primary, 1, 'energy_j', seed),
                            outage=[mean(primary, s, 'outage', seed) for s in range(3)]) for seed in (11, 22)}
    gate = dict(passed=bool(gain >= .02 and all(r['nominal_energy_j'] <= ref_energy and
        max(r['outage']) <= .05 for r in seeds.values())), nominal_saving=gain,
        reference=reference, reference_energy_j=ref_energy, per_seed=seeds,
        inference_domain_ablation_only=True, not_physical_minimum_evidence=True, R1_closed=False)
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    rng, paired = np.random.default_rng(2026092232), []
    for s in range(3):
        for other in ('Energy Myopic min4', 'Energy Value min1', *NONLEARNING):
            for metric in ('energy_j', 'outage'):
                delta = np.array([[lookup[primary, seed, s, i][metric]-
                    lookup[other, 0 if other in NONLEARNING else seed, s, i][metric]
                    for i in range(NUMBER)] for seed in (11, 22)])
                ids, seeds_i = rng.integers(0, NUMBER, (4000, NUMBER)), rng.integers(0, 2, (4000, 2))
                draws = delta[seeds_i[:, :, None], ids[:, None, :]].mean((1, 2))
                paired.append(dict(scenario=s, reference=other, metric=metric, difference=float(delta.mean()),
                    ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975))))
    return gate, paired


def main():
    torch.set_num_threads(1)
    checks()
    ROOT.mkdir(exist_ok=False)
    _, checkpoints = models()
    files = [*Path('joint_power/matched_four').glob('*.py'), Path('joint_power/matched_four/PLAN.md'),
             PREVIOUS/'output_hashes.json', PREVIOUS/'target_budget_v2/output_hashes.json',
             Path('joint_power/baseline_anchor/control.py'), Path('joint_power/floor_projection/control.py')]
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        stream=STREAM, trajectories_per_scenario=NUMBER, specifications=specifications(),
        no_retuning=True, fixed_policy_domain_ablation=True, host_environment_changed=False), indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        output = list(pool.map(one, [(s, i) for s in range(3) for i in range(NUMBER)]))
    rows = [r for group, _, _, _ in output for r in group]
    slots = [r for _, group, _, _ in output for r in group]
    decisions = [r for _, _, group, _ in output for r in group]
    assert len(rows) == 132 and len(slots) == 18480
    gate, paired = analyze(rows)
    errors = {k: max(e[k] for _, _, _, e in output) for k in output[0][3]}
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    write_rows(ROOT/'episodes.csv', rows)
    write_rows(ROOT/'slots.csv', slots)
    write_rows(ROOT/'decisions.csv', decisions)
    write_rows(ROOT/'summary.csv', summaries(rows))
    write_rows(ROOT/'paired.csv', paired)
    (ROOT/'gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    (ROOT/'report.json').write_text(json.dumps(dict(gate=gate, episodes=len(rows), physical_slots=len(slots),
        max_physical_errors=errors, original_reference_unchanged=True), indent=2)+'\n')
    checks()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(r['path']) == r['sha256'] for r in checkpoints.values())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
