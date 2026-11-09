"""Paired fixed-policy heuristic sweep, not a parameter-selection experiment."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.anchor_confirmation.domain_prs import command_at
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.control import episode as learned_episode
from joint_power.model import Environment
from joint_power.risk_only.train import intact as previous_intact
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_sensitivity')
STREAM, NUMBER = 5709220000, 8
SETTINGS = ((.65, .93), (0., .93), (1.3, .93), (.65, .8), (.65, 1.))
REFERENCES = ('PRS4', 'budgetPRS4', 'PRS1', 'budgetPRS1')
LEARNED = ('Value4', 'Value1')


def intact():
    previous_intact()
    old = Path('results/joint_power_20260922_risk_only/training')
    hashes = json.loads((old/'manifest.json').read_text())['input_hashes']
    hashes.update(json.loads((old/'output_hashes.json').read_text()))
    assert all(digest(p) == h for p, h in hashes.items())


def specifications():
    return [(m, seed) for m in LEARNED for seed in (11, 22)]+[(m, 0) for m in REFERENCES]


def reference_episode(cfg, trace, method):
    if method not in (*REFERENCES, 'Full array'):
        raise ValueError('Unknown reference')
    env, past = Environment(cfg, trace), 0
    while env.t < cfg.episode_steps:
        env.prepare()
        if method == 'Full array':
            command = np.ones(cfg.subarrays, bool)
        else:
            threshold = cfg.outage_target
            if method.startswith('budget'):
                allowance = max(0., cfg.outage_target*cfg.users*cfg.episode_steps-past)
                threshold = min(1., allowance/(cfg.users*(cfg.episode_steps-env.t)))
            command = command_at(replace(cfg, outage_target=threshold), env.tracker,
                env.hw, env.req, minimum=int(method[-1]))
        start = len(env.records)
        env.block(command)
        past += sum(round(cfg.users*r['outage']) for r in env.records[start:])
    return env.metrics(), env.records


def one(task):
    scenario, index = task
    torch.set_num_threads(1)
    original = make_config('validation')
    trace_seed = STREAM+scenario*10000+index
    trace = make_trace(original, trace_seed, SCENARIOS[scenario])
    saved = (trace.channel.copy(), trace.requirements_mbps.copy())
    predictor, loaded = load_predictor(), models()[0]
    rows, slots, errors = [], [], defaultdict(float)
    for setting, (beta, discount) in enumerate(SETTINGS):
        cfg = replace(original, uncertainty_penalty=beta, visibility_discount=discount)
        specs = specifications()+([('Full array', 0)] if setting == 0 else [])
        for method, seed in specs:
            if seed:
                metrics, records, ledger = learned_episode(trace, 'Value', .1, predictor,
                    loaded['Visitation-added', seed], int(method[-1]), cfg=cfg)
                assert ledger[-1]['cumulative_failed'] == sum(round(3*r['outage']) for r in records)
            else:
                metrics, records = reference_episode(cfg, trace, method)
            minimum = 16 if method == 'Full array' else int(method[-1])
            assert min(r['command_bits'].bit_count() for r in records) >= minimum
            for key, error in independent(records, trace).items():
                errors[key] = max(errors[key], error)
            identity = dict(scenario=scenario, trajectory=index, trace_seed=trace_seed,
                setting=setting, beta=beta, discount=discount, method=method, seed=seed)
            rows.append(dict(**identity, **metrics))
            slots.extend(dict(**identity, **r) for r in records)
    for actual, before in zip((trace.channel, trace.requirements_mbps), saved):
        np.testing.assert_array_equal(actual, before)
    print(f'sensitivity scenario={scenario} trace={index}: {len(rows)} episodes verified', flush=True)
    return rows, slots, dict(errors)


def summarize(rows):
    fields = ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')
    result = []
    for scenario, setting, method in dict.fromkeys((r['scenario'], r['setting'], r['method']) for r in rows):
        group = [r for r in rows if (r['scenario'], r['setting'], r['method']) == (scenario, setting, method)]
        result.append(dict(scenario=scenario, setting=setting, beta=SETTINGS[setting][0],
            discount=SETTINGS[setting][1], method=method, trajectories=NUMBER,
            training_seeds=2 if method in LEARNED else 0,
            **{key: float(np.mean([r[key] for r in group])) for key in fields}))
    return result


def comparisons(rows):
    lookup = {(r['scenario'], r['setting'], r['method'], r['seed'], r['trajectory']): r for r in rows}
    rng, output = np.random.default_rng(2026092235), []
    for scenario in range(3):
        for setting in range(len(SETTINGS)):
            for method in LEARNED:
                comparators = [('same-range budgetPRS', setting, 'budgetPRS'+method[-1])]
                if setting:
                    comparators.append(('default parameter', 0, method))
                for comparison, ref_setting, ref_method in comparators:
                    for metric in ('energy_j', 'outage'):
                        delta = np.array([[lookup[scenario, setting, method, seed, i][metric]-
                            lookup[scenario, ref_setting, ref_method,
                                   seed if ref_method in LEARNED else 0, i][metric]
                            for i in range(NUMBER)] for seed in (11, 22)])
                        ids = rng.integers(0, NUMBER, (4000, NUMBER))
                        seeds = rng.integers(0, 2, (4000, 2))
                        boot = delta[seeds[:, :, None], ids[:, None, :]].mean((1, 2))
                        output.append(dict(scenario=scenario, setting=setting, method=method,
                            comparison=comparison, reference=ref_method, metric=metric,
                            difference=float(delta.mean()), ci_low=float(np.quantile(boot, .025)),
                            ci_high=float(np.quantile(boot, .975))))
    return output


def main():
    torch.set_num_threads(1)
    intact()
    ROOT.mkdir(exist_ok=False)
    _, checkpoints = models()
    sources = [*Path('joint_power/sensitivity').glob('*.py'), Path('joint_power/sensitivity/PLAN.md'),
        Path('joint_power/matched_four/control.py'), Path('joint_power/anchor_confirmation/domain_prs.py'),
        Path('results/joint_power_20260922_risk_only/training/output_hashes.json')]
    hashes = {str(p): digest(p) for p in sources}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        settings=SETTINGS, stream=STREAM, trajectories_per_condition=NUMBER,
        primary='Value4', unrestricted_domain_ablation='Value1', alpha=.1,
        no_retraining_or_selection=True, full_array_once_per_trace=True,
        host_environment_changed=False), indent=2)+'\n')
    tasks = [(s, i) for s in range(3) for i in range(NUMBER)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, tasks))
    rows = [r for group, _, _ in results for r in group]
    slots = [r for _, group, _ in results for r in group]
    assert len(rows) == 984 and len(slots) == 137760
    key = lambda r: tuple(r[k] for k in ('scenario', 'trajectory', 'setting', 'method', 'seed'))
    assert len({key(r) for r in rows}) == len(rows)
    summary, paired = summarize(rows), comparisons(rows)
    errors = {k: max(e[k] for _, _, e in results) for k in results[0][2]}
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for name, data in (('episodes', rows), ('slots', slots), ('summary', summary), ('paired', paired)):
        write_rows(ROOT/f'{name}.csv', data)
    report = dict(unique_traces=24, episodes=len(rows), physical_slots=len(slots),
        settings=SETTINGS, maximum_errors=errors, all_results_retained=True,
        no_parameter_selection=True, calibration_guarantee=False, R1_closed=False,
        scope='Five-point fixed-policy heuristic sensitivity, not simultaneous confidence or standardized-channel validation')
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(v['path']) == v['sha256'] for v in checkpoints.values())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
