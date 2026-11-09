"""Expand PRS counts fairly; all other forecast/ranking objectives are intact."""
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _prediction
from joint_power.aggregation.control import read_rows, summaries
from joint_power.anchor_confirmation.study import ROOT, STREAM, NUMBER, input_checks
from joint_power.model import Environment
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from major_revision.budget_search import predicted_rates


def command_at(cfg, tracker, hardware, requirements, minimum=1):
    if not 1 <= minimum <= cfg.subarrays:
        raise ValueError('Invalid minimum count')
    score, _ = _prediction(cfg, tracker, hardware, requirements)
    counts = np.arange(minimum, cfg.subarrays+1)
    rates = predicted_rates(cfg, tracker, hardware, score, counts)
    outage = (rates < requirements).mean((1, 2))
    shortfall = (np.maximum(requirements-rates, 0.)/requirements).mean((1, 2))
    best = min(range(len(counts)), key=lambda i: (outage[i] > cfg.outage_target,
        counts[i] if outage[i] <= cfg.outage_target else shortfall[i], counts[i]))
    return select_top(score, int(counts[best]))


def episode(trace, adaptive):
    cfg = make_config('validation')
    env, failed = Environment(cfg, trace), 0
    while env.t < cfg.episode_steps:
        env.prepare()
        threshold = min(1., max(0., 21-failed)/(3*(140-env.t))) if adaptive else .05
        command = command_at(replace(cfg, outage_target=threshold), env.tracker, env.hw, env.req)
        start = len(env.records)
        env.block(command)
        failed += sum(round(3*r['outage']) for r in env.records[start:])
    return env.metrics(), env.records


def main():
    input_checks()
    previous = json.loads((ROOT/'output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in previous.items())
    out = ROOT/'expanded_prs'
    out.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/anchor_confirmation/DOMAIN_PLAN.md'), ROOT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (out/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, stream=STREAM,
        action_domain='all counts1..16', post_confirmation_baseline_addition=True,
        retuned=False, host_environment_changed=False), indent=2)+'\n')
    cfg, rows, slots, errors = make_config('validation'), [], [], defaultdict(float)
    for s, scenario in enumerate(SCENARIOS):
        for i in range(NUMBER):
            trace = make_trace(cfg, STREAM+10000*s+i, scenario)
            for adaptive in (False, True):
                metrics, records = episode(trace, adaptive)
                ident = dict(method='Expanded budgetPRS' if adaptive else 'Expanded PRS',
                             seed=0, scenario=s, trajectory=i, parameter=1. if adaptive else .05)
                rows.append(dict(**ident, **metrics))
                slots.extend(dict(**ident, **r) for r in records)
                for name, error in independent(records, trace).items():
                    errors[name] = max(errors[name], error)
        print(f'expanded PRS scenario={s} complete', flush=True)
    assert len(rows) == 48 and len(slots) == 6720
    assert errors['rate_mbps'] < 1e-5 and max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    write_rows(out/'episodes.csv', rows)
    write_rows(out/'slots.csv', slots)
    write_rows(out/'summary.csv', summaries(rows))
    report = dict(episodes=48, verified_slots=6720, max_physical_errors=dict(errors),
        summary=summaries(rows), primary_policy_unchanged=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in previous.items())
    assert all(digest(p) == h for p, h in hashes.items())
    (out/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in out.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
