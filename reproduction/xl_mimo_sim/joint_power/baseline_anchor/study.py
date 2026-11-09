"""Two inspected research streams, locked admission rules and full references."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models, read_rows, summaries
from joint_power.baseline_anchor.control import episode
from joint_power.calibrated.histogram import load_predictor
from joint_power.floor_confirmation.study import ROOT as DEVELOPMENT, input_checks as old_checks
from joint_power.floor_projection.study import ROOT as SCREEN
from joint_power.rollout.verify import independent
from joint_power.study import SCENARIOS, digest, make_trace, write_rows

ROOT = Path('results/joint_power_20260922_baseline_anchor')
STREAMS = (3909220000, 4009220000)
BASELINES = ('PRS original', 'PRS tuned', 'PRS budget', 'Raw full', 'Raw coarse',
             'Full power', 'Raw budget', 'Floor Raw budget', 'Empty Raw budget', 'Energy Raw budget')


def input_checks():
    old_checks()
    for name in ('manifest.json', 'output_hashes.json'):
        hashes = json.loads((DEVELOPMENT/name).read_text())
        hashes = hashes['input_hashes'] if name == 'manifest.json' else hashes
        assert all(digest(p) == h for p, h in hashes.items())


def references(stream):
    if stream == STREAMS[0]:
        rows = read_rows(SCREEN/'all_episodes.csv')
        names = {'Floor Value Visitation-added': 'Floor Value', 'Floor Myopic Visitation-added': 'Floor Myopic',
                 'Value Visitation-added': 'Value', 'Visitation-added': 'Fixed-price Value',
                 'Myopic Visitation-added': 'Myopic'}
        rows = [dict(r, method=names.get(r['method'], r['method'])) for r in rows]
    else:
        rows = read_rows(DEVELOPMENT/'episodes.csv')
    return [dict(stream=stream, **r) for r in rows]


def analyze(rows):
    def mean(method, s, key, seed=None):
        values = [r[key] for r in rows if r['method'] == method and r['scenario'] == s
                  and (seed is None or seed == r['seed'])]
        assert values, (method, s, key, seed)
        return float(np.mean(values))
    present = {r['method'] for r in rows}
    feasible = []
    for method in BASELINES:
        if method not in present:
            continue
        worst = max(mean(method, s, 'outage') for s in range(3))
        if worst <= .05+1e-12:
            feasible.append(dict(method=method, energy_j=mean(method, 1, 'energy_j'), worst_outage=worst))
    reference = min(feasible, key=lambda r: (r['energy_j'], r['worst_outage']))
    per_seed = {str(seed): [mean('Energy Value', s, 'outage', seed) for s in range(3)] for seed in (11, 22)}
    energy = {str(seed): mean('Energy Value', 1, 'energy_j', seed) for seed in (11, 22)}
    saving = 1-mean('Energy Value', 1, 'energy_j')/reference['energy_j']
    return dict(passed=bool(max(max(v) for v in per_seed.values()) <= .05+1e-12 and saving >= .02
                          and max(energy.values()) <= reference['energy_j']),
        reference=reference, feasible_baselines=feasible, per_seed_outage=per_seed,
        per_seed_nominal_energy=energy, nominal_energy_j=mean('Energy Value', 1, 'energy_j'),
        nominal_saving=saving, required_saving=.02, independent_confirmation=False)


def paired(rows, stream):
    rng = np.random.default_rng(2026092225)
    lookup = {(r['method'], r['seed'], r['scenario'], r['trajectory']): r for r in rows}
    assert len(lookup) == len(rows)
    output = []
    for s in range(3):
        for reference in ('Empty Value', 'Energy Myopic', 'Floor Value', 'PRS original', 'PRS budget'):
            for key in ('energy_j', 'outage'):
                delta = np.array([[lookup['Energy Value', seed, s, i][key]-lookup[
                    reference, 0 if reference.startswith('PRS ') else seed, s, i][key]
                    for i in range(4)] for seed in (11, 22)])
                si, ti = rng.integers(0, 2, (4000, 2)), rng.integers(0, 4, (4000, 4))
                draws = delta[si[:, :, None], ti[:, None, :]].mean((1, 2))
                output.append(dict(stream=stream, scenario=s, reference=reference, metric=key,
                    difference=float(delta.mean()), ci_low=float(np.quantile(draws, .025)),
                    ci_high=float(np.quantile(draws, .975))))
    return output


def main():
    torch.set_num_threads(1)
    input_checks()
    ROOT.mkdir(exist_ok=False)
    loaded, checkpoints = models()
    files = [*Path('joint_power/baseline_anchor').glob('*.py'), Path('joint_power/baseline_anchor/PLAN.md'),
        SCREEN/'output_hashes.json', DEVELOPMENT/'output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        streams=STREAMS, both_streams_previously_inspected=True, primary='Energy Value',
        host_environment_changed=False, retraining=False, parameter_search=False), indent=2)+'\n')
    cfg, predictor = make_config('validation'), load_predictor()
    rows, slots, decisions, ledgers, errors = [], [], [], [], defaultdict(float)
    specs = [(proposer, rule, seed, .1) for proposer in ('Value', 'Myopic')
             for rule in ('Empty', 'Energy') for seed in (11, 22)]
    specs += [('Raw budget', rule, 0, 0.) for rule in ('Empty', 'Energy')]
    for stream in STREAMS:
        for s, scenario in enumerate(SCENARIOS):
            for i in range(4):
                trace = make_trace(cfg, stream+s*10000+i, scenario)
                for proposer, rule, seed, alpha in specs:
                    model = loaded['Visitation-added', seed] if seed else None
                    metrics, records, ledger = episode(trace, proposer, rule, alpha, predictor, model)
                    ident = dict(stream=stream, method=rule+' '+proposer, seed=seed,
                                 scenario=s, trajectory=i, parameter=alpha)
                    rows.append(dict(**ident, **metrics))
                    slots.extend(dict(**ident, **r) for r in records)
                    decisions.extend(dict(**ident, **r) for r in ledger)
                    for name, value in independent(records, trace).items():
                        errors[name] = max(errors[name], value)
                    past = 0
                    for j, item in enumerate(ledger):
                        block = records[4*j:4*j+4]
                        floor = round(3*block[0]['outage'])+sum(3 for r in block[1:] if r['active_count'] == 0)
                        fail = sum(round(3*r['outage']) for r in block)
                        assert floor == item['executed_floor'] <= fail
                        assert item['executed_bits'] == block[0]['command_bits']
                        assert item['past_failed'] == past and item['block_failed'] == fail
                        assert abs(item['residual_budget']-max(0., 21-past)) < 1e-10
                        if not item['accepted']:
                            assert item['executed_bits'] == item['reference_bits']
                        if rule == 'Energy' and item['accepted']:
                            assert item['proposal_relative_energy'] < -1e-8
                        past += fail
                        assert item['cumulative_failed'] == past
                    ledgers.append(dict(**ident, final_failed=past, pathwise_target_exceeded=int(past > 21),
                        admission_count=sum(r['accepted'] for r in ledger),
                        differs_from_reference_count=sum(r['differs_from_reference'] for r in ledger),
                        rejected_empty=sum(r['reason'] == 'empty' for r in ledger),
                        rejected_energy=sum(r['reason'] == 'no_predicted_saving' for r in ledger)))
                print(f'anchor stream={stream} scenario={s} trace={i}, episodes={len(rows)}', flush=True)
    assert len(rows) == 240 and len(slots) == 33600 and len(decisions) == 8400
    assert errors['rate_mbps'] < 1e-5 and max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    all_rows = rows+sum((references(s) for s in STREAMS), [])
    write_rows(ROOT/'new_episodes.csv', rows)
    write_rows(ROOT/'all_episodes.csv', all_rows)
    write_rows(ROOT/'slots.csv', slots)
    write_rows(ROOT/'decisions.csv', decisions)
    write_rows(ROOT/'episode_ledgers.csv', ledgers)
    summary, paired_rows, gates = [], [], []
    for stream in STREAMS:
        group = [r for r in all_rows if r['stream'] == stream]
        summary.extend(dict(stream=stream, **r) for r in summaries(group))
        paired_rows.extend(paired(group, stream))
        gates.append(dict(stream=stream, **analyze(group)))
    write_rows(ROOT/'summary.csv', summary)
    write_rows(ROOT/'paired_differences.csv', paired_rows)
    result = dict(passed=all(r['passed'] for r in gates), primary='Energy Value', streams=gates,
                  both_streams_previously_inspected=True, independent_confirmation_launched=False)
    (ROOT/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    report = dict(new_episodes=240, retained_reference_episodes=len(all_rows)-240,
        verified_slots=33600, verified_decisions=8400, max_physical_errors=dict(errors),
        actual_commands_and_floors_verified=True, gate=result)
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    input_checks()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
