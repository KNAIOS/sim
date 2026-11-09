"""Post-hoc target-risk opportunity; no new simulation or deployable policy."""
import csv
import json
from pathlib import Path

import numpy as np

from joint_power.strong_headroom.study import ROOT, NUMBER, FIELDS, intact
from joint_power.study import digest, write_rows


def choose(y, reference, past, allowance, horizon):
    valid = np.flatnonzero(y[:, -1, 1]+past <= allowance)
    if not len(valid):
        return reference, False
    return min(valid, key=lambda a: (y[a, horizon, 0], a != reference, a)), True


def parse_episode(row):
    return dict(scenario=int(row['scenario']), trajectory=int(row['trajectory']),
                reference=row['reference'], energy_j=float(row['energy_j']),
                outage=float(row['outage']))


def main():
    intact()
    hashes = json.loads((ROOT/'output_hashes.json').read_text())
    manifest = json.loads((ROOT/'manifest.json').read_text())
    assert all(digest(p) == h for p, h in (hashes | manifest['input_hashes']).items())
    out = ROOT/'target_budget_v2'
    out.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/strong_headroom/BUDGET_PLAN.md'), ROOT/'output_hashes.json']
    own = {str(p): digest(p) for p in files}
    (out/'manifest.json').write_text(json.dumps(dict(input_hashes=own, posthoc=True,
        no_new_simulation=True, uses_future_outcomes=True), indent=2)+'\n')
    with (ROOT/'episodes.csv').open(newline='', encoding='utf-8') as stream:
        episodes = {(r['scenario'], r['trajectory'], r['reference']): r
                    for r in map(parse_episode, csv.DictReader(stream))}
    rows = []
    for path in sorted(ROOT.glob('labels_*.npz')):
        with np.load(path) as data:
            for n, (s, i, adaptive, slot) in enumerate(data['identity']):
                name = 'Expanded budgetPRS' if adaptive else 'Expanded PRS'
                y, ref, past = data['y'][n], int(data['reference'][n]), int(data['past'][n])
                episode = episodes[int(s), int(i), name]
                for horizon in (0, 2):
                    a, feasible = choose(y, ref, past, 21, horizon)
                    delta = y[a, -1]-y[ref, -1]
                    full_energy = episode['energy_j']+delta[0]
                    failed = past+y[a, -1, 1]
                    assert failed <= 21 if feasible else a == ref
                    rows.append(dict(scenario=int(s), trajectory=int(i), reference=name,
                        slot=int(slot), horizon=4 if horizon == 0 else -1, feasible=feasible,
                        past_failed=past, selected_action=int(a), selected_count=int(data['masks'][n, a].sum()),
                        reference_episode_j=episode['energy_j'], intervention_episode_j=float(full_energy),
                        reference_outage=episode['outage'], intervention_outage=float(failed/420),
                        relative_saving=float(-delta[0]/episode['energy_j']),
                        **{f'delta_{key}': float(delta[k]) for k, key in enumerate(FIELDS)}))
    assert len(rows) == 144
    rng, summary = np.random.default_rng(2026092231), []
    for s in range(3):
        for name in ('Expanded PRS', 'Expanded budgetPRS'):
            for horizon in (4, -1):
                group = [r for r in rows if (r['scenario'], r['reference'], r['horizon']) == (s, name, horizon)]
                trace_delta = np.array([np.mean([r['delta_energy_j'] for r in group if r['trajectory'] == i])
                                        for i in range(NUMBER)])
                draws = trace_delta[rng.integers(0, NUMBER, (4000, NUMBER))].mean(1)
                summary.append(dict(scenario=s, reference=name, horizon=horizon, roots=len(group),
                    empty_sets=sum(not r['feasible'] for r in group),
                    **{key: float(np.mean([r[key] for r in group])) for key in
                       ('reference_episode_j', 'intervention_episode_j', 'reference_outage',
                        'intervention_outage', 'relative_saving', *[f'delta_{key}' for key in FIELDS])},
                    ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975))))
    selected = {s: {r['horizon']: r for r in summary
                    if r['scenario'] == s and r['reference'] == 'Expanded budgetPRS'} for s in range(3)}
    gain = selected[1][-1]['relative_saving']
    gap = gain-selected[1][4]['relative_saving']
    gate = dict(budget_conditioned_probe_justified=bool(gain >= .02 and all(
        selected[s][-1]['intervention_outage'] <= .05 for s in range(3))),
        long_energy_head_motivated=bool(gap >= .01), nominal_relative_saving=gain,
        long_energy_gap_fraction=gap, original_conservative_gate_unchanged=True,
        hindsight_training_diagnostic_only=True, R1_closed=False)
    write_rows(out/'roots.csv', rows)
    write_rows(out/'summary.csv', summary)
    (out/'report.json').write_text(json.dumps(dict(gate=gate, summary=summary), indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in (hashes | manifest['input_hashes'] | own).items())
    (out/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in out.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
