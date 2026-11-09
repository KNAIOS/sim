"""Fresh NLOS/half-load comparison, reusing unchanged controllers and physics."""
import importlib.metadata
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

from future_vr.channel import Trajectory, generate_trajectory
from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.calibrated.histogram import ROOT as HELPER_ROOT, load_predictor
from joint_power.standard_transfer.control import episode
from joint_power.standard_transfer.study import GENERATOR, array_hash, intact
from joint_power.standard_transfer.verify import independent
from joint_power.study import digest, write_rows

ROOT = Path('results/joint_power_20260922_focus_confirmation')
PREVIOUS = Path('results/joint_power_20260922_standard_transfer')
NUMBER, CHANNEL_STREAM, DEMAND_STREAM = 40, 6009220000, 6109220000
SPECS = (('Value4', 11), ('Value4', 22), ('WBbudgetPRS4', 0))
METRICS = ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j',
           'mean_active', 'numerical_fallbacks')


def previous_intact():
    intact()
    before = json.loads((PREVIOUS/'manifest.json').read_text())['input_hashes']
    before.update(json.loads((PREVIOUS/'output_hashes.json').read_text()))
    assert all(digest(p) == h for p, h in before.items())


def configuration():
    return replace(make_config('validation'), array_aperture_m=15*.5*299792458/14e9)


def analyze(rows, number=NUMBER):
    expected = {(i, method, seed) for i in range(number) for method, seed in SPECS}
    assert len(rows) == len(expected)
    lookup = {(r['trajectory'], r['method'], r['seed']): r for r in rows}
    assert set(lookup) == expected and all(r['state'] == 'NLOS' and r['load'] == .5 for r in rows)
    summaries = []
    for method, seed in (('Value4', -1), *SPECS):
        selected = [r for r in rows if r['method'] == method and (seed == -1 or r['seed'] == seed)]
        summaries.append(dict(method=method, seed=seed, independent_drops=number,
            executions=len(selected), **{key: float(np.mean([r[key] for r in selected])) for key in METRICS}))
    rng = np.random.default_rng(2026092260)
    drops = rng.integers(0, number, (10000, number))
    seeds = rng.integers(0, 2, (10000, 2))
    comparisons, bootstraps = [], {}
    for metric in ('energy_j', 'outage'):
        learned = np.array([[lookup[i, 'Value4', seed][metric] for i in range(number)] for seed in (11, 22)])
        baseline = np.array([lookup[i, 'WBbudgetPRS4', 0][metric] for i in range(number)])
        lboot = learned[seeds[:, :, None], drops[:, None, :]].mean((1, 2))
        bboot = baseline[drops].mean(1)
        delta = lboot-bboot
        comparisons.append(dict(reference='WBbudgetPRS4', metric=metric,
            difference=float(learned.mean()-baseline.mean()),
            ci_low=float(np.quantile(delta, .025)), ci_high=float(np.quantile(delta, .975))))
        bootstraps[metric] = (lboot, bboot)
    learned, baseline = summaries[0], summaries[-1]
    upper_l, upper_b = [float(np.quantile(x, .95)) for x in bootstraps['outage']]
    energy_ok = comparisons[0]['ci_high'] < 0
    reliability_ok = upper_l <= .05 and upper_b <= .05
    report = dict(independent_drops=number, episodes=len(rows), slots=140*len(rows),
        energy_saving_percent=100*(baseline['energy_j']-learned['energy_j'])/baseline['energy_j'],
        energy_difference_ci95=[comparisons[0]['ci_low'], comparisons[0]['ci_high']],
        learned_mean_outage=learned['outage'], reference_mean_outage=baseline['outage'],
        learned_outage_upper95=upper_l, reference_outage_upper95=upper_b,
        energy_supported=energy_ok, reliability_supported=reliability_ok,
        independent_confirmation_passed=bool(energy_ok and reliability_ok),
        globally_optimal_proven=False, equal_reliability_proven=False,
        old_exploratory_sample_pooled=False, resamples=10000, bootstrap_seed=2026092260)
    return summaries, comparisons, report


def one(index):
    torch.set_num_threads(1)
    sys.path.insert(0, str(GENERATOR.parent))
    from umi_channel import generate
    out = ROOT/f'NLOS_{index}'
    out.mkdir(exist_ok=False)
    arrays, metadata = generate(CHANNEL_STREAM+index, frequencies=99, los=False)
    physical = arrays['channel']
    demand = generate_trajectory(make_config('validation'), DEMAND_STREAM+index, 2.).requirements_mbps*.5
    before = array_hash(physical), array_hash(demand)
    metadata.update(demand_seed=DEMAND_STREAM+index, load=.5,
                    channel_sha256=before[0], half_load_demand_sha256=before[1])
    (out/'channel_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if index in (0, NUMBER-1):
        np.savez_compressed(out/'example.npz', **arrays, requirements_mbps=demand)
    centre = physical[:, 49]
    gain = 10*np.log10(np.maximum((abs(centre.reshape(140, 3, 16, 8))**2).mean(3), 1e-30)).transpose(0, 2, 1)
    positions = np.repeat(arrays['ut_positions_m'][None, :, :2], 140, axis=0)
    velocities = np.repeat(arrays['ut_velocity_mps'][None, :, :2], 140, axis=0)
    trace = Trajectory(centre, gain, np.ones_like(gain), positions, velocities, demand)
    loaded, predictor = models()[0], load_predictor()
    rows, slots, decisions, errors, modes = [], [], [], defaultdict(float), Counter()
    for method, seed in SPECS:
        metrics, records, ledger = episode(configuration(), trace, physical, method, predictor,
            loaded['Visitation-added', seed] if seed else None)
        ident = dict(state='NLOS', trajectory=index, load=.5, method=method, seed=seed)
        rows.append(dict(**ident, **metrics))
        slots.extend(dict(**ident, **r) for r in records)
        decisions.extend(dict(**ident, **r) for r in ledger)
        modes.update(r['allocator_mode'] for r in records)
        for key, error in independent(configuration(), records, physical, demand).items():
            errors[key] = max(errors[key], error)
    assert before == (array_hash(physical), array_hash(demand))
    assert len(rows) == 3 and len(slots) == 420 and len(decisions) == 105
    for name, data in (('episodes', rows), ('slots', slots), ('decisions', decisions)):
        write_rows(out/f'{name}.csv', data)
    (out/'verification.json').write_text(json.dumps(dict(maximum_errors=dict(errors),
        episodes=3, slots=420, modes=dict(modes)), indent=2)+'\n')
    print(f'Independent focused drop {index+1}/{NUMBER}: 3 episodes, 420 slots verified', flush=True)
    return rows, dict(errors), dict(modes)


def main():
    torch.set_num_threads(1)
    previous_intact()
    ROOT.mkdir(exist_ok=False)
    _, all_models = models()
    checkpoints = {key: value for key, value in all_models.items() if key.startswith('Visitation-added ')}
    assert {v['epoch'] for v in checkpoints.values()} == {20}
    paths = [p for folder in ('future_vr', 'joint_power', 'major_revision') for p in Path(folder).rglob('*.py')]
    paths += [GENERATOR, Path('joint_power/focus_confirmation/PLAN.md'),
              HELPER_ROOT/'calibration/selected.pt', PREVIOUS/'manifest.json', PREVIOUS/'output_hashes.json']
    hashes = {str(p): digest(p) for p in paths}
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), input_hashes=hashes,
        checkpoints=checkpoints, number=NUMBER, channel_stream=CHANNEL_STREAM, demand_stream=DEMAND_STREAM,
        specifications=SPECS, state='NLOS', load=.5, minimum_command=4, alpha=.1,
        sionna=importlib.metadata.version('sionna-no-rt'), torch=torch.__version__,
        numpy=np.__version__, python=sys.version, workers=2, threads_per_worker=1,
        new_training=False, previous_data_reused=False, host_environment_changed=False)
    # Written before ProcessPoolExecutor can generate any fresh experimental data.
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, range(NUMBER)))
    rows = [r for group, _, _ in results for r in group]
    summaries, paired, report = analyze(rows)
    errors = {key: max(local[key] for _, local, _ in results) for key in results[0][1]}
    modes = Counter()
    for _, _, counts in results:
        modes.update(counts)
    assert errors['rate_mbps'] < 1e-5 and max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    assert sum(modes.values()) == 16800
    report.update(maximum_replay_errors=errors, allocator_modes=dict(modes),
                  all_prespecified_samples_retained=True, completed_utc=datetime.now(timezone.utc).isoformat())
    for name, data in (('episodes', rows), ('summary', summaries), ('paired', paired)):
        write_rows(ROOT/f'{name}.csv', data)
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    previous_intact()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(v['path']) == v['sha256'] for v in checkpoints.values())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.rglob('*')
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
