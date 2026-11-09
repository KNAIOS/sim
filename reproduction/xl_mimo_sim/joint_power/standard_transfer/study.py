"""Locked two-state/two-load standardized-channel transfer, no model selection."""
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

from future_vr.channel import Trajectory, generate_trajectory
from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.calibrated.histogram import load_predictor
from joint_power.sensitivity.study import intact as previous_intact
from joint_power.standard_transfer.control import REFERENCES, episode
from joint_power.standard_transfer.verify import independent
from joint_power.study import digest, write_rows

ROOT = Path('results/joint_power_20260922_standard_transfer')
CHANNEL_STREAM, DEMAND_STREAM, NUMBER = 5809220000, 5909220000, 20
LOADS, STATES = (.5, 1.), ('LOS', 'NLOS')
GENERATOR = Path('../standard_channel_20260920/umi_channel.py').resolve()


def intact():
    previous_intact()
    for root in (Path('results/joint_power_20260922_sensitivity'),
                 Path('results/joint_wideband_environment_checks_20260922')):
        hashes = json.loads((root/'output_hashes.json').read_text())
        hashes.update(json.loads((root/'manifest.json').read_text())['input_hashes'])
        assert all(digest(p) == h for p, h in hashes.items())


def specifications():
    return [('Value4', 11), ('Value4', 22)]+[(method, 0) for method in REFERENCES]


def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def one(task):
    state_index, index = task
    torch.set_num_threads(1)
    sys.path.insert(0, str(GENERATOR.parent))
    from umi_channel import generate, FC
    state = STATES[state_index]
    out = ROOT/f'{state}_{index}'
    out.mkdir(exist_ok=False)
    cfg = replace(make_config('validation'), array_aperture_m=15*.5*299792458/FC)
    channel_seed = CHANNEL_STREAM+state_index*10000+index
    demand_seed = DEMAND_STREAM+state_index*10000+index
    arrays, metadata = generate(channel_seed, frequencies=99, los=state == 'LOS')
    physical = arrays['channel']
    demands = generate_trajectory(make_config('validation'), demand_seed, 2.).requirements_mbps
    before = (array_hash(physical), array_hash(demands))
    metadata.update(demand_seed=demand_seed, channel_sha256=before[0], demand_sha256=before[1])
    (out/'channel_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if index == 0:
        np.savez_compressed(out/'example.npz', **arrays, requirements_mbps=demands)
    centre = physical[:, 49]
    gain = 10*np.log10(np.maximum((abs(centre.reshape(140, 3, 16, 8))**2).mean(3), 1e-30)).transpose(0, 2, 1)
    positions = np.repeat(arrays['ut_positions_m'][None, :, :2], 140, axis=0)
    velocities = np.repeat(arrays['ut_velocity_mps'][None, :, :2], 140, axis=0)
    loaded, predictor = models()[0], load_predictor()
    rows, slots, decisions, errors, modes = [], [], [], defaultdict(float), Counter()
    for load in LOADS:
        trace = Trajectory(centre, gain, np.ones_like(gain), positions, velocities, demands*load)
        for method, seed in specifications():
            metrics, records, ledger = episode(cfg, trace, physical, method, predictor,
                loaded['Visitation-added', seed] if seed else None)
            ident = dict(state=state, trajectory=index, load=load, method=method, seed=seed)
            rows.append(dict(**ident, **metrics))
            slots.extend(dict(**ident, **r) for r in records)
            decisions.extend(dict(**ident, **r) for r in ledger)
            modes.update(r['allocator_mode'] for r in records)
            for key, error in independent(cfg, records, physical, trace.requirements_mbps).items():
                errors[key] = max(errors[key], error)
    assert before == (array_hash(physical), array_hash(demands))
    assert len(rows) == 18 and len(slots) == 2520 and len(decisions) == 630
    for name, data in (('episodes', rows), ('slots', slots), ('decisions', decisions)):
        write_rows(out/f'{name}.csv', data)
    (out/'verification.json').write_text(json.dumps(dict(episodes=len(rows), slots=len(slots),
        maximum_errors=dict(errors), modes=dict(modes)), indent=2)+'\n')
    print(f'UMi transfer {state} drop {index+1}/{NUMBER}: 18 episodes/2520 slots verified', flush=True)
    return rows, dict(errors), dict(modes)


def analyze(rows):
    summaries, paired = [], []
    rng = np.random.default_rng(2026092236)
    fields = ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active', 'numerical_fallbacks')
    for state in STATES:
        for load in LOADS:
            group = [r for r in rows if (r['state'], r['load']) == (state, load)]
            lookup = {(r['method'], r['seed'], r['trajectory']): r for r in group}
            for method in ('Value4', *REFERENCES):
                values = [r for r in group if r['method'] == method]
                summaries.append(dict(state=state, load=load, method=method, independent_drops=NUMBER,
                    training_seeds=2 if method == 'Value4' else 0,
                    **{k: float(np.mean([r[k] for r in values])) for k in fields}))
            for reference in REFERENCES:
                for metric in ('energy_j', 'outage'):
                    delta = np.array([[lookup['Value4', seed, i][metric]-lookup[reference, 0, i][metric]
                        for i in range(NUMBER)] for seed in (11, 22)])
                    drops = rng.integers(0, NUMBER, (4000, NUMBER))
                    seeds = rng.integers(0, 2, (4000, 2))
                    boot = delta[seeds[:, :, None], drops[:, None, :]].mean((1, 2))
                    paired.append(dict(state=state, load=load, reference=reference, metric=metric,
                        difference=float(delta.mean()), ci_low=float(np.quantile(boot, .025)),
                        ci_high=float(np.quantile(boot, .975))))
    return summaries, paired


def main():
    torch.set_num_threads(1)
    intact()
    ROOT.mkdir(exist_ok=False)
    _, checkpoints = models()
    sources = [*Path('joint_power/standard_transfer').glob('*.py'),
        Path('joint_power/standard_transfer/PLAN.md'), Path('joint_power/standard_transfer/README.md'),
        GENERATOR, Path('joint_power/wideband_power.py'), Path('major_revision/budget_search.py'),
        Path('joint_power/model.py'), Path('joint_power/postdecision/encoding.py'),
        Path('joint_power/matched_four/control.py')]
    hashes = {str(p): digest(p) for p in sources}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, checkpoints=checkpoints,
        channel_stream=CHANNEL_STREAM, demand_stream=DEMAND_STREAM, independent_drops_per_state=NUMBER,
        loads=LOADS, method_specs=specifications(), alpha=.1, minimum_primary_command=4,
        no_retraining_or_selection=True, sionna=importlib.metadata.version('sionna-no-rt'),
        torch=torch.__version__, numpy=np.__version__, python=sys.version,
        host_environment_changed=False, original_references_preserved=True,
        scope='UMi compact far-field fading transfer, not moving near-field VR or field calibration'), indent=2)+'\n')
    tasks = [(state, i) for state in range(2) for i in range(NUMBER)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, tasks))
    rows = [r for group, _, _ in results for r in group]
    assert len(rows) == 720
    assert len({tuple(r[k] for k in ('state', 'load', 'method', 'seed', 'trajectory')) for r in rows}) == 720
    summary, paired = analyze(rows)
    errors = {k: max(e[k] for _, e, _ in results) for k in results[0][1]}
    modes = Counter()
    for _, _, local in results:
        modes.update(local)
    assert sum(modes.values()) == 100800
    assert errors['rate_mbps'] < 1e-5
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    for name, data in (('episodes', rows), ('summary', summary), ('paired', paired)):
        write_rows(ROOT/f'{name}.csv', data)
    report = dict(independent_channel_demand_pairs=40, load_cases=80, episodes=720, slots=100800,
        maximum_errors=errors, allocator_modes=dict(modes), all_results_retained=True,
        no_parameter_selection=True, R1_closed=False, R3_policy_evaluation_completed=True,
        limitations=['compact far-field array, no XL visibility-region calibration',
            'fixed topology and large-scale parameters, Doppler evolution within one CIR',
            'centre-frequency policy features, current active wideband CSI for physical floor',
            'analytical circuit/baseband power, not measured hardware energy',
            'two training seeds, no multiplicity-adjusted or per-episode safety guarantee'])
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(v['path']) == v['sha256'] for v in checkpoints.values())
    (ROOT/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in ROOT.rglob('*')
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
