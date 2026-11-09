"""Complete base-policy continuations: hindsight labels, not a deployed oracle."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.calibration import branch
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment, search_action
from joint_power.postdecision.encoding import decision
from joint_power.postdecision.study import intact as previous_intact
from joint_power.study import SCENARIOS, digest, make_trace, write_rows
from major_revision.budget_search import search

ROOT = Path('results/joint_power_20260922_rollout')
STREAMS = {'train': (3609220000, 8), 'validation': (3709220000, 4)}
SLOTS = tuple(range(0, 113, 16))


def base_command(env):
    env.prepare()
    return search(env.cfg, env.tracker, env.hw, env.req)


def rollout(env, command):
    """One intervention, then causal PRS on action-dependent observations."""
    alternative = branch(env)
    alternative.block(command)
    while alternative.t < alternative.cfg.episode_steps:
        alternative.block(base_command(alternative))
    energy = np.array([r['energy_j'] for r in alternative.records])
    failures = np.rint([r['outage']*env.cfg.users for r in alternative.records]).astype(int)
    return np.array([[energy[:h].sum(), failures[:h].sum()]
                     for h in (4, 16, len(energy))])


def state_labels(env, predictor):
    candidates = env.candidates()
    state = decision(env, candidates, predictor)
    baseline = base_command(env)
    reference = int(baseline.sum())-1
    np.testing.assert_array_equal(state.masks[reference], baseline)
    # Finish every current observation/feature before looking at outcome labels.
    features = state.x.copy()
    unique = {}
    labels = []
    for command in state.masks:
        bits = command.tobytes()
        if bits not in unique:
            unique[bits] = rollout(env, command)
        labels.append(unique[bits])
    return dict(x=features, y=np.array(labels), reference=reference,
                raw_action=search_action(candidates), masks=state.masks)


def collect_one(task):
    split, scenario, index = task
    torch.set_num_threads(1)
    cfg, predictor = make_config('validation'), load_predictor()
    seed = STREAMS[split][0]+scenario*10000+index
    trace = make_trace(cfg, seed, SCENARIOS[scenario])
    before = (trace.channel.copy(), trace.requirements_mbps.copy())
    env, data = Environment(cfg, trace), []
    while env.t < cfg.episode_steps:
        if env.t in SLOTS:
            start = env.t
            labels = state_labels(env, predictor)
            assert env.t == start
            data.append(dict(slot=start, **labels))
        env.block(base_command(env))
    for row in data:
        suffix = env.records[row['slot']:]
        reference = [sum(r['energy_j'] for r in suffix),
                     sum(round(r['outage']*cfg.users) for r in suffix)]
        np.testing.assert_allclose(row['y'][row['reference'], -1], reference, atol=1e-9)
    np.testing.assert_array_equal(trace.channel, before[0])
    np.testing.assert_array_equal(trace.requirements_mbps, before[1])
    identity = np.array([[scenario, index, r['slot']] for r in data], dtype=np.int32)
    output = dict(x=np.array([r['x'] for r in data]), y=np.array([r['y'] for r in data]),
        reference=np.array([r['reference'] for r in data], dtype=np.int64),
        raw_action=np.array([r['raw_action'] for r in data], dtype=np.int64),
        masks=np.array([r['masks'] for r in data]), identity=identity)
    np.savez_compressed(ROOT/f'{split}_{scenario}_{index}.npz', **output)
    print(f'rollout labels {split} scenario={scenario} trajectory={index}: '
          f'{len(data)} states, complete causal-PRS suffix check passed', flush=True)
    return dict(split=split, scenario=scenario, trajectory=index, states=len(data),
                trajectory_seed=seed, baseline_energy_j=env.metrics()['energy_j'],
                baseline_outage=env.metrics()['outage'])


def main():
    previous_intact()
    ROOT.mkdir(parents=True, exist_ok=False)
    files = [*Path('joint_power/rollout').glob('*.py'), Path('joint_power/rollout/PLAN.md'),
             Path('joint_power/model.py'), Path('major_revision/budget_search.py'),
             *Path('future_vr').glob('*.py'), *Path('joint_power/calibrated').glob('*.py'),
             Path('joint_power/postdecision/encoding.py'),
             Path('results/joint_power_20260922_phase2b/calibration/selected.pt'),
             Path('results/joint_power_20260922_phase3/output_hashes.json')]
    hashes = {str(p): digest(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        streams=STREAMS, slots=SLOTS, config=make_config('validation').as_dict(),
        scope='Offline simulator outcome labels, not deployable hindsight performance',
        host_environment_changed=False), indent=2)+'\n')
    tasks = [(split, scenario, i) for split, (_, number) in STREAMS.items()
             for scenario in range(3) for i in range(number)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(collect_one, tasks))
    for split, (_, number) in STREAMS.items():
        payloads = []
        for scenario in range(3):
            for i in range(number):
                with np.load(ROOT/f'{split}_{scenario}_{i}.npz') as saved:
                    payloads.append({key: saved[key] for key in saved.files})
        np.savez_compressed(ROOT/f'{split}.npz', **{key: np.concatenate([p[key] for p in payloads])
                                                  for key in payloads[0]})
    write_rows(ROOT/'trajectories.csv', rows)
    previous_intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (ROOT/'collection_complete.json').write_text(json.dumps(dict(trajectories=len(rows),
        states=sum(r['states'] for r in rows), baseline_suffixes_verified=True), indent=2)+'\n')
    outputs = {str(p): digest(p) for p in ROOT.rglob('*') if p.is_file()}
    (ROOT/'collection_hashes.json').write_text(json.dumps(outputs, indent=2)+'\n')


if __name__ == '__main__':
    main()
