"""Small causal margin policy, optimized by complete episodic returns (no critic)."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import full_slot_energy, hold_predictions
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260921_direct_policy')
TRAIN_STREAM = 1120000000
VAL_STREAM = 1130000000
MARGINS = (.85, .90, 1., 1.04, 1.08, 1.16, 1.24, 1.35)
ITERATIONS, DIRECTIONS, TOP = 12, 4, 2
NOISE, STEP, PRICE = .08, .04, 4.
VARIANTS = ('state', 'bias')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_index(prediction, requirements, margin, outage_target=.05):
    """Identical lexicographic rule to existing cost-prefix search."""
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('Positive finite prediction margin required')
    req = requirements[None, None, :] * margin
    outage = np.mean(prediction['rates'] < req, axis=(1, 2))
    shortfall = np.mean(np.maximum(req - prediction['rates'], 0.) / req, axis=(1, 2))
    energy = prediction['energy'].sum(axis=1)
    def key(i):
        feasible = outage[i] <= outage_target
        return (not feasible, energy[i] if feasible else shortfall[i], energy[i], i)
    return min(range(len(energy)), key=key)


def features(requirements, age, active, countdown, velocity, last_outage, prediction,
             reference):
    """Only provided causal arrays; no access to the trajectory or simulator."""
    ratios = prediction['rates'][reference] / requirements[None, :]
    start, end = np.clip([ratios[0].min(), ratios[-1].min()], 0., 2.)
    return np.asarray([1., requirements.mean() / 560. - 1., start - 1., end - start,
        np.minimum(age, 32.).mean() / 32., np.mean(countdown > 0),
        np.mean(active) - .5, np.clip(np.mean(np.abs(velocity)) / .55, 0., 1.),
        last_outage], dtype=np.float64)


def margin_value(theta, phi, base_margin=1.04):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape not in ((1,), (9,)) or not np.isfinite(theta).all():
        raise ValueError('One or nine finite policy weights required')
    if not np.isfinite(phi).all() or not .85 <= base_margin <= 1.35:
        raise ValueError('Invalid causal feature or base margin')
    return float(np.clip(base_margin * np.exp(np.clip(theta @ phi[:len(theta)], -2., 2.)),
                         .85, 1.35))


def decision(run, theta, base_margin=1.04):
    _, score = run.base_observation()
    p = hold_predictions(run.cfg, run.tracker, run.hw, score)
    reference = select_index(p, run.req, 1.04, run.cfg.outage_target)
    phi = features(run.req, run.age, run.hw.active, run.hw.countdown,
                   run.tracker.velocity, run.last_outage, p, reference)
    margin = margin_value(theta, phi, base_margin)
    action = select_index(p, run.req, margin, run.cfg.outage_target)
    return p['masks'][action].copy(), phi, margin, action


def episode(cfg, tr, theta, base_margin=1.04, record=False):
    run, decisions = FullDomainRollout(cfg, tr), []
    margins = []
    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
            continue
        command, phi, margin, action = decision(run, theta, base_margin)
        if record:
            decisions.append(dict(t=run.t, action=action, margin=margin, features=phi.tolist(),
                                  command=command.astype(int).tolist()))
        margins.append(margin)
        run.step(command)
    result = run.metrics()
    result.update(mean_margin=float(np.mean(margins)), min_margin=float(np.min(margins)),
                  max_margin=float(np.max(margins)))
    if record:
        return result, decisions
    return result


def ars_update(theta, directions, rewards):
    """ARS-V1-t-style ascent; rewards columns are plus, minus."""
    directions, rewards = np.asarray(directions), np.asarray(rewards)
    if directions.shape != (DIRECTIONS, len(theta)) or rewards.shape != (DIRECTIONS, 2):
        raise ValueError('Unexpected perturbation/return dimensions')
    if not np.isfinite(directions).all() or not np.isfinite(rewards).all():
        raise ValueError('Non-finite training data')
    order = np.argsort(-rewards.max(axis=1), kind='stable')[:TOP]
    sigma = float(rewards[order].std(ddof=0))
    if sigma <= 1e-12:
        return theta.copy(), order, sigma
    update = STEP / (TOP * sigma) * np.sum(
        (rewards[order, 0] - rewards[order, 1])[:, None] * directions[order], axis=0)
    return theta + update, order, sigma


def initialize(part, extra=()):
    folder = ROOT / part
    folder.mkdir(parents=True, exist_ok=False)
    files = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path(__file__).with_name('DIRECT_POLICY_PLAN.md'), *map(Path, extra)]
    hashes = {str(p): digest(p) for p in files}
    manifest = dict(part=part, hashes=hashes, config=make_config('validation').as_dict(),
                    pid=os.getpid(), threads=1, train_stream=TRAIN_STREAM, validation_stream=VAL_STREAM,
                    iterations=ITERATIONS, directions=DIRECTIONS, top=TOP, noise=NOISE, step=STEP,
                    price=PRICE, variants=VARIANTS, margins=MARGINS,
                    numpy=np.__version__, torch=torch.__version__, independent_confirmation=False)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return folder, hashes


def verify(folder):
    assert (folder / 'COMPLETE').exists(), folder
    for path, expected in json.loads((folder / 'manifest.json').read_text())['hashes'].items():
        assert digest(path) == expected, path
    for path, expected in json.loads((folder / 'output_hashes.json').read_text()).items():
        assert digest(path) == expected, path


def finish(folder, hashes):
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    outputs = {str(p): digest(p) for p in folder.iterdir() if p.is_file()}
    (folder / 'output_hashes.json').write_text(json.dumps(outputs, indent=2))
    (folder / 'COMPLETE').write_text('Stage complete; not scientific promotion.\n')


def train(seed):
    folder, hashes = initialize(f'train_seed{seed}')
    cfg = make_config('validation')
    full_energy = cfg.episode_steps * full_slot_energy(cfg)
    started, raw, log = time.monotonic(), [], []
    for variant in VARIANTS:
        # Deliberately reset to the same seed: fixed matched optimizer budget,
        # not a claim that different-dimensional Gaussian arrays are identical.
        rng = np.random.default_rng(seed)
        theta = np.zeros(9 if variant == 'state' else 1, dtype=np.float64)
        weights, perturbations, returns = [theta.copy()], [], []
        for iteration in range(ITERATIONS):
            traces = [(j, i, trajectory(cfg, TRAIN_STREAM + j*10000 + 2*iteration+i, sc))
                      for j, sc in enumerate(SCENARIOS) for i in range(2)]
            directions = rng.normal(size=(DIRECTIONS, len(theta)))
            rewards = np.zeros((DIRECTIONS, 2), dtype=np.float64)
            for k, delta in enumerate(directions):
                for sign_index, sign in enumerate((1., -1.)):
                    for j, i, tr in traces:
                        metric = episode(cfg, tr, theta + sign*NOISE*delta)
                        reward = -(metric['energy_j']/full_energy + PRICE*metric['outage'])
                        rewards[k, sign_index] += reward/len(traces)
                        raw.append(dict(seed=seed, variant=variant, iteration=iteration,
                            direction=k, sign=int(sign), scenario=j, trajectory=2*iteration+i,
                            reward=reward, **metric))
            theta, order, sigma = ars_update(theta, directions, rewards)
            weights.append(theta.copy())
            perturbations.append(directions)
            returns.append(rewards)
            log.append(dict(seed=seed, variant=variant, iteration=iteration+1, sigma=sigma,
                selected=order.tolist(), mean_reward=float(rewards.mean()),
                elapsed_s=time.monotonic()-started))
            np.savez_compressed(folder/f'{variant}_history.npz', weights=np.asarray(weights),
                directions=np.asarray(perturbations), rewards=np.asarray(returns))
            _write_rows(folder/'training_raw.csv', raw)
            (folder/'progress.json').write_text(json.dumps(log, indent=2))
            print('direct policy', seed, variant, iteration+1, '/', ITERATIONS,
                  'mean return', round(float(rewards.mean()), 6),
                  'elapsed', round(time.monotonic()-started, 1), flush=True)
        assert len(weights) == 13
    assert len(raw) == 1152
    finish(folder, hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', choices=(11, 22), type=int, required=True)
    torch.set_num_threads(1)
    train(parser.parse_args().seed)
