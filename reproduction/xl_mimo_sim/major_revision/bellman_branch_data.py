"""All-action four-slot transitions for a NEW bounded Bellman reconstruction.

This module does not train a policy, change the physical simulator, or turn an
evaluation cutoff into a terminal state. Future samples are labels only.
"""
import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.action_value import (
    action_size, baseline_command, full_slot_energy, observation_size,
)
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import representation_size
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.policy_consistent_value import (
    ROOT as PREDECESSOR_ROOT, decide, load_model, settings, verify_folder,
)

ROOT = Path('results/major_revision_20260921_bellman_branch')
TRAIN_STREAM = 1100000000
FEATURE_SETTING = dict(margin=1., minimum=1)
BLOCK = 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def known_costs(cfg, features):
    """Exact block energy and CURRENT-slot outage in the causal feature vector."""
    x = np.asarray(features)
    rows = x[..., representation_size(cfg):observation_size(cfg)]
    rows = rows.reshape(*x.shape[:-1], cfg.subarrays, action_size(cfg))
    energy = rows[..., cfg.subarrays:cfg.subarrays+BLOCK].astype(float).mean(-1)
    current = x[..., observation_size(cfg)+2:]
    current = current.reshape(*x.shape[:-1], cfg.subarrays, cfg.users+1)[..., -1]
    return energy, current


def block_transition(root, command):
    """Return actual costs and a causal successor, without mutating the root."""
    if root.t % BLOCK or root.t+BLOCK >= root.cfg.episode_steps:
        raise ValueError('Need a decision root and an available successor sample')
    trial = root.clone()
    start, energy = trial.t, trial.energy
    trial.step(command)
    first_outage = trial.outages[-1]
    for _ in range(BLOCK-1):
        trial.step()
    cost = np.array([(trial.energy-energy)/(BLOCK*full_slot_energy(root.cfg)),
                     np.mean(trial.outages[start:start+BLOCK])], dtype=np.float64)
    successor, _ = extend_observation(trial, FEATURE_SETTING)
    return cost, successor, float(first_outage)


def all_action_transitions(root):
    """Keep the action mapping, root features and all 16 counterfactual labels."""
    features, score = extend_observation(root, FEATURE_SETTING)
    costs, successors, current_outages = [], [], []
    for count in range(1, root.cfg.subarrays+1):
        cost, successor, current = block_transition(root, select_top(score, count))
        costs.append(cost)
        successors.append(successor)
        current_outages.append(current)
    after, after_score = extend_observation(root, FEATURE_SETTING)
    np.testing.assert_array_equal(features, after)
    np.testing.assert_array_equal(score, after_score)
    costs, successors = np.asarray(costs), np.asarray(successors)
    predicted_energy, predicted_current = known_costs(root.cfg, features)
    energy_error = float(np.max(np.abs(predicted_energy-costs[:, 0])))
    current_error = float(np.max(np.abs(predicted_current-current_outages)))
    if energy_error > 5e-7 or current_error > 5e-7:
        raise AssertionError('Known-cost identity failed')
    unknown_risk = (costs[:, 1]-np.asarray(current_outages)/BLOCK)/(1-1/BLOCK)
    if unknown_risk.min() < -1e-10 or unknown_risk.max() > 1+1e-10:
        raise AssertionError('Invalid remaining-three-slot risk label')
    return features, score, costs, successors, energy_error, current_error


def initialize(seed, smoke):
    verify_folder(PREDECESSOR_ROOT/f'fit_seed{seed}')
    folder = ROOT/(f'smoke_seed{seed}' if smoke else f'collect_seed{seed}')
    folder.mkdir(parents=True, exist_ok=False)
    files = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path(__file__).with_name('BELLMAN_BRANCH_PLAN.md'),
             PREDECESSOR_ROOT/f'fit_seed{seed}/mix_pi.pt']
    # The imported settings() table and all predecessor dependencies are verified
    # by verify_folder above; record its manifest as a chain-of-custody pointer.
    files += [PREDECESSOR_ROOT/f'fit_seed{seed}/manifest.json',
              PREDECESSOR_ROOT/f'fit_seed{seed}/output_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(
        seed=seed, smoke=smoke, stream=TRAIN_STREAM, threads=1, pid=os.getpid(),
        config=make_config('validation').as_dict(), feature_setting=FEATURE_SETTING,
        source_hashes=hashes, torch=str(torch.__version__), numpy=np.__version__,
        role='training transitions only; no new policy or performance claim'), indent=2))
    return folder, hashes


def collect(seed, smoke=False):
    folder, hashes = initialize(seed, smoke)
    base_cfg = make_config('validation')
    scored_slots = 12 if smoke else base_cfg.episode_steps
    cfg = replace(base_cfg, episode_steps=scored_slots+1)
    model, _ = load_model(seed, 'mix_pi')
    frozen_weights = {k: v.clone() for k, v in model.state_dict().items()}
    configs = settings()
    scenarios = [(1, SCENARIOS[1])] if smoke else list(enumerate(SCENARIOS))
    count = 1 if smoke else 4
    features, costs, successors, keys, chosen_actions, metrics = [], [], [], [], [], []
    energy_error = current_error = replay_error = 0.
    for rollin in (0, 1):
        for j, scenario in scenarios:
            setting = configs[j]
            for i in range(count):
                run = FullDomainRollout(cfg, trajectory(cfg, TRAIN_STREAM+j*10000+i, scenario))
                while run.t < scored_slots:
                    t = run.t
                    if rollin == 0:
                        chosen_command = baseline_command(run, 'cost', setting['margin'], setting['minimum'])
                        chosen = int(chosen_command.sum())-1
                    else:
                        _, policy_score, _, chosen = decide(run, setting, model)
                        chosen_command = select_top(policy_score, chosen+1)
                    x, score, c, nx, ee, ce = all_action_transitions(run)
                    np.testing.assert_array_equal(chosen_command, select_top(score, chosen+1))
                    energy_error, current_error = max(energy_error, ee), max(current_error, ce)
                    features.append(x)
                    costs.append(c)
                    successors.append(nx)
                    keys.append((rollin, j, i, t))
                    chosen_actions.append(chosen)
                    energy = run.energy
                    run.step(chosen_command)
                    for _ in range(BLOCK-1):
                        run.step()
                    actual = np.array([(run.energy-energy)/(BLOCK*full_slot_energy(cfg)),
                                       np.mean(run.outages[t:t+BLOCK])])
                    replay_error = max(replay_error, float(np.max(np.abs(actual-c[chosen]))))
                    np.testing.assert_array_equal(extend_observation(run, FEATURE_SETTING)[0], nx[chosen])
                    if replay_error > 1e-10:
                        raise AssertionError('Roll-in branch replay mismatch')
                metrics.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, **run.metrics()))
                print('Bellman transitions', seed, rollin, j, i+1, '/', count,
                      'roots', len(keys), flush=True)
    x, c, nx, key = map(np.asarray, (features, costs, successors, keys))
    n = 2*len(scenarios)*count*(scored_slots//BLOCK)
    assert x.shape == (n, 1500) and c.shape == (n, cfg.subarrays, 2)
    assert nx.shape == (n, cfg.subarrays, 1500) and len(set(map(tuple, key))) == n
    assert np.isfinite(x).all() and np.isfinite(c).all() and np.isfinite(nx).all()
    assert (c >= 0).all() and (c[:, :, 1] <= 1).all()
    assert np.all(key[:, 3]+BLOCK <= scored_slots)
    assert np.count_nonzero(key[:, 3] == scored_slots-BLOCK) == 2*len(scenarios)*count
    for k, v in frozen_weights.items():
        assert torch.equal(v, model.state_dict()[k])
    np.savez_compressed(folder/'transitions.npz', features=x, costs=c, successors=nx,
                        keys=key, rollin_action=np.array(chosen_actions, np.int64),
                        bootstrap=np.ones((n, cfg.subarrays), dtype=bool))
    _write_rows(folder/'rollin_episodes.csv', metrics)
    audit = dict(seed=seed, smoke=smoke, roots=n, transitions=n*cfg.subarrays,
                 branch_slots=n*cfg.subarrays*BLOCK,
                 exogenous_trajectories=len(scenarios)*count, scored_slots=scored_slots,
                 generated_samples=scored_slots+1, analytic_energy_error=energy_error,
                 current_outage_error=current_error, diagonal_replay_error=replay_error,
                 successor_replay_exact=True, real_cutoff_successor=True,
                 all_bootstrap_true=True, snapshots_unchanged=True, model_unchanged=True,
                 independent_test_episodes=0, learned_policy_performance_available=False)
    (folder/'audit.json').write_text(json.dumps(audit, indent=2))
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    (folder/'output_hashes.json').write_text(json.dumps(
        {str(p): digest(p) for p in folder.iterdir() if p.is_file()}, indent=2))
    (folder/'COMPLETE').write_text('Training transition collection complete; no performance claim.\n')
    print(json.dumps(audit), flush=True)


def audit_both():
    """Independent shape, accounting, successor and cross-seed duplicate checks."""
    data = []
    sources = {}
    cfg = make_config('validation')
    for seed in (11, 22):
        folder = ROOT/f'collect_seed{seed}'
        assert (folder/'COMPLETE').exists()
        manifest = json.loads((folder/'manifest.json').read_text())
        for path, expected in manifest['source_hashes'].items():
            assert digest(path) == expected, path
        for path, expected in json.loads((folder/'output_hashes.json').read_text()).items():
            assert digest(path) == expected, path
        d = np.load(folder/'transitions.npz', allow_pickle=False)
        assert d['features'].shape == (840, 1500)
        assert d['successors'].shape == (840, 16, 1500)
        assert d['bootstrap'].all()
        e, o = known_costs(cfg, d['features'])
        np.testing.assert_allclose(e, d['costs'][:, :, 0], atol=5e-7, rtol=0)
        residual = (d['costs'][:, :, 1]-o/4)/.75
        assert residual.min() >= -5e-7 and residual.max() <= 1+5e-7
        for rollin in (0, 1):
            for scenario in range(3):
                for i in range(4):
                    indices = np.flatnonzero(np.all(d['keys'][:, :3] == (rollin, scenario, i), axis=1))
                    assert len(indices) == 35
                    np.testing.assert_array_equal(d['keys'][indices, 3], np.arange(0, 140, 4))
                    selected = d['successors'][indices[:-1], d['rollin_action'][indices[:-1]]]
                    np.testing.assert_array_equal(selected, d['features'][indices[1:]])
        data.append(d)
        sources[str(folder/'transitions.npz')] = digest(folder/'transitions.npz')
    for name in ('features', 'costs', 'successors', 'keys', 'rollin_action', 'bootstrap'):
        left = data[0][name][data[0]['keys'][:, 0] == 0]
        right = data[1][name][data[1]['keys'][:, 0] == 0]
        np.testing.assert_array_equal(left, right)
    folder = ROOT/'data_audit'
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(passed=True, seed_count=2, roots=1680, action_transitions=26880,
                  unique_exogenous_trajectories=12, repeated_mu_roots_per_seed=420,
                  original_root_successor_chain_exact=True, all_actions_present=True,
                  duplicate_mu_arrays_exact=True, known_costs_verified=True,
                  no_terminal_cutoff=True, no_independent_performance_evidence=True,
                  input_hashes=sources, training_authorized_only_after_separate_protocol=True)
    (folder/'audit.json').write_text(json.dumps(report, indent=2))
    (folder/'COMPLETE').write_text('Transition contract passed; training protocol still required.\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('collect', 'smoke', 'audit'), required=True)
    parser.add_argument('--seed', choices=(11, 22), type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'audit':
        audit_both()
    else:
        collect(args.seed, smoke=args.stage == 'smoke')
