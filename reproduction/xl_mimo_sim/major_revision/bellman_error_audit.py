"""Frozen Bellman error decomposition; hindsight appears only in diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.absolute_return_screen import search_command
from major_revision.action_value import full_slot_energy
from major_revision.bellman_branch_data import (
    ROOT as FIT_ROOT, FEATURE_SETTING, all_action_transitions, digest,
)
from major_revision.bellman_screen import STREAM, read_rows, verify_fit
from major_revision.bellman_value import BellmanValue, DEPTH, greedy_cost
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260921_bellman_error')
TIMES = (0, 32, 64)
NAMES = ('learned', 'search', 'immediate_oracle')


def freeze_model(seed):
    verify_fit(seed)
    checkpoint = torch.load(FIT_ROOT/f'fit_seed{seed}/final.pt', weights_only=True)
    assert checkpoint['depth'] == DEPTH
    model = BellmanValue(make_config('validation'))
    model.load_state_dict(checkpoint['online'], strict=True)
    return model.eval()


def policy_values(run, model, price):
    x, score = extend_observation(run, FEATURE_SETTING)
    with torch.no_grad():
        tx = torch.from_numpy(x[None])
        immediate = model.immediate(tx)[0].numpy()
        long = model(tx, price, DEPTH)[0].numpy()
    action = int(np.argmin(long[:, 0]+price*long[:, 1]))
    return x, score, immediate, long, action


def oracle_action(predicted_immediate, predicted_long, actual_immediate, price):
    """NONCAUSAL diagnostic: replace root risk only, never a deployed policy."""
    changed = predicted_long[:, 0]+price*(predicted_long[:, 1]-predicted_immediate[:, 1]
                                         +actual_immediate[:, 1])
    return int(np.argmin(changed))


def deployment_return(root, command, model, price, slots=64):
    if root.t % 4 or slots % 4 or root.t+slots > root.cfg.episode_steps:
        raise ValueError('Need complete four-slot blocks within physical trajectory')
    trial = root.clone()
    start = trial.t
    blocks = []
    while trial.t < start+slots:
        if trial.t == start:
            chosen = command
        else:
            _, score, _, _, action = policy_values(trial, model, price)
            chosen = select_top(score, action+1)
        energy = trial.energy
        trial.step(chosen)
        for _ in range(3):
            trial.step()
        blocks.append([(trial.energy-energy)/(4*full_slot_energy(root.cfg)),
                       np.mean(trial.outages[-4:])])
    blocks = np.asarray(blocks)
    total = np.sum(blocks*root.cfg.rl_gamma**np.arange(len(blocks))[:, None], axis=0)
    return total, blocks


def error_components(pred4, pred64, actual4, actual64, price):
    """Signed components of predicted minus actual cost, never MAE percentages."""
    return np.array([pred4[0]-actual4[0], price*(pred4[1]-actual4[1]),
                     (pred64[0]-pred4[0])-(actual64[0]-actual4[0]),
                     price*((pred64[1]-pred4[1])-(actual64[1]-actual4[1]))])


def configurations(seed):
    selected = read_rows(FIT_ROOT/'screen_audit/selected.csv')
    means = read_rows(FIT_ROOT/'screen_audit/all_means.csv')
    prices, searches = {}, {}
    for j in range(3):
        prices[j] = next(r['price'] for r in selected if r['scenario'] == j and
                         r['seed'] == seed and r['method'] == 'bellman')
        candidates = [r for r in means if r['scenario'] == j and r['seed'] == 0 and
                      r['method'] != 'full' and r['outage'] <= .05]
        searches[j] = min(candidates, key=lambda r: (r['energy_j'], r['outage']))
    return prices, searches


def initialize(seed):
    gate = json.loads((FIT_ROOT/'screen_audit/gate.json').read_text())
    assert gate['permit_confirmation'] is False
    folder = ROOT/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    files = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path(__file__).with_name('BELLMAN_ERROR_PLAN.md'),
             FIT_ROOT/f'fit_seed{seed}/final.pt', FIT_ROOT/f'collect_seed{seed}/transitions.npz',
             *[FIT_ROOT/'screen_audit'/f for f in ('selected.csv', 'all_means.csv', 'gate.json')],
             FIT_ROOT/'screen_learned/raw.csv', FIT_ROOT/'screen_baselines/raw.csv']
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(seed=seed, stream=STREAM, trajectories_per_scenario=4,
        times=TIMES, threads=1, config=make_config('validation').as_dict(), input_hashes=hashes,
        role='viewed-data frozen-model diagnosis; oracle is not deployable; not training'), indent=2))
    return folder, hashes


def run_audit(seed):
    model = freeze_model(seed)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    prices, searches = configurations(seed)
    folder, hashes = initialize(seed)
    cfg = make_config('validation')
    previous_l = read_rows(FIT_ROOT/'screen_learned/raw.csv')
    previous_b = read_rows(FIT_ROOT/'screen_baselines/raw.csv')
    features, keys, immediate, long, actual4, actions, actual64, all_blocks = [], [], [], [], [], [], [], []
    episodes = []
    episode_error = suffix_error = first_block_error = known_energy_error = 0.
    unique_long_branches = 0
    for rollin in (0, 1):
        for j, scenario in enumerate(SCENARIOS):
            price, search = prices[j], searches[j]
            for i in range(4):
                run = FullDomainRollout(cfg, trajectory(cfg, STREAM+j*10000+i, scenario))
                snapshots = []
                while run.t < cfg.episode_steps:
                    if run.t % 4:
                        run.step()
                        continue
                    x, score, pi4, pi64, learned = policy_values(run, model, price)
                    scmd = search_command(run, search['method'], search['margin'], int(search['minimum']))
                    sa = int(scmd.sum())-1
                    np.testing.assert_array_equal(scmd, select_top(score, sa+1))
                    if run.t in TIMES:
                        snapshots.append((run.clone(), x, score, pi4, pi64, learned, sa))
                    run.step(select_top(score, learned+1) if rollin == 0 else scmd)
                if rollin == 0:
                    reference = next(r for r in previous_l if r['scenario'] == j and r['trajectory'] == i and
                        r['seed'] == seed and r['method'] == 'bellman' and r['price'] == price)
                else:
                    reference = next(r for r in previous_b if r['scenario'] == j and r['trajectory'] == i and
                        (r['method'], r['minimum'], r['margin']) == (search['method'], search['minimum'], search['margin']))
                result = run.metrics()
                for field in result:
                    if field != 'method':
                        episode_error = max(episode_error, abs(result[field]-reference[field]))
                assert episode_error < 1e-10
                episodes.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, **result))
                for root, x, score, pi4, pi64, learned, sa in snapshots:
                    before = x.copy()
                    ax, _, c4, _, ee, _ = all_action_transitions(root)
                    np.testing.assert_array_equal(ax, before)
                    known_energy_error = max(known_energy_error, ee)
                    oracle = oracle_action(pi4, pi64, c4, price)
                    chosen = (learned, sa, oracle)
                    cache = {}
                    for a in set(chosen):
                        total, blocks = deployment_return(root, select_top(score, a+1), model, price)
                        first_block_error = max(first_block_error, float(np.max(np.abs(blocks[0]-c4[a]))))
                        assert first_block_error < 1e-10
                        if rollin == 0 and a == learned:
                            expected = np.array([[sum(run.slot_energy[t:t+4])/(4*full_slot_energy(cfg)),
                                                  np.mean(run.outages[t:t+4])]
                                                 for t in range(root.t, root.t+64, 4)])
                            suffix_error = max(suffix_error, float(np.max(np.abs(expected-blocks))))
                            assert suffix_error < 1e-10
                        cache[a] = (total, blocks)
                    unique_long_branches += len(cache)
                    features.append(x)
                    keys.append((rollin, j, i, root.t))
                    immediate.append(pi4)
                    long.append(pi64)
                    actual4.append(c4)
                    actions.append(chosen)
                    actual64.append([cache[a][0] for a in chosen])
                    all_blocks.append([cache[a][1] for a in chosen])
                    np.testing.assert_array_equal(extend_observation(root, FEATURE_SETTING)[0], before)
                print('Bellman error audit', seed, rollin, j, i+1, '/4', 'roots', len(keys), flush=True)
    assert len(keys) == 72 and len(set(keys)) == 72
    arrays = dict(features=np.asarray(features), keys=np.asarray(keys, np.int64),
                  predicted4=np.asarray(immediate), predicted64=np.asarray(long), actual4=np.asarray(actual4),
                  actions=np.asarray(actions, np.int64), actual64=np.asarray(actual64), blocks=np.asarray(all_blocks))
    np.savez_compressed(folder/'roots.npz', **arrays)
    training = np.load(FIT_ROOT/f'collect_seed{seed}/transitions.npz', allow_pickle=False)
    selected = np.isin(training['keys'][:, 3], TIMES)
    tx = torch.from_numpy(training['features'][selected])
    training_keys = training['keys'][selected]
    with torch.no_grad():
        pred4 = model.immediate(tx).numpy()
        ta = np.empty(len(tx), dtype=np.int64)
        for j in range(3):
            index = np.flatnonzero(training_keys[:, 1] == j)
            ta[index] = greedy_cost(model(tx[index], prices[j], DEPTH), prices[j])[0].numpy()
    assert len(tx) == 72
    np.savez_compressed(folder/'training_reference.npz', keys=training_keys, predicted4=pred4,
                        actual4=training['costs'][selected], selected_action=ta)
    for k, value in frozen.items():
        assert torch.equal(value, model.state_dict()[k])
    _write_rows(folder/'episodes.csv', episodes)
    audit = dict(seed=seed, roots=72, short_branches=72*16, unique_long_branches=unique_long_branches,
                 max_possible_long_branches=216, long_branch_slots=64*unique_long_branches,
                 episode_replay_error=episode_error, suffix_replay_error=suffix_error,
                 first_block_error=first_block_error, known_energy_error=known_energy_error,
                 prices=prices, searches=searches, snapshots_unchanged=True, model_unchanged=True,
                 training_updates=0, independent_test_episodes=0, oracle_is_deployable=False)
    (folder/'audit.json').write_text(json.dumps(audit, indent=2))
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    (folder/'output_hashes.json').write_text(json.dumps(
        {str(p): digest(p) for p in folder.iterdir() if p.is_file()}, indent=2))
    (folder/'COMPLETE').write_text('Frozen-model diagnosis complete; failed policy not promoted.\n')
    print(json.dumps(audit), flush=True)


def verified_data(seed):
    folder = ROOT/f'seed{seed}'
    assert (folder/'COMPLETE').exists()
    for path, expected in json.loads((folder/'manifest.json').read_text())['input_hashes'].items():
        assert digest(path) == expected, path
    for path, expected in json.loads((folder/'output_hashes.json').read_text()).items():
        assert digest(path) == expected, path
    with np.load(folder/'roots.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(folder/'training_reference.npz', allow_pickle=False) as data:
        training = {key: data[key] for key in data.files}
    return arrays, training, json.loads((folder/'audit.json').read_text())


def summarize():
    result, errors, interventions, predictions, cluster_rows = [], [], [], [], []
    source = {}
    shared = []
    identity_error = 0.
    cfg = make_config('validation')
    for seed in (11, 22):
        data, training, audit = verified_data(seed)
        source[str(ROOT/f'seed{seed}/roots.npz')] = digest(ROOT/f'seed{seed}/roots.npz')
        source[str(ROOT/f'seed{seed}/training_reference.npz')] = digest(ROOT/f'seed{seed}/training_reference.npz')
        shared.append(data)
        for dataset, d in (('train', training), ('visited', data)):
            for rollin in (0, 1):
                for j in range(3):
                    idx = np.flatnonzero((d['keys'][:, 0] == rollin) & (d['keys'][:, 1] == j))
                    assert len(idx) == 12
                    selections = {'all': None, 'learned': d['selected_action'][idx] if dataset == 'train' else d['actions'][idx, 0]}
                    if dataset == 'visited':
                        selections['search'] = d['actions'][idx, 1]
                    for which, action in selections.items():
                        pred, real = d['predicted4'][idx, :, 1], d['actual4'][idx, :, 1]
                        if action is not None:
                            pred = pred[np.arange(len(idx)), action]
                            real = real[np.arange(len(idx)), action]
                        error = pred-real
                        predictions.append(dict(seed=seed, dataset=dataset, rollin=rollin, scenario=j,
                            selection=which, roots=12, scalar_labels=int(error.size),
                            predicted=float(np.mean(pred)), actual=float(np.mean(real)),
                            bias=float(np.mean(error)), mae=float(np.mean(np.abs(error))),
                            rmse=float(np.sqrt(np.mean(error**2)))))
        for n, key in enumerate(data['keys']):
            rollin, j, i, t = map(int, key)
            price = float(audit['prices'][str(j)])
            a, b, oracle = data['actions'][n]
            pred4, pred64 = data['predicted4'][n], data['predicted64'][n]
            real4, real64 = data['actual4'][n], data['actual64'][n]
            parts = (error_components(pred4[a], pred64[a], real4[a], real64[0], price)
                     -error_components(pred4[b], pred64[b], real4[b], real64[1], price))
            pred_delta = (pred64[a]-pred64[b]) @ np.array([1., price])
            true_delta = (real64[0]-real64[1]) @ np.array([1., price])
            identity_error = max(identity_error, abs(float(parts.sum()-(pred_delta-true_delta))))
            assert identity_error < 1e-9
            row = dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, time=t, price=price,
                       action=int(a), search_action=int(b), oracle_action=int(oracle),
                       predicted_delta=float(pred_delta), actual_delta=float(true_delta),
                       immediate_energy_error=float(parts[0]), immediate_risk_error=float(parts[1]),
                       tail_energy_error=float(parts[2]), tail_risk_error=float(parts[3]),
                       actual_energy_delta=float(real64[0, 0]-real64[1, 0]),
                       actual_outage_cost_delta=float(real64[0, 1]-real64[1, 1]))
            errors.append(row)
            interventions.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, time=t,
                price=price, changed=int(oracle != a),
                current_risk_change=float(real4[oracle, 1]-real4[a, 1]),
                energy_change=float(real64[2, 0]-real64[0, 0]),
                outage_cost_change=float(real64[2, 1]-real64[0, 1]),
                scalar_change=float((real64[2]-real64[0]) @ np.array([1., price]))))
        np.testing.assert_allclose(data['actual64'],
            (data['blocks']*cfg.rl_gamma**np.arange(16)[None, None, :, None]).sum(2), atol=1e-12, rtol=0)
        for rollin in (0, 1):
            for j in range(3):
                group = [r for r in errors if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, j)]
                intr = [r for r in interventions if (r['seed'], r['rollin'], r['scenario']) == (seed, rollin, j)]
                changed = [r for r in group if r['action'] != r['search_action']]
                row = dict(seed=seed, rollin=rollin, scenario=j, roots=len(group),
                    different_from_search=len(changed),
                    wrong_sign=sum(r['predicted_delta'] < -1e-7 and r['actual_delta'] > 1e-7 for r in group),
                    predicted_delta=float(np.mean([r['predicted_delta'] for r in group])),
                    actual_delta=float(np.mean([r['actual_delta'] for r in group])),
                    oracle_changes=sum(r['changed'] for r in intr),
                    oracle_better=sum(r['changed'] and r['scalar_change'] < -1e-7 for r in intr),
                    oracle_worse=sum(r['changed'] and r['scalar_change'] > 1e-7 for r in intr),
                    oracle_scalar_change=float(np.mean([r['scalar_change'] for r in intr])))
                for component in ('immediate_energy_error', 'immediate_risk_error', 'tail_energy_error', 'tail_risk_error'):
                    row[component] = float(np.mean([r[component] for r in group]))
                    row[component+'_mae'] = float(np.mean([abs(r[component]) for r in group]))
                result.append(row)
                for i in range(4):
                    roots = [r for r in group if r['trajectory'] == i]
                    assert len(roots) == 3
                    cluster_rows.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i,
                        predicted_delta=float(np.mean([r['predicted_delta'] for r in roots])),
                        actual_delta=float(np.mean([r['actual_delta'] for r in roots]))))
    for field in ('features', 'keys', 'actual4'):
        left = shared[0][field][shared[0]['keys'][:, 0] == 1]
        right = shared[1][field][shared[1]['keys'][:, 0] == 1]
        np.testing.assert_array_equal(left, right)
    folder = ROOT/'summary'
    folder.mkdir(parents=True, exist_ok=False)
    for filename, values in (('risk_prediction.csv', predictions), ('relative_errors.csv', errors),
                             ('oracle_interventions.csv', interventions), ('summary.csv', result),
                             ('trajectory_clusters.csv', cluster_rows)):
        _write_rows(folder/filename, values)
    report = dict(roots=144, distinct_exogenous_trajectories=12, error_sum_residual=identity_error,
                  duplicate_search_states_exact=True, return_reconstruction_exact=True,
                  input_hashes=source, oracle_is_deployable=False, training_updates=0,
                  independent_confirmation=False, failed_candidate_gate_unchanged=True)
    (folder/'audit.json').write_text(json.dumps(report, indent=2))
    (folder/'COMPLETE').write_text('Error diagnosis summarized; no promotion or new policy claim.\n')
    print(json.dumps(dict(audit=report, summary=result), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('collect', 'summarize'), required=True)
    parser.add_argument('--seed', choices=(11, 22), default=11, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'collect':
        run_audit(args.seed)
    else:
        summarize()
