"""One bounded, factorial approximate-policy-improvement reconstruction."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.action_value import baseline_command, full_slot_energy
from major_revision.action_value_report import numeric_rows
from major_revision.action_value_study import ROOT as BASE_ROOT, MARGINS
from major_revision.absolute_return_screen import episode, METHODS
from major_revision.absolute_return_frontier import mixture_frontier
from major_revision.factored_value import ROOT as FIT_ROOT, FactoredNetwork, extend_observation
from major_revision.factored_value_fit import objective
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.rollout_value_data import branch_return, TIMES
from major_revision.rollout_value_fit import select_action

ROOT = Path('results/major_revision_20260921_policy_consistent')
VARIANTS = ('mu_mu', 'mix_mu', 'mu_pi', 'mix_pi')
TRAIN_STREAM, VALIDATION_STREAM = 1080000000, 1090000000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(hashes):
    for path, expected in hashes.items():
        assert digest(path) == expected, path


def initialize(part, extra=()):
    upstream = json.loads((FIT_ROOT / 'fit_audit/audit.json').read_text())
    verify(upstream['input_hashes'])
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in
        ('POLICY_CONSISTENT_VALUE_PLAN.md', 'test_policy_consistent_value.py',
         'factored_value.py', 'factored_value_fit.py', 'rollout_value_data.py', 'rollout_value_fit.py',
         'absolute_return_screen.py', 'absolute_return_frontier.py', 'action_value.py',
         'gram_control.py', 'full_domain.py', 'audit_rollout.py', 'pilot.py',
         'spatial_actions.py', 'constraint_training.py')], *Path('future_vr').glob('*.py'),
         BASE_ROOT / 'baselines/selected.csv', FIT_ROOT / 'fit_audit/audit.json',
         *[FIT_ROOT / f'fit_seed{s}/free_final.pt' for s in (11, 22)], *extra]
    hashes = {str(p): digest(p) for p in files}
    folder = ROOT / part
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'manifest.json').write_text(json.dumps(dict(hashes=hashes, part=part,
        config=make_config('validation').as_dict(), train_stream=TRAIN_STREAM,
        validation_stream=VALIDATION_STREAM, threads=1, pid=os.getpid(),
        torch=str(torch.__version__), numpy=np.__version__, variants=VARIANTS,
        role='bounded factorial reconstruction; not independent confirmation'), indent=2))
    return folder, hashes


def finish(folder, hashes):
    verify(hashes)
    generated = {str(p): digest(p) for p in folder.iterdir() if p.is_file()}
    (folder / 'output_hashes.json').write_text(json.dumps(generated, indent=2))
    (folder / 'COMPLETE').write_text('Frozen stage complete; no automatic scientific promotion.\n')


def verify_folder(folder):
    assert (folder / 'COMPLETE').exists()
    verify(json.loads((folder / 'manifest.json').read_text())['hashes'])
    verify(json.loads((folder / 'output_hashes.json').read_text()))


def load_model(seed, variant=None):
    source = (FIT_ROOT / f'fit_seed{seed}/free_final.pt' if variant is None
              else ROOT / f'fit_seed{seed}/{variant}.pt')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 100
    model = FactoredNetwork(make_config('validation'), structured=False)
    model.load_state_dict(checkpoint['online'], strict=True)
    model.eval()
    return model, checkpoint


def settings():
    rows = numeric_rows(BASE_ROOT / 'baselines/selected.csv')
    return {j: next(r for r in rows if r['method'] == 'cost' and r['scenario'] == j)
            for j in range(3)}


def decide(run, setting, model):
    x, score = extend_observation(run, setting)
    ref = int(baseline_command(run, 'cost', setting['margin'], setting['minimum']).sum()) - 1
    with torch.no_grad():
        costs = model(torch.from_numpy(x[None]))[0, :, 1].numpy()
    action = select_action(costs - costs[ref], ref, 2.)
    return x, score, ref, action


def continuation_return(root, command, setting, model=None):
    """Physical return of a frozen continuation, not a learned value prediction."""
    if model is None:
        return branch_return(root, command, setting)
    if root.t % 4 or root.t + 16 > root.cfg.episode_steps:
        raise ValueError('Need a decision root and sixteen complete physical slots')
    trial = root.clone()
    start = trial.t
    blocks = []
    while trial.t < start + 16:
        energy = trial.energy
        if trial.t == start:
            action = command
        else:
            _, score, _, chosen = decide(trial, setting, model)
            action = select_top(score, chosen + 1)
        trial.step(action)
        for _ in range(3):
            trial.step()
        blocks.append([(trial.energy - energy) / (4 * full_slot_energy(root.cfg)),
                       float(np.mean(trial.outages[-4:]))])
    blocks = np.asarray(blocks)
    cost = np.sum(blocks * root.cfg.rl_gamma**np.arange(4)[:, None], axis=0)
    return np.stack((blocks[0], cost)), blocks


def collect(seed):
    cfg = make_config('validation')
    model, _ = load_model(seed)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    folder, hashes = initialize(f'collect_seed{seed}')
    config = settings()
    all_x, all_costs, all_blocks, keys, refs, policy_actions, metrics = [], [], [], [], [], [], []
    replay_error = 0.
    for rollin in (0, 1):
        for j, scenario in enumerate(SCENARIOS):
            setting = config[j]
            for i in range(4):
                run = FullDomainRollout(cfg, trajectory(cfg, TRAIN_STREAM+j*10000+i, scenario))
                snapshots = []
                while run.t < cfg.episode_steps:
                    if run.t % 4:
                        run.step()
                        continue
                    x, score, ref, chosen = decide(run, setting, model)
                    if run.t in TIMES:
                        masks = np.asarray([select_top(score, n) for n in range(1, cfg.subarrays+1)])
                        snapshots.append((run.t, x.copy(), ref, chosen, masks, run.clone()))
                    action = ref if rollin == 0 else chosen
                    run.step(select_top(score, action+1))
                metrics.append(dict(seed=seed, rollin=rollin, scenario=j, trajectory=i, **run.metrics()))
                for t, x, ref, chosen, masks, root in snapshots:
                    values, block_values = [], []
                    for continuation, controller in enumerate((None, model)):
                        local, local_blocks = [], []
                        for a, mask in enumerate(masks):
                            cost, blocks = continuation_return(root, mask, setting, controller)
                            if continuation == rollin and a == (ref if rollin == 0 else chosen):
                                expected = np.asarray([[sum(run.slot_energy[n:n+4])/(4*full_slot_energy(cfg)),
                                                       float(np.mean(run.outages[n:n+4]))]
                                                       for n in range(t, t+16, 4)])
                                replay_error = max(replay_error, float(np.max(np.abs(blocks-expected))))
                                assert replay_error < 1e-10
                            local.append(cost)
                            local_blocks.append(blocks)
                        values.append(local)
                        block_values.append(local_blocks)
                    values = np.asarray(values)
                    np.testing.assert_array_equal(values[0, :, 0], values[1, :, 0])
                    np.testing.assert_array_equal(x, extend_observation(root, setting)[0])
                    all_x.append(x)
                    all_costs.append(values)
                    all_blocks.append(block_values)
                    keys.append((rollin, j, i, t))
                    refs.append(ref)
                    policy_actions.append(chosen)
                print('policy-consistent collect', seed, rollin, j, i+1, '/ 4', flush=True)
    x, costs, blocks = np.asarray(all_x), np.asarray(all_costs), np.asarray(all_blocks)
    assert x.shape == (192, 1500) and costs.shape == (192, 2, 16, 2, 2)
    assert blocks.shape == (192, 2, 16, 4, 2) and len(set(keys)) == 192
    assert np.isfinite(costs).all() and (costs >= 0).all()
    for k, v in frozen.items():
        assert torch.equal(model.state_dict()[k], v)
    np.savez_compressed(folder/'labels.npz', features=x, costs=costs, blocks=blocks,
        keys=np.asarray(keys, np.int64), reference=np.asarray(refs, np.int64),
        policy_action=np.asarray(policy_actions, np.int64))
    _write_rows(folder/'rollin_episodes.csv', metrics)
    (folder/'audit.json').write_text(json.dumps(dict(seed=seed, roots=192, branches=6144,
        branch_slots=98304, distinct_exogenous_traces=12, suffix_replay_error=replay_error,
        C4_continuation_equal=True, snapshots_unchanged=True, model_unchanged=True), indent=2))
    finish(folder, hashes)


def epoch_indices(keys, variant, rng):
    if variant not in VARIANTS:
        raise ValueError('Unknown factorial variant')
    if variant.startswith('mu_'):
        available = np.flatnonzero(keys[:, 0] == 0)
        assert len(available)*2 == len(keys)
        return np.concatenate((rng.permutation(available), rng.permutation(available)))
    return rng.permutation(len(keys))


def fit(seed):
    data_folder = ROOT/f'collect_seed{seed}'
    verify_folder(data_folder)
    data = np.load(data_folder/'labels.npz', allow_pickle=False)
    folder, hashes = initialize(f'fit_seed{seed}', extra=(data_folder/'labels.npz', data_folder/'audit.json'))
    x = torch.from_numpy(data['features']).float()
    reference = torch.from_numpy(data['reference']).long()
    history, summaries = [], []
    for variant in VARIANTS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model, checkpoint = load_model(seed)
        model.train()
        scales = (checkpoint['absolute_scale'], checkpoint['relative_scale'])
        continuation = 0 if variant.endswith('_mu') else 1
        target = torch.from_numpy(data['costs'][:, continuation]).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
        steps = 0
        for epoch in range(100):
            order = epoch_indices(data['keys'], variant, rng)
            total = 0.
            for start in range(0, len(order), 64):
                take = order[start:start+64]
                pred = model(x[take])
                loss = objective(pred, target[take], reference[take], scales)
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach())*len(take)
                steps += 1
            history.append(dict(seed=seed, variant=variant, epoch=epoch+1,
                                loss=total/len(order), root_presentations=len(order)))
        assert steps == 300
        model.eval()
        with torch.no_grad():
            prediction = model(x).numpy()
        np.savez_compressed(folder/f'{variant}_predictions.npz', training=prediction)
        torch.save(dict(online=model.state_dict(), seed=seed, variant=variant, epoch=100,
                        update_steps=steps, parameters=sum(p.numel() for p in model.parameters()),
                        absolute_scale=scales[0], relative_scale=scales[1]), folder/f'{variant}.pt')
        for rollin in (0, 1):
            take = data['keys'][:, 0] == rollin
            delta = prediction-prediction[np.arange(192), data['reference']][:, None]
            truth = data['costs'][:, continuation]
            truth = truth-truth[np.arange(192), data['reference']][:, None]
            summaries.append(dict(seed=seed, variant=variant, rollin=rollin,
                energy_relative_mse=float(((delta[take, :, 1, 0]-truth[take, :, 1, 0])**2).mean()),
                outage_relative_mse=float(((delta[take, :, 1, 1]-truth[take, :, 1, 1])**2).mean())))
        print('policy-consistent fit', seed, variant, '300 updates complete', flush=True)
    _write_rows(folder/'history.csv', history)
    _write_rows(folder/'training_metrics.csv', summaries)
    finish(folder, hashes)


def screen(part):
    for seed in (11, 22):
        verify_folder(ROOT/f'fit_seed{seed}')
    extra = [ROOT/f'fit_seed{s}/{v}.pt' for s in (11, 22) for v in VARIANTS]
    folder, hashes = initialize(f'screen_{part}', extra=extra)
    cfg, config = make_config('validation'), settings()
    traces = [(j, i, trajectory(cfg, VALIDATION_STREAM+j*10000+i, sc))
              for j, sc in enumerate(SCENARIOS) for i in range(8)]
    raw, summaries = [], []
    if part == 'learned':
        for seed in (11, 22):
            for variant in (*VARIANTS, 'predecessor'):
                model, _ = load_model(seed, None if variant == 'predecessor' else variant)
                for j in range(3):
                    group = [dict(seed=seed, variant=variant, scenario=j, trajectory=i,
                        **episode(cfg, tr, model=model, setting=config[j])) for sc, i, tr in traces if sc == j]
                    raw.extend(group)
                    summaries.append(dict(seed=seed, variant=variant, scenario=j, **mean(group)))
                    print('policy screen', seed, variant, j, summaries[-1], flush=True)
                _write_rows(folder/'raw.csv', raw)
                _write_rows(folder/'summary.csv', summaries)
        assert len(raw) == 240
    else:
        for mode in METHODS:
            for j in range(3):
                for minimum in (1, 4):
                    for margin in MARGINS:
                        group = [dict(scenario=j, trajectory=i, minimum=minimum, margin=margin,
                            **episode(cfg, tr, mode, dict(minimum=minimum, margin=margin)))
                            for sc, i, tr in traces if sc == j]
                        raw.extend(group)
                        summaries.append(dict(scenario=j, method=mode, minimum=minimum, margin=margin, **mean(group)))
                _write_rows(folder/'raw.csv', raw)
                _write_rows(folder/'summary.csv', summaries)
                print('policy screen baseline', mode, j, '12 settings complete', flush=True)
        for j in range(3):
            for mode in ('mu', 'full'):
                setting = config[j] if mode == 'mu' else dict(minimum=1, margin=1.)
                group = []
                for sc, i, tr in traces:
                    if sc != j:
                        continue
                    row = episode(cfg, tr, 'cost' if mode == 'mu' else 'full', setting)
                    row['method'] = mode
                    group.append(dict(scenario=j, trajectory=i, minimum=setting['minimum'], margin=setting['margin'], **row))
                raw.extend(group)
                summaries.append(dict(scenario=j, method=mode, minimum=setting['minimum'], margin=setting['margin'], **mean(group)))
        assert len(raw) == 1200
        _write_rows(folder/'raw.csv', raw)
        _write_rows(folder/'summary.csv', summaries)
    finish(folder, hashes)


def mean(group):
    return {k: float(np.mean([r[k] for r in group])) for k in ('energy_j', 'outage')}


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return [{k: v if k in ('method', 'variant') else float(v) for k, v in r.items()}
                for r in csv.DictReader(stream)]


def promotion(candidate, pure, frontier, controls):
    base_pass = bool(pure is not None and frontier is not None and candidate['outage'] <= .05
        and candidate['outage'] <= pure['outage']+.005
        and candidate['energy_j'] <= .98*pure['energy_j']
        and candidate['energy_j'] <= .98*frontier['energy_j'])
    control_pass = bool(candidate['outage'] <= .05 and all(c['outage'] > .05 or
        (candidate['energy_j'] < c['energy_j'] and candidate['outage'] <= c['outage']+.005)
        for c in controls))
    return dict(baseline_pass=base_pass, ablation_pass=control_pass, passed=base_pass and control_pass)


def audit():
    extra = []
    for part in (*[f'collect_seed{s}' for s in (11, 22)], *[f'fit_seed{s}' for s in (11, 22)],
                 'screen_learned', 'screen_baselines'):
        folder = ROOT/part
        verify_folder(folder)
        extra.extend(p for p in folder.iterdir() if p.is_file())
    output, hashes = initialize('audit', extra=extra)
    data = {s: np.load(ROOT/f'collect_seed{s}/labels.npz', allow_pickle=False) for s in (11, 22)}
    for name in ('features', 'reference', 'keys'):
        np.testing.assert_array_equal(data[11][name][:96], data[22][name][:96])
    np.testing.assert_array_equal(data[11]['costs'][:96, 0], data[22]['costs'][:96, 0])
    diagnostics = []
    for seed in (11, 22):
        ds = data[seed]
        for rollin in (0, 1):
            for j in range(3):
                take = (ds['keys'][:, 0] == rollin) & (ds['keys'][:, 1] == j)
                ids = np.flatnonzero(take)
                actions = ds['policy_action'][take]
                gap = ds['costs'][ids, 1, actions, 1]-ds['costs'][ids, 0, actions, 1]
                diagnostics.append(dict(seed=seed, rollin=rollin, scenario=j, roots=len(ids),
                    continuation_energy_gap_mean=float(gap[:, 0].mean()),
                    continuation_outage_gap_mean=float(gap[:, 1].mean()),
                    continuation_outage_gap_mae=float(np.abs(gap[:, 1]).mean()),
                    mean_active_count=float(ds['features'][take, 288:304].sum(1).mean()),
                    mean_waking_count=float((ds['features'][take, 320:336] > 0).sum(1).mean())))
    baseline = read_rows(ROOT/'screen_baselines/raw.csv')
    learned = read_rows(ROOT/'screen_learned/raw.csv')
    assert len(baseline) == 1200 and len(learned) == 240
    assert len({(r['method'], r['scenario'], r['minimum'], r['margin'], r['trajectory']) for r in baseline}) == 1200
    assert len({(r['variant'], r['seed'], r['scenario'], r['trajectory']) for r in learned}) == 240
    identity = max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in baseline+learned)
    assert identity < 1e-8 and all(0 <= r['outage'] <= 1 for r in baseline+learned)
    summaries, frontiers, gates, paired = [], [], [], []
    for j in range(3):
        points = []
        for mode in (*METHODS, 'full'):
            configs = sorted({(r['minimum'], r['margin']) for r in baseline if r['scenario'] == j and r['method'] == mode})
            for minimum, margin in configs:
                group = [r for r in baseline if r['scenario'] == j and r['method'] == mode
                         and r['minimum'] == minimum and r['margin'] == margin]
                assert len(group) == 8
                points.append(dict(method=mode, minimum=minimum, margin=margin, **mean(group)))
        assert len(points) == 49
        feasible = [p for p in points if p['method'] != 'full' and p['outage'] <= .05]
        best = min(feasible, key=lambda p: (p['energy_j'], p['method'], p['minimum'], p['margin'])) if feasible else None
        for point in points:
            summaries.append(dict(scenario=j, kind='search', seed=0, variant=point['method'], **point))
        for seed in (11, 22):
            candidates = {}
            for variant in (*VARIANTS, 'predecessor'):
                group = [r for r in learned if r['scenario'] == j and r['seed'] == seed and r['variant'] == variant]
                assert len(group) == 8
                candidates[variant] = mean(group)
                summaries.append(dict(scenario=j, kind='learned', seed=seed, variant=variant, **mean(group)))
                match = mixture_frontier(points, candidates[variant]['outage'])
                frontiers.append(dict(scenario=j, seed=seed, variant=variant,
                    target_outage=candidates[variant]['outage'],
                    learned_energy_j=candidates[variant]['energy_j'], frontier=match,
                    a_setting=points[match['a']] if match is not None else None,
                    b_setting=points[match['b']] if match is not None else None))
            candidate = candidates['mix_pi']
            frontier = mixture_frontier(points, candidate['outage'])
            row = dict(scenario=j, seed=seed, primary=j == 1, candidate=candidate, pure=best,
                frontier=frontier, **promotion(candidate, best, frontier,
                    [v for name, v in candidates.items() if name != 'mix_pi']))
            gates.append(row)
            if best is not None:
                for i in range(8):
                    p = next(r for r in learned if r['scenario'] == j and r['seed'] == seed
                             and r['variant'] == 'mix_pi' and r['trajectory'] == i)
                    b = next(r for r in baseline if r['scenario'] == j and r['trajectory'] == i
                             and r['method'] == best['method'] and r['minimum'] == best['minimum'] and r['margin'] == best['margin'])
                    paired.append(dict(scenario=j, seed=seed, trajectory=i,
                        energy_difference_j=p['energy_j']-b['energy_j'], outage_difference=p['outage']-b['outage']))
        match = mixture_frontier(points, .05)
        frontiers.append(dict(scenario=j, seed=0, variant='risk_limit', target_outage=.05,
            learned_energy_j=None, frontier=match,
            a_setting=points[match['a']] if match is not None else None,
            b_setting=points[match['b']] if match is not None else None))
    assert len(frontiers) == 33
    # Heterogeneous summary rows are rectangularized without modifying raw files.
    fields = list(dict.fromkeys(k for r in summaries for k in r))
    _write_rows(output/'summary.csv', [{k: r.get(k, '') for k in fields} for r in summaries])
    _write_rows(output/'continuation_diagnostic.csv', diagnostics)
    _write_rows(output/'paired.csv', paired)
    (output/'frontiers.json').write_text(json.dumps(frontiers, indent=2))
    permit = all(r['passed'] for r in gates if r['primary'])
    (output/'gate.json').write_text(json.dumps(dict(permit_confirmation=permit, gates=gates), indent=2))
    (output/'audit.json').write_text(json.dumps(dict(episode_records=1440, exogenous_validation_traces=24,
        duplicate_mu_data_exact=True, energy_identity_error=identity,
        roots_per_seed=192, unique_training_traces=12, learning_updates_per_variant=300,
        permit_confirmation=permit, all_variants_retained=True), indent=2))
    finish(output, hashes)
    if permit:
        (ROOT/'PERMIT_CONFIRMATION').write_text('Freeze independent confirmation before further evaluation.\n')
    print(json.dumps(dict(gates=gates, continuation=diagnostics), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('collect', 'fit', 'learned', 'baselines', 'audit'))
    parser.add_argument('--seed', type=int, choices=(11, 22))
    args = parser.parse_args()
    if args.stage in ('collect', 'fit') and args.seed is None:
        parser.error('Collection/fitting needs an explicit fixed seed')
    torch.set_num_threads(1)
    if args.stage == 'collect':
        collect(args.seed)
    elif args.stage == 'fit':
        fit(args.seed)
    elif args.stage == 'audit':
        audit()
    else:
        screen(args.stage)
