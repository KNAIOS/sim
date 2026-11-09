"""Fixed-root diagnosis of finite-return truncation; never an online oracle."""
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
from major_revision.action_value_study import ROOT as BASE_ROOT
from major_revision.factored_value import ROOT as FIT_ROOT, FactoredNetwork, extend_observation
from major_revision.factored_value_fit import load_data
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.rollout_value_data import branch_return
from major_revision.rollout_value_fit import select_action

ROOT = Path('results/major_revision_20260921_long_horizon')
TIMES = (0, 32, 64)
HORIZONS = (4, 16, 32, 64)
PRICE = 2.


def horizon_returns(blocks, gamma, horizons=HORIZONS):
    """Each block contains 4-slot normalized energy and mean user-slot outage."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim < 2 or blocks.shape[-1] != 2 or not np.isfinite(blocks).all():
        raise ValueError('Expected finite [..., blocks, 2] costs')
    if not 0 < gamma <= 1 or any(h <= 0 or h % 4 or h // 4 > blocks.shape[-2] for h in horizons):
        raise ValueError('Invalid discount or physical horizon')
    weights = gamma ** np.arange(blocks.shape[-2])
    return np.stack([(blocks[..., :h // 4, :] * weights[:h // 4, None]).sum(-2)
                     for h in horizons], axis=-2)


def error_parts(prediction, target16, target64):
    """MSE identity includes the cross term; values may be scalar or vectors."""
    prediction, target16, target64 = np.broadcast_arrays(
        np.asarray(prediction, float), np.asarray(target16, float), np.asarray(target64, float))
    e = prediction - target16
    z = target64 - target16
    mse16, tail_mse = float(np.mean(e**2)), float(np.mean(z**2))
    cross = float(-2 * np.mean(e * z))
    mse64 = float(np.mean((prediction - target64)**2))
    return dict(prediction_error_mean=float(e.mean()), prediction_error_mae=float(np.abs(e).mean()),
                omitted_tail_mean=float(z.mean()), omitted_tail_mae=float(np.abs(z).mean()),
                prediction_error_mse=mse16, omitted_tail_mse=tail_mse,
                cross_term=cross, error_to_C64_mse=mse64,
                identity_error=abs(mse64 - (mse16 + tail_mse + cross)))


def scalar_cost(costs):
    return costs[..., 0] + PRICE * costs[..., 1]


def main():
    cfg = make_config('validation')
    data = load_data()['validation']
    upstream = json.loads((FIT_ROOT / 'fit_audit/audit.json').read_text())
    for path, digest in upstream['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ('LONG_HORIZON_AUDIT_PLAN.md', 'test_long_horizon_audit.py', 'rollout_value_data.py',
              'rollout_value_fit.py', 'factored_value.py', 'factored_value_fit.py', 'action_value.py',
              'gram_control.py', 'full_domain.py', 'audit_rollout.py', 'pilot.py', 'spatial_actions.py',
              'constraint_training.py')], *Path('future_vr').glob('*.py'),
             FIT_ROOT / 'fit_audit/audit.json', BASE_ROOT / 'baselines/selected.csv']
    files.extend(FIT_ROOT / f'fit_seed{s}/free_{n}' for s in (11, 22)
                 for n in ('final.pt', 'predictions.npz'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    # Model decisions are fixed on causal archived inputs before ANY new future label.
    predictions, decisions = {}, {}
    for seed in (11, 22):
        model = FactoredNetwork(cfg, structured=False)
        checkpoint = torch.load(FIT_ROOT / f'fit_seed{seed}/free_final.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == 100
        model.load_state_dict(checkpoint['online'], strict=True)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(data['features']).float()).numpy()
        saved = np.load(FIT_ROOT / f'fit_seed{seed}/free_predictions.npz', allow_pickle=False)['validation']
        np.testing.assert_array_equal(pred, saved)
        pred = pred.astype(float)
        delta = pred - pred[np.arange(len(pred)), data['reference']][:, None]
        predictions[seed] = delta[:, :, 1]
        decisions[seed] = np.asarray([select_action(d[:, 1], int(ref), PRICE)
                                     for d, ref in zip(delta, data['reference'])])
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'manifest.json').write_text(json.dumps(dict(input_hashes=hashes, config=cfg.as_dict(),
        times=TIMES, horizons=HORIZONS, trajectories_per_scenario=4, stream=1020000000,
        price=PRICE, threads=1, process_id=os.getpid(), numpy=np.__version__, torch=torch.__version__,
        role='viewed single-future root diagnostic; no training, online oracle or policy performance'), indent=2))
    settings = numeric_rows(BASE_ROOT / 'baselines/selected.csv')
    key_indices = {tuple(k): i for i, k in enumerate(data['keys'])}
    blocks_all, keys, references, indices = [], [], [], []
    archived_error = reference_error = 0.
    for j, scenario in enumerate(SCENARIOS):
        setting = next(r for r in settings if r['method'] == 'cost' and r['scenario'] == j)
        for trace in range(4):
            run = FullDomainRollout(cfg, trajectory(cfg, 1020000000 + j * 10000 + trace, scenario))
            roots = []
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                if run.t in TIMES:
                    key = (j, trace, run.t)
                    index = key_indices[key]
                    x, score = extend_observation(run, setting)
                    np.testing.assert_array_equal(x, data['features'][index])
                    command = baseline_command(run, 'cost', setting['margin'], setting['minimum'])
                    assert int(command.sum()) - 1 == data['reference'][index]
                    masks = np.asarray([select_top(score, n) for n in range(1, cfg.subarrays + 1)])
                    roots.append((key, index, run.clone(), masks))
                else:
                    command = baseline_command(run, 'cost', setting['margin'], setting['minimum'])
                run.step(command)
            for key, index, root, masks in roots:
                before = extend_observation(root, setting)[0]
                local = []
                ref = int(data['reference'][index])
                for action, mask in enumerate(masks):
                    _, blocks = branch_return(root, mask, setting, slots=64)
                    returns = horizon_returns(blocks, cfg.rl_gamma)
                    archived_error = max(archived_error,
                        float(np.max(np.abs(returns[:2] - data['costs'][index, action]))))
                    assert archived_error < 1e-10
                    if action == ref:
                        expected = np.asarray([[sum(run.slot_energy[t:t+4]) / (4 * full_slot_energy(cfg)),
                                               float(np.mean(run.outages[t:t+4]))]
                                               for t in range(root.t, root.t + 64, 4)])
                        reference_error = max(reference_error, float(np.max(np.abs(blocks - expected))))
                        assert reference_error < 1e-10
                    local.append(blocks)
                np.testing.assert_array_equal(before, extend_observation(root, setting)[0])
                blocks_all.append(local)
                keys.append(key)
                references.append(ref)
                indices.append(index)
                print('tail diagnostic', j, trace, key[2], len(keys), '/ 36', flush=True)
    blocks_all = np.asarray(blocks_all)
    keys, refs, indices = np.asarray(keys), np.asarray(references), np.asarray(indices)
    assert blocks_all.shape == (36, 16, 16, 2) and len(set(map(tuple, keys))) == 36
    costs = horizon_returns(blocks_all, cfg.rl_gamma)
    delta = costs - costs[np.arange(36), refs][:, None]
    np.testing.assert_array_equal(delta[np.arange(36), refs], 0.)
    np.savez_compressed(ROOT / 'branches.npz', keys=keys, reference=refs, blocks=blocks_all,
                        costs=costs, difference=delta, horizons=np.asarray(HORIZONS),
                        predictions=np.stack([predictions[s][indices] for s in (11, 22)]),
                        choices=np.stack([decisions[s][indices] for s in (11, 22)]))
    action_rows, selected_rows, summaries, errors, rank_rows = [], [], [], [], []
    for i, (j, trace, t) in enumerate(keys):
        for a in range(16):
            for h, horizon in enumerate(HORIZONS):
                action_rows.append(dict(scenario=int(j), trajectory=int(trace), step=int(t),
                    action=a+1, reference=int(refs[i])+1, horizon=horizon,
                    energy_return=float(costs[i, a, h, 0]), outage_return=float(costs[i, a, h, 1]),
                    energy_difference=float(delta[i, a, h, 0]), outage_difference=float(delta[i, a, h, 1]),
                    cost_difference=float(scalar_cost(delta[i, a, h]))))
        d16, d64 = scalar_cost(delta[i, :, 1]), scalar_cost(delta[i, :, 3])
        valid = np.arange(16) != refs[i]
        rank_rows.append(dict(scenario=int(j), trajectory=int(trace), step=int(t), alternatives=15,
            helpful_C16=int(((d16 < -1e-6) & valid).sum()), helpful_C64=int(((d64 < -1e-6) & valid).sum()),
            helpful_to_harmful=int(((d16 < -1e-6) & (d64 > 1e-6) & valid).sum()),
            harmful_to_helpful=int(((d16 > 1e-6) & (d64 < -1e-6) & valid).sum()),
            sampled_best_C16=int(np.argmin(d16))+1, sampled_best_C64=int(np.argmin(d64))+1))
    for seed in (11, 22):
        chosen = decisions[seed][indices]
        prediction = predictions[seed][indices, chosen]
        actual = delta[np.arange(36), chosen]
        for i, (j, trace, t) in enumerate(keys):
            for h, horizon in enumerate(HORIZONS):
                selected_rows.append(dict(seed=seed, scenario=int(j), trajectory=int(trace), step=int(t),
                    action=int(chosen[i])+1, reference=int(refs[i])+1, horizon=horizon,
                    predicted_C16_energy=float(prediction[i, 0]), predicted_C16_outage=float(prediction[i, 1]),
                    predicted_C16_cost=float(scalar_cost(prediction[i])),
                    energy_difference=float(actual[i, h, 0]), outage_difference=float(actual[i, h, 1]),
                    cost_difference=float(scalar_cost(actual[i, h]))))
        for j in (-1, 0, 1, 2):
            take = np.ones(36, bool) if j == -1 else keys[:, 0] == j
            p, a = prediction[take], actual[take]
            for h, horizon in enumerate(HORIZONS):
                q = scalar_cost(a[:, h])
                summaries.append(dict(seed=seed, scenario=j, roots=int(take.sum()),
                    trajectories=12 if j == -1 else 4, horizon=horizon,
                    changed=int((chosen[take] != refs[take]).sum()),
                    mean_energy_difference=float(a[:, h, 0].mean()),
                    mean_outage_difference=float(a[:, h, 1].mean()), mean_cost_difference=float(q.mean()),
                    helpful_roots=int((q < -1e-6).sum()), harmful_roots=int((q > 1e-6).sum()),
                    tail_after_C16_energy=float((a[:, h, 0] - a[:, 1, 0]).mean()),
                    tail_after_C16_outage=float((a[:, h, 1] - a[:, 1, 1]).mean())))
            for k, name in ((0, 'energy'), (1, 'outage'), (2, 'price2_cost')):
                value = (p[:, k], a[:, 1, k], a[:, 3, k]) if k < 2 else (
                    scalar_cost(p), scalar_cost(a[:, 1]), scalar_cost(a[:, 3]))
                errors.append(dict(seed=seed, scenario=j, roots=int(take.sum()), component=name,
                                   **error_parts(*value)))
    assert len(action_rows) == 2304 and len(selected_rows) == 288 and len(errors) == 24
    identity = max(r['identity_error'] for r in errors)
    assert identity < 1e-10
    for filename, rows in (('all_actions.csv', action_rows), ('selected.csv', selected_rows),
                           ('summary.csv', summaries), ('error_decomposition.csv', errors),
                           ('direction_reversals.csv', rank_rows)):
        _write_rows(ROOT / filename, rows)
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    audit = dict(roots=36, trajectories=12, branches=576, branch_slots=36864,
        archived_C4_C16_max_error=archived_error, reference_suffix_max_error=reference_error,
        error_decomposition_identity_max_error=identity, features_exact=True, predictions_exact=True,
        hashes_match=True, input_hashes=hashes,
        data_hash=hashlib.sha256((ROOT / 'branches.npz').read_bytes()).hexdigest(),
        role='single-future viewed validation diagnosis; no new training, promotion or policy evidence')
    (ROOT / 'audit.json').write_text(json.dumps(audit, indent=2))
    (ROOT / 'COMPLETE').write_text('Finite-return tail diagnostic complete; no model promoted.\n')
    print(json.dumps(dict(audit={k:v for k,v in audit.items() if k != 'input_hashes'},
                          summary=summaries, error_decomposition=errors), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
