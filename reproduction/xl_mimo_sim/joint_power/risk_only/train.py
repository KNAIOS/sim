"""Frozen 2-arm x 2-seed prediction experiment; no control-performance selection."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from joint_power.risk_only.collect import ROOT, intact as previous_intact
from joint_power.risk_only.model import ReturnCritic, batches, frequency, load_checkpoint, losses
from joint_power.study import digest, write_rows

ARMS, SEEDS, EPOCHS = ('risk_only', 'joint'), (11, 22), (0, 20, 40, 60, 80)


def intact():
    previous_intact()
    for name in ('manifest.json', 'collection_hashes.json'):
        data = json.loads((ROOT/name).read_text())
        hashes = data.get('input_hashes', data)
        assert all(digest(p) == h for p, h in hashes.items()), name


def dataset(split):
    with np.load(ROOT/f'{split}.npz', allow_pickle=False) as data:
        assert data['x'].shape[1:] == (17, 342)
        assert np.all(data['identity'][:, 3] == data['slot'])
        assert len(np.unique(data['trace_seed'])) == (24 if split == 'train' else 12)
        inputs = (torch.tensor(data['x'], dtype=torch.float32),
                  torch.tensor(data['reference'], dtype=torch.long),
                  torch.tensor(140-data['slot'], dtype=torch.float32),
                  torch.tensor(data['y'][:, :, -1], dtype=torch.float32))
        return inputs, data['identity'][:, 0].copy(), data['trace_seed'].copy()


@torch.no_grad()
def scores(model, tensors):
    x, ref, remaining, outcome = tensors
    count, _, _ = model(x, ref, remaining)
    risk, auxiliary = losses(model, *tensors)
    error = count-outcome[..., 1]
    delta_error = error-error.gather(1, ref[:, None])
    rate_error = error/(3*remaining[:, None])*100
    return dict(risk_loss=float(risk), energy_huber=float(auxiliary),
        failed_user_slots_mae=float(error.abs().mean()),
        failed_user_slots_bias=float(error.mean()),
        reference_difference_mae=float(delta_error.abs().mean()),
        outage_percentage_points_mae=float(rate_error.abs().mean()),
        outage_percentage_points_bias=float(rate_error.mean()))


def checkpoint(model, arm, seed, epoch, rate):
    return dict(state=copy.deepcopy(model.state_dict()), arm=arm, seed=seed,
                epoch=epoch, frequency=rate, architecture=[342, 64, 64, 2 if model.joint else 1])


def main():
    torch.set_num_threads(1)
    intact()
    training, train_scenario, train_seeds = dataset('train')
    validation, valid_scenario, valid_seeds = dataset('validation')
    assert set(train_seeds).isdisjoint(valid_seeds)
    assert len(training[0]) == 192 and len(validation[0]) == 96
    x, _, remaining, outcome = training
    mean, std = x.mean((0, 1)), x.std((0, 1), unbiased=False).clamp_min(.05)
    rate = frequency(outcome, remaining)
    folder = ROOT/'training'
    folder.mkdir(exist_ok=False)
    sources = [Path(__file__), Path('joint_power/risk_only/model.py'),
               Path('joint_power/risk_only/test_model.py'), Path('joint_power/risk_only/PLAN.md'),
               Path('joint_power/rollout/probe.py'), ROOT/'collection_hashes.json']
    hashes = {str(p): digest(p) for p in sources}
    manifest = dict(input_hashes=hashes, arms=ARMS, seeds=SEEDS, checkpoints=EPOCHS,
        training_frequency=rate, frequency_definition='sum failures / sum capacity over root-actions',
        normalization='training mean/std; std floor 0.05; standardized clamp [-10,10]',
        selection='minimum validation risk loss, earlier epoch on ties',
        epochs=80, batch_roots=16, optimizer='AdamW', learning_rate=.001, weight_decay=.0001,
        gradient_clip=5, permutation_seed='same as model seed, dedicated torch.Generator',
        prediction_only=True, host_environment_changed=False)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    histories, metric_rows, selections, checks = [], [], [], []
    initial_trunks, order_hashes = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = ReturnCritic(mean, std, rate, joint=arm == 'joint')
            initial = {k: v.clone() for k, v in model.trunk.state_dict().items()}
            if arm == 'risk_only':
                initial_trunks[seed] = initial
            else:
                assert all(torch.equal(v, initial_trunks[seed][k]) for k, v in initial.items())
            generator = torch.Generator().manual_seed(seed)
            order_digest = hashlib.sha256()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            best, best_loss = None, float('inf')
            for epoch in range(81):
                if epoch:
                    model.train()
                    for ids in batches(len(x), generator):
                        order_digest.update(ids.numpy().tobytes())
                        risk, energy = losses(model, *(t[ids] for t in training))
                        loss = risk+energy
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('Nonfinite prediction objective')
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                if epoch not in EPOCHS:
                    continue
                model.eval()
                train_score, valid_score = scores(model, training), scores(model, validation)
                if not all(np.isfinite(v) for v in (*train_score.values(), *valid_score.values())):
                    raise RuntimeError('Nonfinite prediction metric')
                payload = checkpoint(model, arm, seed, epoch, rate)
                path = folder/f'{arm}_seed{seed}_epoch{epoch:03d}.pt'
                torch.save(payload, path)
                reloaded, _ = load_checkpoint(path)
                with torch.no_grad():
                    original = model(*validation[:3])
                    restored = reloaded(*validation[:3])
                for a, b in zip(original, restored):
                    if a is not None:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                if valid_score['risk_loss'] < best_loss:
                    best, best_loss = payload, valid_score['risk_loss']
                histories.append(dict(arm=arm, seed=seed, epoch=epoch,
                    validation_risk_loss=valid_score['risk_loss'],
                    training_risk_loss=train_score['risk_loss'],
                    validation_energy_huber=valid_score['energy_huber'],
                    training_energy_huber=train_score['energy_huber']))
                for split, tensors, scenario in (('train', training, train_scenario),
                                                 ('validation', validation, valid_scenario)):
                    for s in (-1, 0, 1, 2):
                        ids = np.ones(len(scenario), dtype=bool) if s == -1 else scenario == s
                        metric_rows.append(dict(arm=arm, seed=seed, epoch=epoch,
                            split=split, scenario=s, roots=int(ids.sum()),
                            **scores(model, tuple(t[ids] for t in tensors))))
                print(f'{arm} seed={seed} epoch={epoch}: risk={valid_score["risk_loss"]:.6f}, '
                      f'MAE={valid_score["failed_user_slots_mae"]:.3f}, '
                      f'selected={best["epoch"]}', flush=True)
            torch.save(best, folder/f'{arm}_seed{seed}_selected.pt')
            selected_model, _ = load_checkpoint(folder/f'{arm}_seed{seed}_selected.pt')
            selected_score = scores(selected_model, validation)
            assert selected_score['risk_loss'] == best_loss
            selections.append(dict(arm=arm, seed=seed, epoch=best['epoch'], **selected_score))
            order = order_digest.hexdigest()
            if arm == 'risk_only':
                order_hashes[seed] = order
            else:
                assert order == order_hashes[seed]
            checks.append(dict(arm=arm, seed=seed, batch_order_sha256=order,
                parameter_count=sum(p.numel() for p in model.parameters()),
                checkpoint_reload_exact=True, selected_reload_exact=True))
    write_rows(folder/'prediction_history.csv', histories)
    write_rows(folder/'prediction_metrics.csv', metric_rows)
    (folder/'selection.json').write_text(json.dumps(selections, indent=2)+'\n')
    report = dict(selection=selections, training_frequency=rate, checks=checks,
        initial_trunks_matched=True, batch_orders_matched=True,
        train_validation_trace_disjoint=True, normalization_training_only=True,
        total_fits=4, saved_checkpoint_count=24, control_evidence=False, R1_closed=False)
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (folder/'output_hashes.json').write_text(json.dumps({str(p): digest(p)
        for p in folder.iterdir() if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
