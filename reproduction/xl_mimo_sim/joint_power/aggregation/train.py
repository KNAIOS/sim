"""Equal-size/equal-update critics differing in collected training-state visits."""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from joint_power.aggregation.collect import ROOT, load, previous_intact
from joint_power.rollout.absolute import AbsoluteValue, loss
from joint_power.study import digest, write_rows

ARMS = {'PRS-added': 'prs_added', 'Visitation-added': 'visitation_added'}


def intact():
    previous_intact()
    for name in ('manifest.json', 'collection_hashes.json'):
        data = json.loads((ROOT/name).read_text())
        hashes = data.get('input_hashes', data)
        assert all(digest(p) == h for p, h in hashes.items()), name


def tensors(data):
    return (torch.tensor(data['x'], dtype=torch.float32),
            torch.tensor(data['reference'], dtype=torch.long),
            torch.tensor(140-data['identity'][:, 2], dtype=torch.float32),
            torch.tensor(data['y'][:, :, -1], dtype=torch.float32))


def main():
    torch.set_num_threads(1)
    intact()
    folder = ROOT/'training'
    folder.mkdir(exist_ok=False)
    files = [Path(__file__), Path('joint_power/aggregation/PLAN.md'), ROOT/'collection_hashes.json']
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    validation = tensors(load(ROOT/'validation.npz'))
    history, selections = [], []
    for arm, stem in ARMS.items():
        x, reference, remaining, outcome = tensors(load(ROOT/f'{stem}.npz'))
        assert len(x) == 384
        mean, std = x.mean((0, 1)), x.std((0, 1), unbiased=False).clamp_min(.05)
        for seed in (11, 22):
            torch.manual_seed(seed)
            model = AbsoluteValue(mean, std)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            best, best_epoch, best_loss = None, None, float('inf')
            for epoch in range(81):
                if epoch:
                    model.train()
                    for ids in torch.randperm(len(x)).split(16):
                        value = loss(model, x[ids], reference[ids], remaining[ids], outcome[ids])
                        if not torch.isfinite(value):
                            raise RuntimeError('Nonfinite aggregation loss')
                        optimizer.zero_grad()
                        value.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_loss = float(loss(model, *validation))
                        train_loss = float(loss(model, x, reference, remaining, outcome))
                    torch.save(dict(state=model.state_dict(), epoch=epoch),
                               folder/f'{stem}_seed{seed}_epoch{epoch:03d}.pt')
                    if valid_loss < best_loss:
                        best, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epoch, valid_loss
                    history.append(dict(arm=arm, seed=seed, epoch=epoch,
                        validation_prediction_loss=valid_loss, training_prediction_loss=train_loss))
                    print(f'{arm} seed={seed} epoch={epoch} validation={valid_loss:.5f} best={best_epoch}', flush=True)
            torch.save(dict(state=best, epoch=best_epoch), folder/f'{stem}_seed{seed}_selected.pt')
            selections.append(dict(arm=arm, seed=seed, epoch=best_epoch, validation_prediction_loss=best_loss))
    write_rows(folder/'prediction_history.csv', history)
    (folder/'selection.json').write_text(json.dumps(selections, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    (folder/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in folder.iterdir()
        if p.is_file()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
