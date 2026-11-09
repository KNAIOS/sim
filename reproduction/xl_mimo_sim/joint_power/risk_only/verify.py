"""Post-fit integrity and constant-checkpoint audit, not a control experiment."""
import json
from pathlib import Path

import numpy as np
import torch

from joint_power.risk_only.collect import ROOT
from joint_power.risk_only.model import load_checkpoint
from joint_power.risk_only.train import dataset, intact, scores
from joint_power.study import digest, write_rows


def main():
    torch.set_num_threads(1)
    intact()
    inputs = json.loads((ROOT/'training/manifest.json').read_text())['input_hashes']
    outputs = json.loads((ROOT/'training/output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in (inputs | outputs).items())
    training, _, train_seeds = dataset('train')
    validation, _, valid_seeds = dataset('validation')
    assert set(train_seeds).isdisjoint(valid_seeds)
    mean = training[0].mean((0, 1))
    std = training[0].std((0, 1), unbiased=False).clamp_min(.05)
    selected = json.loads((ROOT/'training/selection.json').read_text())
    checkpoints = []
    for row in selected:
        path = ROOT/f'training/{row["arm"]}_seed{row["seed"]}_selected.pt'
        model, payload = load_checkpoint(path)
        assert payload['epoch'] == row['epoch']
        torch.testing.assert_close(model.mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(model.std, std, rtol=0, atol=0)
        assert scores(model, validation)['risk_loss'] == row['risk_loss']
        with torch.no_grad():
            count, logits, energy = model(*validation[:3])
        action_span = float((count.max(1).values-count.min(1).values).max())
        logit_span = float(logits.max()-logits.min())
        checkpoints.append(dict(arm=row['arm'], seed=row['seed'], epoch=row['epoch'],
            action_failure_count_span=action_span, all_input_logit_span=logit_span,
            head_weights_nonzero=int(torch.count_nonzero(model.head.weight)),
            energy_difference_max_abs=0. if energy is None else float(energy.abs().max()),
            checkpoint_sha256=digest(path)))
    summaries = []
    for split in ('train', 'validation'):
        with np.load(ROOT/f'{split}.npz', allow_pickle=False) as data:
            for s in range(3):
                use = data['identity'][:, 0] == s
                failed = data['y'][use, :, -1, 1]
                capacity = 3*(140-data['slot'][use, None])
                summaries.append(dict(split=split, scenario=s,
                    unique_traces=len(np.unique(data['trace_seed'][use])), roots=int(use.sum()),
                    candidate_entries=int(failed.size),
                    label_frequency=float(failed.sum()/np.broadcast_to(capacity, failed.shape).sum()),
                    label_mean_failed_user_slots=float(failed.mean()),
                    label_max_failed_user_slots=float(failed.max())))
    constant = all(r['epoch'] == 0 and r['head_weights_nonzero'] == 0 and
                   r['all_input_logit_span'] == 0 for r in checkpoints)
    paper = Path('../ICECAI2026_4pages_20260922')
    frozen = {'icecai2026_energy_rl.tex': 'ad0938f0e7f0ba70d1a9829d85f60cfd1db7cc96e2da3cad2a8264ae1abc647f',
        'output/pdf/clean.pdf': '504801338e74ba19e2c9aae348902eab5f29a230c7d329b9d181b16c9f1d7926',
        'output/pdf/icecai2026_energy_rl.pdf': '63c30f9ec214a01d1c1a6a792f3233e0642e4e64e31916dfe0426ccae833db08'}
    assert all(len(value) == 64 for value in frozen.values())
    assert all(digest(paper/p).lower() == expected for p, expected in frozen.items())
    folder = ROOT/'audit'
    folder.mkdir(exist_ok=False)
    write_rows(folder/'label_summary.csv', summaries)
    report = dict(training_manifest_and_output_hashes_verified=len(inputs | outputs),
        recursive_previous_evidence_intact=True, original_manuscript_hashes_unchanged=True,
        normalization_training_only=True, selected_checkpoints=checkpoints,
        all_selected_heads_input_independent=constant, closed_loop_not_run=True,
        development_stream5609220000_unused=True,
        limitation='Reference-return prediction fit only, not new energy savings. '
            'No evidence that dropping the energy head improves closed-loop control.',
        R1_closed=False)
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    hashes = {str(p): digest(p) for p in [Path(__file__), ROOT/'training/output_hashes.json',
                                        folder/'report.json', folder/'label_summary.csv']}
    (folder/'hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
