"""Independent numerical/accounting checks, not another candidate selection."""
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.policy_consistent_value import ROOT, VARIANTS, load_model, read_rows, verify_folder, digest


def main():
    verify_folder(ROOT/'audit')
    cfg = make_config('validation')
    files = [Path(__file__), ROOT/'audit/output_hashes.json']
    labels_error = 0.
    reloads = []
    for seed in (11, 22):
        folder = ROOT/f'fit_seed{seed}'
        ds = np.load(ROOT/f'collect_seed{seed}/labels.npz', allow_pickle=False)
        blocks, costs = ds['blocks'], ds['costs']
        reproduced = np.stack((blocks[..., 0, :],
                               (blocks*cfg.rl_gamma**np.arange(4)[:, None]).sum(-2)), axis=-2)
        labels_error = max(labels_error, float(np.max(np.abs(reproduced-costs))))
        assert labels_error < 1e-12
        assert np.all(costs[..., 0, 1] <= 1.)
        assert np.all(costs[..., 1, 1] <= sum(cfg.rl_gamma**n for n in range(4))+1e-12)
        history = read_history(folder/'history.csv')
        for variant in VARIANTS:
            model, checkpoint = load_model(seed, variant)
            assert checkpoint['update_steps'] == 300 and checkpoint['parameters'] == 32324
            with torch.no_grad():
                prediction = model(torch.from_numpy(ds['features']).float()).numpy()
            saved = np.load(folder/f'{variant}_predictions.npz', allow_pickle=False)['training']
            np.testing.assert_array_equal(prediction, saved)
            rows = [r for r in history if r['variant'] == variant]
            assert {int(r['epoch']) for r in rows} == set(range(1, 101)) and len(rows) == 100
            assert sum(int(r['root_presentations']) for r in rows) == 19200
            assert all(np.isfinite(float(r['loss'])) for r in rows)
            reloads.append(dict(seed=seed, variant=variant, parameters=32324,
                                updates=300, predictions_exact=True))
        files.extend(p for p in folder.iterdir() if p.is_file())
        files.extend(p for p in (ROOT/f'collect_seed{seed}').iterdir() if p.is_file())
    baseline = read_rows(ROOT/'screen_baselines/raw.csv')
    learned = read_rows(ROOT/'screen_learned/raw.csv')
    frontiers = json.loads((ROOT/'audit/frontiers.json').read_text())
    paired, summary = [], []
    mixture_error = 0.
    for row in frontiers:
        match = row['frontier']
        if match is None:
            continue
        j, seed, variant = row['scenario'], row['seed'], row['variant']
        components = []
        for i in range(8):
            def get(setting):
                return next(r for r in baseline if r['scenario'] == j and r['trajectory'] == i
                            and r['method'] == setting['method'] and r['minimum'] == setting['minimum']
                            and r['margin'] == setting['margin'])
            a, b, weight = get(row['a_setting']), get(row['b_setting']), match['weight_a']
            energy = weight*a['energy_j']+(1-weight)*b['energy_j']
            outage = weight*a['outage']+(1-weight)*b['outage']
            components.append([energy, outage])
            if variant == 'mix_pi':
                learner = next(r for r in learned if r['scenario'] == j and r['trajectory'] == i
                               and r['seed'] == seed and r['variant'] == variant)
                paired.append(dict(scenario=j, seed=seed, trajectory=i,
                    expected_mixture_energy_j=energy, expected_mixture_outage=outage,
                    learned_energy_j=learner['energy_j'], learned_outage=learner['outage'],
                    energy_difference_j=learner['energy_j']-energy,
                    outage_difference=learner['outage']-outage))
        mixture_error = max(mixture_error, float(np.max(np.abs(np.mean(components, axis=0)
                                - [match['energy_j'], match['outage']]))))
    for j in range(3):
        for seed in (11, 22):
            group = [r for r in paired if r['scenario'] == j and r['seed'] == seed]
            if not group:
                continue
            d = np.asarray([[r['energy_difference_j'], r['outage_difference']] for r in group])
            ref_energy = np.mean([r['expected_mixture_energy_j'] for r in group])
            summary.append(dict(scenario=j, seed=seed, trajectories=8,
                energy_difference_mean_j=float(d[:, 0].mean()), energy_difference_sd_j=float(d[:, 0].std(ddof=1)),
                outage_difference_mean=float(d[:, 1].mean()), outage_difference_sd=float(d[:, 1].std(ddof=1)),
                lower_energy_traces=int((d[:, 0] < 0).sum()), saving_fraction=float(-d[:, 0].mean()/ref_energy)))
    assert mixture_error < 1e-10
    files.extend((ROOT/'screen_baselines/raw.csv', ROOT/'screen_learned/raw.csv', ROOT/'audit/frontiers.json'))
    paper = Path('../ICECAI2026_Revision_20260920')
    expected = {'icecai2026_energy_rl.pdf': 'd67cd247f149692f6ce5e25153d1f003823f709ee48f087e04fce7c2a9424c0e',
                'clean.pdf': 'd6e892d71eb8c786b8425bb8c59a03b9a53d6f4b908f95004e55b13c779cfefa'}
    for name, value in expected.items():
        assert digest(paper/name) == value
    output = ROOT/'verified'
    output.mkdir(exist_ok=False)
    _write_rows(output/'frontier_expected_paired.csv', paired)
    _write_rows(output/'frontier_paired_summary.csv', summary)
    _write_rows(output/'model_reload.csv', reloads)
    report = dict(input_hashes={str(p): digest(p) for p in files},
        models_reloaded=8, predictions_exact=True, label_reconstruction_error=labels_error,
        mixture_recalculation_error=mixture_error, paper_pdf_hashes=expected,
        interpretation='Paired sample SD only; mixture parameters fitted on same validation, no independent confirmation')
    (output/'audit.json').write_text(json.dumps(report, indent=2))
    (output/'COMPLETE').write_text('Numerical provenance checks complete; original failed gate unchanged.\n')
    print(json.dumps(dict(summary=summary, label_error=labels_error, mixture_error=mixture_error,
                          reloads=reloads), indent=2))


def read_history(path):
    import csv
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
