"""Frozen fit and new-trajectory evaluation of two causal probe predictors."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe import causal_snapshot, probe_choices, branch
from major_revision.information_probe_study import digest, finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.probe_predictability import (
    FEATURE_NAMES, KINDS, RIDGE, FIRST_COST, OutcomePredictor, features, choose,
)


ROOT = Path('results/major_revision_20260921_probe_predictability')
SOURCE = Path('results/major_revision_20260921_information_probe')
STREAM = 1150000000
TIMES = (32, 64, 96)
MARGINS = (1., 1.08, 1.04)
MINIMUM = (1, 1, 4)
RULES = ('none', 'score', 'age', *KINDS)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def integer(row, key):
    return int(float(row[key]))


def initialize(stage, extra=()):
    folder = ROOT/stage
    folder.mkdir(parents=True, exist_ok=False)
    paths = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path('major_revision/PROBE_PREDICTABILITY_PLAN.md'), *extra]
    hashes = {str(p): digest(p) for p in paths}
    manifest = dict(pid=os.getpid(), threads=1, stage=stage, hashes=hashes,
        config=make_config('validation').as_dict(), scenarios=SCENARIOS,
        stream=STREAM, times=TIMES, features=FEATURE_NAMES, models=KINDS,
        ridge=RIDGE, known_first_cost_j=FIRST_COST, margins=MARGINS, minimum=MINIMUM,
        rl_updates=0, role='supervised single-intervention predictability diagnostic',
        scientific_completion=False)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return folder, hashes


def snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def development_data():
    for part in ('shard0', 'shard1', 'verified'):
        verify(SOURCE/part)
    raw = rows(SOURCE/'verified/all_branches.csv')
    indexed = {(integer(r, 'scenario'), integer(r, 'trajectory'), integer(r, 't'),
                integer(r, 'q')): r for r in raw if r['variant'] == 'normal'}
    cfg = make_config('validation')
    xs, ys, metadata = [], [], []
    for j in range(3):
        for i in range(4):
            for t in TIMES:
                path = SOURCE/f'shard{i//2}/root_s{j}_i{i}_t{t}.npz'
                qids, x = features(cfg, snapshot(path), t)
                for q, value in zip(qids, x):
                    r = indexed[j, i, t, int(q)]
                    xs.append(value)
                    ys.append([float(r['energy_difference_j']), 100*float(r['outage_difference'])])
                    metadata.append(dict(scenario=j, trajectory=i, t=t, q=int(q)))
    assert len(xs) == 423
    return np.asarray(xs), np.asarray(ys), metadata


def error_metrics(truth, prediction, constant):
    residual = prediction-truth
    output = {}
    for k, name in enumerate(('energy_j', 'outage_pp')):
        mse = np.mean(residual[:, k]**2)
        denominator = np.mean((truth[:, k]-constant[:, k])**2)
        output.update({f'{name}_rmse': float(np.sqrt(mse)),
            f'{name}_mae': float(np.mean(np.abs(residual[:, k]))),
            f'{name}_skill_vs_development_mean': float(1-mse/max(denominator, 1e-20))})
    return output


def fit():
    x, y, metadata = development_data()
    extras = [p for p in SOURCE.rglob('*') if p.is_file()]
    folder, hashes = initialize('fit', extras)
    groups = np.array([r['scenario']*4+r['trajectory'] for r in metadata])
    oof, errors = [], []
    for kind in KINDS:
        predictions = np.zeros_like(y)
        constant = np.zeros_like(y)
        for group in range(12):
            held = groups == group
            model = OutcomePredictor.fit(x[~held], y[~held], kind)
            predictions[held] = model.predict(x[held])
            constant[held] = y[~held].mean(0)
        errors.append(dict(model=kind, partition='viewed_grouped_development',
                           **error_metrics(y, predictions, constant)))
        for meta, truth, prediction in zip(metadata, y, predictions):
            oof.append(dict(model=kind, **meta, energy_truth=truth[0], outage_truth_pp=truth[1],
                energy_prediction=prediction[0], outage_prediction_pp=prediction[1]))
        model = OutcomePredictor.fit(x, y, kind)
        model.save(folder/f'{kind}.npz')
        np.testing.assert_array_equal(model.predict(x),
                                      OutcomePredictor.load(folder/f'{kind}.npz').predict(x))
    np.savez_compressed(folder/'development.npz', x=x, y=y, groups=groups,
                        ids=np.array([[r[k] for k in ('scenario', 'trajectory', 't', 'q')]
                                      for r in metadata]))
    _write_rows(folder/'grouped_development_predictions.csv', oof)
    _write_rows(folder/'grouped_development_errors.csv', errors)
    (folder/'audit.json').write_text(json.dumps(dict(candidates=len(x), roots=36,
        external_trajectories=12, supervised_fits=26, rl_updates=0,
        selection_uses_validation=False, validation_generated=False), indent=2))
    finish(folder, hashes)
    print('Frozen models complete', json.dumps(errors), flush=True)


def evaluate(shard):
    verify(ROOT/'fit')
    extra = [p for p in (ROOT/'fit').rglob('*') if p.is_file()]
    folder, hashes = initialize(f'shard{shard}', extra)
    models = {kind: OutcomePredictor.load(ROOT/f'fit/{kind}.npz') for kind in KINDS}
    cfg, raw, selections, prediction_rows, baseline_rows = make_config('validation'), [], [], [], []
    overhead_error = 0.
    for j, scenario in enumerate(SCENARIOS):
        for i in range(4*shard, 4*shard+4):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            original = {key: getattr(tr, key).copy() for key in (
                'channel', 'gain_db', 'requirements_mbps')}
            run = FullDomainRollout(cfg, tr)
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                reference = baseline_command(run, 'cost', MARGINS[j], MINIMUM[j])
                if run.t in TIMES:
                    root = run.clone()
                    eligible, simple = probe_choices(root, reference)
                    snap = causal_snapshot(root, reference, simple)
                    qids, x = features(cfg, snap, root.t)
                    np.testing.assert_array_equal(qids, eligible)
                    predictions = {kind: model.predict(x) for kind, model in models.items()}
                    choices = dict(none=-1, **simple,
                        **{kind: choose(qids, pred) for kind, pred in predictions.items()})
                    # Save causally chosen actions and all predictions before outcome simulation.
                    np.savez_compressed(folder/f'root_s{j}_i{i}_t{root.t}.npz',
                        **snap, eligible=eligible, x=x,
                        **{f'prediction_{k}': v for k, v in predictions.items()})
                    selections.extend(dict(scenario=j, trajectory=i, t=root.t, rule=k, q=q)
                                      for k, q in choices.items())
                    _write_rows(folder/'choices.csv', selections)
                    for kind, pred in predictions.items():
                        prediction_rows.extend(dict(scenario=j, trajectory=i, t=root.t,
                            model=kind, q=int(q), energy_prediction=p[0], outage_prediction_pp=p[1])
                            for q, p in zip(qids, pred))
                    base = branch(root, reference, -1, MARGINS[j], MINIMUM[j])
                    candidates = [(-1, 'normal', base)]
                    for q in qids:
                        result = branch(root, reference, int(q), MARGINS[j], MINIMUM[j])
                        candidates.append((int(q), 'normal', result))
                        overhead_error = max(overhead_error,
                            abs(result['first_block_energy_j']-base['first_block_energy_j']-FIRST_COST))
                        assert result['first_block_outage'] == base['first_block_outage']
                    for q in sorted({choices[k] for k in KINDS if choices[k] >= 0}):
                        candidates.append((q, 'withheld',
                            branch(root, reference, q, MARGINS[j], MINIMUM[j], True)))
                    raw.extend(dict(scenario=j, trajectory=i, t=root.t, q=q, variant=variant,
                        energy_difference_j=value['energy_j']-base['energy_j'],
                        outage_difference=value['outage']-base['outage'], **value)
                        for q, variant, value in candidates)
                    _write_rows(folder/'raw.csv', raw)
                run.step(reference)
            baseline = run.metrics()
            baseline['method'] = 'cost'
            baseline_rows.append(dict(scenario=j, trajectory=i, **baseline))
            for key, value in original.items():
                np.testing.assert_array_equal(getattr(tr, key), value)
            print('predictability validation', shard, j, i, 'branches', len(raw), flush=True)
    assert len(selections) == 36*len(RULES) and overhead_error < 1e-9
    _write_rows(folder/'predictions.csv', prediction_rows)
    _write_rows(folder/'baseline_episodes.csv', baseline_rows)
    (folder/'audit.json').write_text(json.dumps(dict(roots=36, trajectories=12,
        branches=len(raw), overhead_error_j=overhead_error,
        choices_saved_before_replay=True, physical_inputs_unchanged=True,
        model_updates=0, rl_updates=0), indent=2))
    finish(folder, hashes)
    print('validation shard complete', shard, len(raw), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit', 'evaluate'))
    parser.add_argument('--shard', type=int, choices=(0, 1))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'fit':
        fit()
    else:
        if args.shard is None:
            parser.error('evaluate requires --shard')
        evaluate(args.shard)
