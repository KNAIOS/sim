"""Independent replays, grouped uncertainty, and all-result accounting."""
import json

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe import causal_snapshot, probe_choices
from major_revision.information_probe_study import finish, verify
from major_revision.information_probe_verify import replay
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.probe_predictability import KINDS, OutcomePredictor, features, choose
from major_revision.probe_predictability_study import (
    ROOT, SOURCE, STREAM, TIMES, MARGINS, MINIMUM, RULES, rows, integer,
    initialize, snapshot, development_data, error_metrics,
)


def key(row):
    return tuple(integer(row, k) for k in ('scenario', 'trajectory', 't'))


def policy_rows(raw, choices):
    indexed = {(*key(r), integer(r, 'q'), r['variant']): r for r in raw}
    result = []
    for row in choices:
        k, q, rule = key(row), integer(row, 'q'), row['rule']
        base, selected = indexed[*k, -1, 'normal'], indexed[*k, q, 'normal']
        record = dict(scenario=k[0], trajectory=k[1], t=k[2], rule=rule, q=q,
            energy_j=float(selected['energy_j']), outage_pct=100*float(selected['outage']),
            baseline_energy_j=float(base['energy_j']),
            baseline_outage_pct=100*float(base['outage']),
            energy_difference_j=float(selected['energy_j'])-float(base['energy_j']),
            outage_difference_pp=100*(float(selected['outage'])-float(base['outage'])),
            selected_probe=int(q >= 0))
        if rule in KINDS:
            blinded = indexed[*k, q, 'withheld'] if q >= 0 else base
            record.update(
                information_energy_difference_j=float(selected['energy_j'])-float(blinded['energy_j']),
                information_outage_difference_pp=100*(float(selected['outage'])-float(blinded['outage'])))
        else:
            record.update(information_energy_difference_j='', information_outage_difference_pp='')
        result.append(record)
    return result


def summarize(policies):
    per_trajectory, summaries = [], []
    ordinary = ('energy_j', 'outage_pct', 'baseline_energy_j', 'baseline_outage_pct',
                'energy_difference_j', 'outage_difference_pp', 'selected_probe')
    information = ('information_energy_difference_j', 'information_outage_difference_pp')
    for j in range(3):
        for rule in RULES:
            for i in range(8):
                group = [r for r in policies if (r['scenario'], r['rule'], r['trajectory']) == (j, rule, i)]
                assert len(group) == 3
                item = dict(scenario=j, rule=rule, trajectory=i,
                            **{k: float(np.mean([r[k] for r in group])) for k in ordinary})
                item.update({k: float(np.mean([r[k] for r in group])) if rule in KINDS else ''
                             for k in information})
                per_trajectory.append(item)
            group = [r for r in per_trajectory if r['scenario'] == j and r['rule'] == rule]
            item = dict(scenario=j, rule=rule, trajectories=8, roots=24,
                        **{k: float(np.mean([r[k] for r in group])) for k in ordinary})
            item.update({k: float(np.mean([r[k] for r in group])) if rule in KINDS else ''
                         for k in information})
            # Same resampled trajectory indices for every metric/rule/scenario.
            draw = np.random.default_rng(1150900001).integers(0, 8, size=(10000, 8))
            for metric in ('energy_difference_j', 'outage_difference_pp'):
                value = np.array([r[metric] for r in group])
                interval = np.quantile(value[draw].mean(1), [.025, .975])
                item[metric+'_ci95_low'], item[metric+'_ci95_high'] = map(float, interval)
                item[metric+'_trajectory_sd'] = float(value.std(ddof=1))
            item['relative_mean_energy_saving_pct'] = -100*item['energy_difference_j']/item['baseline_energy_j']
            summaries.append(item)
    return per_trajectory, summaries


def development_policy_summary():
    data = rows(ROOT/'fit/grouped_development_predictions.csv')
    source = rows(SOURCE/'verified/all_branches.csv')
    indexed = {(*key(r), integer(r, 'q')): r for r in source if r['variant'] == 'normal'}
    selections, summary = [], []
    for j in range(3):
        for kind in KINDS:
            for i in range(4):
                for t in TIMES:
                    group = [r for r in data if key(r) == (j, i, t) and r['model'] == kind]
                    qids = np.array([integer(r, 'q') for r in group], dtype=int)
                    pred = np.array([[float(r['energy_prediction']), float(r['outage_prediction_pp'])]
                                     for r in group]).reshape(-1, 2)
                    q = choose(qids, pred)
                    value = indexed[j, i, t, q]
                    selections.append(dict(scenario=j, trajectory=i, t=t, model=kind, q=q,
                        energy_difference_j=float(value['energy_difference_j']),
                        outage_difference_pp=100*float(value['outage_difference'])))
            group = [r for r in selections if r['scenario'] == j and r['model'] == kind]
            summary.append(dict(scenario=j, model=kind, roots=12, trajectories=4,
                probes=sum(r['q'] >= 0 for r in group),
                energy_difference_j=float(np.mean([r['energy_difference_j'] for r in group])),
                outage_difference_pp=float(np.mean([r['outage_difference_pp'] for r in group]))))
    return selections, summary


def main():
    verify(ROOT/'fit')
    raw, selections, prediction_rows, extras = [], [], [], []
    for shard in (0, 1):
        folder = ROOT/f'shard{shard}'
        verify(folder)
        raw.extend(rows(folder/'raw.csv'))
        selections.extend(rows(folder/'choices.csv'))
        prediction_rows.extend(rows(folder/'predictions.csv'))
        extras.extend(p for p in folder.rglob('*') if p.is_file())
    extras.extend(p for p in (ROOT/'fit').rglob('*') if p.is_file())
    folder, hashes = initialize('verified', extras)
    indexed = {(*key(r), integer(r, 'q'), r['variant']): r for r in raw}
    selected = {(*key(r), r['rule']): integer(r, 'q') for r in selections}
    recorded_predictions = {(*key(r), r['model'], integer(r, 'q')):
        np.array([float(r['energy_prediction']), float(r['outage_prediction_pp'])])
        for r in prediction_rows}
    assert len(indexed) == len(raw) and len(selected) == len(selections) == 72*len(RULES)
    assert len(recorded_predictions) == len(prediction_rows)
    x, y, _ = development_data()
    models = {kind: OutcomePredictor.load(ROOT/f'fit/{kind}.npz') for kind in KINDS}
    for kind, model in models.items():
        independently_fit = OutcomePredictor.fit(x, y, kind)
        np.testing.assert_array_equal(model.predict(x), independently_fit.predict(x))
    cfg, replays, roots = make_config('validation'), 0, 0
    errors = dict(replay=0., decomposition=0., delta=0., baseline=0.)
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            run = FullDomainRollout(cfg, trajectory(cfg, STREAM+j*10000+i, scenario))
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                reference = baseline_command(run, 'cost', MARGINS[j], MINIMUM[j])
                if run.t in TIMES:
                    roots += 1
                    qids, simple = probe_choices(run, reference)
                    live = causal_snapshot(run, reference, simple)
                    saved = snapshot(ROOT/f'shard{i//4}/root_s{j}_i{i}_t{run.t}.npz')
                    for name, value in live.items():
                        np.testing.assert_array_equal(saved[name], value)
                    qq, xx = features(cfg, live, run.t)
                    np.testing.assert_array_equal(qids, qq)
                    np.testing.assert_array_equal(saved['eligible'], qq)
                    np.testing.assert_array_equal(saved['x'], xx)
                    expected = dict(none=-1, **simple)
                    for kind, model in models.items():
                        pred = model.predict(xx)
                        np.testing.assert_array_equal(saved[f'prediction_{kind}'], pred)
                        for q, value in zip(qq, pred):
                            np.testing.assert_allclose(
                                recorded_predictions[j, i, run.t, kind, int(q)], value,
                                rtol=1e-14, atol=1e-14)
                        expected[kind] = choose(qq, pred)
                    for rule, q in expected.items():
                        assert selected[j, i, run.t, rule] == q
                    expected_keys = {(j, i, run.t, int(q), 'normal') for q in [-1, *qids]}
                    expected_keys |= {(j, i, run.t, q, 'withheld')
                                      for k, q in expected.items() if k in KINDS and q >= 0}
                    found = {k for k in indexed if k[:3] == (j, i, run.t)}
                    assert found == expected_keys
                    base = indexed[j, i, run.t, -1, 'normal']
                    for k in sorted(expected_keys):
                        row = indexed[k]
                        actual = replay(run, reference, k[3], k[4] == 'withheld', MARGINS[j], MINIMUM[j])
                        for name, value in actual.items():
                            errors['replay'] = max(errors['replay'], abs(float(row[name])-value))
                        errors['decomposition'] = max(errors['decomposition'], abs(
                            float(row['energy_j'])-float(row['switching_j'])
                            -float(row['pilot_j'])-float(row['operating_j'])))
                        for metric, delta in (('energy_j', 'energy_difference_j'), ('outage', 'outage_difference')):
                            errors['delta'] = max(errors['delta'],
                                abs(float(row[delta])-(float(row[metric])-float(base[metric]))))
                        replays += 1
                run.step(reference)
            baseline_rows = rows(ROOT/f'shard{i//4}/baseline_episodes.csv')
            old = next(r for r in baseline_rows if integer(r, 'scenario') == j and integer(r, 'trajectory') == i)
            for name, value in run.metrics().items():
                if name != 'method':
                    errors['baseline'] = max(errors['baseline'], abs(value-float(old[name])))
            print('predictability verified', j, i, 'branches', replays, flush=True)
    assert roots == 72 and replays == len(raw) and max(errors.values()) < 1e-9
    assert len(prediction_rows) == 2*sum(k[3] >= 0 and k[4] == 'normal' for k in indexed)
    policies = policy_rows(raw, selections)
    trajectories, summaries = summarize(policies)
    prediction_errors = []
    labelled_predictions = []
    for kind in KINDS:
        group = [r for r in prediction_rows if r['model'] == kind]
        truth, predictions = [], []
        for r in group:
            actual = indexed[*key(r), integer(r, 'q'), 'normal']
            target = [float(actual['energy_difference_j']), 100*float(actual['outage_difference'])]
            prediction = [float(r['energy_prediction']), float(r['outage_prediction_pp'])]
            truth.append(target)
            predictions.append(prediction)
            labelled_predictions.append(dict(**r, energy_truth=target[0], outage_truth_pp=target[1]))
        truth, predictions = np.array(truth), np.array(predictions)
        prediction_errors.append(dict(model=kind, candidates=len(truth),
            **error_metrics(truth, predictions, np.tile(y.mean(0), (len(truth), 1)))))
    cv_choices, cv_summary = development_policy_summary()
    gate = dict(positive_nominal_exploratory_mean=[r['rule'] for r in summaries
            if r['scenario'] == 1 and r['rule'] in KINDS and
            r['energy_difference_j'] < -1e-9 and r['outage_difference_pp'] <= 1e-10],
        nominal_both_descriptive_ci_upper_bounds_pass=[r['rule'] for r in summaries
            if r['scenario'] == 1 and r['rule'] in KINDS and
            r['energy_difference_j_ci95_high'] < 0 and r['outage_difference_pp_ci95_high'] <= 0],
        permit_rl_training=False, permit_full_policy_screen=False, scientific_completion=False,
        reason='Single-intervention supervised diagnostic; not a closed-loop RL evaluation')
    for name, content in (
        ('all_branches.csv', raw), ('all_choices.csv', selections),
        ('policy_roots.csv', policies), ('policy_trajectories.csv', trajectories),
        ('policy_summary.csv', summaries), ('labelled_predictions.csv', labelled_predictions),
        ('prediction_errors.csv', prediction_errors),
        ('grouped_development_choices.csv', cv_choices),
        ('grouped_development_policy_summary.csv', cv_summary),
    ):
        _write_rows(folder/name, content)
    (folder/'gate.json').write_text(json.dumps(gate, indent=2))
    (folder/'audit.json').write_text(json.dumps(dict(errors=errors, roots=roots,
        external_trajectories=24, all_branch_replays=replays, choices=len(selections),
        choices_reconstructed=True, original_training_only_refit_identical=True,
        normal_probe_candidates=len(labelled_predictions)//2, model_updates=0,
        rl_updates=0, scientific_completion=False), indent=2))
    finish(folder, hashes)
    print(json.dumps(dict(summary=summaries, prediction_errors=prediction_errors,
                          gate=gate, errors=errors), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
