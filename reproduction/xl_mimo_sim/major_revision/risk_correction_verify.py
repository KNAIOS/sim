"""Verified calibration statistics and paired full-episode control outcomes."""
import json

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe_study import finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction import RateRiskModel, select_action
from major_revision.risk_correction_study import (
    ROOT, MODES, initialize, all_files, csv_rows, load_future, causal_data,
)


def probability_metrics(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    clipped = np.clip(p, 1e-6, 1-1e-6)
    bins, ece = [], 0.
    for b in range(10):
        selected = (p >= b/10) & (p < (b+1)/10 if b < 9 else p <= 1)
        n = int(selected.sum())
        predicted = float(np.mean(p[selected])) if n else 0.
        observed = float(np.mean(y[selected])) if n else 0.
        ece += n*abs(predicted-observed)/len(y)
        bins.append(dict(bin=b, samples=n, predicted=predicted, observed=observed))
    return dict(samples=len(y), brier=float(np.mean((p-y)**2)),
        log_loss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
        mean_prediction=float(p.mean()), mean_observed=float(y.mean()), ece=ece), bins


def prediction_audit():
    _, fit_y = load_future('fit_data')
    prevalence = float(fit_y.mean())
    models = {k: RateRiskModel.load(ROOT/f'model/{k}.npz') for k in ('uncalibrated', 'calibrated')}
    metrics, reliability, groups = [], [], []
    for j in range(3):
        for i in range(8):
            with np.load(ROOT/f'audit_data/data_s{j}_i{i}.npz', allow_pickle=False) as z:
                predictions = {k: [] for k in ('old', 'anchored', 'constant', *models)}
                for r in range(len(z['x'])):
                    data = dict(features=z['x'][r], rates=z['rates'][r], available=z['available'][r])
                    req = z['requirements'][r]
                    predictions['old'].append((z['old_rates'][r] < req).astype(float))
                    predictions['anchored'].append((z['rates'][r] < req).astype(float))
                    constant = np.full_like(z['rates'][r], prevalence)
                    constant[:, 0] = z['rates'][r][:, 0] < req
                    constant[z['available'][r].sum(2) == 0] = 1.
                    predictions['constant'].append(constant)
                    for kind, model in models.items():
                        predictions[kind].append(model.predict(data, req))
                y = z['y'].copy()
                groups.append((j, i, y, {k: np.stack(v) for k, v in predictions.items()},
                               z['x'][..., 4].copy()*32.))
    for scenario in (-1, 0, 1, 2):
        chosen = [g for g in groups if scenario == -1 or g[0] == scenario]
        truth = np.concatenate([g[2] for g in chosen])
        ages = np.concatenate([g[4] for g in chosen])
        h = np.broadcast_to(np.arange(8)[None, None, :, None], truth.shape)
        strata = dict(future=h > 0, current=h == 0,
            early_future=(h > 0) & (h < 4), ready_future=h >= 4,
            fresh=(h > 0) & (ages <= 1e-9), stale_1_8=(h > 0) & (ages > 1e-9) & (ages <= 8),
            stale_over_8=(h > 0) & (ages > 8))
        for method in ('old', 'anchored', 'constant', *models):
            pred = np.concatenate([g[3][method] for g in chosen])
            for stratum, selected in strata.items():
                if not np.any(selected):
                    continue
                values, bins = probability_metrics(truth[selected], pred[selected])
                metrics.append(dict(scenario=scenario, method=method, stratum=stratum, **values))
                if stratum == 'future':
                    reliability.extend(dict(scenario=scenario, method=method, **r) for r in bins)
    return metrics, reliability


def replay_all_selected():
    raw = csv_rows(ROOT/'policy_audit/raw.csv')
    cfg = make_config('validation')
    model = RateRiskModel.load(ROOT/'model/calibrated.npz')
    error = 0.
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, 1180000000+j*10000+i, scenario)
            for mode in MODES:
                row = next(r for r in raw if int(r['scenario']) == j and
                           int(r['trajectory']) == i and r['method'] == mode)
                run = FullDomainRollout(cfg, tr)
                actions = []
                while run.t < cfg.episode_steps:
                    if run.t % 4:
                        run.step()
                    else:
                        data = causal_data(run)
                        action = select_action(data, run.req, mode, float(row['parameter']),
                                               int(row['minimum']), model)
                        actions.append(action)
                        run.step(data['masks'][action])
                with np.load(ROOT/f'policy_audit/trace_{mode}_s{j}_i{i}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(actions, z['actions'])
                    np.testing.assert_array_equal(run.slot_energy, z['energy'])
                    np.testing.assert_array_equal(run.outages, z['outage'])
                    error = max(error, abs(float(row['energy_j'])-z['energy'].sum()),
                                abs(float(row['outage'])-z['outage'].mean()))
                for k, value in run.metrics().items():
                    if k != 'method':
                        error = max(error, abs(value-float(row[k])))
            print('risk replay verified', j, i, flush=True)
    assert error < 1e-9, error
    return raw, error


def main():
    extras = []
    for part in ('fit_data', 'calibration_data', 'model', 'audit_data', 'policy_audit',
                 *[f'tune_{m}' for m in MODES]):
        verify(ROOT/part)
        extras.extend(all_files(ROOT/part))
    folder, hashes = initialize('verified', extras)
    raw, error = replay_all_selected()
    metrics, reliability = prediction_audit()
    summary, paired = [], []
    draw = np.random.default_rng(1180900001).integers(0, 8, (10000, 8))
    for j in range(3):
        for mode in MODES:
            group = sorted([r for r in raw if int(r['scenario']) == j and r['method'] == mode],
                           key=lambda r: int(r['trajectory']))
            assert len(group) == 8
            summary.append(dict(scenario=j, method=mode, trajectories=8,
                **{k: float(np.mean([float(r[k]) for r in group])) for k in (
                    'energy_j', 'outage', 'switching_j', 'pilot_j', 'operating_j',
                    'transitions', 'short_sleep_reactivations', 'mean_count')}))
        for reference in ('old', 'anchored'):
            a = sorted([r for r in raw if int(r['scenario']) == j and r['method'] == 'calibrated'],
                       key=lambda r: int(r['trajectory']))
            b = sorted([r for r in raw if int(r['scenario']) == j and r['method'] == reference],
                       key=lambda r: int(r['trajectory']))
            energy = np.array([float(u['energy_j'])-float(v['energy_j']) for u, v in zip(a, b)])
            outage = np.array([100*(float(u['outage'])-float(v['outage'])) for u, v in zip(a, b)])
            ci_e, ci_o = np.quantile(energy[draw].mean(1), [.025, .975]), np.quantile(outage[draw].mean(1), [.025, .975])
            base_e = np.mean([float(r['energy_j']) for r in b])
            paired.append(dict(scenario=j, reference=reference, energy_difference_j=float(energy.mean()),
                saving_pct=float(-100*energy.mean()/base_e), outage_difference_pp=float(outage.mean()),
                energy_ci95_low=float(ci_e[0]), energy_ci95_high=float(ci_e[1]),
                outage_ci95_low=float(ci_o[0]), outage_ci95_high=float(ci_o[1])))
    def metric(j, method):
        return next(r for r in metrics if r['scenario'] == j and r['method'] == method and r['stratum'] == 'future')
    brier_pass = [j for j in range(3) if metric(j, 'calibrated')['brier'] < min(
        metric(j, 'old')['brier'], metric(j, 'constant')['brier'])]
    nominal = next(r for r in summary if r['scenario'] == 1 and r['method'] == 'calibrated')
    ece = metric(-1, 'calibrated')['ece']
    passed = len(brier_pass) == 3 and ece <= .03 and nominal['outage'] <= .05
    gate = dict(brier_pass_scenarios=brier_pass, pooled_future_ece=ece,
        nominal_closed_loop_outage=nominal['outage'], estimator_screen_pass=passed,
        permit_matched_rl_protocol_design=passed, rl_superiority_established=False,
        scientific_completion=False)
    _write_rows(folder/'prediction_metrics.csv', metrics)
    _write_rows(folder/'reliability.csv', reliability)
    _write_rows(folder/'policy_summary.csv', summary)
    _write_rows(folder/'paired_summary.csv', paired)
    (folder/'gate.json').write_text(json.dumps(gate, indent=2))
    (folder/'audit.json').write_text(json.dumps(dict(replay_error=error,
        complete_episode_replays=72, external_policy_trajectories=24,
        prediction_audit_trajectories=24, same_external_stream=True,
        model_updates=0, rl_updates=0, scientific_completion=False), indent=2))
    finish(folder, hashes)
    print(json.dumps(dict(summary=summary, paired=paired, gate=gate, replay_error=error), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
