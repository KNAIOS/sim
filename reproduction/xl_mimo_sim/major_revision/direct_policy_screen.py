"""Frozen direct-policy validation and complete strong-search comparison."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.absolute_return_frontier import mixture_frontier
from major_revision.absolute_return_screen import METHODS, episode as search_episode
from major_revision.direct_policy import (ROOT, VAL_STREAM, MARGINS, VARIANTS,
    episode, initialize, finish, verify)
from major_revision.pilot import SCENARIOS, trajectory

COUNT = 8


def load_weights(seed, variant):
    folder = ROOT/f'train_seed{seed}'
    verify(folder)
    with np.load(folder/f'{variant}_history.npz', allow_pickle=False) as z:
        assert z['weights'].shape == (13, 9 if variant == 'state' else 1)
        return z['weights'][-1].copy()


def learned():
    weights = {(s, v): load_weights(s, v) for s in (11, 22) for v in VARIANTS}
    extra = [ROOT/f'train_seed{s}/{v}_history.npz' for s, v in weights]
    folder, hashes = initialize('validation_learned', extra)
    cfg, raw = make_config('validation'), []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(COUNT):
            tr = trajectory(cfg, VAL_STREAM+j*10000+i, scenario)
            for (seed, variant), theta in weights.items():
                for margin in MARGINS:
                    metric = episode(cfg, tr, theta, margin)
                    metric.update(method=variant, seed=seed, scenario=j, trajectory=i, margin=margin)
                    raw.append(metric)
            _write_rows(folder/'raw.csv', raw)
            print('direct validation learned', j, i+1, '/', COUNT, 'rows', len(raw), flush=True)
    assert len(raw) == 768
    finish(folder, hashes)


def baselines():
    folder, hashes = initialize('validation_baselines')
    cfg, raw = make_config('validation'), []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(COUNT):
            tr = trajectory(cfg, VAL_STREAM+j*10000+i, scenario)
            for method in METHODS:
                for minimum in (1, 4):
                    for margin in MARGINS:
                        metric = search_episode(cfg, tr, method, dict(margin=margin, minimum=minimum))
                        metric.update(scenario=j, trajectory=i, minimum=minimum, margin=margin)
                        raw.append(metric)
            metric = search_episode(cfg, tr, 'full')
            metric.update(scenario=j, trajectory=i, minimum=1, margin=1.)
            raw.append(metric)
            _write_rows(folder/'raw.csv', raw)
            print('direct validation baseline', j, i+1, '/', COUNT, 'rows', len(raw), flush=True)
    assert len(raw) == 1560
    finish(folder, hashes)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [{k: v if k in ('method', 'variant') else float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def aggregate(group):
    assert len(group) == COUNT
    return {k: float(np.mean([r[k] for r in group])) for k in ('energy_j', 'outage')}


def select_point(points):
    feasible = [p for p in points if p['outage'] <= .05]
    if feasible:
        return min(feasible, key=lambda p: (p['energy_j'], p['outage'], p['margin']))
    return min(points, key=lambda p: (p['outage'], p['energy_j'], p['margin']))


def promotion(state, pure, matched, bias, all_scenarios):
    exists = all(p is not None for p in (pure, matched, bias))
    checks = dict(all_scenarios_feasible=bool(all_scenarios), controls_exist=exists,
        nominal_feasible=state['outage'] <= .05,
        pure_energy=bool(exists and state['energy_j'] <= .98*pure['energy_j']),
        pure_risk=bool(exists and state['outage'] <= pure['outage']+.005),
        matched_search_energy=bool(exists and state['energy_j'] <= .98*matched['energy_j']),
        matched_bias_energy=bool(exists and state['energy_j'] < bias['energy_j']))
    return dict(**checks, passed=all(checks.values()))


def audit():
    for part in ('validation_learned', 'validation_baselines'):
        verify(ROOT/part)
    learner = read_rows(ROOT/'validation_learned/raw.csv')
    search = read_rows(ROOT/'validation_baselines/raw.csv')
    assert len(learner) == 768 and len(search) == 1560
    assert len({(r['method'],r['seed'],r['scenario'],r['trajectory'],r['margin']) for r in learner}) == 768
    assert len({(r['method'],r['minimum'],r['scenario'],r['trajectory'],r['margin']) for r in search}) == 1560
    error = max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in learner+search)
    assert error < 1e-9
    summary, selected, points_by_sc = [], {}, {}
    for j in range(3):
        points = []
        keys = sorted({(r['method'],r['minimum'],r['margin']) for r in search if r['scenario'] == j})
        for method, minimum, margin in keys:
            group = [r for r in search if r['scenario'] == j and
                     (r['method'],r['minimum'],r['margin']) == (method,minimum,margin)]
            p = dict(scenario=j, method=method, minimum=minimum, margin=margin, seed=0, **aggregate(group))
            points.append(p)
            summary.append(p)
        assert len(points) == 65
        points_by_sc[j] = points
        for seed in (11,22):
            for variant in VARIANTS:
                choices = []
                for margin in MARGINS:
                    group = [r for r in learner if r['scenario'] == j and r['seed'] == seed and
                             r['method'] == variant and r['margin'] == margin]
                    p = dict(scenario=j, method=variant, minimum=1, margin=margin, seed=seed, **aggregate(group))
                    choices.append(p)
                    summary.append(p)
                selected[j,seed,variant] = select_point(choices)
    comparisons, pairs, gates, frontiers = [], [], [], []
    for j in range(3):
        points = points_by_sc[j]
        feasible = [p for p in points if p['method'] != 'full' and p['outage'] <= .05]
        pure = min(feasible, key=lambda p:(p['energy_j'],p['outage'])) if feasible else None
        for seed in (11,22):
            state = selected[j,seed,'state']
            matched = mixture_frontier(points,state['outage'])
            bias_points = [p for p in summary if p['scenario'] == j and p['seed'] == seed and p['method'] == 'bias']
            bias_points += [p for p in points if p['method'] == 'full']
            bias = mixture_frontier(bias_points,state['outage'])
            all_ok = all(selected[s,seed,'state']['outage'] <= .05 for s in range(3))
            gates.append(dict(scenario=j,seed=seed,primary=j==1,
                              **promotion(state,pure,matched,bias,all_ok)))
            row = dict(**state, pure_energy_j=None if pure is None else pure['energy_j'],
                       pure_outage=None if pure is None else pure['outage'],
                       matched_search_energy_j=None if matched is None else matched['energy_j'],
                       matched_bias_energy_j=None if bias is None else bias['energy_j'],
                       difference_j=None, paired_sd_j=None, lower_energy_trajectories=None)
            frontiers.append(dict(scenario=j,seed=seed,pure=pure,matched=matched,points=points,
                                  bias=bias,bias_points=bias_points))
            if matched is not None:
                a,b = points[matched['a']],points[matched['b']]
                w = matched['weight_a']
                diff = []
                for i in range(COUNT):
                    def get(p):
                        return next(r for r in search if r['scenario']==j and r['trajectory']==i and
                                    (r['method'],r['minimum'],r['margin'])==(p['method'],p['minimum'],p['margin']))
                    expected = w*get(a)['energy_j']+(1-w)*get(b)['energy_j']
                    actual = next(r for r in learner if r['scenario']==j and r['trajectory']==i and
                                  r['seed']==seed and r['method']=='state' and r['margin']==state['margin'])
                    diff.append(actual['energy_j']-expected)
                    pairs.append(dict(scenario=j,seed=seed,trajectory=i,energy_j=actual['energy_j'],
                        expected_search_j=expected,difference_j=diff[-1]))
                row.update(difference_j=float(np.mean(diff)),paired_sd_j=float(np.std(diff,ddof=1)),
                           lower_energy_trajectories=int(np.sum(np.asarray(diff)<0)))
            comparisons.append(row)
    folder, hashes = initialize('audit', [ROOT/'validation_learned/raw.csv',ROOT/'validation_baselines/raw.csv'])
    _write_rows(folder/'summary.csv',summary)
    _write_rows(folder/'selected.csv',list(selected.values()))
    _write_rows(folder/'comparisons.csv',comparisons)
    _write_rows(folder/'paired.csv',pairs)
    (folder/'frontiers.json').write_text(json.dumps(frontiers,indent=2))
    gate = dict(rows=gates, permit_confirmation_protocol=all(g['passed'] for g in gates if g['primary']),
                independent_confirmation=False, scientific_completion=False)
    (folder/'gate.json').write_text(json.dumps(gate,indent=2))
    (folder/'audit.json').write_text(json.dumps(dict(learner_records=768,search_records=1560,
        external_trajectories=24,energy_identity_error=error,validation_only=True),indent=2))
    finish(folder,hashes)
    print(json.dumps(dict(comparisons=comparisons,gate=gate),indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('learned','baselines','audit'),required=True)
    torch.set_num_threads(1)
    globals()[parser.parse_args().stage]()
