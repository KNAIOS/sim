"""New closed-loop validation; frozen complete grids, no favourable-point rerun."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _write_rows
from major_revision.absolute_return_frontier import mixture_frontier
from major_revision.absolute_return_screen import METHODS, episode as search_episode
from major_revision.action_value_study import MARGINS
from major_revision.bellman_branch_data import ROOT, FEATURE_SETTING, digest
from major_revision.bellman_value import BellmanValue, DEPTH, PRICES, greedy_cost
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.policy_consistent_value import decide, load_model, settings

STREAM = 1110000000
COUNT = 8


def verify_fit(seed):
    folder = ROOT/f'fit_seed{seed}'
    assert (folder/'COMPLETE').exists()
    for path, expected in json.loads((folder/'manifest.json').read_text())['input_hashes'].items():
        assert digest(path) == expected, path
    for path, expected in json.loads((folder/'output_hashes.json').read_text()).items():
        assert digest(path) == expected, path


def initialize(part):
    for seed in (11, 22):
        verify_fit(seed)
    folder = ROOT/part
    folder.mkdir(parents=True, exist_ok=False)
    files = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path(__file__).with_name('BELLMAN_SCREEN_PLAN.md'),
             *[ROOT/f'fit_seed{s}/final.pt' for s in (11, 22)]]
    hashes = {str(p): digest(p) for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(part=part, stream=STREAM,
        count=COUNT, prices=PRICES, depth=DEPTH, threads=1, input_hashes=hashes,
        config=make_config('validation').as_dict(), independent_confirmation=False), indent=2))
    return folder, hashes


def finish(folder, hashes):
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    (folder/'output_hashes.json').write_text(json.dumps(
        {str(p): digest(p) for p in folder.iterdir() if p.is_file()}, indent=2))
    (folder/'COMPLETE').write_text('Frozen validation stage complete; no automatic scientific promotion.\n')


def learned_episode(cfg, tr, model, price, depth):
    run = FullDomainRollout(cfg, tr)
    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
            continue
        x, score = extend_observation(run, FEATURE_SETTING)
        with torch.no_grad():
            action, _ = greedy_cost(model(torch.from_numpy(x[None]), price, depth), price)
        run.step(select_top(score, int(action[0])+1))
    return run.metrics()


def predecessor_episode(cfg, tr, model, setting):
    run = FullDomainRollout(cfg, tr)
    while run.t < cfg.episode_steps:
        if run.t % 4:
            run.step()
        else:
            _, score, _, action = decide(run, setting, model)
            run.step(select_top(score, action+1))
    return run.metrics()


def learned():
    folder, hashes = initialize('screen_learned')
    cfg = make_config('validation')
    models, predecessors = {}, {}
    for seed in (11, 22):
        model = BellmanValue(cfg)
        final = torch.load(ROOT/f'fit_seed{seed}/final.pt', weights_only=True)
        assert final['depth'] == DEPTH
        model.load_state_dict(final['online'], strict=True)
        model.eval()
        myopic = torch.load(ROOT/f'fit_seed{seed}/myopic.pt', weights_only=True)['online']
        for k, value in final['online'].items():
            if k.startswith('risk.'):
                assert torch.equal(value, myopic[k])
        models[seed] = model
        predecessors[seed] = load_model(seed, 'mix_pi')[0]
    config, rows = settings(), []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(COUNT):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            for seed in (11, 22):
                for depth in (1, DEPTH):
                    for price in PRICES:
                        result = learned_episode(cfg, tr, models[seed], price, depth)
                        result.update(seed=seed, scenario=j, trajectory=i,
                                      method='myopic' if depth == 1 else 'bellman',
                                      depth=depth, price=price)
                        rows.append(result)
                result = predecessor_episode(cfg, tr, predecessors[seed], config[j])
                result.update(seed=seed, scenario=j, trajectory=i, method='predecessor', depth=0, price=2.)
                rows.append(result)
            _write_rows(folder/'raw.csv', rows)
            print('Bellman learned screen', j, i+1, '/', COUNT, 'rows', len(rows), flush=True)
    assert len(rows) == 624
    finish(folder, hashes)


def baselines():
    folder, hashes = initialize('screen_baselines')
    cfg, rows = make_config('validation'), []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(COUNT):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            for method in METHODS:
                for minimum in (1, 4):
                    for margin in MARGINS:
                        result = search_episode(cfg, tr, mode=method,
                                                setting=dict(margin=margin, minimum=minimum))
                        result.update(method=method, scenario=j, trajectory=i, minimum=minimum, margin=margin)
                        rows.append(result)
            result = search_episode(cfg, tr, mode='full')
            result.update(method='full', scenario=j, trajectory=i, minimum=1, margin=1.)
            rows.append(result)
            _write_rows(folder/'raw.csv', rows)
            print('Bellman baseline screen', j, i+1, '/', COUNT, 'rows', len(rows), flush=True)
    assert len(rows) == 1176
    finish(folder, hashes)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key != 'method':
                row[key] = float(row[key])
    return rows


def mean(group):
    assert len(group) == COUNT
    return dict(energy_j=float(np.mean([r['energy_j'] for r in group])),
                outage=float(np.mean([r['outage'] for r in group])))


def select_price(points):
    feasible = [p for p in points if p['outage'] <= .05]
    if feasible:
        return min(feasible, key=lambda p: (p['energy_j'], p['outage'], p['price']))
    return min(points, key=lambda p: (p['outage'], p['energy_j'], p['price']))


def promotion(selected, pure, frontier, myopic_frontier, all_scenarios_feasible):
    valid = all(p is not None for p in (pure, frontier, myopic_frontier))
    checks = dict(all_scenarios_feasible=bool(all_scenarios_feasible), controls_exist=valid,
                  nominal_feasible=selected['outage'] <= .05)
    checks.update(pure_gain=bool(valid and selected['energy_j'] <= .98*pure['energy_j']),
                  pure_risk=bool(valid and selected['outage'] <= pure['outage']+.005),
                  matched_search_gain=bool(valid and selected['energy_j'] <= .98*frontier['energy_j']),
                  matched_myopic_gain=bool(valid and selected['energy_j'] < myopic_frontier['energy_j']))
    return {**checks, 'passed': all(checks.values())}


def audit():
    for part in ('screen_learned', 'screen_baselines'):
        folder = ROOT/part
        assert (folder/'COMPLETE').exists()
        for path, expected in json.loads((folder/'manifest.json').read_text())['input_hashes'].items():
            assert digest(path) == expected, path
        for path, expected in json.loads((folder/'output_hashes.json').read_text()).items():
            assert digest(path) == expected, path
    learner = read_rows(ROOT/'screen_learned/raw.csv')
    search = read_rows(ROOT/'screen_baselines/raw.csv')
    assert len(learner) == 624 and len(search) == 1176
    assert len({(r['scenario'], r['trajectory'], r['seed'], r['method'], r['price']) for r in learner}) == len(learner)
    assert len({(r['scenario'], r['trajectory'], r['method'], r['minimum'], r['margin']) for r in search}) == len(search)
    error = max(abs(r['energy_j']-r['operating_j']-r['switching_j']-r['pilot_j']) for r in learner+search)
    assert error < 1e-9
    summaries, selected, baselines_by_sc, pure_by_sc = [], {}, {}, {}
    for j in range(3):
        points = []
        for method, minimum, margin in sorted({(r['method'], r['minimum'], r['margin']) for r in search if r['scenario'] == j}):
            group = [r for r in search if r['scenario'] == j and (r['method'], r['minimum'], r['margin']) == (method, minimum, margin)]
            point = dict(method=method, minimum=minimum, margin=margin, scenario=j, seed=0, price=0, **mean(group))
            points.append(point)
            summaries.append(point)
        assert len(points) == 49
        baselines_by_sc[j] = points
        feasible = [p for p in points if p['method'] != 'full' and p['outage'] <= .05]
        pure_by_sc[j] = min(feasible, key=lambda p: (p['energy_j'], p['outage'])) if feasible else None
        for seed in (11, 22):
            for method in ('myopic', 'bellman', 'predecessor'):
                points = []
                for price in (PRICES if method != 'predecessor' else (2.,)):
                    group = [r for r in learner if r['scenario'] == j and r['seed'] == seed and r['method'] == method and r['price'] == price]
                    point = dict(method=method, minimum=0, margin=0, scenario=j, seed=seed, price=price, **mean(group))
                    points.append(point)
                    summaries.append(point)
                selected[j, seed, method] = select_price(points)
    comparisons, paired, gates = [], [], []
    for j in range(3):
        for seed in (11, 22):
            candidate = selected[j, seed, 'bellman']
            points = baselines_by_sc[j]
            frontier = mixture_frontier(points, candidate['outage'])
            myopic_points = [p for p in summaries if p['scenario'] == j and p['seed'] == seed and p['method'] == 'myopic']
            myopic_points += [p for p in points if p['method'] == 'full']
            mf = mixture_frontier(myopic_points, candidate['outage'])
            all_ok = all(selected[sc, seed, 'bellman']['outage'] <= .05 for sc in range(3))
            checks = promotion(candidate, pure_by_sc[j], frontier, mf, all_ok)
            gates.append(dict(scenario=j, seed=seed, primary=j == 1, **checks))
            row = dict(scenario=j, seed=seed, price=candidate['price'], energy_j=candidate['energy_j'],
                       outage=candidate['outage'], pure_energy_j=None if pure_by_sc[j] is None else pure_by_sc[j]['energy_j'],
                       pure_outage=None if pure_by_sc[j] is None else pure_by_sc[j]['outage'],
                       matched_search_energy_j=None if frontier is None else frontier['energy_j'],
                       matched_myopic_energy_j=None if mf is None else mf['energy_j'],
                       matched_energy_difference_j=None, paired_sd_j=None, lower_energy_trajectories=None)
            if frontier is not None:
                a, b = points[frontier['a']], points[frontier['b']]
                differences = []
                for i in range(COUNT):
                    def get(point):
                        return next(r for r in search if r['scenario'] == j and r['trajectory'] == i and
                                    (r['method'], r['minimum'], r['margin']) == (point['method'], point['minimum'], point['margin']))
                    ra, rb = get(a), get(b)
                    expected = frontier['weight_a']*ra['energy_j']+(1-frontier['weight_a'])*rb['energy_j']
                    actual = next(r for r in learner if r['scenario'] == j and r['trajectory'] == i and
                                  r['seed'] == seed and r['method'] == 'bellman' and r['price'] == candidate['price'])
                    difference = actual['energy_j']-expected
                    differences.append(difference)
                    paired.append(dict(scenario=j, seed=seed, trajectory=i, learned_energy_j=actual['energy_j'],
                                       expected_mixture_energy_j=expected, difference_j=difference,
                                       a_method=a['method'], a_margin=a['margin'], a_minimum=a['minimum'],
                                       b_method=b['method'], b_margin=b['margin'], b_minimum=b['minimum'],
                                       weight_a=frontier['weight_a']))
                np.testing.assert_allclose(np.mean(differences), candidate['energy_j']-frontier['energy_j'], atol=1e-10)
                row.update(matched_energy_difference_j=float(np.mean(differences)),
                           paired_sd_j=float(np.std(differences, ddof=1)),
                           lower_energy_trajectories=int(np.sum(np.asarray(differences) < 0)))
            comparisons.append(row)
    folder = ROOT/'screen_audit'
    folder.mkdir(parents=True, exist_ok=False)
    _write_rows(folder/'all_means.csv', summaries)
    _write_rows(folder/'selected.csv', list(selected.values()))
    _write_rows(folder/'comparisons.csv', comparisons)
    _write_rows(folder/'expected_paired.csv', paired)
    permit = all(g['passed'] for g in gates if g['primary'])
    (folder/'gate.json').write_text(json.dumps(dict(permit_confirmation=permit, primary='nominal', rows=gates), indent=2))
    (folder/'audit.json').write_text(json.dumps(dict(learned_records=len(learner), baseline_records=len(search),
        external_trajectories=24, accounting_error=error, no_significance_claim=True,
        mixture_is_validation_fitted_expectation=True,
        input_hashes={str(ROOT/p/'raw.csv'): digest(ROOT/p/'raw.csv') for p in ('screen_learned', 'screen_baselines')}), indent=2))
    (folder/'COMPLETE').write_text('Complete bounded screen; consult gate.json before further research.\n')
    if permit:
        (folder/'PERMIT_CONFIRMATION').write_text('Requires separate frozen independent confirmation protocol.\n')
    print(json.dumps(dict(comparisons=comparisons, permit_confirmation=permit, gates=gates), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('learned', 'baselines', 'audit'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {'learned': learned, 'baselines': baselines, 'audit': audit}[args.stage]()
