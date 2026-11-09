"""Matched-trajectory checks, complete policy replays, and seed-wise accounting."""
import json
import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.information_probe_study import finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction_study import csv_rows, all_files, episode as search_episode
from major_revision.risk_rl import RiskDQN, episode
from major_revision.risk_rl_study import ROOT, VARIANTS, SEEDS, MODES, initialize, setup


def main():
    settings, risk = setup()
    extras = []
    for part in ('audit', *[f'{m}_{s}' for m in VARIANTS for s in SEEDS]):
        verify(ROOT/part)
        extras += all_files(ROOT/part)
    folder, hashes = initialize('verified', extras)
    for seed in SEEDS:
        old, new = (csv_rows(ROOT/f'{m}_{seed}/training.csv') for m in VARIANTS)
        assert len(old) == len(new) == 120
        assert all(a['trace_hash'] == b['trace_hash'] and a['scenario'] == b['scenario']
                   and a['epsilon'] == b['epsilon'] for a, b in zip(old, new))
    cfg, controllers, chosen = make_config('validation'), {}, {}
    for mode in VARIANTS:
        for seed in SEEDS:
            saved = torch.load(ROOT/f'{mode}_{seed}/selected.pt', map_location='cpu', weights_only=True)
            model = RiskDQN(cfg, seed)
            model.online.load_state_dict(saved['online'])
            controllers[mode, seed] = model
            chosen[mode, seed] = saved['episode']
            selection = csv_rows(ROOT/f'{mode}_{seed}/selection.csv')
            independent = min(selection, key=lambda r: (float(r['violation']), float(r['energy_j']), int(r['episode'])))
            assert int(independent['episode']) == saved['episode']
    raw = csv_rows(ROOT/'audit/raw.csv')
    assert len(raw) == 168
    error, replays = 0., 0
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, 1210000000+j*10000+i, scenario)
            group = [r for r in raw if int(r['scenario']) == j and int(r['trajectory']) == i]
            assert len(group) == 7
            for row in group:
                method, seed = row['method'], int(row['seed'])
                if seed == 0:
                    result, actions, energy, outage = search_episode(cfg, tr, method,
                        *settings[method, j], risk, True)
                else:
                    mode = method.removesuffix('_rl')
                    result, actions, energy, outage = episode(cfg, tr, mode,
                        *settings[mode, j], risk, controllers[mode, seed], trace=True)
                with np.load(ROOT/f'audit/trace_{method}_{seed}_s{j}_i{i}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(actions, z['actions'])
                    np.testing.assert_array_equal(energy, z['energy'])
                    np.testing.assert_array_equal(outage, z['outage'])
                for k, value in result.items():
                    if k != 'method':
                        error = max(error, abs(value-float(row[k])))
                error = max(error, abs(energy.sum()-float(row['energy_j'])),
                    abs(outage.mean()-float(row['outage'])),
                    abs(float(row['energy_j'])-float(row['switching_j'])-float(row['pilot_j'])-float(row['operating_j'])))
                replays += 1
            print('risk RL replay verified', j, i, flush=True)
    assert error < 1e-9
    def group(j, method, seed):
        return sorted([r for r in raw if int(r['scenario']) == j and r['method'] == method
                       and int(r['seed']) == seed], key=lambda r: int(r['trajectory']))
    summary, comparisons = [], []
    identities = [(m, 0) for m in MODES]+[(f'{m}_rl', s) for m in VARIANTS for s in SEEDS]
    draw = np.random.default_rng(1180900001).integers(0, 8, (10000, 8))
    for j in range(3):
        for method, seed in identities:
            data = group(j, method, seed)
            summary.append(dict(scenario=j, method=method, seed=seed, trajectories=8,
                checkpoint=int(data[0]['checkpoint']),
                **{k: float(np.mean([float(r[k]) for r in data])) for k in (
                    'energy_j', 'outage', 'switching_j', 'pilot_j', 'operating_j',
                    'transitions', 'short_sleep_reactivations', 'mean_count')}))
        comparisons_for_scenario = [(f'{m}_rl', s, m, 0) for m in VARIANTS for s in SEEDS]
        comparisons_for_scenario += [('calibrated_rl', s, 'old_rl', s) for s in SEEDS]
        for method, seed, reference, ref_seed in comparisons_for_scenario:
            a, b = group(j, method, seed), group(j, reference, ref_seed)
            e = np.array([float(u['energy_j'])-float(v['energy_j']) for u, v in zip(a, b)])
            o = np.array([100*(float(u['outage'])-float(v['outage'])) for u, v in zip(a, b)])
            ei, oi = np.quantile(e[draw].mean(1), [.025, .975]), np.quantile(o[draw].mean(1), [.025, .975])
            comparisons.append(dict(scenario=j, method=method, seed=seed, reference=reference,
                reference_seed=ref_seed, energy_difference_j=float(e.mean()),
                saving_pct=float(-100*e.mean()/np.mean([float(r['energy_j']) for r in b])),
                outage_difference_pp=float(o.mean()), energy_ci95_low=float(ei[0]),
                energy_ci95_high=float(ei[1]), outage_ci95_low=float(oi[0]), outage_ci95_high=float(oi[1])))
    passing = []
    for seed in SEEDS:
        values = next(r for r in summary if r['scenario'] == 1 and r['method'] == 'calibrated_rl' and r['seed'] == seed)
        delta = next(r for r in comparisons if r['scenario'] == 1 and r['method'] == 'calibrated_rl'
                     and r['reference'] == 'calibrated' and r['seed'] == seed)
        if (values['checkpoint'] > 0 and values['outage'] <= .05 and delta['energy_difference_j'] < -1e-9
                and delta['outage_difference_pp'] <= 1e-10):
            passing.append(seed)
    gate = dict(nominal_strict_mean_pass_seeds=passing,
        both_calibrated_seeds_pass=len(passing) == 2,
        scientific_completion=False, statistical_superiority_established=False,
        standard_channel_transfer_completed=False)
    _write_rows(folder/'summary.csv', summary)
    _write_rows(folder/'paired.csv', comparisons)
    (folder/'gate.json').write_text(json.dumps(gate, indent=2))
    (folder/'verification.json').write_text(json.dumps(dict(complete_replays=replays,
        external_trajectories=24, matched_training_trajectory_pairs=240,
        independent_checkpoint_selection_verified=True, maximum_error=error,
        no_model_updates_during_audit=True), indent=2))
    finish(folder, hashes)
    print(json.dumps(dict(summary=summary, paired=comparisons, gate=gate), indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
