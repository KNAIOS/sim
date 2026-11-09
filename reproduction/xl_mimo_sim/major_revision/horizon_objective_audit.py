"""Post-hoc objective diagnostic of the failed fixed H8 prior, not new tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from future_vr.controllers import select_top
from major_revision.action_value import observation, action_size, full_slot_energy
from major_revision.action_value_report import numeric_rows
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import representation_size
from major_revision.horizon_prior import horizon_values
from major_revision.horizon_prior_screen import ROOT
from major_revision.pilot import SCENARIOS, trajectory


def inspect_choice(cfg, state):
    rows = state[representation_size(cfg):].reshape(cfg.subarrays, action_size(cfg))
    rates = rows[:, cfg.subarrays + cfg.horizon:
        cfg.subarrays + cfg.horizon * (1 + cfg.users)].reshape(cfg.subarrays, cfg.horizon, cfg.users)
    predicted_outage = (rates < 1.).mean((1, 2))
    energy = rows[:, cfg.subarrays:cfg.subarrays + cfg.horizon].sum(1) * full_slot_energy(cfg)
    q = horizon_values(cfg, torch.as_tensor(state)).numpy()
    chosen = int(q.argmax())
    feasible = np.flatnonzero(predicted_outage <= cfg.outage_target)
    ref = min(feasible, key=lambda i:(energy[i], i)) if len(feasible) else None
    category = ('chosen_predicted_feasible' if predicted_outage[chosen] <= cfg.outage_target
        else 'unsafe_with_feasible_alternative' if ref is not None else 'no_predicted_feasible_action')
    row = dict(chosen=chosen + 1, category=category, feasible_actions=len(feasible),
        chosen_predicted_outage=float(predicted_outage[chosen]), chosen_predicted_energy_j=float(energy[chosen]),
        chosen_q=float(q[chosen]), alternative_count=int(ref + 1) if ref is not None else 0,
        alternative_predicted_outage=float(predicted_outage[ref]) if ref is not None else None,
        alternative_predicted_energy_j=float(energy[ref]) if ref is not None else None,
        alternative_q=float(q[ref]) if ref is not None else None)
    if ref is not None:
        assert q[chosen] >= q[ref]
    return chosen, row


def main():
    folder = ROOT / 'objective_audit'
    screen = ROOT / 'screen'
    assert (screen / 'COMPLETE').exists()
    assert not json.loads((screen / 'gate.json').read_text())['permit_training']
    old = json.loads((screen / 'manifest.json').read_text())
    for kind in ('source_hashes', 'input_hashes'):
        for path, digest in old[kind].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    original = numeric_rows(screen / 'raw.csv')
    files = [Path(__file__), Path(__file__).with_name('HORIZON_OBJECTIVE_AUDIT_PLAN.md'),
             Path(__file__).with_name('test_horizon_objective.py'), screen / 'manifest.json', screen / 'raw.csv']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    folder.mkdir(exist_ok=False)
    (folder / 'manifest.json').write_text(json.dumps(dict(input_and_source_hashes=hashes,
        role='post-hoc diagnostic of fixed H8; not trained policy or independent performance test',
        development_stream=998000000, trajectories_per_scenario=4, threads=1), indent=2))
    cfg = make_config('validation')
    states, episodes, counts = [], [], []
    max_error = 0.
    for j, sc in enumerate(SCENARIOS):
        for i in range(4):
            run = FullDomainRollout(cfg, trajectory(cfg, 998000000 + j*10000 + i, sc))
            local = []
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                state, score = observation(run)
                chosen, row = inspect_choice(cfg, state)
                local.append(dict(scenario=j, trajectory=i, step=run.t, **row))
                run.step(select_top(score, chosen + 1))
            actual = run.metrics()
            expected = next(r for r in original if r['split'] == 'development' and r['scenario'] == j and r['trajectory'] == i)
            for metric in ('energy_j', 'outage', 'operating_j', 'switching_j', 'pilot_j'):
                error = abs(actual[metric] - expected[metric])
                assert error < 1e-9
                max_error = max(max_error, error)
            episodes.append(dict(scenario=j, trajectory=i, **actual))
            states.extend(local)
            counts.append(dict(scenario=j, trajectory=i, decisions=len(local), **{
                category:sum(r['category'] == category for r in local) for category in
                ('chosen_predicted_feasible','unsafe_with_feasible_alternative','no_predicted_feasible_action')}))
        print('objective audit scenario', j, 'complete', flush=True)
    summaries = []
    for j in range(3):
        group = [r for r in states if r['scenario'] == j]
        for category in sorted({r['category'] for r in group}):
            cell = [r for r in group if r['category'] == category]
            summaries.append(dict(scenario=j, category=category, count=len(cell), total=len(group),
                fraction=len(cell)/len(group), mean_predicted_outage=float(np.mean([r['chosen_predicted_outage'] for r in cell]))))
    assert len(states) == 420 and len(episodes) == 12
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    for name, data in (('states',states),('episodes',episodes),('per_trajectory',counts),('summary',summaries)):
        _write_rows(folder / f'{name}.csv', data)
    (folder / 'audit.json').write_text(json.dumps(dict(states=420, trajectories=12,
        replay_error=max_error, source_hashes_match=True, no_counterfactual_performance_claim=True), indent=2))
    (folder / 'COMPLETE').write_text('Post-hoc objective audit completed without training or tuning.\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
