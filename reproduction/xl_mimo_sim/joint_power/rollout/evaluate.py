"""Causal closed-loop screen; no diagnostic outcome labels enter deployment."""
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.calibrated.study import episode as baseline_episode
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.postdecision.learning import Learner as Previous, episode as previous_episode
from joint_power.postdecision.study import ROOT as PREVIOUS
from joint_power.rollout.analyze import intact
from joint_power.rollout.collect import ROOT, base_command
from joint_power.rollout.probe import DifferenceValue, select
from joint_power.study import SCENARIOS, digest, make_trace, write_rows


def episode(trace, predictor, model, price, ridge=None):
    env = Environment(make_config('validation'), trace)
    while env.t < env.cfg.episode_steps:
        candidates = env.candidates()
        state = decision(env, candidates, predictor)
        reference = int(base_command(env).sum())-1
        np.testing.assert_array_equal(state.masks[reference], base_command(env))
        x = state.x[None]
        if ridge is None:
            with torch.no_grad():
                predicted = model(torch.from_numpy(x), torch.tensor([reference])).numpy()
        else:
            standardized = np.clip((x-ridge['mean'])/ridge['std'], -10, 10)
            predicted = (standardized-standardized[:, reference, None])@ridge['weights']
        action = int(select(predicted, np.array([reference]), price)[0])
        env.block(state.masks[action])
    return env.metrics(), env.records


def screen(rows):
    mean = lambda s, m, metric: float(np.mean([r[metric] for r in rows if r['scenario'] == s and r['method'] == m]))
    per_seed = {str(seed): [float(np.mean([r['outage'] for r in rows if r['scenario'] == s
                 and r['method'] == 'MC improvement' and r['seed'] == seed])) for s in range(3)] for seed in (11, 22)}
    energy = {str(seed): float(np.mean([r['energy_j'] for r in rows if r['scenario'] == 1
              and r['method'] == 'MC improvement' and r['seed'] == seed])) for seed in (11, 22)}
    prs = mean(1, 'PRS power', 'energy_j')
    gain = 1-mean(1, 'MC improvement', 'energy_j')/prs
    return dict(passed=bool(max(max(v) for v in per_seed.values()) <= .05 and
                gain >= .02 and max(energy.values()) <= prs), per_seed_outage=per_seed,
                nominal_energy_by_seed=energy, nominal_prs_energy=prs, nominal_saving_vs_prs=gain,
                manuscript_ready=False, scope='Bounded development screen; not final confirmation')


def main():
    torch.set_num_threads(1)
    intact()
    hashes = json.loads((ROOT/'probe/output_hashes.json').read_text())
    assert all(digest(p) == h for p, h in hashes.items())
    output = ROOT/'evaluation'
    output.mkdir(exist_ok=False)
    predictor, cfg = load_predictor(), make_config('validation')
    models, previous = {}, {}
    for seed in (11, 22):
        checkpoint = torch.load(ROOT/f'probe/seed{seed}_selected.pt', weights_only=True)
        model = DifferenceValue(checkpoint['state']['mean'], checkpoint['state']['std'])
        model.load_state_dict(checkpoint['state'])
        models[seed] = (model.eval(), checkpoint['price'])
        checkpoint = torch.load(PREVIOUS/f'PDS-branch_seed{seed}/selected.pt', weights_only=True)
        learner = Previous(seed)
        learner.online.load_state_dict(checkpoint['state'])
        previous[seed] = (learner, checkpoint['parameter'])
    with np.load(ROOT/'probe/ridge.npz') as saved:
        ridge = {k: saved[k] for k in saved.files}
    ridge_price = next(r['price'] for r in json.loads((ROOT/'probe/selection.json').read_text()) if r['method'] == 'Ridge')
    myopic_price = json.loads((PREVIOUS/'baseline_selection.json').read_text())['Myopic']['parameter']
    specs = [(m, 0) for m in ('Full power', 'PRS power', 'Raw full', 'Raw coarse', 'Myopic', 'Ridge')]
    specs += [(m, seed) for seed in (11, 22) for m in ('Previous branch RL', 'MC improvement')]
    files = [Path(__file__), ROOT/'probe/output_hashes.json', ROOT/'probe/selection.json',
             PREVIOUS/'output_hashes.json']
    manifest = {str(p): digest(p) for p in files}
    (output/'manifest.json').write_text(json.dumps(dict(input_hashes=manifest,
        development_stream=3809220000, trajectories_per_scenario=4,
        no_hindsight_at_inference=True, no_retraining=True), indent=2)+'\n')
    rows, slots = [], []
    for s, spec in enumerate(SCENARIOS):
        for i in range(4):
            trace = make_trace(cfg, 3809220000+s*10000+i, spec)
            for method, seed in specs:
                if method == 'MC improvement':
                    model, price = models[seed]
                    metrics, records = episode(trace, predictor, model, price)
                elif method == 'Ridge':
                    metrics, records = episode(trace, predictor, None, ridge_price, ridge)
                elif method == 'Previous branch RL':
                    learner, price = previous[seed]
                    metrics, records = previous_episode(trace, predictor, learner, price)
                else:
                    metrics, records = baseline_episode(trace, method, predictor, parameter=myopic_price)
                identity = dict(scenario=s, trajectory=i, method=method, seed=seed)
                rows.append(dict(**identity, **{k: v for k, v in metrics.items() if k != 'loss'}))
                slots.extend(dict(**identity, **r) for r in records)
            print(f'causal MC screen scenario={s} trajectory={i}', flush=True)
    write_rows(output/'episodes.csv', rows)
    write_rows(output/'slots.csv', slots)
    summary = []
    for s in range(3):
        for method in dict.fromkeys(m for m, _ in specs):
            group = [r for r in rows if r['scenario'] == s and r['method'] == method]
            summary.append(dict(scenario=s, method=method, **{k: float(np.mean([r[k] for r in group]))
                for k in ('energy_j', 'outage', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'mean_active')}))
    write_rows(output/'summary.csv', summary)
    result = screen(rows)
    (output/'screen.json').write_text(json.dumps(result, indent=2)+'\n')
    intact()
    assert all(digest(p) == h for p, h in hashes.items())
    assert all(digest(p) == h for p, h in manifest.items())
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file() and p.name != 'output_hashes.json'}, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, screen=result), indent=2))


if __name__ == '__main__':
    main()
