"""Frozen paired RL training and untouched stream-121 policy audit."""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.constraint_training import preserved_exploration_rng, selection_key
from major_revision.information_probe_study import finish, verify
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.risk_correction import RateRiskModel
from major_revision.risk_correction_study import (
    ROOT as RISK_ROOT, MODES, csv_rows, all_files, initialize as risk_initialize,
    episode as search_episode,
)
from major_revision.risk_rl import RiskDQN, episode


ROOT = Path('results/major_revision_20260922_risk_rl')
VARIANTS = ('old', 'calibrated')
SEEDS = (11, 22)
CHECKPOINTS = (0, 30, 60, 90, 120)


def initialize(part, extras=()):
    import os
    from major_revision.information_probe_study import digest
    folder = ROOT/part
    folder.mkdir(parents=True, exist_ok=False)
    paths = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path('major_revision/RISK_RL_PLAN.md'), *extras]
    hashes = {str(p): digest(p) for p in paths}
    (folder/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(), threads=1,
        hashes=hashes, part=part, config=make_config('validation').as_dict(),
        train_stream=1200000000, selection_stream=1190000000, audit_stream=1210000000,
        episodes=120, checkpoints=CHECKPOINTS, price=2., state_dimension=1658,
        scientific_completion=False), indent=2))
    return folder, hashes


def setup():
    for part in ('model', 'verified', *[f'tune_{m}' for m in MODES]):
        verify(RISK_ROOT/part)
    gate = json.loads((RISK_ROOT/'verified/gate.json').read_text())
    assert gate['permit_matched_rl_protocol_design']
    settings = {}
    for mode in MODES:
        for r in csv_rows(RISK_ROOT/f'tune_{mode}/selected.csv'):
            settings[mode, int(r['scenario'])] = (float(r['parameter']), int(r['minimum']))
    return settings, RateRiskModel.load(RISK_ROOT/'model/calibrated.npz')


def train(mode, seed):
    settings, risk = setup()
    extras = all_files(RISK_ROOT/'model')+all_files(RISK_ROOT/'verified')
    extras += [RISK_ROOT/f'tune_{m}/selected.csv' for m in MODES]
    folder, hashes = initialize(f'{mode}_{seed}', extras)
    cfg = make_config('validation')
    controller = RiskDQN(cfg, seed)
    validation = [(j, i, trajectory(cfg, 1190000000+j*10000+i, s))
                  for j, s in enumerate(SCENARIOS) for i in range(4)]
    best, selected = (float('inf'), float('inf')), None
    history, evaluations, selections = [], [], []
    def validate(checkpoint):
        nonlocal best, selected
        with preserved_exploration_rng(controller):
            current = [dict(scenario=j, trajectory=i, **episode(cfg, tr, mode,
                *settings[mode, j], risk, controller)) for j, i, tr in validation]
        violation, energy, worst = selection_key(current)
        key = (violation, energy)
        saved = dict(online=copy.deepcopy(controller.online.state_dict()),
            target=copy.deepcopy(controller.target.state_dict()), mode=mode, seed=seed,
            episode=checkpoint, updates=controller.steps, selection_key=key,
            parameters=sum(p.numel() for p in controller.online.parameters()))
        torch.save(saved, folder/f'checkpoint_{checkpoint:03d}.pt')
        if key < best:
            best, selected = key, checkpoint
            torch.save(saved, folder/'selected.pt')
        selections.append(dict(episode=checkpoint, violation=violation, energy_j=energy,
            worst_outage=worst, selected_episode=selected, updates=controller.steps))
        evaluations.extend(dict(checkpoint=checkpoint, **r) for r in current)
        _write_rows(folder/'selection.csv', selections)
        _write_rows(folder/'validation.csv', evaluations)
        print('risk RL validation', mode, seed, checkpoint, key, 'selected', selected, flush=True)
    validate(0)
    for ep in range(120):
        j = ep % 3
        tr = trajectory(replace(cfg, episode_steps=cfg.episode_steps+1),
                        1200000000+seed*10000+ep, SCENARIOS[j])
        trace_hash = hashlib.sha256(tr.channel.tobytes()+tr.requirements_mbps.tobytes()).hexdigest()
        epsilon = .04+.46*np.exp(-4*ep/119)
        result = episode(cfg, tr, mode, *settings[mode, j], risk, controller, True, epsilon)
        history.append(dict(episode=ep+1, scenario=j, epsilon=epsilon, trace_hash=trace_hash, **result))
        if (ep+1) % 10 == 0:
            _write_rows(folder/'training.csv', history)
            print('risk RL training', mode, seed, ep+1, flush=True)
        if ep+1 in CHECKPOINTS:
            validate(ep+1)
    (folder/'audit.json').write_text(json.dumps(dict(episodes=120, updates=controller.steps,
        selected_episode=selected, checkpoint_zero_is_search=(selected == 0),
        selected_key=best, state_dimension=1658, seed=seed, mode=mode), indent=2))
    finish(folder, hashes)


def evaluate():
    settings, risk = setup()
    extras = all_files(RISK_ROOT/'model')
    controllers, checkpoints = {}, {}
    cfg = make_config('validation')
    for mode in VARIANTS:
        for seed in SEEDS:
            folder = ROOT/f'{mode}_{seed}'
            verify(folder)
            extras += all_files(folder)
            saved = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
            controller = RiskDQN(cfg, seed)
            controller.online.load_state_dict(saved['online'])
            controllers[mode, seed] = controller
            checkpoints[mode, seed] = saved['episode']
    folder, hashes = initialize('audit', extras)
    raw = []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(8):
            tr = trajectory(cfg, 1210000000+j*10000+i, scenario)
            for mode in MODES:
                result, actions, energy, outage = search_episode(cfg, tr, mode,
                    *settings[mode, j], risk, True)
                raw.append(dict(scenario=j, trajectory=i, seed=0, checkpoint=-1, **result, loss=0.))
                np.savez_compressed(folder/f'trace_{mode}_0_s{j}_i{i}.npz',
                                    actions=actions, energy=energy, outage=outage)
            for (mode, seed), controller in controllers.items():
                result, actions, energy, outage = episode(cfg, tr, mode,
                    *settings[mode, j], risk, controller, trace=True)
                raw.append(dict(scenario=j, trajectory=i, seed=seed,
                    checkpoint=checkpoints[mode, seed], **result))
                np.savez_compressed(folder/f'trace_{mode}_rl_{seed}_s{j}_i{i}.npz',
                                    actions=actions, energy=energy, outage=outage)
            _write_rows(folder/'raw.csv', raw)
            print('risk RL audit', j, i, flush=True)
    assert len(raw) == 168
    finish(folder, hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'evaluate'))
    parser.add_argument('--mode', choices=VARIANTS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'train':
        if args.mode is None or args.seed is None:
            parser.error('train requires --mode and --seed')
        train(args.mode, args.seed)
    else:
        evaluate()
