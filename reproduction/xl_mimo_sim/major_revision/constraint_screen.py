"""Prespecified development screen; keep every training arm and tuned baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS, episode, trajectory
from major_revision.rate_context import candidate_rate_context
from major_revision.confirm_candidate import MARGINS


TRAINING = Path('results/major_revision_20260920_constraint_training')


def baseline(args, cfg, out):
    method = 'PRS' if args.part == 'prs' else 'Switch-aware PRS'
    val_rows, val_means, selections, dev = [], [], [], []
    for j, scenario in enumerate(SCENARIOS):
        validation = [trajectory(cfg, 972000000+j*10000+i, scenario) for i in range(12)]
        means = []
        for margin in MARGINS:
            metrics = [episode(cfg, tr, method, rate_margin=margin) for tr in validation]
            val_rows.extend(dict(scenario=j, trajectory=i, margin=margin, **r)
                            for i, r in enumerate(metrics))
            mean = dict(scenario=j, method=method, margin=margin,
                        energy_j=float(np.mean([r['energy_j'] for r in metrics])),
                        outage=float(np.mean([r['outage'] for r in metrics])))
            means.append(mean)
            val_means.append(mean)
            _write_rows(out/'validation_raw.csv', val_rows)
            _write_rows(out/'validation_summary.csv', val_means)
            print('validation', method, j, margin, round(mean['outage'], 5), flush=True)
        feasible = [m for m in means if m['outage'] <= cfg.outage_target]
        selected = min(feasible, key=lambda r:r['energy_j']) if feasible else min(means,key=lambda r:r['outage'])
        selections.append(dict(**selected, validation_feasible=bool(feasible)))
        _write_rows(out/'selected.csv', selections)
        for i in range(16):
            tr = trajectory(cfg, 973000000+j*10000+i, scenario)
            r = episode(cfg, tr, method, rate_margin=selected['margin'])
            dev.append(dict(scenario=j, trajectory=i, seed=0, margin=selected['margin'], **r))
            if args.part == 'prs':
                full = episode(cfg, tr, 'Full array')
                dev.append(dict(scenario=j, trajectory=i, seed=0, margin=1., **full))
            _write_rows(out/'development_raw.csv', dev)
        print('development complete', method, j, flush=True)


def learned(args, cfg, out):
    models, hashes = [], {}
    for price in (4, 8):
        for seed in (11, 22):
            folder = TRAINING/f'price{price}_seed{seed}'
            assert (folder/'COMPLETE').exists(), folder
            path = folder/'selected.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            model = ConstrainedDQN(cfg, 178, seed)
            model.online.load_state_dict(saved['online'])
            models.append((price, seed, model))
            hashes[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                selected_episode=saved['episode'], validation_key=saved['validation_key'],
                validation_feasible=saved['validation_feasible'])
    (out/'frozen_models.json').write_text(json.dumps(hashes,indent=2))
    rows = []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(16):
            tr = trajectory(cfg, 973000000+j*10000+i, scenario)
            for price, seed, model in models:
                r = episode(cfg, tr, 'Enriched-DQN', model, state_features=candidate_rate_context)
                r['method'] = f'Rate-DQN price{price}'
                rows.append(dict(scenario=j, trajectory=i, seed=seed, margin=1., **r))
            _write_rows(out/'development_raw.csv', rows)
            print('development learned', j, i+1, flush=True)


def run(args):
    torch.set_num_threads(1)
    cfg = make_config('validation')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(part=args.part, scenarios=SCENARIOS, validation_per_scenario=12,
        development_per_scenario=16, margins=MARGINS, config=cfg.as_dict(),
        validation_stream=972000000, development_stream=973000000,
        status='development only; no test-driven training or tuning',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('pilot.py'),Path(__file__).with_name('rate_context.py'))})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.part == 'learned':
        learned(args,cfg,out)
    else:
        baseline(args,cfg,out)
    (out/'COMPLETE').write_text('Prespecified development screen component completed.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=('prs','cost','learned'), required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
