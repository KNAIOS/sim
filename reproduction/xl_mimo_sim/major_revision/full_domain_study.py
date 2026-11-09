"""Bounded, validation-selected full-domain study; no outcome-driven resweeps."""
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
from major_revision.full_domain import FullDomainDQN, full_domain_episode
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260920_full_domain')
MARGINS = (.90, 1., 1.04, 1.08, 1.16, 1.24)
PRICES = (1, 2, 4)


def initialize(part):
    out = ROOT/part
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in (
        'full_domain.py', 'audit_rollout.py', 'pilot.py', 'spatial_actions.py',
        'constraint_training.py', 'FULL_DOMAIN_PLAN.md')], *Path('future_vr').glob('*.py')]
    cfg = make_config('validation')
    manifest = dict(config=cfg.as_dict(), part=part, minimum_count=1,
        state_dimension=235, actions=16, train_stream=981000000,
        validation_stream=982000000, development_stream=983000000,
        validation_per_scenario=12, development_per_scenario=16,
        prices=PRICES, margins=MARGINS, episodes_per_arm=300, checkpoints=[100,200,300],
        status='fixed development protocol; not final confirmation',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    validation = [(j,i,trajectory(cfg,982000000+j*10000+i,sc))
                  for j,sc in enumerate(SCENARIOS) for i in range(12)]
    return cfg,out,validation,files,manifest


def verify_sources(files, manifest):
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == manifest['source_hashes'][str(p)]
               for p in files)


def baselines():
    cfg,out,validation,files,manifest = initialize('baselines')
    raw, means, selected = [], [], []
    for mode in ('prs','cost'):
        for j in range(3):
            candidates = []
            for margin in MARGINS:
                group = [dict(scenario=j,trajectory=i,margin=margin,
                    **full_domain_episode(cfg,tr,margin=margin,mode=mode))
                    for scenario,i,tr in validation if scenario==j]
                raw.extend(group)
                mean = dict(scenario=j,mode=mode,margin=margin,
                    energy_j=float(np.mean([r['energy_j'] for r in group])),
                    outage=float(np.mean([r['outage'] for r in group])))
                candidates.append(mean)
                means.append(mean)
                _write_rows(out/'validation_raw.csv',raw)
                _write_rows(out/'validation_summary.csv',means)
                print('full-domain baseline',mode,j,margin,round(mean['outage'],5),flush=True)
            feasible = [r for r in candidates if r['outage']<=cfg.outage_target]
            chosen = min(feasible,key=lambda r:r['energy_j']) if feasible else min(
                candidates,key=lambda r:(r['outage'],r['energy_j']))
            selected.append(dict(**chosen,validation_feasible=bool(feasible)))
    # Freeze all tuning choices before any development trajectory is evaluated.
    _write_rows(out/'selected.csv',selected)
    rows = []
    for j,sc in enumerate(SCENARIOS):
        for i in range(16):
            tr = trajectory(cfg,983000000+j*10000+i,sc)
            for setting in [r for r in selected if r['scenario']==j]+[dict(mode='full',margin=1.)]:
                result=full_domain_episode(cfg,tr,mode=setting['mode'],margin=setting['margin'])
                rows.append(dict(scenario=j,trajectory=i,seed=0,margin=setting['margin'],**result))
            _write_rows(out/'development_raw.csv',rows)
        print('full-domain baseline development',j,'complete',flush=True)
    verify_sources(files,manifest)
    (out/'COMPLETE').write_text('Full-domain baselines finished, including all development traces.\n')


def training(seed):
    cfg,out,validation,files,manifest = initialize(f'seed{seed}')
    extended = replace(cfg,episode_steps=cfg.episode_steps+1)
    history, validation_rows, selections = [], [], []
    best_key = (float('inf'),float('inf'))
    for price in PRICES:
        model = FullDomainDQN(cfg,seed)
        model.dual = float(price)
        rng = np.random.default_rng(seed)
        for ep in range(300):
            j = int(rng.integers(3))
            tr = trajectory(extended,981000000+seed*10000+ep,SCENARIOS[j])
            epsilon = .04+.76*np.exp(-4*ep/299)
            result = full_domain_episode(cfg,tr,model,True,epsilon)
            history.append(dict(seed=seed,price=price,episode=ep+1,scenario=j,epsilon=epsilon,**result))
            if (ep+1)%10==0:
                _write_rows(out/'training.csv',history)
                print('full-domain train',seed,price,ep+1,flush=True)
            if (ep+1)%100:
                continue
            current=[]
            with preserved_exploration_rng(model):
                for j,i,tr in validation:
                    current.append(dict(scenario=j,trajectory=i,
                        **full_domain_episode(cfg,tr,model)))
            violation,energy,worst = selection_key(current)
            key = (violation,energy)
            saved = dict(online=copy.deepcopy(model.online.state_dict()),
                target=copy.deepcopy(model.target.state_dict()),state_dimension=235,actions=16,
                minimum_count=1,seed=seed,price=price,episode=ep+1,validation_key=key,
                worst_validation_outage=worst,validation_feasible=violation==0.,
                bootstrap_truncation=True)
            torch.save(saved,out/f'price{price}_checkpoint_{ep+1:03d}.pt')
            if key < best_key:
                best_key=key
                torch.save(saved,out/'selected.pt')
            selections.append(dict(price=price,episode=ep+1,violation=violation,
                energy_j=energy,worst_outage=worst))
            validation_rows.extend(dict(price=price,seed=seed,episode=ep+1,**r) for r in current)
            _write_rows(out/'validation.csv',validation_rows)
            _write_rows(out/'selection.csv',selections)
            print('full-domain validation',seed,price,ep+1,'outage',round(worst,5),
                  'energy',round(energy,3),flush=True)
    saved=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    model.online.load_state_dict(saved['online'])
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(16):
            tr=trajectory(cfg,983000000+j*10000+i,sc)
            rows.append(dict(scenario=j,trajectory=i,seed=seed,price=saved['price'],
                selected_episode=saved['episode'],**full_domain_episode(cfg,tr,model)))
            _write_rows(out/'development_raw.csv',rows)
        print('full-domain learned development',seed,j,'complete',flush=True)
    verify_sources(files,manifest)
    (out/'COMPLETE').write_text('Full-domain training and development finished.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=('baselines','seed11','seed22'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    baselines() if args.part=='baselines' else training(int(args.part[4:]))
