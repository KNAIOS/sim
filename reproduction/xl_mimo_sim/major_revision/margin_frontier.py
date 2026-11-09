"""Development-only search reliability frontier; true demand is never inflated.

Margins are chosen on validation, then evaluated once on a separate development
stream. Existing pilot DQNs are evaluated on that same stream without retraining.
This is a diagnostic screen, not a statistically certified final comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import SCENARIOS, episode, trajectory

METHODS = ('PRS', 'Switch-aware PRS')
MARGINS = (1., 1.05, 1.10, 1.20, 1.35, 1.50)


def run(args):
    torch.set_num_threads(1)
    cfg = make_config('validation')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name('pilot.py')]
    manifest = dict(margins=MARGINS, scenarios=SCENARIOS,
                    validation_stream=924000000, development_stream=925000000,
                    trajectories=args.trajectories, config=cfg.as_dict(),
                    status='development screen, not final publication evidence',
                    selection='per-scenario minimum mean energy among mean-outage-feasible margins; otherwise minimum outage',
                    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    validation, selected, development = [], [], []
    controllers = []
    for path in sorted(Path(args.checkpoints).glob('*DQN_*.pt')):
        # Pilot metadata contains a NumPy float64 validation score. Keep the
        # restricted loader; allow only the specific scalar/dtype constructors.
        with torch.serialization.safe_globals([np._core.multiarray.scalar,
                                               np.dtype, np.dtypes.Float64DType]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        method, seed = path.stem.rsplit('_', 1)
        model = ConstrainedDQN(cfg, saved['state_dimension'], int(seed))
        model.online.load_state_dict(saved['online'])
        controllers.append((method, int(seed), model))
    if len(controllers) != 4:
        raise ValueError('Expected all four pilot checkpoints')
    for j, scenario in enumerate(SCENARIOS):
        val = [trajectory(cfg,924000000+j*10000+i,scenario) for i in range(args.trajectories)]
        choices = []
        for method in METHODS:
            means = []
            for margin in MARGINS:
                group = [episode(cfg,tr,method,rate_margin=margin) for tr in val]
                validation.extend(dict(scenario=j,trajectory=i,margin=margin,**r) for i,r in enumerate(group))
                means.append(dict(scenario=j,method=method,margin=margin,
                                  energy_j=float(np.mean([r['energy_j'] for r in group])),
                                  outage=float(np.mean([r['outage'] for r in group]))))
                _write_rows(out/'validation_raw.csv',validation)
                print('validation',j,method,margin,flush=True)
            feasible = [r for r in means if r['outage']<=cfg.outage_target]
            best = min(feasible,key=lambda r:r['energy_j']) if feasible else min(means,key=lambda r:r['outage'])
            selected.append(dict(**best,validation_feasible=bool(feasible)))
            choices.append((method,best['margin']))
        _write_rows(out/'selected.csv',selected)
        for i in range(args.trajectories):
            tr = trajectory(cfg,925000000+j*10000+i,scenario)
            for method,margin in choices:
                development.append(dict(scenario=j,trajectory=i,seed=0,margin=margin,
                                        **episode(cfg,tr,method,rate_margin=margin)))
            for method,seed,model in controllers:
                development.append(dict(scenario=j,trajectory=i,seed=seed,margin=1.,
                                        **episode(cfg,tr,method,model)))
            _write_rows(out/'development_raw.csv',development)
        print('development scenario complete',j,flush=True)
    summary=[]
    for j in range(len(SCENARIOS)):
        for method in (*METHODS,'Original-DQN','Enriched-DQN'):
            group=[r for r in development if r['scenario']==j and r['method']==method]
            summary.append(dict(scenario=j,method=method,**{k:float(np.mean([r[k] for r in group]))
                            for k in ('energy_j','outage','switching_j','mean_count')}))
    _write_rows(out/'development_summary.csv',summary)
    (out/'COMPLETE').write_text('Development diagnostic only.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoints',default='results/major_revision_20260920_pilot')
    p.add_argument('--trajectories',type=int,default=8)
    run(p.parse_args())
