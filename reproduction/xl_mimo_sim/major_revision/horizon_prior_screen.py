"""Frozen, one-candidate screen; never trains or overwrites existing outputs."""
import hashlib
import json
import os
import platform
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import action_episode
from major_revision.action_value_report import numeric_rows, means
from major_revision.action_value_study import ROOT as OLD_ROOT
from major_revision.horizon_prior import FixedHorizon
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260921_horizon_prior')


def gate(candidate, previous, full):
    comparisons = [dict(scenario=j,
        no_worse=candidate[j]['outage'] <= previous[j]['outage'],
        improvement=previous[j]['outage'] - candidate[j]['outage'],
        feasible=candidate[j]['outage'] <= .05,
        energy_ok=candidate[j]['energy_j'] <= full[j]['energy_j']) for j in range(3)]
    allowed = (all(r['no_worse'] and r['energy_ok'] for r in comparisons)
        and any(r['improvement'] >= .05 for r in comparisons)
        and sum(r['feasible'] for r in comparisons) >= 2)
    return dict(permit_training=bool(allowed), comparisons=comparisons)


def main():
    cfg = make_config('validation')
    old_folder = OLD_ROOT / 'baselines'
    assert (OLD_ROOT / 'COMPLETE').exists()
    old_manifest = json.loads((old_folder / 'manifest.json').read_text())
    for path, digest in old_manifest['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    old_audit = json.loads((OLD_ROOT / 'audit.json').read_text())
    # Reuse old records only after their audit hashes have also been checked.
    for path, digest in old_audit['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    old_validation = numeric_rows(old_folder / 'validation_raw.csv')
    old_development = numeric_rows(old_folder / 'development.csv')
    assert len(old_validation) == 936 and len(old_development) == 192
    folder = ROOT / 'screen'
    folder.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[Path(__file__).with_name(n) for n in
        ('horizon_prior.py', 'HORIZON_PRIOR_PLAN.md', 'action_value.py',
         'action_value_report.py', 'action_value_study.py', 'gram_control.py',
         'full_domain.py', 'audit_rollout.py', 'pilot.py', 'spatial_actions.py',
         'constraint_training.py', 'test_horizon_prior.py')], *Path('future_vr').glob('*.py')]
    inputs = [old_folder / 'manifest.json', OLD_ROOT / 'audit.json',
              old_folder / 'validation_raw.csv', old_folder / 'development.csv']
    manifest = dict(config=cfg.as_dict(), gamma_per_decision_block=cfg.rl_gamma,
        price=2., blocks=2, block_slots=4, parameters=0,
        validation_stream=997000000, development_stream=998000000,
        trajectories_per_scenario=dict(validation=12, development=16),
        role='viewed method-development streams; not untouched confirmation',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        process_id=os.getpid(), python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, threads=1)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, summaries = [], []
    model = FixedHorizon(cfg)
    gate_result = None
    for split, stream, count, old in (
        ('validation', 997000000, 12, old_validation),
        ('development', 998000000, 16, old_development)):
        for j, sc in enumerate(SCENARIOS):
            group = []
            for i in range(count):
                tr = trajectory(cfg, stream + j * 10000 + i, sc)
                result = action_episode(cfg, tr, model)
                result['method'] = 'H8 prior (no learning)'
                group.append(dict(split=split, scenario=j, trajectory=i, **result))
            rows.extend(group)
            summary = dict(split=split, scenario=j, method='H8', **means(group))
            summaries.append(summary)
            for mode in ('prior', 'full'):
                matched = [r for r in old if r['method'] == mode and r['scenario'] == j]
                assert len(matched) == count
                summaries.append(dict(split=split, scenario=j, method=mode, **means(matched)))
            _write_rows(folder / 'raw.csv', rows)
            _write_rows(folder / 'summary.csv', summaries)
            print('horizon prior', summary, flush=True)
        if split == 'validation':
            sets = [[next(r for r in summaries if r['split'] == split
                and r['scenario'] == j and r['method'] == mode) for j in range(3)]
                for mode in ('H8', 'prior', 'full')]
            gate_result = gate(*sets)
            # Freeze before reading development performance; never modify gate.
            (folder / 'gate.json').write_text(json.dumps(gate_result, indent=2))
            print('validation-only training gate', gate_result, flush=True)
    assert len(rows) == 84
    for key in ('source_hashes', 'input_hashes'):
        for path, digest in manifest[key].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert len({(r['split'], r['scenario'], r['trajectory']) for r in rows}) == 84
    error = max(abs(r['energy_j'] - r['operating_j'] - r['switching_j'] - r['pilot_j']) for r in rows)
    assert error < 1e-8
    (folder / 'audit.json').write_text(json.dumps(dict(records=84,
        energy_identity_error=error, hashes_match=True, gate_validation_only=True), indent=2))
    (folder / 'COMPLETE').write_text('Fixed H8 screen complete; no training performed.\n')
    if gate_result['permit_training']:
        (ROOT / 'PERMIT_TRAINING').write_text('Frozen validation gate passed.\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
