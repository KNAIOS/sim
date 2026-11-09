"""Bounded exhaustive single-probe mechanism study; no model training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import _write_rows
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe import (probe_choices, branch, independent_reference,
                                              causal_snapshot)
from major_revision.pilot import SCENARIOS, trajectory

ROOT = Path('results/major_revision_20260921_information_probe')
STREAM = 1140000000
TIMES = (32, 64, 96)
MARGINS = (1., 1.08, 1.04)
MINIMUM = (1, 1, 4)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(part, extras=()):
    folder = ROOT / part
    folder.mkdir(parents=True, exist_ok=False)
    files = [*Path('major_revision').glob('*.py'), *Path('future_vr').glob('*.py'),
             Path('major_revision/INFORMATION_PROBE_PLAN.md'), *map(Path, extras)]
    hashes = {str(p): digest(p) for p in files}
    manifest = dict(pid=os.getpid(), threads=1, stream=STREAM, times=TIMES,
                    margins=MARGINS, minimum=MINIMUM, scenarios=SCENARIOS,
                    config=make_config('validation').as_dict(), hashes=hashes,
                    training_updates=0, independent_confirmation=False,
                    role='new development mechanism data; not RL performance')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return folder, hashes


def finish(folder, hashes):
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    outputs = {str(p): digest(p) for p in folder.rglob('*') if p.is_file()}
    (folder / 'output_hashes.json').write_text(json.dumps(outputs, indent=2))
    (folder / 'COMPLETE').write_text('Mechanism stage complete; not learned superiority.\n')


def verify(folder):
    assert (folder / 'COMPLETE').exists()
    for name, key in (('manifest.json', 'hashes'), ('output_hashes.json', None)):
        values = json.loads((folder / name).read_text())
        for path, expected in (values[key] if key else values).items():
            assert digest(path) == expected, path


def collect(shard):
    folder, hashes = initialize(f'shard{shard}')
    cfg, rows, baseline_rows = make_config('validation'), [], []
    errors = dict(reference=0., overhead=0., first_block_outage=0.)
    root_count = 0
    for j, scenario in enumerate(SCENARIOS):
        for i in range(2*shard, 2*shard+2):
            tr = trajectory(cfg, STREAM+j*10000+i, scenario)
            tr_hashes = {name: hashlib.sha256(getattr(tr, name).tobytes()).hexdigest()
                         for name in ('channel', 'gain_db', 'requirements_mbps')}
            run = FullDomainRollout(cfg, tr)
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    continue
                reference = baseline_command(run, 'cost', MARGINS[j], MINIMUM[j])
                if run.t in TIMES:
                    root = run.clone()
                    eligible, choices = probe_choices(root, reference)
                    # Save causal choices BEFORE evaluating any hidden future.
                    np.savez_compressed(folder/f'root_s{j}_i{i}_t{run.t}.npz',
                        eligible=eligible, **causal_snapshot(root, reference, choices))
                    base = branch(root, reference, -1, MARGINS[j], MINIMUM[j])
                    original = independent_reference(root, reference, MARGINS[j], MINIMUM[j])
                    for key in original:
                        if key != 'method':
                            errors['reference'] = max(errors['reference'], abs(base[key]-original[key]))
                    candidates = [(-1, False, base)]
                    for q in eligible:
                        for withheld in (False, True):
                            result = branch(root, reference, int(q), MARGINS[j], MINIMUM[j], withheld)
                            errors['overhead'] = max(errors['overhead'],
                                abs(result['first_block_energy_j']-base['first_block_energy_j']-.33))
                            errors['first_block_outage'] = max(errors['first_block_outage'],
                                abs(result['first_block_outage']-base['first_block_outage']))
                            candidates.append((int(q), withheld, result))
                    for q, withheld, result in candidates:
                        rows.append(dict(scenario=j, trajectory=i, t=run.t, q=q,
                            variant='withheld' if withheld else 'normal',
                            score_choice=choices['score'], age_choice=choices['age'],
                            energy_difference_j=result['energy_j']-base['energy_j'],
                            outage_difference=result['outage']-base['outage'], **result))
                    root_count += 1
                    _write_rows(folder/'raw.csv', rows)
                    print('probe', shard, j, i, run.t, 'candidates', len(eligible),
                          'rows', len(rows), flush=True)
                run.step(reference)
            baseline = run.metrics()
            baseline['method'] = 'cost'
            baseline_rows.append(dict(scenario=j, trajectory=i, **baseline))
            for name, expected in tr_hashes.items():
                assert hashlib.sha256(getattr(tr, name).tobytes()).hexdigest() == expected, name
    assert root_count == 18 and max(errors.values()) < 1e-9, errors
    _write_rows(folder/'baseline_episodes.csv', baseline_rows)
    (folder/'audit.json').write_text(json.dumps(dict(roots=root_count, rows=len(rows),
        errors=errors, trajectory_hashes_unchanged=True, training_updates=0), indent=2))
    finish(folder, hashes)
    print('probe shard complete', shard, json.dumps(errors), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    collect(args.shard)
