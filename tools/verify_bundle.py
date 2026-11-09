"""Read-only verification of archived bytes and the paper's primary result."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT/'reproduction'
SIM = REPRO/'xl_mimo_sim'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def relocate(name, original):
    name = name.replace('\\', '/')
    original = original.replace('\\', '/').rstrip('/')
    if name.lower().startswith(original.lower()+'/'):
        return REPRO/name[len(original)+1:]
    return SIM/name


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    package = json.loads((ROOT/'PACKAGING_MANIFEST.json').read_text(encoding='utf-8'))
    for name, record in package['files'].items():
        assert digest(ROOT/name) == record['sha256'], name
    folder = SIM/'results/joint_power_20260922_focus_confirmation'
    manifest = json.loads((folder/'manifest.json').read_text())
    frozen = dict(manifest['input_hashes'])
    frozen.update(json.loads((folder/'output_hashes.json').read_text()))
    for record in manifest['checkpoints'].values():
        frozen[record['path']] = record['sha256']
    for name, expected in frozen.items():
        assert digest(relocate(name, package['original_root'])) == expected, name
    episodes = rows(folder/'episodes.csv')
    assert len(episodes) == 120
    expected = {(str(i), m, s) for i in range(40)
                for m,s in (('Value4','11'), ('Value4','22'), ('WBbudgetPRS4','0'))}
    assert {(r['trajectory'],r['method'],r['seed']) for r in episodes} == expected
    summary = {r['method']:r for r in rows(folder/'summary.csv') if r['seed'] in ('-1','0')}
    for method in summary:
        subset = [r for r in episodes if r['method'] == method]
        for key in ('energy_j','outage','circuit_j','pa_j','switching_j','pilot_j','mean_active'):
            assert math.isclose(sum(float(r[key]) for r in subset)/len(subset),
                                float(summary[method][key]), abs_tol=1e-9), (method,key)
    for row in episodes:
        assert math.isclose(float(row['energy_j']), sum(float(row[k]) for k in
            ('circuit_j','pa_j','switching_j','pilot_j')), abs_tol=1e-9)
    slot_count = sum(len(rows(folder/f'NLOS_{i}/slots.csv')) for i in range(40))
    assert slot_count == 16800
    gain = 100*(1-float(summary['Value4']['energy_j'])/float(summary['WBbudgetPRS4']['energy_j']))
    original_report = json.loads((folder/'report.json').read_text())
    assert math.isclose(gain, original_report['energy_saving_percent'], abs_tol=1e-10)
    report = dict(copied_files_checked=len(package['files']), frozen_records_checked=len(frozen),
                  confirmation_episodes=120, independent_drops=40, slots=slot_count,
                  energy_saving_percent=gain, original_data_modified=False,
                  verification='hash, saved episode means, energy ledger and slot counts; no new simulation')
    out = ROOT/'runs'
    out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
