"""Normalize both directional probes without changing their archived raw CSVs."""
import hashlib
import json
from pathlib import Path
from future_vr.experiment import _write_rows
from major_revision.ready_report import read


def run():
    output=Path('results/major_revision_20260920_headroom_combined')
    output.mkdir(parents=True,exist_ok=False)
    inputs={-1:Path('results/major_revision_20260920_search_headroom/pairs.csv'),
             1:Path('results/major_revision_20260920_search_plus_one/pairs.csv')}
    rows=[]
    originals={}
    for direction,path in inputs.items():
        for source in read(path):
            key=tuple(int(source[k]) for k in ('scenario','trajectory','slot'))
            baseline=(float(source['prs_energy_j']),float(source['prs_outage']),int(source['prs_count']))
            if key in originals:
                assert originals[key]==baseline
            originals[key]=baseline
            row=dict(direction=direction,scenario=key[0],trajectory=key[1],slot=key[2],
                baseline_count=int(source['prs_count']),alternative_count=int(source['reduced_count']),
                boundary_clipped=source['same_action']=='True',
                baseline_energy_j=float(source['prs_energy_j']),
                alternative_energy_j=float(source['reduced_energy_j']),
                baseline_outage=float(source['prs_outage']),alternative_outage=float(source['reduced_outage']))
            for metric in ('energy_delta_j','outage_delta','immediate_energy_delta_j','immediate_outage_delta'):
                row[metric]=float(source[metric])
            row['dominance_on_realized_trace']=source['dominance_on_realized_trace']=='True'
            assert abs(row['alternative_energy_j']-row['baseline_energy_j']-row['energy_delta_j'])<1e-9
            rows.append(row)
    assert len(rows)==len({(r['direction'],r['scenario'],r['trajectory'],r['slot']) for r in rows})==96
    _write_rows(output/'paired_interventions.csv',rows)
    report=dict(records=96,matched_baseline_starts=len(originals),baseline_identity='exact',
        scope='post-hoc local-action audit; no online-policy superiority claim',
        inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs.values()})
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    (output/'COMPLETE').write_text('Directional probes normalized; original records unchanged.\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
