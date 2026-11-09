"""Summarize frozen diagnostics, without changing policies or selecting trials."""
import hashlib
import json
import csv
from pathlib import Path
import numpy as np
from future_vr.experiment import _write_rows
from major_revision.structural_value_audit import ROOT


def numeric_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def summarize():
    assert (ROOT/'COMPLETE').exists()
    sources=[ROOT/name for name in ('manifest.json','audit.json','analytic_energy.csv',
        'selected_error.csv','conditional_raw.csv','conditional_roots.csv')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for path,digest in manifest['hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    errors=numeric_rows(ROOT/'selected_error.csv')
    roots=numeric_rows(ROOT/'conditional_roots.csv')
    raw=numeric_rows(ROOT/'conditional_raw.csv')
    assert len(errors)==576 and len(roots)==96 and len(raw)==1536
    summary=[]
    for seed in (11,22):
        for scenario in range(3):
            e=[r for r in errors if r['seed']==seed and r['scenario']==scenario]
            group=[r for r in roots if r['model_seed']==seed and r['scenario']==scenario]
            assert len(e)==96 and len(group)==16
            for r in group:
                samples=[x['cost_difference'] for x in raw if x['root']==r['root'] and x['model_seed']==seed]
                assert len(samples)==16
                assert abs(np.mean(samples)-r['conditional_mean_cost'])<1e-12
                assert abs(np.std(samples,ddof=1)/4-r['mc_standard_error'])<1e-12
            changed=[r for r in group if r['action_count']!=r['reference_count']]
            summary.append(dict(seed=seed,scenario=scenario,all_error_roots=len(e),
                mean_energy_error=float(np.mean([r['energy_error'] for r in e])),
                mean_priced_outage_error=float(np.mean([r['priced_outage_error'] for r in e])),
                mean_total_error=float(np.mean([r['total_error'] for r in e])),
                conditional_roots=len(group),changed=len(changed),
                predicted_mean=float(np.mean([r['predicted_cost'] for r in group])),
                original_sample_mean=float(np.mean([r['original_sample_cost'] for r in group])),
                conditional_mean=float(np.mean([r['conditional_mean_cost'] for r in group])),
                changed_positive_mean=sum(r['conditional_mean_cost']>1e-6 for r in changed),
                changed_negative_mean=sum(r['conditional_mean_cost']< -1e-6 for r in changed),
                changed_all_draws_positive=sum(r['positive_samples']==16 for r in changed),
                changed_sign_disagreement=sum((r['original_sample_cost']>1e-6)!=(r['conditional_mean_cost']>1e-6) for r in changed),
                median_mc_se=float(np.median([r['mc_standard_error'] for r in group]))))
    output=ROOT/'summary'
    output.mkdir(exist_ok=False)
    _write_rows(output/'summary.csv',summary)
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (output/'audit.json').write_text(json.dumps(dict(input_hashes=hashes,
        grouped_future_recalculation=True,independent_base_roots=48,
        note='Descriptive latent-state diagnostic on viewed validation data, not a superiority test.'),indent=2))
    (output/'COMPLETE').write_text('All groups retained; no model or policy change.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    summarize()
