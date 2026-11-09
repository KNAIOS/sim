"""Post-study capacity sanity check on ALL fixed validation/development traces.

No controller is tuned, retrained or filtered. Cauchy--Schwarz supplies an
interference-free per-user upper bound, not an achievable joint rate vector.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from future_vr.config import make_config
from future_vr.experiment import _write_rows
from future_vr.system import XLSystem
from major_revision.pilot import SCENARIOS,trajectory
from major_revision.ready_report import read
from major_revision.full_domain_study import ROOT


def rate_upper_bound(cfg,channel,power_per_user):
    noise=XLSystem(cfg).noise_w
    channel_norm_sq=np.sum(np.abs(channel)**2,axis=-1)
    return cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+power_per_user*channel_norm_sq/noise)


def main():
    assert (ROOT/'COMPLETE').exists(),'Freeze and audit the candidate outcomes first'
    out=ROOT/'feasibility_audit'
    out.mkdir(exist_ok=False)
    cfg=make_config('validation')
    system=XLSystem(cfg)
    reference={(int(r['scenario']),int(r['trajectory'])):float(r['outage'])
        for r in read(ROOT/'baselines_v2/development_raw.csv') if r['method']=='full'}
    files=[Path(__file__),Path(__file__).with_name('pilot.py'),*Path('future_vr').glob('*.py')]
    manifest=dict(config=cfg.as_dict(),post_hoc=True,
        scope='all predeclared validation and development traces; no filtering or policy changes',
        interpretation='lower bound on unavoidable user-slot outage; upper-rate vectors need not be jointly achievable',
        derivation='|h_A w|^2 <= ||h_A||^2 <= ||h_full||^2 and interference >= 0',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for split,stream,n in (('validation',982000000,12),('development',983000000,16)):
        for j,sc in enumerate(SCENARIOS):
            for i in range(n):
                tr=trajectory(cfg,stream+j*10000+i,sc)
                rates=np.array([system.achievable_rates(h,np.ones(cfg.subarrays,bool)) for h in tr.channel])
                equal=rate_upper_bound(cfg,tr.channel,cfg.tx_power_w/cfg.users)
                all_power=rate_upper_bound(cfg,tr.channel,cfg.tx_power_w)
                assert np.all(rates <= equal+1e-6)
                outage=float(np.mean(rates<tr.requirements_mbps))
                if split=='development':
                    assert abs(outage-reference[j,i])<1e-12
                rows.append(dict(split=split,scenario=j,trajectory=i,
                    full_rzf_outage=outage,
                    equal_power_unavoidable=float(np.mean(equal<tr.requirements_mbps)),
                    total_power_unavoidable=float(np.mean(all_power<tr.requirements_mbps))))
    summary=[]
    for split in ('validation','development'):
        for j in range(3):
            group=[r for r in rows if r['split']==split and r['scenario']==j]
            summary.append(dict(split=split,scenario=j,trajectories=len(group),
                **{k:float(np.mean([r[k] for r in group])) for k in
                   ('full_rzf_outage','equal_power_unavoidable','total_power_unavoidable')}))
    _write_rows(out/'raw.csv',rows)
    _write_rows(out/'summary.csv',summary)
    assert len(rows)==84
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_hashes'][str(p)] for p in files)
    (out/'COMPLETE').write_text('All 84 fixed traces checked without changing any policy or dataset.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
