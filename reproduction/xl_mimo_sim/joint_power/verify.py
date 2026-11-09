"""Independent antenna-domain and hardware replay of saved joint-control data.

Does not call the new Environment, effective_gains, allocator or slot_energy.
It checks physical accounting, not whether the scientific hypothesis succeeded.
"""
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from future_vr.channel import generate_trajectory
from future_vr.config import make_config

ROOT=Path('results/joint_power_20260922_phase1')


def read(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return int(row['scenario']),int(row['trajectory']),row['method'],int(row['seed'])


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    hashes=json.loads((ROOT/'output_hashes.json').read_text())
    assert all(sha(p)==h for p,h in manifest['input_hashes'].items())
    assert all(sha(p)==h for p,h in hashes.items())
    cfg=make_config('validation')
    noise=10**((-174+10*np.log10(40e6)+7-30)/10)
    scenarios=((.5,.85),(2.,1.),(3.,1.15))
    traces={}
    for scenario,spec in enumerate(scenarios):
        for i in range(4):
            trace=generate_trajectory(cfg,2409220000+scenario*10000+i,spec[0])
            traces[scenario,i]=replace(trace,requirements_mbps=trace.requirements_mbps*spec[1])
    groups=defaultdict(list)
    for row in read(ROOT/'evaluation/slots.csv'):
        groups[key(row)].append(row)
    episodes={key(r):r for r in read(ROOT/'evaluation/episodes.csv')}
    assert len(groups)==len(episodes)==108
    errors=defaultdict(float)
    total_slots=0
    for identity,records in groups.items():
        records.sort(key=lambda r:int(r['slot']))
        assert [int(r['slot']) for r in records]==list(range(140))
        trace=traces[identity[0],identity[1]]
        active=np.ones(16,bool)
        countdown=np.zeros(16,int)
        energy=outage=0.
        for t,row in enumerate(records):
            observed=int(active.sum())
            bits=int(row['command_bits'])
            command=np.array([bool(bits & 1<<q) for q in range(16)])
            off=~command & (active|(countdown>0))
            on=command & ~active & (countdown==0)
            active[off]=False
            countdown[off]=0
            countdown[on]=4
            assert int(row['active_bits'])==sum(int(b)<<q for q,b in enumerate(active))
            assert int(row['observed_count'])==observed
            assert int(row['active_count'])==int(active.sum())
            assert int(row['waking_count'])==int((countdown>0).sum())
            power=np.array([float(row[f'power_{k}']) for k in range(3)])
            assert np.isfinite(power).all() and (power>=0).all() and power.sum()<=1.2+1e-10
            h=trace.channel[t][:,np.repeat(active,8)]
            if not h.shape[1]:
                rate=np.zeros(3)
            else:
                beam=h.conj().T @ np.linalg.inv(h@h.conj().T + 3*noise/1.2*np.eye(3))
                beam/=np.maximum(np.linalg.norm(beam,axis=0,keepdims=True),1e-15)
                # Direct received complex amplitudes, not the batched Gram formula.
                amplitude=h@(beam*np.sqrt(power)[None,:])
                received=np.abs(amplitude)**2
                desired=np.diag(received)
                interfering=received.copy()
                np.fill_diagonal(interfering,0.)
                rate=39.4*np.log2(1+desired/(noise+interfering.sum(1)))
            saved=np.array([float(row[f'rate_{k}']) for k in range(3)])
            errors['rate_mbps']=max(errors['rate_mbps'],float(np.max(abs(rate-saved))))
            expected_outage=float(np.mean(rate<trace.requirements_mbps[t]))
            assert expected_outage==float(row['outage']), (identity,t,rate,saved)
            na,nw=int(active.sum()),int((countdown>0).sum())
            processing=(8*na*8*9+8*27/3)*40e6/200/75e9
            circuit=.05*(8+2*na+1.2*nw+.05*(16-na-nw)+processing)
            pa=.05*power.sum()/.4
            switching=.1*on.sum()+.01*off.sum()
            pilot=.002*observed
            expected=circuit+pa+switching+pilot
            for name,value in [('energy_j',expected),('circuit_j',circuit),('pa_j',pa),
                               ('switching_j',switching),('pilot_j',pilot)]:
                errors[name]=max(errors[name],abs(float(value)-float(row[name])))
            energy+=expected
            outage+=expected_outage/140
            ready=countdown==1
            countdown[countdown>0]-=1
            active[ready]=True
            total_slots+=1
        errors['episode_energy_j']=max(errors['episode_energy_j'],abs(energy-float(episodes[identity]['energy_j'])))
        assert abs(outage-float(episodes[identity]['outage']))<1e-12
    assert total_slots==15120
    assert errors['rate_mbps']<1e-5
    assert max(value for name,value in errors.items() if name!='rate_mbps')<1e-8
    for row in read(ROOT/'evaluation/summary.csv'):
        group=[r for identity,r in episodes.items() if identity[0]==int(row['scenario']) and identity[2]==row['method']]
        for metric in ('energy_j','outage','circuit_j','pa_j','switching_j','pilot_j','mean_active','numerical_fallbacks'):
            assert abs(np.mean([float(r[metric]) for r in group])-float(row[metric]))<1e-10

    # Paired development intervals, not independent confirmation or equivalence.
    paired=[]
    rng=np.random.default_rng(2026092202)
    seed_draw=rng.integers(0,2,(4000,2))
    trace_draw=rng.integers(0,4,(4000,4))
    for scenario in range(3):
        for baseline in ('Joint full','Joint coarse','Prior','PRS power'):
            for metric,scale in [('energy_j',1.),('outage',100.)]:
                diff=np.array([[scale*(float(episodes[scenario,i,'Joint RL',seed][metric])-
                    float(episodes[scenario,i,baseline,0][metric])) for i in range(4)] for seed in (11,22)])
                bootstrap=diff[seed_draw[:,:,None],trace_draw[:,None,:]].mean((1,2))
                lo,hi=np.quantile(bootstrap,[.025,.975])
                paired.append(dict(scenario=scenario,baseline=baseline,metric=metric,
                    mean=float(diff.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    folder=ROOT/'verified'
    folder.mkdir(exist_ok=False)
    with (folder/'paired.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    report=dict(episodes=108,independent_environment_trajectories=12,slots=total_slots,
        errors=dict(errors),frozen_input_files=len(manifest['input_hashes']),verified_output_files=len(hashes),
        outcome=json.loads((ROOT/'evaluation/screen.json').read_text()),
        sources_preserved=True,scope='Physical replay and statistics, not scientific success')
    (folder/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
