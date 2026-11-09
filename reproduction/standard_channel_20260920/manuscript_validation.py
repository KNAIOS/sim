"""Frozen manuscript/candidate policies on independent standardized UMi drops.

This supplements rather than validates moving near-field visibility regions.
Both selected offered loads are retained, including overload/failure cases.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import numpy as np
import torch
from umi_channel import generate, FC

SIM=Path(__file__).resolve().parents[1]/'xl_mimo_sim'
sys.path.insert(0,str(SIM))
from future_vr.channel import Trajectory, generate_trajectory
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import _write_rows
from major_revision.pilot import episode
from major_revision.rate_context import candidate_rate_context
from major_revision.wideband import WidebandSystem

LOADS=(.5,1.)


def models(cfg):
    result=[]
    for seed in (11,22,33,44,55):
        path=SIM/f'results/revision_20260905/checkpoints/RA-FVV_{seed}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        controller=ConstrainedDQN(cfg,72,seed)
        controller.online.load_state_dict(payload['online'])
        result.append(('Manuscript RA-DQN',seed,controller,None,path,'Original-DQN'))
    for seed in (11,22):
        path=SIM/f'results/major_revision_20260920_rate_ablation/rate_seed{seed}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        controller=ConstrainedDQN(cfg,178,seed)
        controller.online.load_state_dict(payload['online'])
        result.append(('Candidate rate-DQN',seed,controller,candidate_rate_context,path,'Enriched-DQN'))
    return result


def summary(rows,out):
    rng=np.random.default_rng(967000000)
    means=[]
    for state in ('LOS','NLOS'):
        for load in LOADS:
            for method in sorted({r['method'] for r in rows}):
                group=[r for r in rows if r['state']==state and r['load']==load and r['method']==method]
                seeds=sorted({r['seed'] for r in group})
                ids=sorted({r['trajectory'] for r in group})
                lookup={(r['seed'],r['trajectory']):r for r in group}
                si=rng.integers(len(seeds),size=(4000,len(seeds)))
                ti=rng.integers(len(ids),size=(4000,len(ids)))
                row=dict(state=state,load=load,method=method,training_seeds=len(seeds),trajectories=len(ids))
                for metric in ('energy_j','outage'):
                    values=np.array([[lookup[s,i][metric] for i in ids] for s in seeds])
                    boot=values[si[:,:,None],ti[:,None,:]].mean((1,2))
                    row[metric]=float(values.mean())
                    row[metric+'_low']=float(np.quantile(boot,.025))
                    row[metric+'_high']=float(np.quantile(boot,.975))
                means.append(row)
    _write_rows(out/'summary.csv',means)


def run(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    # Horizontal column-centre span of the actual compact panel, not the old 12m aperture.
    cfg=replace(make_config('validation'),array_aperture_m=15*.5*299792458./FC)
    controllers=models(cfg)
    manifest=dict(channel_stream=965000000,demand_stream=966000000,loads=LOADS,
        trajectories_per_state=args.trajectories,frequency_samples=99,
        config=cfg.as_dict(),sionna=metadata.version('sionna-no-rt'),torch=torch.__version__,
        status='frozen policy evaluation on new independent standardized drops; no retuning on these outcomes',
        scenario='UMi TR38.901 V16.1, fixed topology, per-drop LSPs, 2m/s Doppler within CIR',
        limitations=['compact far-field planar array, not original 12m near-field model',
                     'fixed topology/LSPs over each 7s episode, not stateful large-scale mobility',
                     'common active centre-frequency CSI for decisions; ideal active per-bin CSI for physical RZF',
                     '33-to-297 quadrature pilot motivated 99 bins; this is not an OFDM waveform simulation',
                     'analytical RF/baseband power proxy, not measured power',
                     'candidate has only two training seeds; manuscript method has five',
                     'no continuous-time fading reconstruction or intra-slot reliability guarantee'],
        checkpoints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for *_,p,_ in controllers},
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('umi_channel.py'),SIM/'major_revision/pilot.py',
             SIM/'major_revision/rate_context.py',SIM/'major_revision/wideband.py']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for state_index,los in enumerate((True,False)):
        for i in range(args.trajectories):
            arrays,details=generate(965000000+state_index*10000+i,frequencies=99,los=los)
            h=arrays['channel']
            central=h[:,49]
            gain=10*np.log10(np.maximum((abs(central.reshape(140,3,16,8))**2).mean(3),1e-30)).transpose(0,2,1)
            requirements=generate_trajectory(make_config('validation'),966000000+state_index*10000+i,2.).requirements_mbps
            label='LOS' if los else 'NLOS'
            details['channel_sha256']=hashlib.sha256(h.tobytes()).hexdigest()
            details['demand_sha256']=hashlib.sha256(requirements.tobytes()).hexdigest()
            (out/f'{label}_{i}_metadata.json').write_text(json.dumps(details,indent=2))
            # One full trace per state for independent numerical audit; remaining
            # traces can be regenerated from pinned versions, seeds and hashes.
            if i==0:
                np.savez_compressed(out/f'{label}_example.npz',**arrays,requirements_mbps=requirements)
            for load in LOADS:
                tr=Trajectory(central,gain,np.ones_like(gain),
                    np.repeat(arrays['ut_positions_m'][None,:,:2],140,axis=0),
                    np.repeat(arrays['ut_velocity_mps'][None,:,:2],140,axis=0),requirements*load)
                specs=controllers+[(m,0,None,None,None,m) for m in ('PRS','Full array')]
                for method,seed,controller,features,_,engine_method in specs:
                    result=episode(cfg,tr,engine_method,controller,physical_channel=h,
                                   physical_system=WidebandSystem(cfg),state_features=features)
                    result['method']=method
                    rows.append(dict(state=label,trajectory=i,load=load,seed=seed,**result))
            _write_rows(out/'raw.csv',rows)
            print('standard validation',label,i+1,'/',args.trajectories,flush=True)
    expected=args.trajectories*2*len(LOADS)*9
    assert len(rows)==expected
    assert len({(r['state'],r['trajectory'],r['load'],r['seed'],r['method']) for r in rows})==expected
    for r in rows:
        assert abs(r['energy_j']-r['operating_j']-r['pilot_j']-r['switching_j'])<1e-7
        assert 0<=r['outage']<=1
    summary(rows,out)
    (out/'COMPLETE').write_text(f'{expected} unique evaluations; energy accounting passed.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--trajectories',type=int,default=20)
    run(p.parse_args())
