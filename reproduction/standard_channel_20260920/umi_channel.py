"""Pinned TR 38.901 UMi supplementary channel, not a moving-VR XL model.

Fixed topology and large-scale parameters, continuous Doppler within one CIR
call. No re-sampling topology at every slot. Absolute pathloss is retained.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import numpy as np
import torch
from sionna.phy import config
from sionna.phy.channel.tr38901 import PanelArray, UMi

FC = 14e9
BANDWIDTH = 40e6


def model(seed, speed=2., batch=1, small_array=False, los=True):
    config.seed = int(seed)
    torch.set_num_threads(1)
    kwargs = dict(polarization='single', polarization_type='V',
                  antenna_pattern='omni', carrier_frequency=FC,
                  precision='double', device='cpu')
    bs = PanelArray(num_rows_per_panel=1 if small_array else 8,
                    num_cols_per_panel=1 if small_array else 16, **kwargs)
    ut = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, **kwargs)
    channel = UMi(FC, 'low', ut, bs, 'downlink', spec_version='16.1',
                  enable_pathloss=True, enable_shadow_fading=True,
                  enable_spatial_consistency=False, precision='double', device='cpu')
    # Three outdoor UEs; compact planar BS array has 16 contiguous 8-element columns.
    xyz = torch.tensor([[25.,-12.,1.5],[35.,0.,1.5],[28.,14.,1.5]],dtype=torch.float64)
    xyz = xyz[None].repeat(batch,1,1)
    velocity = torch.zeros_like(xyz)
    velocity[:,:,1] = speed
    channel.set_topology(ut_loc=xyz,
        bs_loc=torch.tensor([[[0.,0.,10.]]],dtype=torch.float64).repeat(batch,1,1),
        ut_orientations=torch.zeros_like(xyz),
        bs_orientations=torch.zeros((batch,1,3),dtype=torch.float64),
        ut_velocities=velocity, in_state=torch.zeros((batch,3),dtype=torch.bool),los=los)
    return channel, bs, xyz, velocity


def generate(seed, speed=2., steps=140, frequencies=33, los=True):
    ch, bs, xyz, velocity = model(seed,speed,los=los)
    a,tau=ch(num_time_samples=steps,sampling_frequency=20.)
    # [K,N,P,T] and [K,P], no per-realization normalization.
    a=a[0,:,0,0].detach().numpy()
    tau=tau[0,:,0].detach().numpy()
    f=((np.arange(frequencies)+.5)/frequencies-.5)*BANDWIDTH
    phase=np.exp(-2j*np.pi*tau[:,:,None]*f[None,None,:])
    h=np.einsum('knpt,kpf->tfkn',a,phase,optimize=True)
    assert h.shape==(steps,frequencies,3,128)
    assert np.isfinite(h).all()
    details=dict(seed=seed,speed_mps=speed,steps=steps,slot_s=.05,
                 carrier_hz=FC,bandwidth_hz=BANDWIDTH,spec_version='16.1',
                 scenario='UMi outdoor '+('LOS' if los else 'NLOS'),
                 topology='fixed; Doppler evolution only; no moving visibility regions',
                 antenna='8x16 single-V omnidirectional planar half-wavelength array',
                 subarrays='16 columns of 8 elements in native Sionna antenna order',
                 normalization=False,pathloss=True,shadow_fading=True,
                 h_shape=list(h.shape),frequency_samples=frequencies,
                 sionna=metadata.version('sionna-no-rt'),torch=torch.__version__,
                 pathloss_db=ch._scenario._pl_b.detach().numpy().tolist(),
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return dict(channel=h,frequencies_hz=f,delays_s=tau,
                antenna_positions_m=bs.ant_pos.detach().numpy(),
                ut_positions_m=xyz[0].numpy(),ut_velocity_mps=velocity[0].numpy()),details


def checks():
    results={}
    for los in (True,False):
        ch,_,xyz,_=model(941000001,small_array=True,los=los)
        distance=np.sqrt(np.sum((xyz[0].numpy()-np.array([0,0,10]))**2,axis=1))
        los_loss=32.4+21*np.log10(distance)+20*np.log10(FC/1e9)
        expected=los_loss if los else np.maximum(los_loss,35.3*np.log10(distance)+22.4+21.3*np.log10(FC/1e9))
        actual=ch._scenario._pl_b.detach().numpy()[0,0]
        error=float(np.max(abs(actual-expected)))
        assert error<1e-8
        results['LOS' if los else 'NLOS']=dict(pathloss_max_error_db=error)
    # Independent topology batches give independent LSP draws. Check one user
    # per batch, not correlated users within a drop, for distribution moments.
    ch,_,_,_=model(941000002,batch=1024,small_array=True)
    lsp=ch.sample_lsp()
    ds=np.log10(lsp.ds[:,0,0].numpy())
    sf=10*np.log10(lsp.sf[:,0,0].numpy())
    expected_ds=-.24*np.log10(1+FC/1e9)-7.14
    assert abs(ds.mean()-expected_ds)<.06
    assert abs(ds.std(ddof=1)-.38)<.06
    assert abs(sf.mean())<.5
    assert abs(sf.std(ddof=1)-4.)<.5
    results['lsp']=dict(independent_drops=1024,log10_ds_mean=float(ds.mean()),
                       target_log10_ds_mean=float(expected_ds),log10_ds_std=float(ds.std(ddof=1)),
                       target_log10_ds_std=.38,sf_mean_db=float(sf.mean()),
                       sf_std_db=float(sf.std(ddof=1)),target_sf_std_db=4.)
    x,_=generate(941000003,speed=0.,steps=4,frequencies=5)
    y,_=generate(941000003,speed=0.,steps=4,frequencies=5)
    np.testing.assert_array_equal(x['channel'],y['channel'])
    np.testing.assert_allclose(x['channel'][1:],np.repeat(x['channel'][:1],3,axis=0),rtol=1e-10,atol=1e-15)
    z,_=generate(941000004,speed=2.,steps=4,frequencies=5)
    change=float(np.linalg.norm(np.diff(z['channel'],axis=0))/np.linalg.norm(z['channel']))
    assert change>1e-3
    results.update(reproducibility='exact',zero_doppler='constant',
                   nonzero_doppler_relative_change=change,
                   scope='implementation checks, not complete 3GPP multi-cell calibration')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--check',action='store_true')
    p.add_argument('--seed',type=int,default=942000000)
    p.add_argument('--steps',type=int,default=140)
    p.add_argument('--frequencies',type=int,default=33)
    args=p.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    if args.check:
        result=checks()
        (out/'checks.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    else:
        arrays,details=generate(args.seed,steps=args.steps,frequencies=args.frequencies)
        np.savez_compressed(out/'channel.npz',**arrays)
        (out/'metadata.json').write_text(json.dumps(details,indent=2))
        print(json.dumps(details,indent=2))
