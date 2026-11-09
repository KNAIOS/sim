"""Snapshot adapter of the ORIGINAL controlled channel, for offline diagnosis.

The step order and random draws deliberately match future_vr.channel exactly.
This exposes latent simulator state to an auditor, NEVER to an online policy.
It is not a calibrated standard channel or a new physical-model assumption.
"""
import copy
from dataclasses import fields
import numpy as np
from future_vr.channel import Trajectory,_reflect


class BranchingChannel:
    def __init__(self,cfg,seed,speed):
        self.cfg,self.speed,self.t=cfg,speed,0
        self.rng=np.random.default_rng(seed)
        rng=self.rng
        self.wavelength=299_792_458.0/(cfg.carrier_ghz*1e9)
        self.q_x=np.linspace(-cfg.array_aperture_m/2.,cfg.array_aperture_m/2.,cfg.subarrays)
        local=(np.arange(cfg.antennas_per_subarray)-(cfg.antennas_per_subarray-1)/2.)*self.wavelength/2.
        antenna_x=(self.q_x[:,None]+local[None,:]).reshape(-1)
        self.array_xy=np.column_stack((antenna_x,np.zeros(cfg.antennas)))
        self.position=np.column_stack((rng.uniform(-.35*cfg.array_aperture_m,.35*cfg.array_aperture_m,cfg.users),
                                       rng.uniform(8.,17.,cfg.users)))
        direction=rng.choice((-1.,1.),cfg.users)
        self.velocity=np.column_stack((direction*speed,rng.normal(0.,.08*speed,cfg.users)))
        self.scatterer=np.empty((cfg.users,cfg.clusters_per_user,2))
        self.cluster_width=rng.uniform(1.2,3.,(cfg.users,cfg.clusters_per_user))
        self.cluster_power=rng.exponential(1.,(cfg.users,cfg.clusters_per_user))
        self.cluster_phase=rng.uniform(0.,2.*np.pi,(cfg.users,cfg.clusters_per_user))
        self.active=rng.random((cfg.users,cfg.clusters_per_user))<.7
        for k in range(cfg.users):
            self.scatterer[k,:,0]=rng.uniform(-.55*cfg.array_aperture_m,.55*cfg.array_aperture_m,cfg.clusters_per_user)
            self.scatterer[k,:,1]=rng.uniform(2.,14.,cfg.clusters_per_user)
        self.demand_state=rng.integers(0,len(cfg.demand_levels_mbps),cfg.users)
        self.shadow_db=rng.normal(0.,2.5,cfg.users)
        self.rho_shadow=np.exp(-speed*cfg.slot_s/8.)

    def fork(self,seed=None):
        other=copy.deepcopy(self)
        if seed is not None:
            other.rng=np.random.default_rng(seed)
        return other

    def step(self):
        cfg,rng=self.cfg,self.rng
        if self.t and rng.random()<.035:
            changed=rng.integers(cfg.users)
            self.velocity[changed,0]*=-1.
        if self.t:
            changes=rng.random(cfg.users)<.04
            if changes.any():
                self.demand_state[changes]=np.clip(self.demand_state[changes]+rng.choice((-1,1),int(changes.sum())),
                                                  0,len(cfg.demand_levels_mbps)-1)
            death=self.active&(rng.random(self.active.shape)<.006+.002*self.speed)
            birth=(~self.active)&(rng.random(self.active.shape)<.004+.0015*self.speed)
            self.active[death]=False
            self.active[birth]=True
            self.cluster_phase+=rng.normal(0.,.05+.02*self.speed,self.cluster_phase.shape)
        requirements=np.asarray(cfg.demand_levels_mbps)[self.demand_state]
        xy,velocity=self.position.copy(),self.velocity.copy()
        self.shadow_db=self.rho_shadow*self.shadow_db+np.sqrt(max(1.-self.rho_shadow**2,0.))*rng.normal(0.,2.5,cfg.users)
        channel=np.empty((cfg.users,cfg.antennas),dtype=np.complex128)
        visibility=np.empty((cfg.subarrays,cfg.users))
        gain=np.empty_like(visibility)
        for k in range(cfg.users):
            los_width=2.+.06*self.position[k,1]
            los_vr_q=np.exp(-.5*((self.q_x-self.position[k,0])/los_width)**2)
            los_vr_m=np.repeat(los_vr_q,cfg.antennas_per_subarray)
            distance=np.linalg.norm(self.array_xy-self.position[k],axis=1)
            pl_db=32.4+20.*np.log10(np.maximum(distance,1.))+20.*np.log10(cfg.carrier_ghz)
            los_amp=np.sqrt(10.**(-(pl_db+self.shadow_db[k])/10.))
            h_k=np.sqrt(.45*los_vr_m)*los_amp*np.exp(-1j*2.*np.pi*distance/self.wavelength)
            v_q=.45*los_vr_q
            normalizer=max(float(self.cluster_power[k,self.active[k]].sum()),1e-9)
            for ell in range(cfg.clusters_per_user):
                if not self.active[k,ell]:
                    continue
                vr_q=np.exp(-.5*((self.q_x-self.scatterer[k,ell,0])/self.cluster_width[k,ell])**2)
                ue_leg=np.linalg.norm(self.position[k]-self.scatterer[k,ell])
                user_visibility=np.exp(-ue_leg/25.)
                vr_q*=user_visibility
                vr_m=np.repeat(vr_q,cfg.antennas_per_subarray)
                bs_leg=np.linalg.norm(self.array_xy-self.scatterer[k,ell],axis=1)
                total=bs_leg+ue_leg
                nlos_pl_db=32.4+22.*np.log10(np.maximum(total,1.))+20.*np.log10(cfg.carrier_ghz)+5.
                amp=np.sqrt(10.**(-(nlos_pl_db+self.shadow_db[k])/10.))
                weight=.55*self.cluster_power[k,ell]/normalizer
                h_k+=np.sqrt(weight*vr_m)*amp*np.exp(-1j*2.*np.pi*total/self.wavelength+1j*self.cluster_phase[k,ell])
                v_q+=weight*vr_q
            channel[k]=h_k
            v_q/=max(float(v_q.max()),1e-12)
            visibility[:,k]=np.clip(v_q,0.,1.)
            gain[:,k]=np.mean(np.abs(h_k.reshape(cfg.subarrays,cfg.antennas_per_subarray))**2,axis=1)
        self.position+=self.velocity*cfg.slot_s
        _reflect(self.position,self.velocity,.48*cfg.array_aperture_m)
        self.t+=1
        return channel,10.*np.log10(np.maximum(gain,1e-20)),visibility,xy,velocity,requirements

    def generate(self,steps,capture=()):
        samples,snapshots=[],{}
        for _ in range(steps):
            samples.append(self.step())
            if self.t-1 in capture:
                snapshots[self.t-1]=self.fork()
        return Trajectory(*[np.stack([sample[i] for sample in samples]) for i in range(6)]),snapshots


def replace_future(original,root_time,continuation,load):
    """Copy the trajectory, preserve all observations through the root time."""
    length=len(continuation.channel)
    if root_time+1+length>len(original.channel):
        raise ValueError('Continuation exceeds original evaluation window')
    arrays=[]
    for field in fields(Trajectory):
        current=getattr(original,field.name).copy()
        future=getattr(continuation,field.name)
        current[root_time+1:root_time+1+length]=future*load if field.name=='requirements_mbps' else future
        arrays.append(current)
    return Trajectory(*arrays)
