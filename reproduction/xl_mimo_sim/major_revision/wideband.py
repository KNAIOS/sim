"""Frequency-quadrature rate evaluation with equal power spectral density.

Each bin uses P/F and noise/F, so its SINR equals the full-band-normalized
formula below; B/F rates sum to B times average spectral efficiency.
The original analytical baseband-power proxy is retained, not measured power.
"""
import numpy as np
from future_vr.system import XLSystem


class WidebandSystem(XLSystem):
    def achievable_rates(self, channel, available):
        if channel.ndim == 2:
            return super().achievable_rates(channel,available)
        if channel.ndim != 3:
            raise ValueError('Expected [frequency,user,antenna]')
        indices=np.flatnonzero(np.repeat(available,self.cfg.antennas_per_subarray))
        if len(indices)<self.cfg.users:
            return np.zeros(self.cfg.users)
        h=channel[:,:,indices]
        ht=h.conj().swapaxes(-1,-2)
        gram=h@ht
        regularizer=self.cfg.users*self.noise_w/self.cfg.tx_power_w
        w=ht@np.linalg.inv(gram+regularizer*np.eye(self.cfg.users))
        w/=np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-15)
        power=abs(h@w)**2
        desired=np.diagonal(power,axis1=1,axis2=2)
        interference=power.sum(axis=2)-desired
        per_user=self.cfg.tx_power_w/self.cfg.users
        sinr=per_user*desired/np.maximum(per_user*interference+self.noise_w,1e-30)
        return self.cfg.bandwidth_mhz*(1-self.cfg.pilot_fraction)*np.log2(1+sinr).mean(axis=0)
