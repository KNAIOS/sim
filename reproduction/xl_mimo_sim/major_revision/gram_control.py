"""Causal per-subarray Gram interface for the unchanged equal-power RZF model.

This is exact algebra for the modeled channel estimate, not a new propagation
model, calibrated outage predictor, or proof of a Markov state for the real UE.
No function accepts an exogenous trajectory or future true CSI.
"""
import numpy as np
from future_vr.system import XLSystem


def subarray_grams(cfg, channel):
    blocks=np.asarray(channel).reshape(cfg.users,cfg.subarrays,cfg.antennas_per_subarray)
    return np.einsum('kqn,lqn->qkl',blocks,blocks.conj())


def rates_from_gram(cfg, grams, antenna_counts):
    """Batched RZF, including the original per-column precoder normalization."""
    grams=np.asarray(grams,dtype=np.complex128)
    noise=XLSystem(cfg).noise_w
    regularizer=cfg.users*noise/max(cfg.tx_power_w,1e-12)
    inverse=np.linalg.inv(grams+regularizer*np.eye(cfg.users))
    # W=H^H A, so ||W[:,k]||^2=(A^H G A)[k,k].
    squared_norm=np.diagonal(inverse.conj().swapaxes(-1,-2)@grams@inverse,
                             axis1=-2,axis2=-1).real
    norm=np.maximum(np.sqrt(np.maximum(squared_norm,0.)),1e-15)
    effective=(grams@inverse)/norm[...,None,:]
    power=np.abs(effective)**2
    diagonal=np.diagonal(power,axis1=-2,axis2=-1)
    per_user=cfg.tx_power_w/cfg.users
    interference=per_user*(power.sum(axis=-1)-diagonal)
    sinr=per_user*diagonal/np.maximum(interference+noise,1e-30)
    rates=cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+sinr)
    return np.where(np.asarray(antenna_counts)[...,None]<cfg.users,0.,rates)


def forecast_grams(cfg,tracker):
    grams=subarray_grams(cfg,tracker.channel_estimate)
    power=np.diagonal(grams,axis1=-2,axis2=-1).real/cfg.antennas_per_subarray
    gains,_,_=tracker.predict(cfg.horizon)
    scale=np.sqrt(10.**(gains/10.)/np.maximum(power[None],1e-20))
    return grams[None]*scale[..., :,None]*scale[...,None,:]


def prefix_forecast_rates(cfg,tracker,hardware,score):
    """Shape [count=1..Q, horizon=1..H, user], same masks as select_top."""
    order=np.argsort(score,kind='stable')[::-1]
    forecast=forecast_grams(cfg,tracker)
    ready=hardware.readiness()[None,:]<=np.arange(1,cfg.horizon+1)[:,None]
    ordered=(forecast*ready[...,None,None])[:,order]
    grams=np.cumsum(ordered,axis=1)
    antennas=np.cumsum(ready[:,order],axis=1)*cfg.antennas_per_subarray
    return rates_from_gram(cfg,grams,antennas).transpose(1,0,2)


def gram_belief_features(cfg,tracker,hardware,requirements,previous_requirements,age,last_outage):
    """362 features at Q=16,K=3. No clipping of gain or age.

    Per-module log powers and complex normalized cross-user correlations encode
    the Hermitian Gram matrices (up to floating-point precision and the 1e-30
    zero-power floor). Tracker gain/probability/uncertainty and velocity retain
    its current forecast state. Hardware/age/demand provide control context.
    This is a candidate representation, not a certified real-channel belief.
    """
    grams=subarray_grams(cfg,tracker.channel_estimate)
    diagonal=np.maximum(np.diagonal(grams,axis1=-2,axis2=-1).real,1e-30)
    noise_db=10*np.log10(XLSystem(cfg).noise_w)
    power=(10*np.log10(diagonal/cfg.antennas_per_subarray)-noise_db)/20.
    i,j=np.triu_indices(cfg.users,1)
    correlation=grams[:,i,j]/np.sqrt(diagonal[:,i]*diagonal[:,j])
    channel=np.concatenate((power,correlation.real,correlation.imag),axis=1)
    parts=[channel.ravel(),((tracker.gain_db-noise_db)/20.).ravel(),
           tracker.probability.ravel(),tracker.uncertainty.ravel(),
           hardware.active.astype(float),hardware.command.astype(float),
           hardware.countdown/max(cfg.horizon,1),np.asarray(age)/max(4*cfg.horizon,1),
           tracker.velocity,requirements/max(cfg.demand_levels_mbps),
           previous_requirements/max(cfg.demand_levels_mbps),np.array([last_outage])]
    return np.concatenate(parts).astype(np.float32)


def representation_size(cfg):
    return cfg.subarrays*cfg.users**2+3*cfg.subarrays*cfg.users+4*cfg.subarrays+3*cfg.users+1
