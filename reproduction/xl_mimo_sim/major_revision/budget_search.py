"""Count-stride search with exact batched RZF, not artificial observation noise.

Original sources are untouched. Full stride-one search has the original PRS
candidate set/objective. All variants use the same causal forecast and ranking.
"""
import numpy as np

from future_vr.controllers import build_state, select_top
from future_vr.experiment import _prediction
from future_vr.system import XLSystem

STRIDES = (1, 2, 3, 4, 6, 12)


def stride_counts(minimum, maximum, stride):
    if not 1 <= minimum <= maximum or stride < 1:
        raise ValueError('Invalid count domain or stride')
    return np.unique([*range(minimum, maximum+1, stride), maximum]).astype(int)


def local_counts(minimum, maximum, current, budget):
    """Same count budget: endpoints plus nearest neighbors of previous command."""
    if not minimum <= current <= maximum or not 2 <= budget <= maximum-minimum+1:
        raise ValueError('Invalid previous count or budget')
    order = [minimum, maximum, current]
    for radius in range(1, maximum-minimum+1):
        order.extend((current-radius, current+radius))
    selected = []
    for count in order:
        if minimum <= count <= maximum and count not in selected:
            selected.append(count)
        if len(selected) == budget:
            break
    return np.array(sorted(selected), dtype=int)


def forecasts(cfg, tracker):
    gain, _, _ = tracker.predict(cfg.horizon)
    estimate = tracker.channel_estimate
    previous_power = np.mean(np.abs(estimate.reshape(
        cfg.users, cfg.subarrays, cfg.antennas_per_subarray))**2, axis=2).T
    ratio = np.sqrt(10.**(gain/10.)/np.maximum(previous_power[None], 1e-20))
    return estimate[None]*np.repeat(ratio.transpose(0,2,1), cfg.antennas_per_subarray, axis=2)


def predicted_rates(cfg, tracker, hardware, score, counts):
    """Exact Gram-domain RZF for each selected prefix and all H forecast slots."""
    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1 or not len(counts) or np.any(counts < 1) or np.any(counts > cfg.subarrays):
        raise ValueError('Invalid candidate counts')
    channel = forecasts(cfg, tracker)
    blocks = channel.reshape(cfg.horizon, cfg.users, cfg.subarrays,
                             cfg.antennas_per_subarray).transpose(0,2,1,3)
    module_gram = blocks @ blocks.conj().swapaxes(-2,-1)
    order = np.argsort(score, kind='stable')[::-1]
    ready = hardware.readiness()
    usable = ready[None,order] <= np.arange(1,cfg.horizon+1)[:,None]
    prefix_gram = np.cumsum(module_gram[:,order]*usable[:,:,None,None], axis=1)
    gram = prefix_gram[:,counts-1].transpose(1,0,2,3)
    noise = XLSystem(cfg).noise_w
    inverse = np.linalg.inv(gram + cfg.users*noise/cfg.tx_power_w*np.eye(cfg.users))
    norm_sq = np.diagonal(inverse.conj().swapaxes(-2,-1) @ gram @ inverse,
                          axis1=-2, axis2=-1).real
    norms = np.maximum(np.sqrt(np.maximum(norm_sq,0.)),1e-15)
    effective = (gram @ inverse)/norms[...,None,:]
    powers = np.abs(effective)**2
    diagonal = np.diagonal(powers,axis1=-2,axis2=-1)
    desired = cfg.tx_power_w/cfg.users*diagonal
    interference = cfg.tx_power_w/cfg.users*(powers.sum(-1)-diagonal)
    sinr = desired/np.maximum(interference+noise,1e-30)
    rates = cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+sinr)
    enough = np.cumsum(usable,axis=1)[:,counts-1].T*cfg.antennas_per_subarray >= cfg.users
    return np.where(enough[...,None],rates,0.)


def search(cfg, tracker, hardware, requirements, family='uniform', stride=1):
    score, _ = _prediction(cfg,tracker,hardware,requirements)
    counts = stride_counts(cfg.users+1,cfg.subarrays,stride)
    if family == 'local':
        counts = local_counts(cfg.users+1,cfg.subarrays,int(hardware.command.sum()),len(counts))
    elif family != 'uniform':
        raise ValueError('Unknown count-selection family')
    rates = predicted_rates(cfg,tracker,hardware,score,counts)
    outage = (rates < requirements).mean((1,2))
    shortfall = (np.maximum(requirements-rates,0.)/requirements).mean((1,2))
    best = min(range(len(counts)), key=lambda i: (
        outage[i] > cfg.outage_target,
        counts[i] if outage[i] <= cfg.outage_target else shortfall[i],counts[i]))
    return select_top(score,int(counts[best]))


def learned(cfg, tracker, hardware, requirements, last_outage, controller):
    # Same input/output as original RA-FVV; remove duplicated score construction.
    score,_ = _prediction(cfg,tracker,hardware,requirements)
    observation = build_state(cfg,score,hardware.active,hardware.command,
        hardware.readiness(),requirements,tracker.velocity,last_outage)
    _,count = controller.choose(observation,0.)
    return select_top(score,count)
