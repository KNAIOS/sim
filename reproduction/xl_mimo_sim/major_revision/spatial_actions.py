"""Causal same-count alternatives; no trajectory or future-channel interface."""
import numpy as np
from future_vr.system import XLSystem


def retained_mask(previous, score, count):
    """Minimum Hamming change at fixed count, resolving membership by score."""
    previous=np.asarray(previous,dtype=bool)
    score=np.asarray(score)
    if previous.shape!=score.shape or not 0<=count<=len(previous):
        raise ValueError('Matching one-dimensional mask/score and valid count required')
    order=np.argsort(-score,kind='stable')
    keep=order[previous[order]]
    add=order[~previous[order]]
    selected=np.zeros_like(previous)
    selected[np.concatenate((keep,add))[:count]]=True
    return selected


def causal_forecasts(cfg,tracker):
    gain,_,_=tracker.predict(cfg.horizon)
    estimate=tracker.channel_estimate
    power=np.mean(np.abs(estimate.reshape(cfg.users,cfg.subarrays,
                                         cfg.antennas_per_subarray))**2,axis=2).T
    ratio=np.sqrt(10.**(gain/10.)/np.maximum(power[None],1e-20))
    return estimate[None]*np.repeat(ratio.transpose(0,2,1),cfg.antennas_per_subarray,axis=2)


def predicted_metrics(cfg,hardware,requirements,forecasts,command):
    if np.any(np.asarray(requirements)<=0):
        raise ValueError('Positive prediction requirements required')
    ready=hardware.readiness()
    system=XLSystem(cfg)
    rates=np.asarray([system.achievable_rates(forecasts[h-1],command & (ready<=h))
                      for h in range(1,cfg.horizon+1)])
    newly_off=(~command)&(hardware.active|(hardware.countdown>0))
    newly_on=command&(~hardware.active)&(hardware.countdown==0)
    return dict(outage=float(np.mean(rates<requirements)),
        shortfall=float(np.mean(np.maximum(requirements-rates,0)/requirements)),
        transition_j=float(cfg.sleep_energy_j*newly_off.sum()+cfg.wake_energy_j*newly_on.sum()))


def exchange_mask(cfg,tracker,hardware,requirements,anchor):
    """Best causal one-exchange mask; the original anchor is always admissible."""
    forecasts=causal_forecasts(cfg,tracker)
    original=predicted_metrics(cfg,hardware,requirements,forecasts,anchor)
    key=lambda m,changed:(m['outage']>cfg.outage_target,m['shortfall'],m['transition_j'],changed)
    best,best_metrics=anchor.copy(),original
    best_key=key(original,False)
    for remove in np.flatnonzero(anchor):
        for add in np.flatnonzero(~anchor):
            candidate=anchor.copy()
            candidate[remove],candidate[add]=False,True
            metrics=predicted_metrics(cfg,hardware,requirements,forecasts,candidate)
            candidate_key=key(metrics,True)
            if candidate_key<best_key:
                best,best_metrics,best_key=candidate,metrics,candidate_key
    return best,original,best_metrics


def alternative_command(cfg,tracker,hardware,requirements,score,anchor,mode):
    if mode=='exchange':
        return exchange_mask(cfg,tracker,hardware,requirements,anchor)
    if mode!='retain':
        raise ValueError('Unknown spatial alternative')
    alternative=retained_mask(hardware.command,score,int(anchor.sum()))
    forecasts=causal_forecasts(cfg,tracker)
    return alternative,predicted_metrics(cfg,hardware,requirements,forecasts,anchor),\
        predicted_metrics(cfg,hardware,requirements,forecasts,alternative)
