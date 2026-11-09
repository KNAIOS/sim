"""Candidate-rate context for a future controller ablation, not a new result.

Absolute subarray power does not identify multiuser channel correlation. These
features retain that information via causal, last-observed complex CSI. They
assume each candidate is fully ready, so they are NOT guaranteed future rates;
hardware countdowns must still be represented separately in the controller.
"""
import numpy as np
from future_vr.controllers import select_top
from future_vr.system import XLSystem


def candidate_rate_context(cfg,tracker,score,requirements):
    system=XLSystem(cfg)
    rates=np.array([system.achievable_rates(tracker.channel_estimate,select_top(score,count))
                    for count in range(cfg.users+1,cfg.subarrays+1)])
    return np.clip(rates/np.maximum(requirements[None],1e-9),0,4).ravel().astype(np.float32)
