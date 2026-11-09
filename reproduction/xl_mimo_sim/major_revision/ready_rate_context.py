"""Action-conditional rates after applying a command to the current RF state.

Uses only last-observed CSI and a copy of current hardware. No physical future
channel is available here. Positive-delay newly waking modules contribute no
current rate. This corrects a representation alias; it is not a safety proof.
"""
import copy
import numpy as np
from future_vr.controllers import select_top
from future_vr.system import XLSystem


def ready_rate_context(cfg, tracker, hardware, score, requirements):
    system = XLSystem(cfg)
    rates = []
    for count in range(cfg.users+1, cfg.subarrays+1):
        trial = copy.deepcopy(hardware)
        available, _, _ = trial.apply(select_top(score,count),1.)
        rates.append(system.achievable_rates(tracker.channel_estimate,available))
    return np.clip(np.asarray(rates)/np.maximum(requirements[None],1e-9),0,4).ravel().astype(np.float32)
