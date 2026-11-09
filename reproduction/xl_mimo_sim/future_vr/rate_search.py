"""Non-learning, delay-aware count search with RZF rate evaluation.

Uses the same active-only observations. Forecast amplitudes rescale the last
observed complex channel (phase is held, never taken from future simulation).
Every prefix of the RA-FVV ranking is tested with branch-readiness masks.
This is a transparent constructed baseline, not a reproduction of a paper.
"""
import numpy as np

from .controllers import select_top
from .system import XLSystem


def rate_search_command(cfg, tracker, hardware, requirements, score):
    gain, _, _ = tracker.predict(cfg.horizon)
    estimate = tracker.channel_estimate
    previous_power = np.mean(np.abs(estimate.reshape(
        cfg.users, cfg.subarrays, cfg.antennas_per_subarray)) ** 2, axis=2).T
    ratio = np.sqrt(10.0 ** (gain / 10.0) / np.maximum(previous_power[None], 1e-20))
    forecasts = estimate[None] * np.repeat(ratio.transpose(0, 2, 1), cfg.antennas_per_subarray, axis=2)
    ready = hardware.readiness()
    system = XLSystem(cfg)
    candidates = []
    for count in range(cfg.users + 1, cfg.subarrays + 1):
        command = select_top(score, count)
        rates = np.asarray([system.achievable_rates(forecasts[h-1], command & (ready <= h))
                            for h in range(1, cfg.horizon + 1)])
        outage = float(np.mean(rates < requirements))
        shortfall = float(np.mean(np.maximum(requirements - rates, 0) / requirements))
        candidates.append((outage > cfg.outage_target,
                           count if outage <= cfg.outage_target else shortfall,
                           count, command))
    return min(candidates, key=lambda item: item[:3])[3]
