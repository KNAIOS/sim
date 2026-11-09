"""Causal lower bound on remaining failures, not a future-risk upper bound."""
import numpy as np

from joint_power.model import allocate, effective_gains


def failure_floor(env, masks, interval=4):
    """Known current-slot failures plus later completely unavailable slots.

    All configured users require positive rate. CSI is used only on subarrays
    currently active after the command. Positive wake delay means newly
    requested subarrays cannot provide current-slot observations/service.
    No future channel, demand or realized outcome is accessed.
    """
    env.prepare()
    cfg = env.cfg
    if cfg.wake_delay_slots <= 0 or min(cfg.demand_levels_mbps) <= 0:
        raise ValueError('This bound requires positive wake delay and strictly positive user demand')
    masks = np.asarray(masks, bool)
    if masks.ndim != 2 or masks.shape[1] != cfg.subarrays:
        raise ValueError('Invalid masks')
    active = np.broadcast_to(env.hw.active, masks.shape).copy()
    countdown = np.broadcast_to(env.hw.countdown, masks.shape).copy()
    off = ~masks & (active | (countdown > 0))
    on = masks & ~active & (countdown == 0)
    active[off], countdown[off], countdown[on] = False, 0, cfg.wake_delay_slots
    # Scalar current-slot PHY exactly matches the executed allocator's order.
    first = []
    for available in active:
        channel = env.tracker.channel_estimate[:, np.repeat(available, cfg.antennas_per_subarray)]
        gains = effective_gains(cfg, channel@channel.conj().T)
        _, achieved, _ = allocate(cfg, gains, env.req)
        first.append(int(np.sum(achieved < env.req)))
    lower = np.array(first, dtype=int)
    for _ in range(1, min(interval, cfg.episode_steps-env.t)):
        ready = countdown == 1
        countdown[countdown > 0] -= 1
        active[ready] = True
        lower += cfg.users*(~active.any(1))
    return lower
