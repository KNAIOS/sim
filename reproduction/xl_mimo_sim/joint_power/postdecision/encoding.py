"""Deterministic module-wise state AFTER each candidate command."""
import numpy as np

from joint_power.calibrated.control import decision as aggregate_decision


def hardware_features(hardware, masks, interval=4):
    cfg = hardware.cfg
    active = np.broadcast_to(hardware.active, masks.shape).copy()
    countdown = np.broadcast_to(hardware.countdown, masks.shape).copy()
    off = ~masks & (active | (countdown > 0))
    on = masks & ~active & (countdown == 0)
    active[off], countdown[off], countdown[on] = False, 0, cfg.wake_delay_slots
    if cfg.wake_delay_slots == 0:
        active[on] = True
    next_active = active | ((countdown > 0) & (countdown <= interval))
    next_countdown = np.maximum(countdown-interval, 0)
    scale = max(cfg.wake_delay_slots, 1)
    return np.stack((masks, active, countdown/scale, next_active,
                     next_countdown/scale, on, off), axis=-1).astype(np.float32)


def decision(env, candidates, predictor):
    state = aggregate_decision(candidates, predictor, env.cfg.episode_steps-env.t)
    post = hardware_features(env.hw, candidates.masks).reshape(len(candidates.masks), -1)
    state.x = np.concatenate((state.x, post), axis=-1)
    return state
