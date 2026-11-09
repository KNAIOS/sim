"""Flat-PSD user-power allocation for frequency-selective fixed RZF beams.

Preparation for standardized-channel transfer, not used by phase3 training.
Power per user is TOTAL across bandwidth; effective gains/noise use the same
full-band normalization as the existing per-bin RZF implementation. Rates are
frequency averages, NOT the rate computed from an average channel matrix.
"""
import numpy as np

from future_vr.system import XLSystem


def rates(cfg, gains, power):
    """gains has shape (frequency, receiver, stream); power has shape(users,)."""
    gains, power = np.asarray(gains), np.asarray(power)
    desired = np.diagonal(gains, axis1=-2, axis2=-1)*power
    interference = np.einsum('fkj,j->fk', gains*(1-np.eye(cfg.users)), power)
    per_bin = np.log2(1+desired/(XLSystem(cfg).noise_w+interference))
    return cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*per_bin.mean(0)


def required(cfg, gains, targets, *, iterations=200, bisections=42):
    """Monotone minimum-required-power iteration with explicit failure status.

    With positive target/noise and at least one nonzero direct gain per user,
    each user's minimum power given interference is positive, monotone and
    strictly subhomogeneous. An exact fixed point, when feasible, is the
    componentwise minimum feasible vector. Numerical iteration limits are
    reported as unresolved, never converted into a mathematical infeasibility.
    """
    gains, targets = np.asarray(gains, float), np.asarray(targets, float)
    if gains.ndim != 3 or gains.shape[1:] != (cfg.users, cfg.users):
        raise ValueError('Expected frequency-by-user-by-stream gains')
    if targets.shape != (cfg.users,) or not np.isfinite(gains).all() or (gains < 0).any():
        raise ValueError('Invalid gains or target shape')
    if not np.isfinite(targets).all() or (targets < 0).any() or iterations < 1:
        raise ValueError('Invalid targets/iteration budget')
    diag = np.diagonal(gains, axis1=-2, axis2=-1)
    cross = gains*(1-np.eye(cfg.users))
    factor = cfg.bandwidth_mhz*(1-cfg.pilot_fraction)
    power = np.zeros(cfg.users)
    if not targets.any():
        return power, 'feasible', 0
    for iteration in range(1, iterations+1):
        interference = XLSystem(cfg).noise_w+np.einsum('fkj,j->fk', cross, power)
        def own_rates(candidate):
            return factor*np.log2(1+diag*candidate/interference).mean(0)
        if np.any(own_rates(np.full(cfg.users, cfg.tx_power_w)) < targets-1e-10):
            return power, 'infeasible_cap', iteration
        lower, upper = np.zeros(cfg.users), np.full(cfg.users, cfg.tx_power_w)
        for _ in range(bisections):
            middle = (lower+upper)/2
            insufficient = own_rates(middle) < targets
            lower = np.where(insufficient, middle, lower)
            upper = np.where(insufficient, upper, middle)
        following = np.where(targets > 0, upper, 0.)
        if following.sum() > cfg.tx_power_w+1e-9:
            return following, 'infeasible_cap', iteration
        if (np.max(abs(following-power)) <= 1e-9 and
                np.all(rates(cfg, gains, following) >= targets-1e-7)):
            return following, 'feasible', iteration
        power = following
    return power, 'iteration_limit', iterations


def allocate(cfg, gains, targets):
    """Shared total cap and service-preserving fallback; no rate failures hidden."""
    targets = np.asarray(targets, float)
    equal = np.full(cfg.users, cfg.tx_power_w/cfg.users)
    before = rates(cfg, gains, equal)
    if not np.any(gains):
        return np.zeros(cfg.users), np.zeros(cfg.users), dict(mode='zero_channel', iterations=0)
    power, status, count = required(cfg, gains, targets*(1+1e-8))
    mode = 'full_targets'
    if status != 'feasible' or power.sum() > cfg.tx_power_w:
        safe_target = np.minimum(before, targets*(1+1e-8))
        power, fallback_status, fallback_count = required(cfg, gains, safe_target)
        count += fallback_count
        mode = 'preserve_service'
        if fallback_status != 'feasible' or power.sum() > cfg.tx_power_w:
            power, mode = equal, 'equal_fallback'
    after = rates(cfg, gains, power)
    # Direct final checks protect service against numerical root/fixed-point error.
    if (not np.isfinite(after).all() or np.any(after < np.minimum(before, targets)-1e-7)
            or (power < 0).any() or power.sum() > cfg.tx_power_w+1e-10):
        power, after, mode = equal, before, 'equal_numerical_fallback'
    return power, after, dict(mode=mode, iterations=count, full_target_status=status)
