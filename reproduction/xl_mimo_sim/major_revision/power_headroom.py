"""Optional fixed-beam power diagnostic, not a change to the frozen simulator."""
import numpy as np

from future_vr.system import XLSystem


def beam_gains(cfg, observed_channel, available):
    """Use current available-branch CSI only, and exactly the original RZF beams."""
    idx = np.flatnonzero(np.repeat(available, cfg.antennas_per_subarray))
    if len(idx) < cfg.users:
        return np.zeros((cfg.users, cfg.users))
    h = np.asarray(observed_channel)[:, idx]
    noise = XLSystem(cfg).noise_w
    gram = h @ h.conj().T
    alpha = cfg.users * noise / cfg.tx_power_w
    w = h.conj().T @ np.linalg.inv(gram + alpha * np.eye(cfg.users))
    w /= np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-15)
    return np.abs(h @ w)**2


def rates(cfg, gain, power):
    desired = np.diag(gain) * power
    cross = gain.copy()
    np.fill_diagonal(cross, 0.)
    sinr = desired / np.maximum(cross @ power + XLSystem(cfg).noise_w, 1e-30)
    return cfg.bandwidth_mhz * (1-cfg.pilot_fraction) * np.log2(1+sinr)


def minimum_power(cfg, gain, targets):
    """Minimum sum power for fixed directions and feasible rate targets; no RL."""
    gamma = np.expm1(np.log(2.) * np.asarray(targets) /
                     (cfg.bandwidth_mhz * (1-cfg.pilot_fraction)))
    diagonal = np.diag(gain)
    if np.any((gamma > 0) & (diagonal <= 0)):
        raise ValueError('Positive target with zero direct gain')
    scale = np.divide(gamma, diagonal, out=np.zeros_like(gamma), where=diagonal > 0)
    f = scale[:, None] * gain
    np.fill_diagonal(f, 0.)
    if max(np.abs(np.linalg.eigvals(f))) >= 1:
        raise ValueError('Infeasible interference targets')
    u = scale * XLSystem(cfg).noise_w
    return np.linalg.solve(np.eye(cfg.users)-f, u)


def allocation(cfg, gain, requirements):
    original = np.full(cfg.users, cfg.tx_power_w / cfg.users)
    before = rates(cfg, gain, original)
    target = np.minimum(before, requirements * (1+1e-9))
    fallback = False
    try:
        p = minimum_power(cfg, gain, target)
        # Both endpoints are feasible for the fixed-beam linear inequalities.
        p = (1-1e-8)*p + 1e-8*original
        after = rates(cfg, gain, p)
        valid = (np.isfinite(p).all() and np.all(p >= 0) and
                 np.all(p <= original+1e-10) and p.sum() <= cfg.tx_power_w+1e-10 and
                 np.all(after >= np.minimum(before, requirements)-1e-8) and
                 np.array_equal(after < requirements, before < requirements))
        if not valid:
            raise ValueError('Numerical feasibility or service-identity failure')
    except (ValueError, np.linalg.LinAlgError):
        p, after, fallback = original.copy(), before.copy(), True
    return dict(power=p, before=before, after=after, fallback=fallback)
