from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Trajectory:
    channel: np.ndarray  # [time, user, antenna], complex
    subarray_gain_db: np.ndarray  # [time, subarray, user]
    user_xy: np.ndarray  # [time, user, 2]
    visibility: np.ndarray  # [time, user, cluster-pool, subarray]
    active_clusters: np.ndarray  # [time, user], effective NLOS count
    rate_requirement_mbps: np.ndarray  # [time, user], slowly varying traffic demand


def _reflect(position: np.ndarray, velocity: np.ndarray) -> None:
    x_limits = (-20.0, 20.0)
    y_limits = (10.0, 55.0)
    for axis, limits in enumerate((x_limits, y_limits)):
        low, high = limits
        under = position[:, axis] < low
        over = position[:, axis] > high
        position[under, axis] = 2.0 * low - position[under, axis]
        position[over, axis] = 2.0 * high - position[over, axis]
        velocity[under | over, axis] *= -1.0


def _visibility_profiles(cfg: Config, rng: np.random.Generator, clusters: int | None = None) -> np.ndarray:
    """One-dimensional form of the cluster-level 3GPP SNS attenuation model."""
    q = cfg.subarrays
    centers = np.arange(q, dtype=float) + 0.5
    clusters = cfg.cluster_pool if clusters is None else clusters
    profiles = np.empty((cfg.users, clusters, q), dtype=float)
    relative_power_db = -rng.exponential(scale=7.0, size=(cfg.users, clusters))
    for k in range(cfg.users):
        for ell in range(clusters):
            # UMi parameters reported for the standardized stochastic SNS model.
            vp = 0.12 * np.exp(relative_power_db[k, ell] / 50.0) + 0.48
            vp = float(np.clip(vp + rng.normal(0.0, np.sqrt(0.001)), 0.25, 1.0))
            width = max(1.0, vp * q)
            center = rng.uniform(0.5, q - 0.5)
            left, right = center - width / 2.0, center + width / 2.0
            distance = np.maximum(np.maximum(left - centers, centers - right), 0.0)
            normalizer = max(q - width, 1.0)
            profiles[k, ell] = np.exp(-13.0 * distance / normalizer)
    return profiles


def _redraw_cluster(
    cfg: Config,
    rng: np.random.Generator,
    scatterers: np.ndarray,
    phases: np.ndarray,
    visibility: np.ndarray,
    powers: np.ndarray,
    user: int,
    slot: int,
) -> None:
    """Draw a new NLOS cluster; all parameters change together at a birth."""
    scatterers[user, slot] = (rng.uniform(-25.0, 25.0), rng.uniform(3.0, 50.0))
    phases[user, slot] = rng.uniform(0.0, 2.0 * np.pi)
    visibility[user, slot] = _visibility_profiles(cfg, rng, 1)[user, 0]
    powers[user, slot] = rng.exponential(1.0)


def generate_trajectory(cfg: Config, seed: int, speed_mps: float) -> Trajectory:
    rng = np.random.default_rng(seed)
    c = 299_792_458.0
    wavelength = c / (cfg.carrier_ghz * 1e9)
    element_x = (np.arange(cfg.antennas) - (cfg.antennas - 1) / 2.0) * wavelength / 2.0
    array_xy = np.column_stack((element_x, np.zeros(cfg.antennas)))

    initial = np.column_stack((rng.uniform(-15.0, 15.0, cfg.users), rng.uniform(15.0, 45.0, cfg.users)))
    heading = rng.uniform(0.0, 2.0 * np.pi, cfg.users)
    velocity = speed_mps * np.column_stack((np.cos(heading), np.sin(heading)))

    # NLOS clusters follow the distance-driven birth-death process in Wu et al.,
    # TWC 2015, (46)--(50).  A finite pool is only an implementation device;
    # it is deliberately larger than the stationary mean lambda_G/lambda_R=3.
    scatterers = np.empty((cfg.users, cfg.cluster_pool, 2), dtype=float)
    cluster_phase = np.empty((cfg.users, cfg.cluster_pool), dtype=float)
    visibility_q = _visibility_profiles(cfg, rng)
    cluster_power = rng.exponential(1.0, (cfg.users, cfg.cluster_pool))
    target = np.zeros((cfg.users, cfg.cluster_pool), dtype=bool)
    envelope = np.zeros((cfg.users, cfg.cluster_pool), dtype=float)
    initial_count = np.minimum(rng.poisson(cfg.cluster_generation_rate / cfg.cluster_recombination_rate, cfg.users), cfg.cluster_pool)
    for k in range(cfg.users):
        for ell in range(cfg.cluster_pool):
            _redraw_cluster(cfg, rng, scatterers, cluster_phase, visibility_q, cluster_power, k, ell)
        target[k, : initial_count[k]] = True
        envelope[k, : initial_count[k]] = 1.0

    h = np.empty((cfg.episode_steps, cfg.users, cfg.antennas), dtype=np.complex128)
    large_scale_power = np.empty((cfg.episode_steps, cfg.users, cfg.antennas), dtype=float)
    xy = np.empty((cfg.episode_steps, cfg.users, 2), dtype=float)
    visibility_history = np.empty((cfg.episode_steps, cfg.users, cfg.cluster_pool, cfg.subarrays), dtype=float)
    active_history = np.empty((cfg.episode_steps, cfg.users), dtype=float)
    requirements = np.empty((cfg.episode_steps, cfg.users), dtype=float)
    demand_levels = np.asarray((140.0, 180.0, 220.0))
    demand_state = rng.integers(0, len(demand_levels), cfg.users)
    shadow_db = rng.normal(0.0, 3.0, cfg.users)
    decorrelation_m = 10.0
    rho_shadow = np.exp(-max(speed_mps, 1e-3) * cfg.slot_s / decorrelation_m)
    position = initial.copy()

    for t in range(cfg.episode_steps):
        if t > 0:
            change = rng.random(cfg.users) < 0.05
            direction = rng.choice((-1, 1), cfg.users)
            demand_state = np.where(change, np.clip(demand_state + direction, 0, len(demand_levels) - 1), demand_state)
        requirements[t] = demand_levels[demand_state]
        if t > 0:
            fluctuation = (speed_mps + cfg.moving_cluster_fraction * cfg.cluster_speed_mps) * cfg.slot_s
            survival = np.exp(-cfg.cluster_recombination_rate * fluctuation / cfg.cluster_correlation_distance_m)
            deaths = target & (rng.random(target.shape) > survival)
            target[deaths] = False
            mean_births = (cfg.cluster_generation_rate / cfg.cluster_recombination_rate) * (1.0 - survival)
            for k in range(cfg.users):
                available = np.flatnonzero((~target[k]) & (envelope[k] <= 1e-12))
                births = min(int(rng.poisson(mean_births)), len(available))
                if births:
                    chosen = rng.choice(available, births, replace=False)
                    for ell in chosen:
                        _redraw_cluster(cfg, rng, scatterers, cluster_phase, visibility_q, cluster_power, k, int(ell))
                    target[k, chosen] = True
            step = 1.0 / max(cfg.cluster_fade_slots, 1)
            envelope = np.clip(envelope + np.where(target, step, -step), 0.0, 1.0)
        visibility_history[t] = visibility_q * envelope[:, :, None]
        active_history[t] = envelope.sum(axis=1)
        visibility_m = np.repeat(visibility_q * envelope[:, :, None], cfg.antennas_per_subarray, axis=2)
        xy[t] = position
        shadow_db = rho_shadow * shadow_db + np.sqrt(max(1.0 - rho_shadow**2, 0.0)) * rng.normal(0.0, 3.0, cfg.users)
        for k in range(cfg.users):
            direct_distance = np.linalg.norm(array_xy - position[k], axis=1)
            pl_db = 32.4 + 21.0 * np.log10(np.maximum(direct_distance, 1.0)) + 20.0 * np.log10(cfg.carrier_ghz)
            direct = np.sqrt(10.0 ** (-(pl_db + shadow_db[k]) / 10.0))
            los_weight = 0.4
            channel_k = np.sqrt(los_weight) * direct * np.exp(-1j * 2.0 * np.pi * direct_distance / wavelength)
            power_k = los_weight * direct**2
            normalizer = max(float(cluster_power[k].sum()), 1e-12)
            for ell in range(cfg.cluster_pool):
                bs_leg = np.linalg.norm(array_xy - scatterers[k, ell], axis=1)
                ue_leg = np.linalg.norm(position[k] - scatterers[k, ell])
                total_distance = bs_leg + ue_leg
                nlos_pl_db = 32.4 + 23.0 * np.log10(np.maximum(total_distance, 1.0)) + 20.0 * np.log10(cfg.carrier_ghz) + 6.0
                amplitude = np.sqrt(10.0 ** (-(nlos_pl_db + shadow_db[k]) / 10.0))
                channel_k += (
                    np.sqrt(0.6 * cluster_power[k, ell] / normalizer * visibility_m[k, ell])
                    * amplitude
                    * np.exp(-1j * 2.0 * np.pi * total_distance / wavelength + 1j * cluster_phase[k, ell])
                )
                power_k += 0.6 * cluster_power[k, ell] / normalizer * visibility_m[k, ell] * amplitude**2
            h[t, k] = channel_k
            large_scale_power[t, k] = power_k
        position += velocity * cfg.slot_s
        _reflect(position, velocity)

    reshaped_power = large_scale_power.reshape(cfg.episode_steps, cfg.users, cfg.subarrays, cfg.antennas_per_subarray)
    gain = np.mean(reshaped_power, axis=3).transpose(0, 2, 1)
    gain_db = 10.0 * np.log10(np.maximum(gain, 1e-20))
    return Trajectory(channel=h, subarray_gain_db=gain_db, user_xy=xy, visibility=visibility_history, active_clusters=active_history, rate_requirement_mbps=requirements)
