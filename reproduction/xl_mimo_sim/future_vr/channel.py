from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Trajectory:
    channel: np.ndarray  # [time,user,antenna], complex
    gain_db: np.ndarray  # [time,subarray,user]
    visibility: np.ndarray  # [time,subarray,user], [0,1]
    user_xy: np.ndarray  # [time,user,2]
    velocity_xy: np.ndarray  # [time,user,2]
    requirements_mbps: np.ndarray  # [time,user]


def _reflect(position: np.ndarray, velocity: np.ndarray, x_limit: float) -> None:
    for k in range(len(position)):
        if position[k, 0] < -x_limit:
            position[k, 0] = -2.0 * x_limit - position[k, 0]
            velocity[k, 0] *= -1.0
        elif position[k, 0] > x_limit:
            position[k, 0] = 2.0 * x_limit - position[k, 0]
            velocity[k, 0] *= -1.0
        if position[k, 1] < 7.0 or position[k, 1] > 20.0:
            position[k, 1] = np.clip(position[k, 1], 7.0, 20.0)
            velocity[k, 1] *= -1.0


def generate_trajectory(cfg: Config, seed: int, speed_mps: float) -> Trajectory:
    """Spatially consistent modular near-field channel with moving/multi-modal VRs."""
    rng = np.random.default_rng(seed)
    c = 299_792_458.0
    wavelength = c / (cfg.carrier_ghz * 1e9)
    q_x = np.linspace(-cfg.array_aperture_m / 2.0, cfg.array_aperture_m / 2.0, cfg.subarrays)
    local = (np.arange(cfg.antennas_per_subarray) - (cfg.antennas_per_subarray - 1) / 2.0) * wavelength / 2.0
    antenna_x = (q_x[:, None] + local[None, :]).reshape(-1)
    array_xy = np.column_stack((antenna_x, np.zeros(cfg.antennas)))

    position = np.column_stack(
        (
            rng.uniform(-0.35 * cfg.array_aperture_m, 0.35 * cfg.array_aperture_m, cfg.users),
            rng.uniform(8.0, 17.0, cfg.users),
        )
    )
    direction = rng.choice((-1.0, 1.0), cfg.users)
    velocity = np.column_stack((direction * speed_mps, rng.normal(0.0, 0.08 * speed_mps, cfg.users)))

    scatterer = np.empty((cfg.users, cfg.clusters_per_user, 2))
    cluster_width = rng.uniform(1.2, 3.0, (cfg.users, cfg.clusters_per_user))
    cluster_power = rng.exponential(1.0, (cfg.users, cfg.clusters_per_user))
    cluster_phase = rng.uniform(0.0, 2.0 * np.pi, (cfg.users, cfg.clusters_per_user))
    active = rng.random((cfg.users, cfg.clusters_per_user)) < 0.7
    for k in range(cfg.users):
        scatterer[k, :, 0] = rng.uniform(-0.55 * cfg.array_aperture_m, 0.55 * cfg.array_aperture_m, cfg.clusters_per_user)
        scatterer[k, :, 1] = rng.uniform(2.0, 14.0, cfg.clusters_per_user)

    channel = np.empty((cfg.episode_steps, cfg.users, cfg.antennas), dtype=np.complex128)
    gain = np.empty((cfg.episode_steps, cfg.subarrays, cfg.users))
    visibility = np.empty_like(gain)
    xy = np.empty((cfg.episode_steps, cfg.users, 2))
    velocity_history = np.empty_like(xy)
    requirements = np.empty((cfg.episode_steps, cfg.users))
    demand_state = rng.integers(0, len(cfg.demand_levels_mbps), cfg.users)
    shadow_db = rng.normal(0.0, 2.5, cfg.users)
    rho_shadow = np.exp(-speed_mps * cfg.slot_s / 8.0)

    for t in range(cfg.episode_steps):
        if t and rng.random() < 0.035:
            changed = rng.integers(cfg.users)
            velocity[changed, 0] *= -1.0
        if t:
            changes = rng.random(cfg.users) < 0.04
            if changes.any():
                demand_state[changes] = np.clip(
                    demand_state[changes] + rng.choice((-1, 1), int(changes.sum())),
                    0,
                    len(cfg.demand_levels_mbps) - 1,
                )
            death = active & (rng.random(active.shape) < 0.006 + 0.002 * speed_mps)
            birth = (~active) & (rng.random(active.shape) < 0.004 + 0.0015 * speed_mps)
            active[death] = False
            active[birth] = True
            cluster_phase += rng.normal(0.0, 0.05 + 0.02 * speed_mps, cluster_phase.shape)
        requirements[t] = np.asarray(cfg.demand_levels_mbps)[demand_state]
        xy[t] = position
        velocity_history[t] = velocity
        shadow_db = rho_shadow * shadow_db + np.sqrt(max(1.0 - rho_shadow**2, 0.0)) * rng.normal(0.0, 2.5, cfg.users)

        for k in range(cfg.users):
            # The LoS/local-scattering VR follows the projected UE position.
            los_width = 2.0 + 0.06 * position[k, 1]
            los_vr_q = np.exp(-0.5 * ((q_x - position[k, 0]) / los_width) ** 2)
            los_vr_m = np.repeat(los_vr_q, cfg.antennas_per_subarray)
            distance = np.linalg.norm(array_xy - position[k], axis=1)
            pl_db = 32.4 + 20.0 * np.log10(np.maximum(distance, 1.0)) + 20.0 * np.log10(cfg.carrier_ghz)
            los_amp = np.sqrt(10.0 ** (-(pl_db + shadow_db[k]) / 10.0))
            h_k = np.sqrt(0.45 * los_vr_m) * los_amp * np.exp(-1j * 2.0 * np.pi * distance / wavelength)
            v_q = 0.45 * los_vr_q

            normalizer = max(float(cluster_power[k, active[k]].sum()), 1e-9)
            for ell in range(cfg.clusters_per_user):
                if not active[k, ell]:
                    continue
                # Scatterer-centred VRs make the aggregate support multi-modal;
                # their UE-side weight evolves continuously with mobility.
                vr_q = np.exp(-0.5 * ((q_x - scatterer[k, ell, 0]) / cluster_width[k, ell]) ** 2)
                ue_leg = np.linalg.norm(position[k] - scatterer[k, ell])
                user_visibility = np.exp(-ue_leg / 25.0)
                vr_q *= user_visibility
                vr_m = np.repeat(vr_q, cfg.antennas_per_subarray)
                bs_leg = np.linalg.norm(array_xy - scatterer[k, ell], axis=1)
                total = bs_leg + ue_leg
                nlos_pl_db = 32.4 + 22.0 * np.log10(np.maximum(total, 1.0)) + 20.0 * np.log10(cfg.carrier_ghz) + 5.0
                amp = np.sqrt(10.0 ** (-(nlos_pl_db + shadow_db[k]) / 10.0))
                weight = 0.55 * cluster_power[k, ell] / normalizer
                h_k += np.sqrt(weight * vr_m) * amp * np.exp(
                    -1j * 2.0 * np.pi * total / wavelength + 1j * cluster_phase[k, ell]
                )
                v_q += weight * vr_q
            channel[t, k] = h_k
            v_q /= max(float(v_q.max()), 1e-12)
            visibility[t, :, k] = np.clip(v_q, 0.0, 1.0)
            gain[t, :, k] = np.mean(np.abs(h_k.reshape(cfg.subarrays, cfg.antennas_per_subarray)) ** 2, axis=1)

        position += velocity * cfg.slot_s
        _reflect(position, velocity, 0.48 * cfg.array_aperture_m)

    gain_db = 10.0 * np.log10(np.maximum(gain, 1e-20))
    return Trajectory(channel, gain_db, visibility, xy, velocity_history, requirements)
