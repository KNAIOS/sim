from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Config:
    mode: str = "smoke"
    carrier_ghz: float = 14.0
    bandwidth_mhz: float = 40.0
    noise_figure_db: float = 7.0
    subarrays: int = 10
    antennas_per_subarray: int = 8
    users: int = 2
    array_aperture_m: float = 10.0
    clusters_per_user: int = 4
    slot_s: float = 0.05
    episode_steps: int = 64
    horizon: int = 6
    visibility_discount: float = 0.93
    uncertainty_penalty: float = 0.65
    scan_period: int = 0  # retained for old config files; no sleeping-chain scans
    pilot_fraction: float = 0.015
    symbols_per_processing_block: float = 200.0
    wake_delay_slots: int = 3
    speeds_mps: tuple[float, ...] = (0.8, 2.5)
    wake_delays: tuple[int, ...] = (0, 3, 6)
    train_episodes: int = 36
    test_trajectories: int = 6
    controller_seeds: tuple[int, ...] = (11,)
    outage_target: float = 0.05
    demand_levels_mbps: tuple[float, ...] = (360.0, 460.0, 560.0)
    tx_power_w: float = 1.2
    pa_efficiency: float = 0.4
    fixed_power_w: float = 8.0
    active_subarray_w: float = 2.0
    waking_subarray_w: float = 1.2
    sleeping_subarray_w: float = 0.05
    wake_energy_j: float = 0.10
    sleep_energy_j: float = 0.01
    scan_energy_j_per_subarray: float = 0.002
    compute_efficiency_gflop_w: float = 75.0
    rl_hidden: int = 64
    rl_lr: float = 4e-4
    rl_gamma: float = 0.98
    replay_size: int = 20_000
    epsilon_start: float = 0.8
    epsilon_end: float = 0.04
    dual_initial: float = 8.0
    dual_lr: float = 5.0
    seed: int = 2026

    @property
    def antennas(self) -> int:
        return self.subarrays * self.antennas_per_subarray

    def with_delay(self, delay: int) -> "Config":
        return replace(self, wake_delay_slots=int(delay))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def make_config(mode: str) -> Config:
    if mode == "smoke":
        return Config(mode=mode)
    if mode == "validation":
        return Config(
            mode=mode,
            subarrays=16,
            users=3,
            array_aperture_m=12.0,
            episode_steps=140,
            horizon=8,
            wake_delay_slots=4,
            speeds_mps=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
            wake_delays=(0, 1, 2, 3, 4, 5, 6),
            train_episodes=110,
            test_trajectories=20,
            controller_seeds=(11, 22, 33),
        )
    return Config(
        mode=mode,
        subarrays=20,
        users=3,
        array_aperture_m=16.0,
        episode_steps=180,
        horizon=10,
        wake_delay_slots=4,
        speeds_mps=(0.3, 0.8, 1.5, 2.5, 4.0),
        wake_delays=(0, 1, 2, 4, 6, 8),
        train_episodes=180,
        test_trajectories=40,
        controller_seeds=(11, 22, 33, 44, 55),
    )
