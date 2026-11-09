from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    mode: str
    carrier_ghz: float = 14.0
    bandwidth_mhz: float = 20.0
    noise_figure_db: float = 7.0
    subarrays: int = 8
    antennas_per_subarray: int = 16
    users: int = 3
    clusters_per_user: int = 4
    cluster_pool: int = 8
    cluster_recombination_rate: float = 0.8
    cluster_generation_rate: float = 2.4
    cluster_correlation_distance_m: float = 10.0
    moving_cluster_fraction: float = 0.3
    cluster_speed_mps: float = 0.5
    cluster_fade_slots: int = 5
    slot_s: float = 0.02
    coherence_symbols: int = 200
    pilot_symbols: int = 3
    episode_steps: int = 120
    train_episodes: int = 120
    predictor_trajectories: int = 60
    predictor_epochs: int = 15
    test_trajectories: int = 20
    history: int = 8
    prediction_horizon: int = 4
    scan_period: int = 8
    speeds_mps: tuple[float, ...] = (0.5, 1.5, 3.0)
    switch_scales: tuple[float, ...] = (0.0, 1.0, 3.0)
    rate_requirement_mbps: float = 200.0
    outage_target: float = 0.05
    tx_power_w: float = 0.2
    pa_efficiency: float = 0.4
    fixed_power_w: float = 10.0
    local_oscillator_w: float = 0.2
    antenna_circuit_w: float = 0.2
    user_circuit_w: float = 0.2
    compute_efficiency_gflop_per_w: float = 75.0
    switch_energy_j_per_subarray: float = 0.02
    predictor_hidden: int = 48
    rl_hidden: int = 64
    rl_lr: float = 3e-4
    rl_gamma: float = 0.98
    rl_epsilon_start: float = 0.8
    rl_epsilon_end: float = 0.05
    rl_replay: int = 30_000
    dual_initial: float = 12.0
    dual_lr: float = 8.0
    dual_min: float = 4.0
    dual_max: float = 50.0
    seeds: tuple[int, ...] = (101, 202, 303)

    @property
    def antennas(self) -> int:
        return self.subarrays * self.antennas_per_subarray

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def make_config(mode: str) -> Config:
    if mode == "smoke":
        return Config(
            mode=mode,
            episode_steps=40,
            train_episodes=30,
            predictor_trajectories=12,
            predictor_epochs=4,
            test_trajectories=4,
            speeds_mps=(0.5, 3.0),
            switch_scales=(0.0, 1.0),
            seeds=(101,),
        )
    if mode == "validation":
        return Config(mode=mode)
    return Config(
        mode=mode,
        episode_steps=200,
        train_episodes=250,
        predictor_trajectories=120,
        predictor_epochs=30,
        test_trajectories=40,
        speeds_mps=(0.3, 0.8, 1.5, 2.5, 4.0),
        switch_scales=(0.0, 0.5, 1.0, 2.0, 4.0),
        seeds=(101, 202, 303, 404, 505),
    )
