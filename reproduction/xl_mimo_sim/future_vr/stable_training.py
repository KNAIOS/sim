from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .channel import generate_trajectory
from .config import Config
from .controllers import ConstrainedDQN
from .experiment import _write_rows, simulate_episode


def _validation_scenarios(cfg: Config) -> tuple[tuple[float, int], ...]:
    speeds = (float(cfg.speeds_mps[0]), float(cfg.speeds_mps[len(cfg.speeds_mps) // 2]),
              float(cfg.speeds_mps[-1]))
    delays = (int(cfg.wake_delays[0]), int(cfg.wake_delay_slots), int(cfg.wake_delays[-1]))
    return tuple(sorted({(speed, delay) for speed in speeds for delay in delays}))


def _validate(
    cfg: Config, controller: ConstrainedDQN, controller_seed: int,
    checkpoint_episode: int, trajectories_per_scenario: int,
) -> dict[str, float]:
    energy: list[float] = []
    outage: list[float] = []
    active: list[float] = []
    for scenario_index, (speed, delay) in enumerate(_validation_scenarios(cfg)):
        local_cfg = cfg.with_delay(delay)
        for trajectory_index in range(trajectories_per_scenario):
            # This stream is disjoint from both training and final test streams.
            seed = cfg.seed * 730_000 + scenario_index * 10_000 + trajectory_index
            trajectory = generate_trajectory(local_cfg, seed, speed)
            metric, _ = simulate_episode(
                local_cfg, trajectory, "RA-FVV", controller, 1.0, seed
            )
            energy.append(metric.energy_j)
            outage.append(metric.outage_probability)
            active.append(metric.active_ratio)
    return {
        "controller_seed": controller_seed,
        "checkpoint_episode": checkpoint_episode,
        "validation_samples": len(energy),
        "energy_j": float(np.mean(energy)),
        "outage_probability": float(np.mean(outage)),
        "active_ratio": float(np.mean(active)),
        "feasible": float(np.mean(outage) <= cfg.outage_target),
    }


def _selection_key(row: dict[str, float], target: float) -> tuple[float, float, float]:
    outage = row["outage_probability"]
    energy = row["energy_j"]
    if outage <= target:
        return 0.0, energy, outage
    return 1.0, outage, energy


def _clone_state(controller: ConstrainedDQN) -> dict[str, object]:
    return {
        "online": {key: value.detach().cpu().clone() for key, value in controller.online.state_dict().items()},
        "target": {key: value.detach().cpu().clone() for key, value in controller.target.state_dict().items()},
        "dual": float(controller.dual),
    }


def _train_seed(
    cfg: Config, seed: int, checkpoint: Path, dual_window: int,
    validation_interval: int, validation_trajectories: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    torch.set_num_threads(1)
    state_dim = 4 * cfg.subarrays + 2 * cfg.users + 2
    controller = ConstrainedDQN(cfg, state_dim, seed)
    rng = np.random.default_rng(seed + cfg.seed)
    constraint_window: list[float] = []
    history: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_state: dict[str, object] | None = None
    best_episode = -1

    for episode in range(cfg.train_episodes):
        delay = int(rng.choice(cfg.wake_delays))
        local_cfg = cfg.with_delay(delay)
        speed = float(rng.choice(cfg.speeds_mps))
        trajectory = generate_trajectory(local_cfg, seed * 100_000 + episode, speed)
        fraction = episode / max(cfg.train_episodes - 1, 1)
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * np.exp(-4.0 * fraction)
        metric, trace = simulate_episode(
            local_cfg, trajectory, "RA-FVV", controller, 1.0, seed,
            training=True, epsilon=epsilon,
        )
        constraint_window.append(metric.outage_probability)
        if len(constraint_window) == dual_window:
            # Average first to suppress single-trajectory noise, but preserve
            # the cumulative step of ``dual_window`` episodic updates.
            mean_violation = float(np.mean(constraint_window) - cfg.outage_target)
            controller.dual = float(np.clip(
                controller.dual + cfg.dual_lr * dual_window * mean_violation,
                0.0,
                80.0,
            ))
            constraint_window.clear()
        history.append({
            "controller_seed": seed, "episode": episode,
            "energy_j": metric.energy_j, "outage_probability": metric.outage_probability,
            "active_ratio": metric.active_ratio, "dual": controller.dual,
            "epsilon": epsilon,
            "loss": float(np.mean(trace["loss"])) if trace["loss"].size else 0.0,
        })

        due = (episode + 1) % validation_interval == 0 or episode + 1 == cfg.train_episodes
        if due:
            row = _validate(cfg, controller, seed, episode + 1, validation_trajectories)
            validation.append(row)
            key = _selection_key(row, cfg.outage_target)
            if key < best_key:
                best_key = key
                best_state = _clone_state(controller)
                best_episode = episode + 1

    if best_state is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    controller.online.load_state_dict(best_state["online"])
    controller.target.load_state_dict(best_state["target"])
    controller.dual = float(best_state["dual"])
    torch.save({
        "method": "RA-FVV", "seed": seed,
        "online": controller.online.state_dict(), "target": controller.target.state_dict(),
        "dual": controller.dual, "selected_episode": best_episode,
        "selection_rule": "feasible mean outage first, then minimum validation energy",
    }, checkpoint)
    return history, validation


def run_stable_training(
    cfg: Config, output: Path, workers: int = 5, dual_window: int = 10,
    validation_interval: int = 25, validation_trajectories: int = 4,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    jobs = [
        (seed, checkpoint_dir / f"ra_crl_seed{seed}.pt") for seed in cfg.controller_seeds
    ]
    history: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [
            executor.submit(
                _train_seed, cfg, seed, checkpoint, dual_window,
                validation_interval, validation_trajectories,
            )
            for seed, checkpoint in jobs
        ]
        for future in as_completed(futures):
            seed_history, seed_validation = future.result()
            history.extend(seed_history)
            validation.extend(seed_validation)
    history.sort(key=lambda row: (int(row["controller_seed"]), int(row["episode"])))
    validation.sort(key=lambda row: (int(row["controller_seed"]), int(row["checkpoint_episode"])))
    _write_rows(output / "training_history.csv", history)
    _write_rows(output / "validation_history.csv", validation)
    metadata = {
        "controller_seeds": list(cfg.controller_seeds),
        "training_episodes": cfg.train_episodes,
        "dual_update_window_episodes": dual_window,
        "dual_update_scaling": "window mean times window length (same cumulative step as episodic updates)",
        "validation_interval_episodes": validation_interval,
        "validation_trajectories_per_scenario": validation_trajectories,
        "validation_scenarios": _validation_scenarios(cfg),
        "selection_rule": "feasible mean outage first, then minimum validation energy",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
