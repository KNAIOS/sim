from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .channel import generate_trajectory
from .config import Config
from .controllers import ConstrainedDQN
from .experiment import _write_rows, simulate_episode
from .learned_study import _load_controller
from .matched_policy_study import FIELDS, _hierarchical_ci
from .stable_training import _clone_state, _validate


LOAD_SCALES = (0.85, 1.0, 1.15)
ABLATION_METHODS = ("RA-CRL", "Energy-only DQN")
_CONTROLLERS: dict[str, list[tuple[int, ConstrainedDQN]]] = {}


def _validate_load_aware(
    cfg: Config, controller: ConstrainedDQN, controller_seed: int,
    checkpoint_episode: int, trajectories_per_scenario: int,
) -> dict[str, float]:
    energy: list[float] = []
    outage: list[float] = []
    active: list[float] = []
    speeds = (float(cfg.speeds_mps[0]), 2.0, float(cfg.speeds_mps[-1]))
    scenario_index = 0
    for speed in speeds:
        for load_scale in LOAD_SCALES:
            local_cfg = cfg.with_delay(4)
            for trajectory_index in range(trajectories_per_scenario):
                seed = cfg.seed * 830_000 + scenario_index * 10_000 + trajectory_index
                trajectory = generate_trajectory(local_cfg, seed, speed)
                trajectory = replace(
                    trajectory,
                    requirements_mbps=trajectory.requirements_mbps * load_scale,
                )
                metric, _ = simulate_episode(
                    local_cfg, trajectory, "RA-FVV", controller, 1.0, seed
                )
                energy.append(metric.energy_j)
                outage.append(metric.outage_probability)
                active.append(metric.active_ratio)
            scenario_index += 1
    return {
        "controller_seed": controller_seed,
        "checkpoint_episode": checkpoint_episode,
        "validation_samples": len(energy),
        "energy_j": float(np.mean(energy)),
        "outage_probability": float(np.mean(outage)),
        "active_ratio": float(np.mean(active)),
        "feasible": float(np.mean(outage) <= cfg.outage_target),
    }


def _train_load_aware_seed(
    cfg: Config, seed: int, checkpoint: Path, constrained: bool,
    dual_window: int, validation_interval: int, validation_trajectories: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    torch.set_num_threads(1)
    state_dim = 4 * cfg.subarrays + 2 * cfg.users + 2
    controller = ConstrainedDQN(cfg, state_dim, seed)
    if not constrained:
        controller.dual = 0.0
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
        load_scale = float(rng.choice(LOAD_SCALES))
        trajectory = generate_trajectory(local_cfg, seed * 100_000 + episode, speed)
        trajectory = replace(
            trajectory,
            requirements_mbps=trajectory.requirements_mbps * load_scale,
        )
        fraction = episode / max(cfg.train_episodes - 1, 1)
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * np.exp(-4.0 * fraction)
        metric, trace = simulate_episode(
            local_cfg, trajectory, "RA-FVV", controller, 1.0, seed,
            training=True, epsilon=epsilon,
        )
        if constrained:
            constraint_window.append(metric.outage_probability)
            if len(constraint_window) == dual_window:
                violation = float(np.mean(constraint_window) - cfg.outage_target)
                controller.dual = float(np.clip(
                    controller.dual + cfg.dual_lr * dual_window * violation, 0.0, 80.0
                ))
                constraint_window.clear()
        else:
            controller.dual = 0.0
        history.append({
            "controller_seed": seed, "episode": episode, "load_scale": load_scale,
            "energy_j": metric.energy_j, "outage_probability": metric.outage_probability,
            "active_ratio": metric.active_ratio, "dual": controller.dual,
            "epsilon": epsilon,
            "loss": float(np.mean(trace["loss"])) if trace["loss"].size else 0.0,
        })
        due = (episode + 1) % validation_interval == 0 or episode + 1 == cfg.train_episodes
        if due:
            row = _validate_load_aware(
                cfg, controller, seed, episode + 1, validation_trajectories
            )
            row["objective"] = "constrained" if constrained else "energy_only"
            validation.append(row)
            if constrained:
                key = (
                    0.0 if row["outage_probability"] <= cfg.outage_target else 1.0,
                    row["energy_j"] if row["outage_probability"] <= cfg.outage_target
                    else row["outage_probability"],
                    row["outage_probability"] if row["outage_probability"] <= cfg.outage_target
                    else row["energy_j"],
                )
            else:
                key = (0.0, row["energy_j"], row["outage_probability"])
            if key < best_key:
                best_key = key
                best_state = _clone_state(controller)
                best_episode = episode + 1

    if best_state is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    controller.online.load_state_dict(best_state["online"])
    controller.target.load_state_dict(best_state["target"])
    controller.dual = float(best_state["dual"] if constrained else 0.0)
    label = "ra_crl" if constrained else "energy_only"
    torch.save({
        "method": "RA-FVV", "seed": seed,
        "online": controller.online.state_dict(), "target": controller.target.state_dict(),
        "dual": controller.dual, "selected_episode": best_episode,
        "selection_rule": (
            "feasible mean outage first, then minimum validation energy"
            if constrained else "minimum validation energy without outage price"
        ),
        "training_load_scales": LOAD_SCALES,
    }, checkpoint / f"{label}_seed{seed}.pt")
    return history, validation


def train_load_aware_pair(
    cfg: Config, output: Path, workers: int = 10, dual_window: int = 10,
    validation_interval: int = 25, validation_trajectories: int = 4,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    constrained_dir = output / "constrained_checkpoints"
    energy_dir = output / "energy_only_checkpoints"
    constrained_dir.mkdir()
    energy_dir.mkdir()
    jobs = [
        (label, constrained, seed, constrained_dir if constrained else energy_dir)
        for label, constrained in (("RA-CRL", True), ("Energy-only DQN", False))
        for seed in cfg.controller_seeds
    ]
    history: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                _train_load_aware_seed, cfg, seed, checkpoint_dir, constrained,
                dual_window, validation_interval, validation_trajectories,
            ): label
            for label, constrained, seed, checkpoint_dir in jobs
        }
        for future in as_completed(futures):
            seed_history, seed_validation = future.result()
            label = futures[future]
            history.extend({"method": label, **row} for row in seed_history)
            validation.extend({"method": label, **row} for row in seed_validation)
    history.sort(key=lambda row: (str(row["method"]), int(row["controller_seed"]), int(row["episode"])))
    validation.sort(key=lambda row: (str(row["method"]), int(row["controller_seed"]), int(row["checkpoint_episode"])))
    _write_rows(output / "training_history.csv", history)
    _write_rows(output / "validation_history.csv", validation)
    metadata = {
        "methods": list(ABLATION_METHODS), "controller_seeds": list(cfg.controller_seeds),
        "training_episodes": cfg.train_episodes, "training_load_scales": list(LOAD_SCALES),
        "dual_update_window": dual_window,
        "validation": "3 speeds x 3 loads x 4 trajectories at four-slot delay",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _train_energy_only_seed(
    cfg: Config, seed: int, checkpoint: Path, validation_interval: int,
    validation_trajectories: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    torch.set_num_threads(1)
    state_dim = 4 * cfg.subarrays + 2 * cfg.users + 2
    controller = ConstrainedDQN(cfg, state_dim, seed)
    controller.dual = 0.0
    rng = np.random.default_rng(seed + cfg.seed)
    history: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    best_energy = float("inf")
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
        controller.dual = 0.0
        history.append({
            "controller_seed": seed, "episode": episode,
            "energy_j": metric.energy_j, "outage_probability": metric.outage_probability,
            "active_ratio": metric.active_ratio, "dual": 0.0,
            "epsilon": epsilon,
            "loss": float(np.mean(trace["loss"])) if trace["loss"].size else 0.0,
        })
        due = (episode + 1) % validation_interval == 0 or episode + 1 == cfg.train_episodes
        if due:
            row = _validate(cfg, controller, seed, episode + 1, validation_trajectories)
            row["objective"] = "energy_only"
            validation.append(row)
            if row["energy_j"] < best_energy:
                best_energy = row["energy_j"]
                best_state = _clone_state(controller)
                best_episode = episode + 1

    if best_state is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    controller.online.load_state_dict(best_state["online"])
    controller.target.load_state_dict(best_state["target"])
    controller.dual = 0.0
    torch.save({
        "method": "RA-FVV", "seed": seed,
        "online": controller.online.state_dict(), "target": controller.target.state_dict(),
        "dual": 0.0, "selected_episode": best_episode,
        "selection_rule": "minimum validation energy without outage price",
    }, checkpoint)
    return history, validation


def train_energy_only(
    cfg: Config, output: Path, workers: int = 5,
    validation_interval: int = 25, validation_trajectories: int = 4,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    jobs = [
        (seed, checkpoints / f"energy_only_seed{seed}.pt")
        for seed in cfg.controller_seeds
    ]
    history: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [
            executor.submit(
                _train_energy_only_seed, cfg, seed, checkpoint,
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
        "validation_interval_episodes": validation_interval,
        "selection_rule": "minimum validation energy without outage price",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _init_eval_worker(
    cfg: Config, constrained_specs: list[tuple[int, str]],
    energy_specs: list[tuple[int, str]],
) -> None:
    global _CONTROLLERS
    torch.set_num_threads(1)
    _CONTROLLERS = {
        "RA-CRL": [
            (seed, _load_controller(cfg, "RA-FVV", seed, Path(path)))
            for seed, path in constrained_specs
        ],
        "Energy-only DQN": [
            (seed, _load_controller(cfg, "RA-FVV", seed, Path(path)))
            for seed, path in energy_specs
        ],
    }


def _eval_job(
    cfg: Config, load_scale: float, trajectory_index: int,
) -> list[dict[str, object]]:
    speed = 2.0
    local_cfg = cfg.with_delay(4)
    trajectory_seed = cfg.seed * 510_000 + trajectory_index
    trajectory = generate_trajectory(local_cfg, trajectory_seed, speed)
    trajectory = replace(
        trajectory,
        requirements_mbps=trajectory.requirements_mbps * load_scale,
    )
    rows: list[dict[str, object]] = []
    for method, controllers in _CONTROLLERS.items():
        for controller_seed, controller in controllers:
            metric, trace = simulate_episode(
                local_cfg, trajectory, "RA-FVV", controller, 1.0, trajectory_seed
            )
            rows.append({
                **asdict(metric), "method": method, "load_scale": load_scale,
                "trajectory": trajectory_index, "controller_seed": controller_seed,
                "mean_commanded_count": float(np.mean(trace["action_count"])),
            })
    return rows


def evaluate_constraint_ablation(
    cfg: Config, constrained_dir: Path, energy_dir: Path,
    output: Path, workers: int = 12,
    comparison: str = "same architecture and nominal training distribution; outage price removed only",
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    constrained_specs = [
        (seed, str(constrained_dir / f"ra_crl_seed{seed}.pt"))
        for seed in cfg.controller_seeds
    ]
    energy_specs = [
        (seed, str(energy_dir / f"energy_only_seed{seed}.pt"))
        for seed in cfg.controller_seeds
    ]
    missing = [p for _, p in constrained_specs + energy_specs if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    jobs = [
        (scale, trajectory)
        for scale in LOAD_SCALES
        for trajectory in range(cfg.test_trajectories)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_eval_worker,
        initargs=(cfg, constrained_specs, energy_specs),
    ) as executor:
        futures = [executor.submit(_eval_job, cfg, *job) for job in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (
        float(row["load_scale"]), int(row["trajectory"]),
        int(row["controller_seed"]), str(row["method"]),
    ))
    _write_rows(output / "raw_metrics.csv", rows)

    rng = np.random.default_rng(cfg.seed + 2441)
    summary: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for scale in LOAD_SCALES:
        subset = [row for row in rows if float(row["load_scale"]) == scale]
        matrices: dict[tuple[str, str], np.ndarray] = {}
        for method in ABLATION_METHODS:
            item: dict[str, object] = {
                "method": method, "load_scale": scale,
                "controller_seeds": len(cfg.controller_seeds),
                "test_trajectories": cfg.test_trajectories,
            }
            for field in (*FIELDS, "mean_commanded_count"):
                values = {
                    (int(row["controller_seed"]), int(row["trajectory"])): float(row[field])
                    for row in subset if row["method"] == method
                }
                matrix = np.asarray([
                    [values[(seed, trajectory)] for trajectory in range(cfg.test_trajectories)]
                    for seed in cfg.controller_seeds
                ])
                matrices[(method, field)] = matrix
                low, high = _hierarchical_ci(matrix, rng)
                item[field] = float(matrix.mean())
                item[field + "_ci95_low"] = low
                item[field + "_ci95_high"] = high
            summary.append(item)
        item = {"load_scale": scale}
        for field in ("energy_j", "outage_probability", "mean_commanded_count"):
            delta = matrices[("Energy-only DQN", field)] - matrices[("RA-CRL", field)]
            low, high = _hierarchical_ci(delta, rng)
            item[field + "_energy_only_minus_constrained"] = float(delta.mean())
            item[field + "_ci95_low"] = low
            item[field + "_ci95_high"] = high
        paired.append(item)
    _write_rows(output / "summary.csv", summary)
    _write_rows(output / "paired_summary.csv", paired)

    distribution: list[dict[str, object]] = []
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for method in ABLATION_METHODS:
        item: dict[str, object] = {
            "method": method, "load_distribution": "uniform over 0.85/1.00/1.15",
        }
        for field in (*FIELDS, "mean_commanded_count"):
            values = {
                (int(row["controller_seed"]), int(row["trajectory"]), float(row["load_scale"])):
                float(row[field])
                for row in rows if row["method"] == method
            }
            matrix = np.asarray([
                [
                    np.mean([values[(seed, trajectory, scale)] for scale in LOAD_SCALES])
                    for trajectory in range(cfg.test_trajectories)
                ]
                for seed in cfg.controller_seeds
            ])
            matrices[(method, field)] = matrix
            low, high = _hierarchical_ci(matrix, rng)
            item[field] = float(matrix.mean())
            item[field + "_ci95_low"] = low
            item[field + "_ci95_high"] = high
        distribution.append(item)
    paired_item: dict[str, object] = {
        "method": "Energy-only minus RA-CRL",
        "load_distribution": "uniform over 0.85/1.00/1.15",
    }
    for field in ("energy_j", "outage_probability", "mean_commanded_count"):
        delta = matrices[("Energy-only DQN", field)] - matrices[("RA-CRL", field)]
        low, high = _hierarchical_ci(delta, rng)
        paired_item[field] = float(delta.mean())
        paired_item[field + "_ci95_low"] = low
        paired_item[field + "_ci95_high"] = high
    distribution.append(paired_item)
    _write_rows(output / "distribution_summary.csv", distribution)
    metadata = {
        "load_scales": list(LOAD_SCALES), "speed_mps": 2.0,
        "wake_delay": 4, "controller_seeds": list(cfg.controller_seeds),
        "test_trajectories": cfg.test_trajectories,
        "comparison": comparison,
        "confidence_interval": "two-level percentile bootstrap over seeds and trajectories",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def refresh_distribution_summary(cfg: Config, output: Path) -> None:
    with (output / "raw_metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(cfg.seed + 2441)
    distribution: list[dict[str, object]] = []
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for method in ABLATION_METHODS:
        item: dict[str, object] = {
            "method": method, "load_distribution": "uniform over 0.85/1.00/1.15",
        }
        for field in (*FIELDS, "mean_commanded_count"):
            values = {
                (int(row["controller_seed"]), int(row["trajectory"]), float(row["load_scale"])):
                float(row[field]) for row in rows if row["method"] == method
            }
            matrix = np.asarray([
                [np.mean([values[(seed, trajectory, scale)] for scale in LOAD_SCALES])
                 for trajectory in range(cfg.test_trajectories)]
                for seed in cfg.controller_seeds
            ])
            matrices[(method, field)] = matrix
            low, high = _hierarchical_ci(matrix, rng)
            item[field] = float(matrix.mean())
            item[field + "_ci95_low"] = low
            item[field + "_ci95_high"] = high
        distribution.append(item)
    paired_item: dict[str, object] = {
        "method": "Energy-only minus RA-CRL",
        "load_distribution": "uniform over 0.85/1.00/1.15",
    }
    for field in ("energy_j", "outage_probability", "mean_commanded_count"):
        delta = matrices[("Energy-only DQN", field)] - matrices[("RA-CRL", field)]
        low, high = _hierarchical_ci(delta, rng)
        paired_item[field] = float(delta.mean())
        paired_item[field + "_ci95_low"] = low
        paired_item[field + "_ci95_high"] = high
    distribution.append(paired_item)
    _write_rows(output / "distribution_summary.csv", distribution)
