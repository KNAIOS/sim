from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .channel import generate_trajectory
from .config import Config
from .controllers import ConstrainedDQN, select_top
from .experiment import _write_rows, simulate_episode
from .learned_study import _load_controller
from .matched_policy_study import FIELDS, _hierarchical_ci


METHODS = ("Full array", "Static HRNP", "Current-CSI", "Symmetric guard", "RA-CRL")
CALIBRATION_TRAJECTORIES = 20
_CONTROLLERS: list[tuple[int, ConstrainedDQN]] = []
_STATIC_MASKS: dict[float, np.ndarray] = {}


def _calibration_masks(cfg: Config) -> dict[float, np.ndarray]:
    masks: dict[float, np.ndarray] = {}
    count = max(cfg.users + 1, int(np.ceil(0.65 * cfg.subarrays)))
    for speed in cfg.speeds_mps:
        gains = []
        for index in range(CALIBRATION_TRAJECTORIES):
            trajectory = generate_trajectory(
                cfg, cfg.seed * 90_000 + index, float(speed)
            )
            gains.append(trajectory.gain_db.mean(axis=(0, 2)))
        masks[float(speed)] = select_top(np.mean(gains, axis=0), count)
    return masks


def _init_worker(
    cfg: Config, specs: list[tuple[int, str]], static_masks: dict[float, np.ndarray]
) -> None:
    global _CONTROLLERS, _STATIC_MASKS
    torch.set_num_threads(1)
    _CONTROLLERS = [
        (seed, _load_controller(cfg, "RA-FVV", seed, Path(checkpoint)))
        for seed, checkpoint in specs
    ]
    _STATIC_MASKS = static_masks


def _job(cfg: Config, speed: float, trajectory_index: int) -> list[dict[str, object]]:
    local_cfg = cfg.with_delay(cfg.wake_delay_slots)
    trajectory_seed = cfg.seed * 410_000 + trajectory_index
    trajectory = generate_trajectory(local_cfg, trajectory_seed, speed)
    rows: list[dict[str, object]] = []

    baseline_metrics = {}
    for method in METHODS[:-1]:
        metric, _ = simulate_episode(
            local_cfg,
            trajectory,
            method,
            None,
            1.0,
            trajectory_seed,
            static_mask=_STATIC_MASKS[speed] if method == "Static HRNP" else None,
        )
        baseline_metrics[method] = metric

    for controller_seed, controller in _CONTROLLERS:
        for method, metric in baseline_metrics.items():
            rows.append({
                **asdict(metric),
                "method": method,
                "requested_speed_mps": speed,
                "trajectory": trajectory_index,
                "controller_seed": controller_seed,
            })
        metric, trace = simulate_episode(
            local_cfg, trajectory, "RA-FVV", controller, 1.0, trajectory_seed
        )
        rows.append({
            **asdict(metric),
            "method": "RA-CRL",
            "requested_speed_mps": speed,
            "trajectory": trajectory_index,
            "controller_seed": controller_seed,
        })
    return rows


def _matrix(
    rows: list[dict[str, object]], method: str, field: str,
    seeds: tuple[int, ...], trajectories: int,
) -> np.ndarray:
    values = {
        (int(row["controller_seed"]), int(row["trajectory"])): float(row[field])
        for row in rows if row["method"] == method
    }
    return np.asarray([
        [values[(seed, trajectory)] for trajectory in range(trajectories)]
        for seed in seeds
    ])


def _summaries(
    cfg: Config, rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(cfg.seed + 1881)
    summary: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for speed in cfg.speeds_mps:
        subset = [row for row in rows if float(row["requested_speed_mps"]) == speed]
        matrices: dict[tuple[str, str], np.ndarray] = {}
        for method in METHODS:
            item: dict[str, object] = {
                "method": method,
                "speed_mps": speed,
                "wake_delay": cfg.wake_delay_slots,
                "controller_seeds": len(cfg.controller_seeds),
                "test_trajectories": cfg.test_trajectories,
            }
            for field in FIELDS:
                matrix = _matrix(subset, method, field, cfg.controller_seeds, cfg.test_trajectories)
                matrices[(method, field)] = matrix
                low, high = _hierarchical_ci(matrix, rng)
                item[field] = float(matrix.mean())
                item[field + "_ci95_low"] = low
                item[field + "_ci95_high"] = high
            summary.append(item)

        for baseline in METHODS[:-1]:
            item = {"baseline": baseline, "speed_mps": speed}
            for field in ("energy_j", "outage_probability", "mean_shortfall"):
                delta = matrices[(baseline, field)] - matrices[("RA-CRL", field)]
                low, high = _hierarchical_ci(delta, rng)
                item[field + "_reduction"] = float(delta.mean())
                item[field + "_reduction_ci95_low"] = low
                item[field + "_reduction_ci95_high"] = high
            ee_gain = matrices[("RA-CRL", "energy_efficiency_mbit_j")] - matrices[
                (baseline, "energy_efficiency_mbit_j")
            ]
            low, high = _hierarchical_ci(ee_gain, rng)
            item["energy_efficiency_mbit_j_gain"] = float(ee_gain.mean())
            item["energy_efficiency_mbit_j_gain_ci95_low"] = low
            item["energy_efficiency_mbit_j_gain_ci95_high"] = high
            paired.append(item)
    return summary, paired


def run_absolute_baseline_study(
    cfg: Config, checkpoint_dir: Path, output: Path, workers: int = 12
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    specs = [
        (seed, str(checkpoint_dir / f"ra_crl_seed{seed}.pt"))
        for seed in cfg.controller_seeds
    ]
    missing = [path for _, path in specs if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing RA-CRL checkpoints: {missing}")
    static_masks = _calibration_masks(cfg)
    jobs = [
        (float(speed), trajectory)
        for speed in cfg.speeds_mps
        for trajectory in range(cfg.test_trajectories)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(cfg, specs, static_masks),
    ) as executor:
        futures = [executor.submit(_job, cfg, *job) for job in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (
        float(row["requested_speed_mps"]), int(row["trajectory"]),
        int(row["controller_seed"]), str(row["method"]),
    ))
    _write_rows(output / "raw_metrics.csv", rows)
    summary, paired = _summaries(cfg, rows)
    _write_rows(output / "summary.csv", summary)
    _write_rows(output / "paired_summary.csv", paired)
    metadata = {
        "design": "same trajectories and physical model for absolute baselines and RA-CRL",
        "static_hrnp_calibration": (
            f"{CALIBRATION_TRAJECTORIES} disjoint trajectories per speed; "
            "65% of subarrays retained"
        ),
        "controller_seeds": list(cfg.controller_seeds),
        "test_trajectories": cfg.test_trajectories,
        "confidence_interval": "two-level percentile bootstrap over seeds and trajectories",
        "workers": workers,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
