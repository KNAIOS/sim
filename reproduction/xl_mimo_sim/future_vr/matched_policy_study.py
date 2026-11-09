from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .channel import generate_trajectory
from .config import Config
from .controllers import ConstrainedDQN
from .experiment import _write_rows, simulate_episode
from .learned_study import _load_controller


METHODS = (
    "Current-CSI matched",
    "Future-only matched",
    "RA-FVV matched",
    "Oracle matched",
)
DISPLAY = {
    "Current-CSI matched": "CSI + shared DQN count",
    "Future-only matched": "Future + shared DQN count",
    "RA-FVV matched": "RA-CRL",
    "Oracle matched": "Oracle + shared DQN count",
}
FIELDS = (
    "energy_j",
    "energy_efficiency_mbit_j",
    "outage_probability",
    "mean_shortfall",
    "p95_shortfall",
    "active_ratio",
    "transitions_per_s",
)
_CONTROLLERS: list[tuple[int, ConstrainedDQN]] = []


def _init_worker(cfg: Config, specs: list[tuple[int, str]]) -> None:
    global _CONTROLLERS
    torch.set_num_threads(1)
    _CONTROLLERS = [
        (seed, _load_controller(cfg, "RA-FVV", seed, Path(checkpoint)))
        for seed, checkpoint in specs
    ]


def _job(cfg: Config, speed: float, delay: int, trajectory_index: int) -> list[dict[str, object]]:
    local_cfg = cfg.with_delay(delay)
    trajectory_seed = cfg.seed * 410_000 + trajectory_index
    trajectory = generate_trajectory(local_cfg, trajectory_seed, speed)
    rows: list[dict[str, object]] = []
    for controller_seed, controller in _CONTROLLERS:
        # One learned policy supplies the cardinality sequence.  All ranking
        # methods receive that exact sequence, isolating spatial selection.
        _, policy_trace = simulate_episode(
            local_cfg, trajectory, "RA-FVV", controller, 1.0, trajectory_seed
        )
        counts = policy_trace["action_count"].astype(int, copy=True)
        for method in METHODS:
            metric, _ = simulate_episode(
                local_cfg, trajectory, method, None, 1.0, trajectory_seed,
                forced_counts=counts,
            )
            rows.append({
                **asdict(metric),
                "method": DISPLAY[method],
                "requested_speed_mps": speed,
                "trajectory": trajectory_index,
                "controller_seed": controller_seed,
                "mean_commanded_count": float(np.mean(counts)),
            })
    return rows


def _hierarchical_ci(matrix: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Two-level percentile CI over controller seeds and paired trajectories."""
    seed_count, trajectory_count = matrix.shape
    boot = np.empty(2000, dtype=float)
    for index in range(boot.size):
        seed_sample = rng.integers(0, seed_count, seed_count)
        trajectory_sample = rng.integers(0, trajectory_count, trajectory_count)
        boot[index] = matrix[np.ix_(seed_sample, trajectory_sample)].mean()
    low, high = np.quantile(boot, (0.025, 0.975))
    return float(low), float(high)


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
    ], dtype=float)


def _summaries(cfg: Config, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(cfg.seed + 991)
    summaries: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    scenarios = sorted({
        (float(row["requested_speed_mps"]), int(row["wake_delay"])) for row in rows
    })
    for speed, delay in scenarios:
        subset = [
            row for row in rows
            if float(row["requested_speed_mps"]) == speed and int(row["wake_delay"]) == delay
        ]
        matrices: dict[tuple[str, str], np.ndarray] = {}
        for method in DISPLAY.values():
            item: dict[str, object] = {
                "method": method, "speed_mps": speed, "wake_delay": delay,
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
            summaries.append(item)

        proposed = "RA-CRL"
        for baseline in ("CSI + shared DQN count", "Future + shared DQN count"):
            item: dict[str, object] = {
                "baseline": baseline, "speed_mps": speed, "wake_delay": delay,
            }
            for field in ("energy_j", "outage_probability", "mean_shortfall", "transitions_per_s"):
                # Positive means RA-CRL improves this lower-is-better metric.
                delta = matrices[(baseline, field)] - matrices[(proposed, field)]
                low, high = _hierarchical_ci(delta, rng)
                item[field + "_reduction"] = float(delta.mean())
                item[field + "_reduction_ci95_low"] = low
                item[field + "_reduction_ci95_high"] = high
            paired.append(item)
    return summaries, paired


def run_matched_policy_study(
    cfg: Config, checkpoint_dir: Path, output: Path, workers: int = 12,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    specs = [
        (seed, str(checkpoint_dir / f"ra_crl_seed{seed}.pt"))
        for seed in cfg.controller_seeds
    ]
    missing = [path for _, path in specs if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing RA-CRL checkpoints: {missing}")

    scenarios = {(float(speed), cfg.wake_delay_slots) for speed in cfg.speeds_mps}
    central_speed = float(cfg.speeds_mps[len(cfg.speeds_mps) // 2])
    scenarios.update((central_speed, int(delay)) for delay in cfg.wake_delays)
    jobs = [
        (speed, delay, trajectory)
        for speed, delay in sorted(scenarios)
        for trajectory in range(cfg.test_trajectories)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(cfg, specs),
    ) as executor:
        futures = [executor.submit(_job, cfg, *job) for job in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (
        float(row["requested_speed_mps"]), int(row["wake_delay"]),
        int(row["trajectory"]), int(row["controller_seed"]), str(row["method"]),
    ))
    _write_rows(output / "raw_metrics.csv", rows)
    summary, paired = _summaries(cfg, rows)
    _write_rows(output / "summary.csv", summary)
    _write_rows(output / "paired_summary.csv", paired)
    metadata = {
        "design": "shared RA-CRL dynamic count sequence; ranking is the only changed factor",
        "controller_seeds": list(cfg.controller_seeds),
        "test_trajectories": cfg.test_trajectories,
        "confidence_interval": "two-level percentile bootstrap over seeds and trajectories",
        "bootstrap_repetitions": 2000,
        "workers": workers,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
