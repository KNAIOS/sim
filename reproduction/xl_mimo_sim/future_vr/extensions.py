from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from .belief import VisibilityBelief
from .channel import Trajectory, generate_trajectory
from .config import Config
from .controllers import readiness_aligned_value, select_top
from .experiment import _write_rows
from .system import HardwareState, XLSystem


def _current_score(gain_db: np.ndarray, requirements: np.ndarray) -> np.ndarray:
    linear = 10.0 ** ((gain_db - gain_db.max(axis=0, keepdims=True)) / 10.0)
    weight = requirements / max(float(np.mean(requirements)), 1e-9)
    score = linear @ weight
    return score / max(float(score.max()), 1e-12)


def _future_truth(
    cfg: Config,
    trajectory: Trajectory,
    time_index: int,
    readiness: np.ndarray,
) -> np.ndarray:
    stop = min(time_index + cfg.horizon + 1, cfg.episode_steps)
    gain = trajectory.gain_db[time_index + 1:stop]
    probability = trajectory.visibility[time_index + 1:stop]
    if gain.shape[0] < cfg.horizon:
        missing = cfg.horizon - gain.shape[0]
        gain = np.concatenate((gain, np.repeat(gain[-1:], missing, axis=0)))
        probability = np.concatenate((probability, np.repeat(probability[-1:], missing, axis=0)))
    return readiness_aligned_value(
        cfg, gain, probability, np.zeros_like(probability), readiness,
        trajectory.requirements_mbps[time_index],
    )


def selection_stability(
    cfg: Config,
    trajectory_count: int = 12,
    sample_stride: int = 2,
) -> dict[str, np.ndarray]:
    """Relate the predicted top-M margin to agreement with true future value."""
    count = min(cfg.subarrays - 1, cfg.users + 3)
    records: list[tuple[float, float, float, float, float]] = []
    for trajectory_index in range(trajectory_count):
        speed = float(cfg.speeds_mps[trajectory_index % len(cfg.speeds_mps)])
        trajectory = generate_trajectory(cfg, cfg.seed + 50_000 + trajectory_index, speed)
        tracker = VisibilityBelief(cfg, trajectory.gain_db[0])
        current_mask = select_top(_current_score(trajectory.gain_db[0], trajectory.requirements_mbps[0]), count)
        for t in range(cfg.episode_steps - cfg.horizon - 1):
            observed = np.ones(cfg.subarrays, dtype=bool) if t % cfg.scan_period == 0 else current_mask
            tracker.update(trajectory.gain_db[t], observed)
            if t % sample_stride:
                continue
            readiness = np.where(current_mask, 0, cfg.wake_delay_slots)
            gain, probability, uncertainty = tracker.predict(cfg.horizon)
            predicted = readiness_aligned_value(
                cfg, gain, probability, uncertainty, readiness, trajectory.requirements_mbps[t],
            )
            truth = _future_truth(cfg, trajectory, t, readiness)
            predicted_mask = select_top(predicted, count)
            truth_mask = select_top(truth, count)
            current_value = _current_score(trajectory.gain_db[t], trajectory.requirements_mbps[t])
            current_choice = select_top(current_value, count)
            ordered = np.sort(predicted)[::-1]
            margin = float(ordered[count - 1] - ordered[count])
            delta = float(np.max(np.abs(predicted - truth)))
            overlap = float(np.sum(predicted_mask & truth_mask) / count)
            current_overlap = float(np.sum(current_choice & truth_mask) / count)
            exact = float(np.array_equal(predicted_mask, truth_mask))
            records.append((margin, overlap, current_overlap, exact, delta))
            current_mask = predicted_mask

    data = np.asarray(records)
    margin, overlap, current_overlap, exact, delta = data.T
    edges = np.unique(np.quantile(margin, np.linspace(0.0, 1.0, 6)))
    if len(edges) < 3:
        edges = np.linspace(float(margin.min()), float(margin.max()) + 1e-9, 6)
    bin_index = np.clip(np.digitize(margin, edges[1:-1]), 0, len(edges) - 2)
    centers: list[float] = []
    predicted_mean: list[float] = []
    current_mean: list[float] = []
    exact_mean: list[float] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        selected = bin_index == index
        if not selected.any():
            continue
        centers.append(float(np.mean(margin[selected])))
        predicted_mean.append(float(np.mean(overlap[selected])))
        current_mean.append(float(np.mean(current_overlap[selected])))
        exact_mean.append(float(np.mean(exact[selected])))
        counts.append(int(selected.sum()))
    certified = margin > 2.0 * delta
    return {
        "margin": margin, "overlap": overlap, "current_overlap": current_overlap,
        "exact": exact, "delta": delta, "certified": certified.astype(float),
        "bin_center": np.asarray(centers), "bin_overlap": np.asarray(predicted_mean),
        "bin_current_overlap": np.asarray(current_mean), "bin_exact": np.asarray(exact_mean),
        "bin_count": np.asarray(counts),
    }


@dataclass(frozen=True)
class WindowMetric:
    energy_j: float
    outage: float
    shortfall: float
    delivered_mbit: float


def _fixed_command_window(
    cfg: Config,
    trajectory: Trajectory,
    time_index: int,
    current_mask: np.ndarray,
    command: np.ndarray,
) -> WindowMetric:
    hardware = HardwareState(cfg)
    hardware.active = current_mask.copy()
    hardware.command = current_mask.copy()
    hardware.countdown[:] = 0
    system = XLSystem(cfg)
    energy: list[float] = []
    outage: list[float] = []
    shortfall: list[float] = []
    delivered: list[float] = []
    for h in range(cfg.horizon):
        if h == 0:
            available, switching_energy, _ = hardware.apply(command, 1.0)
        else:
            available = hardware.active.copy()
            switching_energy = 0.0
        step = system.evaluate(
            trajectory.channel[time_index + h], available, hardware.countdown > 0,
            switching_energy, False, trajectory.requirements_mbps[time_index + h],
        )
        energy.append(step.energy_j)
        outage.append(step.outage)
        shortfall.append(step.shortfall)
        delivered.append(step.delivered_mbit)
        hardware.tick()
    return WindowMetric(
        float(np.sum(energy)), float(np.mean(outage)), float(np.mean(shortfall)), float(np.sum(delivered))
    )


def exhaustive_benchmark(base_cfg: Config, output: Path, trajectory_count: int = 10) -> list[dict[str, object]]:
    """Compare fixed-cardinality RA-FVV ranking with all C(Q,M) commands."""
    cfg = replace(
        base_cfg, subarrays=8, users=2, array_aperture_m=8.0, episode_steps=72,
        horizon=6, wake_delay_slots=4, wake_delays=(4,), speeds_mps=(0.8, 2.5),
    )
    activation_count = 3
    rows: list[dict[str, object]] = []
    for trajectory_index in range(trajectory_count):
        speed = float(cfg.speeds_mps[trajectory_index % len(cfg.speeds_mps)])
        trajectory = generate_trajectory(cfg, cfg.seed + 80_000 + trajectory_index, speed)
        tracker = VisibilityBelief(cfg, trajectory.gain_db[0])
        current_mask = select_top(_current_score(trajectory.gain_db[0], trajectory.requirements_mbps[0]), activation_count)
        for t in range(0, cfg.episode_steps - cfg.horizon, 4):
            observed = np.ones(cfg.subarrays, dtype=bool) if t % cfg.scan_period == 0 else current_mask
            tracker.update(trajectory.gain_db[t], observed)
            readiness = np.where(current_mask, 0, cfg.wake_delay_slots)
            gain, probability, uncertainty = tracker.predict(cfg.horizon)
            start = perf_counter()
            score = readiness_aligned_value(
                cfg, gain, probability, uncertainty, readiness, trajectory.requirements_mbps[t],
            )
            proposed_mask = select_top(score, activation_count)
            ranking_ms = 1e3 * (perf_counter() - start)
            current_choice = select_top(
                _current_score(trajectory.gain_db[t], trajectory.requirements_mbps[t]), activation_count
            )

            start = perf_counter()
            candidates: list[tuple[tuple[float, ...], np.ndarray, WindowMetric]] = []
            for indices in combinations(range(cfg.subarrays), activation_count):
                mask = np.zeros(cfg.subarrays, dtype=bool)
                mask[list(indices)] = True
                metric = _fixed_command_window(cfg, trajectory, t, current_mask, mask)
                # With fixed cardinality, compare the communication quality
                # first; energy separates equally useful commands through RF
                # switching cost. This avoids declaring a stale no-switch set
                # "optimal" merely because excess throughput is ignored.
                key = (metric.outage, metric.shortfall, -metric.delivered_mbit, metric.energy_j)
                candidates.append((key, mask, metric))
            exhaustive_ms = 1e3 * (perf_counter() - start)
            _, optimal_mask, optimal = min(candidates, key=lambda item: item[0])
            for method, mask in (("RA-FVV ranking", proposed_mask), ("Current-CSI ranking", current_choice)):
                metric = _fixed_command_window(cfg, trajectory, t, current_mask, mask)
                feasible = metric.outage <= cfg.outage_target
                optimal_feasible = optimal.outage <= cfg.outage_target
                energy_gap = (
                    100.0 * (metric.energy_j - optimal.energy_j) / max(optimal.energy_j, 1e-9)
                    if feasible and optimal_feasible else np.nan
                )
                rows.append({
                    "trajectory": trajectory_index, "time_slot": t, "speed_mps": speed, "method": method,
                    "energy_j": metric.energy_j, "outage": metric.outage, "shortfall": metric.shortfall,
                    "delivered_mbit": metric.delivered_mbit,
                    "feasible": int(feasible), "optimal_energy_j": optimal.energy_j,
                    "optimal_outage": optimal.outage, "conditional_energy_gap_percent": energy_gap,
                    "throughput_gap_percent": 100.0 * (optimal.delivered_mbit - metric.delivered_mbit)
                    / max(optimal.delivered_mbit, 1e-9),
                    "optimal_set_overlap": float(np.sum(mask & optimal_mask) / activation_count),
                    "ranking_time_ms": ranking_ms if method == "RA-FVV ranking" else np.nan,
                    "exhaustive_time_ms": exhaustive_ms,
                })
            current_mask = proposed_mask
    _write_rows(output / "exhaustive_benchmark.csv", rows)
    return rows


def summarize_exhaustive(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in ("RA-FVV ranking", "Current-CSI ranking"):
        selected = [row for row in rows if row["method"] == method]
        gaps = np.asarray([row["conditional_energy_gap_percent"] for row in selected], dtype=float)
        finite = gaps[np.isfinite(gaps)]
        timing = np.asarray([row["ranking_time_ms"] for row in selected], dtype=float)
        timing = timing[np.isfinite(timing)]
        summary.append({
            "method": method,
            "samples": len(selected),
            "feasible_rate": float(np.mean([row["feasible"] for row in selected])),
            "mean_optimal_set_overlap": float(np.mean([row["optimal_set_overlap"] for row in selected])),
            "median_conditional_energy_gap_percent": float(np.median(finite)) if finite.size else np.nan,
            "p95_conditional_energy_gap_percent": float(np.quantile(finite, 0.95)) if finite.size else np.nan,
            "mean_throughput_gap_percent": float(np.mean([row["throughput_gap_percent"] for row in selected])),
            "p95_throughput_gap_percent": float(np.quantile(
                [row["throughput_gap_percent"] for row in selected], 0.95
            )),
            "mean_ranking_time_ms": float(np.mean(timing)) if timing.size else np.nan,
            "mean_exhaustive_time_ms": float(np.mean([row["exhaustive_time_ms"] for row in selected])),
        })
    return summary
