from __future__ import annotations

import json
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .channel import generate_trajectory
from .config import Config
from .experiment import _write_rows, simulate_episode
from .matched_policy_study import _hierarchical_ci


METHODS = (
    "Current-CSI matched",
    "Future-only matched",
    "RA-FVV matched",
    "Oracle matched",
)
DISPLAY = {
    "Current-CSI matched": "Current CSI",
    "Future-only matched": "Unaligned future",
    "RA-FVV matched": "Readiness aligned",
    "Oracle matched": "Noncausal oracle",
}
FIELDS = (
    "energy_j",
    "energy_efficiency_mbit_j",
    "outage_probability",
    "mean_shortfall",
    "active_ratio",
    "transitions_per_s",
)


def _job(
    cfg: Config,
    speed: float,
    delay: int,
    trajectory_index: int,
    counts: tuple[int, ...],
    control_period: int,
) -> list[dict[str, object]]:
    local_cfg = cfg.with_delay(delay)
    trajectory_seed = cfg.seed * 610_000 + trajectory_index
    trajectory = generate_trajectory(local_cfg, trajectory_seed, speed)
    rows: list[dict[str, object]] = []
    for count in counts:
        forced_counts = np.full(cfg.episode_steps, count, dtype=int)
        for method in METHODS:
            metric, _ = simulate_episode(
                local_cfg,
                trajectory,
                method,
                None,
                1.0,
                trajectory_seed,
                forced_counts=forced_counts,
                control_period=control_period,
            )
            rows.append(
                {
                    **asdict(metric),
                    "method": DISPLAY[method],
                    "requested_speed_mps": speed,
                    "trajectory": trajectory_index,
                    "commanded_count": count,
                    "control_period": control_period,
                }
            )
    return rows


def _summary(cfg: Config, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = np.random.default_rng(cfg.seed + 2711)
    output: list[dict[str, object]] = []
    methods = tuple(DISPLAY.values())
    scenarios = sorted(
        {
            (
                str(row["method"]),
                float(row["requested_speed_mps"]),
                int(row["wake_delay"]),
                int(row["commanded_count"]),
            )
            for row in rows
        }
    )
    for method, speed, delay, count in scenarios:
        selected = [
            row
            for row in rows
            if str(row["method"]) == method
            and float(row["requested_speed_mps"]) == speed
            and int(row["wake_delay"]) == delay
            and int(row["commanded_count"]) == count
        ]
        item: dict[str, object] = {
            "method": method,
            "speed_mps": speed,
            "wake_delay": delay,
            "commanded_count": count,
            "control_period": int(selected[0]["control_period"]),
            "test_trajectories": len(selected),
        }
        for field in FIELDS:
            values = np.asarray([float(row[field]) for row in selected], dtype=float)[None, :]
            low, high = _hierarchical_ci(values, rng)
            item[field] = float(values.mean())
            item[field + "_ci95_low"] = low
            item[field + "_ci95_high"] = high
        output.append(item)
    if {str(row["method"]) for row in output} != set(methods):
        raise RuntimeError("Pareto summary is missing a ranking method")
    return output


def _select_target_point(
    rows: list[dict[str, object]], method: str, target: float
) -> tuple[float, float, float]:
    candidates: list[tuple[float, float, float]] = []
    counts = sorted({int(row["commanded_count"]) for row in rows})
    for count in counts:
        selected = [
            row for row in rows
            if str(row["method"]) == method and int(row["commanded_count"]) == count
        ]
        energy = float(np.mean([float(row["energy_j"]) for row in selected]))
        outage = float(np.mean([float(row["outage_probability"]) for row in selected]))
        if outage <= target:
            candidates.append((energy, outage, float(count)))
    if not candidates:
        return float("nan"), float("nan"), float("nan")
    return min(candidates, key=lambda item: (item[0], item[1]))


def write_target_summary(
    raw_path: Path, output_path: Path, target: float = 0.05, repetitions: int = 2000
) -> list[dict[str, object]]:
    """Bootstrap the minimum-energy empirical point satisfying mean outage."""
    with raw_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    methods = tuple(DISPLAY.values())
    trajectories = sorted({int(row["trajectory"]) for row in rows})
    by_trajectory = {
        trajectory: [row for row in rows if int(row["trajectory"]) == trajectory]
        for trajectory in trajectories
    }
    point = {method: _select_target_point(rows, method, target) for method in methods}
    rng = np.random.default_rng(4093)
    bootstrap = {method: [] for method in methods}
    paired = {method: [] for method in methods if method != "Readiness aligned"}
    for _ in range(repetitions):
        sample = rng.choice(trajectories, size=len(trajectories), replace=True)
        sampled_rows = [row for trajectory in sample for row in by_trajectory[int(trajectory)]]
        sampled_points = {
            method: _select_target_point(sampled_rows, method, target) for method in methods
        }
        for method, value in sampled_points.items():
            bootstrap[method].append(value)
        ra_energy = sampled_points["Readiness aligned"][0]
        for method in paired:
            base_energy = sampled_points[method][0]
            saving = (base_energy - ra_energy) / base_energy if np.isfinite(base_energy + ra_energy) else np.nan
            paired[method].append(saving)

    output: list[dict[str, object]] = []
    for method in methods:
        values = np.asarray(bootstrap[method], dtype=float)
        energy, outage, count = point[method]
        item: dict[str, object] = {
            "method": method,
            "outage_target": target,
            "energy_j": energy,
            "outage_probability": outage,
            "commanded_count": count,
            "energy_j_ci95_low": float(np.nanquantile(values[:, 0], 0.025)),
            "energy_j_ci95_high": float(np.nanquantile(values[:, 0], 0.975)),
            "commanded_count_ci95_low": float(np.nanquantile(values[:, 2], 0.025)),
            "commanded_count_ci95_high": float(np.nanquantile(values[:, 2], 0.975)),
        }
        if method != "Readiness aligned":
            savings = np.asarray(paired[method], dtype=float)
            item["ra_energy_saving_fraction"] = float((energy - point["Readiness aligned"][0]) / energy)
            item["ra_energy_saving_fraction_ci95_low"] = float(np.nanquantile(savings, 0.025))
            item["ra_energy_saving_fraction_ci95_high"] = float(np.nanquantile(savings, 0.975))
        output.append(item)
    _write_rows(output_path, output)
    return output


def write_confident_target_summary(
    summary_path: Path,
    raw_path: Path,
    output_path: Path,
    target: float = 0.05,
    repetitions: int = 2000,
) -> list[dict[str, object]]:
    """Select the lowest-energy point whose outage CI upper bound meets target."""
    with summary_path.open(encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    with raw_path.open(encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    selected: dict[str, dict[str, str]] = {}
    for method in DISPLAY.values():
        feasible = [
            row for row in summary
            if row["method"] == method
            and float(row["outage_probability_ci95_high"]) <= target
        ]
        if not feasible:
            raise RuntimeError(f"no confidence-qualified point for {method}")
        selected[method] = min(feasible, key=lambda row: float(row["energy_j"]))

    trajectories = sorted({int(row["trajectory"]) for row in raw})
    energy_by_method: dict[str, np.ndarray] = {}
    for method, point in selected.items():
        count = int(point["commanded_count"])
        values = {
            int(row["trajectory"]): float(row["energy_j"])
            for row in raw
            if row["method"] == method and int(row["commanded_count"]) == count
        }
        energy_by_method[method] = np.asarray([values[index] for index in trajectories])

    rng = np.random.default_rng(8123)
    indices = rng.integers(0, len(trajectories), size=(repetitions, len(trajectories)))
    ra_samples = energy_by_method["Readiness aligned"][indices].mean(axis=1)
    output: list[dict[str, object]] = []
    for method in DISPLAY.values():
        point = selected[method]
        energy_samples = energy_by_method[method][indices].mean(axis=1)
        row: dict[str, object] = {
            "method": method,
            "outage_target": target,
            "selection_rule": "upper 95% outage confidence bound <= target",
            "energy_j": float(point["energy_j"]),
            "energy_j_ci95_low": float(np.quantile(energy_samples, 0.025)),
            "energy_j_ci95_high": float(np.quantile(energy_samples, 0.975)),
            "outage_probability": float(point["outage_probability"]),
            "outage_probability_ci95_high": float(point["outage_probability_ci95_high"]),
            "commanded_count": int(point["commanded_count"]),
        }
        if method != "Readiness aligned":
            saving = (energy_samples - ra_samples) / energy_samples
            row["ra_energy_saving_fraction"] = float(
                (float(point["energy_j"]) - float(selected["Readiness aligned"]["energy_j"]))
                / float(point["energy_j"])
            )
            row["ra_energy_saving_fraction_ci95_low"] = float(np.quantile(saving, 0.025))
            row["ra_energy_saving_fraction_ci95_high"] = float(np.quantile(saving, 0.975))
        output.append(row)
    _write_rows(output_path, output)
    return output


def write_calibration_test_summaries(
    raw_path: Path,
    calibration_summary_path: Path,
    test_summary_path: Path,
    target_summary_path: Path,
    target: float = 0.05,
    calibration_trajectories: int = 40,
    repetitions: int = 2000,
) -> list[dict[str, object]]:
    """Select cardinality on calibration trajectories and report it on held-out tests.

    Trajectory identifiers are split deterministically.  No test sample is used
    to choose the commanded count or to check calibration feasibility.
    """
    with raw_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    trajectories = sorted({int(row["trajectory"]) for row in rows})
    if not 0 < calibration_trajectories < len(trajectories):
        raise ValueError("calibration_trajectories must leave a nonempty test set")
    calibration_ids = set(trajectories[:calibration_trajectories])
    test_ids = set(trajectories[calibration_trajectories:])
    calibration_rows = [row for row in rows if int(row["trajectory"]) in calibration_ids]
    test_rows = [row for row in rows if int(row["trajectory"]) in test_ids]

    def summarize(selected_rows: list[dict[str, str]], seed: int) -> list[dict[str, object]]:
        local_cfg = Config(test_trajectories=len({int(row["trajectory"]) for row in selected_rows}))
        # _summary only uses cfg.seed and the number of selected observations.
        local_cfg = Config(seed=seed, test_trajectories=local_cfg.test_trajectories)
        return _summary(local_cfg, selected_rows)

    calibration_summary = summarize(calibration_rows, 5039)
    test_summary = summarize(test_rows, 6043)
    _write_rows(calibration_summary_path, calibration_summary)
    _write_rows(test_summary_path, test_summary)

    selected: dict[str, dict[str, object]] = {}
    calibration_id_list = sorted(calibration_ids)
    calibration_rng = np.random.default_rng(8107)
    for method in DISPLAY.values():
        feasible: list[dict[str, object]] = []
        for candidate in (row for row in calibration_summary if row["method"] == method):
            count = int(candidate["commanded_count"])
            values_by_id = {
                int(row["trajectory"]): float(row["outage_probability"])
                for row in calibration_rows
                if row["method"] == method and int(row["commanded_count"]) == count
            }
            values = np.asarray([values_by_id[index] for index in calibration_id_list])
            indices = calibration_rng.integers(
                0, len(values), size=(repetitions, len(values))
            )
            # A one-sided 95% upper confidence bound uses the 0.95 quantile;
            # the 0.975 quantile belongs to a two-sided 95% interval.
            ucb = float(np.quantile(values[indices].mean(axis=1), 0.95))
            candidate = {**candidate, "outage_ucb95_one_sided": ucb}
            if ucb <= target:
                feasible.append(candidate)
        if not feasible:
            raise RuntimeError(f"no calibration-feasible point for {method}")
        selected[method] = min(feasible, key=lambda row: float(row["energy_j"]))

    rng = np.random.default_rng(9173)
    output: list[dict[str, object]] = []
    test_id_list = sorted(test_ids)
    for method in DISPLAY.values():
        count = int(selected[method]["commanded_count"])
        method_rows = [
            row for row in test_rows
            if row["method"] == method and int(row["commanded_count"]) == count
        ]
        by_id = {int(row["trajectory"]): row for row in method_rows}
        energy = np.asarray([float(by_id[index]["energy_j"]) for index in test_id_list])
        outage = np.asarray([float(by_id[index]["outage_probability"]) for index in test_id_list])
        indices = rng.integers(0, len(test_id_list), size=(repetitions, len(test_id_list)))
        energy_boot = energy[indices].mean(axis=1)
        outage_boot = outage[indices].mean(axis=1)
        output.append({
            "method": method,
            "outage_target": target,
            "selection_rule": "calibration one-sided 95% outage UCB <= target; independent test report",
            "calibration_trajectories": len(calibration_ids),
            "test_trajectories": len(test_ids),
            "commanded_count": count,
            "calibration_outage_ucb": float(selected[method]["outage_ucb95_one_sided"]),
            "test_energy_j": float(energy.mean()),
            "test_energy_j_ci95_low": float(np.quantile(energy_boot, 0.025)),
            "test_energy_j_ci95_high": float(np.quantile(energy_boot, 0.975)),
            "test_outage_probability": float(outage.mean()),
            "test_outage_probability_ci95_low": float(np.quantile(outage_boot, 0.025)),
            "test_outage_probability_ci95_high": float(np.quantile(outage_boot, 0.975)),
        })
    reference = next(row for row in output if row["method"] == "Readiness aligned")
    for row in output:
        if row["method"] != "Readiness aligned":
            row["ra_test_energy_saving_fraction"] = (
                float(row["test_energy_j"]) - float(reference["test_energy_j"])
            ) / float(row["test_energy_j"])
    _write_rows(target_summary_path, output)
    return output


def run_pareto_study(
    cfg: Config,
    output: Path,
    *,
    speed: float = 2.0,
    delay: int = 4,
    counts: tuple[int, ...] | None = None,
    control_period: int = 4,
    workers: int = 12,
) -> dict[str, object]:
    """Evaluate ranking methods at identical fixed cardinalities.

    The controller is deliberately removed from this mechanism test.  Every
    method receives the same commanded count and decision clock, so its
    energy--outage curve measures only the spatial ranker's consequence.
    """
    output.mkdir(parents=True, exist_ok=False)
    if counts is None:
        counts = tuple(range(1, cfg.subarrays + 1))
    counts = tuple(sorted({int(np.clip(value, 1, cfg.subarrays)) for value in counts}))
    jobs = [
        (cfg, speed, delay, trajectory, counts, control_period)
        for trajectory in range(cfg.test_trajectories)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [executor.submit(_job, *job) for job in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(
        key=lambda row: (
            int(row["commanded_count"]),
            int(row["trajectory"]),
            str(row["method"]),
        )
    )
    _write_rows(output / "raw_metrics.csv", rows)
    _write_rows(output / "summary.csv", _summary(cfg, rows))
    write_target_summary(
        output / "raw_metrics.csv", output / "target_summary.csv", cfg.outage_target
    )
    write_confident_target_summary(
        output / "summary.csv",
        output / "raw_metrics.csv",
        output / "confidence_target_summary.csv",
        cfg.outage_target,
    )
    metadata = {
        "design": "fixed-cardinality energy--outage sweep with a fixed decision clock",
        "speed_mps": speed,
        "wake_delay_slots": delay,
        "control_period_slots": control_period,
        "commanded_counts": list(counts),
        "test_trajectories": cfg.test_trajectories,
        "methods": list(DISPLAY.values()),
        "confidence_interval": "trajectory percentile bootstrap (2000 resamples)",
        "fairness_control": "same trajectories, commanded count, decision epochs, channel model, and power model",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
