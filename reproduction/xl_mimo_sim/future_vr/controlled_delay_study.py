from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import Config
from .experiment import _write_rows
from .matched_policy_study import _hierarchical_ci
from .pareto_study import DISPLAY, FIELDS, _job


def _summaries(
    cfg: Config, rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(cfg.seed + 3317)
    summary: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    methods = tuple(DISPLAY.values())
    for delay in cfg.wake_delays:
        selected_delay = [row for row in rows if int(row["wake_delay"]) == delay]
        matrices: dict[tuple[str, str], np.ndarray] = {}
        for method in methods:
            item: dict[str, object] = {
                "method": method,
                "wake_delay": delay,
                "speed_mps": float(selected_delay[0]["requested_speed_mps"]),
                "commanded_count": int(selected_delay[0]["commanded_count"]),
                "control_period": int(selected_delay[0]["control_period"]),
                "test_trajectories": cfg.test_trajectories,
            }
            method_rows = [row for row in selected_delay if str(row["method"]) == method]
            for field in FIELDS:
                values = np.asarray([float(row[field]) for row in method_rows], dtype=float)[None, :]
                matrices[(method, field)] = values
                low, high = _hierarchical_ci(values, rng)
                item[field] = float(values.mean())
                item[field + "_ci95_low"] = low
                item[field + "_ci95_high"] = high
            summary.append(item)

        for baseline in ("Current CSI", "Unaligned future"):
            item = {"baseline": baseline, "wake_delay": delay}
            for field in ("outage_probability", "energy_j"):
                difference = matrices[(baseline, field)] - matrices[("Readiness aligned", field)]
                low, high = _hierarchical_ci(difference, rng)
                item[field + "_reduction"] = float(difference.mean())
                item[field + "_reduction_ci95_low"] = low
                item[field + "_reduction_ci95_high"] = high
            paired.append(item)
    return summary, paired


def run_controlled_delay_study(
    cfg: Config,
    output: Path,
    *,
    speed: float = 2.0,
    count: int = 5,
    control_period: int = 4,
    workers: int = 20,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    jobs = [
        (cfg, speed, int(delay), trajectory, (count,), control_period)
        for delay in cfg.wake_delays
        for trajectory in range(cfg.test_trajectories)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [executor.submit(_job, *job) for job in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(
        key=lambda row: (
            int(row["wake_delay"]), int(row["trajectory"]), str(row["method"])
        )
    )
    _write_rows(output / "raw_metrics.csv", rows)
    summary, paired = _summaries(cfg, rows)
    _write_rows(output / "summary.csv", summary)
    _write_rows(output / "paired_summary.csv", paired)
    metadata = {
        "design": "wake-up-delay sweep with fixed cardinality and fixed decision clock",
        "speed_mps": speed,
        "commanded_count": count,
        "control_period_slots": control_period,
        "wake_delays": list(cfg.wake_delays),
        "test_trajectories": cfg.test_trajectories,
        "methods": list(DISPLAY.values()),
        "confidence_interval": "paired trajectory percentile bootstrap (2000 resamples)",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
