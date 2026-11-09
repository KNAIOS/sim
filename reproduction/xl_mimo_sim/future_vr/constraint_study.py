from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .channel import generate_trajectory
from .config import Config
from .experiment import _write_rows, simulate_episode, train_controller


LEARNING_MODES = {
    "RA-CRL": {
        "method": "RA-FVV", "dual_mode": "adaptive", "dual_value": None,
    },
    "Vanilla symmetric DQN": {
        "method": "Vanilla symmetric DQN", "dual_mode": "adaptive", "dual_value": None,
    },
    "Energy-only DQN": {
        "method": "RA-FVV", "dual_mode": "none", "dual_value": 0.0,
    },
}


def _summarize_trajectory_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "energy_j",
        "delivered_mbit",
        "energy_efficiency_mbit_j",
        "outage_probability",
        "active_ratio",
        "transitions_per_s",
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summary: list[dict[str, object]] = []
    for method, values in grouped.items():
        item: dict[str, object] = {"method": method, "n": len(values)}
        for field in fields:
            data = np.asarray([float(row[field]) for row in values])
            item[field] = float(data.mean())
            item[field + "_ci95"] = (
                float(1.96 * data.std(ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0
            )
        summary.append(item)
    return summary


def _evaluate_fixed_count(
    cfg: Config,
    speed: float,
    seeds: list[int],
    count: int,
) -> list[dict[str, object]]:
    forced_counts = np.full(cfg.episode_steps, count, dtype=int)
    rows: list[dict[str, object]] = []
    for trajectory_index, seed in enumerate(seeds):
        trajectory = generate_trajectory(cfg, seed, speed)
        metric, _ = simulate_episode(
            cfg,
            trajectory,
            "RA-FVV matched",
            None,
            1.0,
            seed,
            forced_counts=forced_counts,
            control_period=max(cfg.wake_delay_slots, 1),
        )
        row = asdict(metric)
        row.update({"method": f"Fixed $M={count}$", "trajectory": trajectory_index})
        rows.append(row)
    return rows


def run_constraint_study(cfg: Config, output: Path) -> dict[str, object]:
    """Test whether the adaptive multiplier is necessary and effective.

    Controllers receive identical training budgets and randomized operating
    conditions.  During evaluation, the three independently trained networks
    are run on every held-out trajectory; their metrics are averaged per
    trajectory before confidence intervals are computed.  A fixed-cardinality
    comparator is selected on a disjoint validation set under the same 5%
    outage requirement and then evaluated on the held-out test set.
    """
    output.mkdir(parents=True, exist_ok=True)
    # The nominal operating point is deliberately easy enough that the 5%
    # constraint is often inactive.  A separately labelled load-stress test is
    # therefore used to exercise, rather than merely display, the dual update.
    demand_scale = 1.15
    study_cfg = replace(
        cfg,
        demand_levels_mbps=tuple(level * demand_scale for level in cfg.demand_levels_mbps),
    )
    central_speed = float(study_cfg.speeds_mps[len(study_cfg.speeds_mps) // 2])
    local_cfg = study_cfg.with_delay(study_cfg.wake_delay_slots)

    controllers: dict[str, list[object]] = {}
    history_rows: list[dict[str, object]] = []
    for label, settings in LEARNING_MODES.items():
        controllers[label] = []
        for controller_seed in cfg.controller_seeds:
            controller, history = train_controller(
                study_cfg,
                controller_seed,
                str(settings["method"]),
                dual_mode=str(settings["dual_mode"]),
                dual_value=settings["dual_value"],
            )
            controllers[label].append(controller)
            for row in history:
                history_rows.append({"method": label, "controller_seed": controller_seed, **row})

    validation_seeds = [cfg.seed * 320_000 + index for index in range(cfg.test_trajectories)]
    candidates: list[dict[str, object]] = []
    candidate_rows: dict[int, list[dict[str, object]]] = {}
    for count in range(cfg.users + 1, cfg.subarrays + 1):
        values = _evaluate_fixed_count(local_cfg, central_speed, validation_seeds, count)
        candidate_rows[count] = values
        outage_values = np.asarray([float(row["outage_probability"]) for row in values])
        outage_ci = (
            float(1.96 * outage_values.std(ddof=1) / np.sqrt(outage_values.size))
            if outage_values.size > 1 else 0.0
        )
        candidates.append({
            "count": count,
            "energy_j": float(np.mean([row["energy_j"] for row in values])),
            "outage_probability": float(outage_values.mean()),
            "outage_probability_ci95": outage_ci,
        })
    feasible = [
        row for row in candidates
        if float(row["outage_probability"]) + float(row["outage_probability_ci95"]) <= cfg.outage_target
    ]
    selected = min(feasible, key=lambda row: float(row["energy_j"])) if feasible else min(
        candidates, key=lambda row: float(row["outage_probability"])
    )
    selected_count = int(selected["count"])

    test_seeds = [cfg.seed * 330_000 + index for index in range(cfg.test_trajectories)]
    trajectory_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for trajectory_index, seed in enumerate(test_seeds):
        trajectory = generate_trajectory(local_cfg, seed, central_speed)
        for label, mode_controllers in controllers.items():
            repetitions: list[dict[str, object]] = []
            for controller_seed, controller in zip(cfg.controller_seeds, mode_controllers):
                metric, _ = simulate_episode(
                    local_cfg, trajectory, str(LEARNING_MODES[label]["method"]), controller, 1.0, seed
                )
                row = asdict(metric)
                row.update({
                    "method": label,
                    "trajectory": trajectory_index,
                    "controller_seed": controller_seed,
                })
                repetitions.append(row)
                raw_rows.append(row)
            averaged = {
                key: float(np.mean([float(row[key]) for row in repetitions]))
                for key in (
                    "energy_j", "delivered_mbit", "energy_efficiency_mbit_j",
                    "outage_probability", "mean_shortfall", "p95_shortfall",
                    "active_ratio", "transitions_per_s",
                )
            }
            trajectory_rows.append({
                "method": label,
                "trajectory": trajectory_index,
                "speed_mps": central_speed,
                "wake_delay": local_cfg.wake_delay_slots,
                "switch_scale": 1.0,
                "seed": seed,
                **averaged,
            })

    fixed_rows = _evaluate_fixed_count(local_cfg, central_speed, test_seeds, selected_count)
    for row in fixed_rows:
        row["method"] = f"QoS-matched fixed $M={selected_count}$"
    trajectory_rows.extend(fixed_rows)

    summary = _summarize_trajectory_rows(trajectory_rows)
    _write_rows(output / "constraint_training.csv", history_rows)
    _write_rows(output / "constraint_raw_metrics.csv", raw_rows)
    _write_rows(output / "constraint_metrics.csv", trajectory_rows)
    _write_rows(output / "constraint_summary.csv", summary)
    _write_rows(output / "fixed_count_validation.csv", candidates)
    metadata = {
        "speed_mps": central_speed,
        "wake_delay_slots": local_cfg.wake_delay_slots,
        "outage_target": cfg.outage_target,
        "selected_fixed_count": selected_count,
        "controller_seeds": list(cfg.controller_seeds),
        "test_trajectories": cfg.test_trajectories,
        "demand_scale": demand_scale,
    }
    (output / "constraint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
