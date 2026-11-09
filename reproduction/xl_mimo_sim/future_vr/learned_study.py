from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .channel import generate_trajectory
from .config import Config
from .controllers import ConstrainedDQN
from .experiment import (
    EpisodeMetrics,
    _write_rows,
    confidence_summary,
    simulate_episode,
    train_controller,
)


VARIANTS = ("CSI-CRL", "F-CRL", "RA-FVV")
DISPLAY_NAME = {"CSI-CRL": "CSI-CRL", "F-CRL": "F-CRL", "RA-FVV": "RA-CRL"}
_EVAL_CONTROLLERS: dict[str, list[tuple[int, ConstrainedDQN]]] = {}


def _checkpoint_path(directory: Path, method: str, seed: int) -> Path:
    return directory / f"{DISPLAY_NAME[method].lower().replace('-', '_')}_seed{seed}.pt"


def _train_job(cfg: Config, method: str, seed: int, checkpoint: Path) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    controller, history = train_controller(cfg, seed, method)
    torch.save(
        {
            "method": method,
            "seed": seed,
            "online": controller.online.state_dict(),
            "target": controller.target.state_dict(),
            "dual": controller.dual,
        },
        checkpoint,
    )
    return [
        {"method": DISPLAY_NAME[method], "controller_seed": seed, **row}
        for row in history
    ]


def _load_controller(cfg: Config, method: str, seed: int, checkpoint: Path) -> ConstrainedDQN:
    state_dim = 4 * cfg.subarrays + 2 * cfg.users + 2
    controller = ConstrainedDQN(cfg, state_dim, seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    controller.online.load_state_dict(payload["online"])
    controller.target.load_state_dict(payload["target"])
    controller.dual = float(payload["dual"])
    return controller


def _init_eval_worker(cfg: Config, specs: list[tuple[str, int, str]]) -> None:
    global _EVAL_CONTROLLERS
    torch.set_num_threads(1)
    _EVAL_CONTROLLERS = {method: [] for method in VARIANTS}
    for method, seed, checkpoint in specs:
        _EVAL_CONTROLLERS[method].append(
            (seed, _load_controller(cfg, method, seed, Path(checkpoint)))
        )


def _evaluate_job(
    cfg: Config,
    speed: float,
    delay: int,
    trajectory_index: int,
) -> tuple[list[dict[str, object]], list[EpisodeMetrics], list[dict[str, object]]]:
    local_cfg = cfg.with_delay(delay)
    test_seed = cfg.seed * 410_000 + trajectory_index
    trajectory = generate_trajectory(local_cfg, test_seed, speed)
    raw: list[dict[str, object]] = []
    averaged: list[EpisodeMetrics] = []
    actions: list[dict[str, object]] = []
    central_speed = float(cfg.speeds_mps[len(cfg.speeds_mps) // 2])
    record_actions = np.isclose(speed, central_speed) and delay == cfg.wake_delay_slots

    for method in VARIANTS:
        repetitions: list[EpisodeMetrics] = []
        for controller_seed, controller in _EVAL_CONTROLLERS[method]:
            metric, trace = simulate_episode(
                local_cfg, trajectory, method, controller, 1.0, test_seed
            )
            repetitions.append(metric)
            raw.append(
                {
                    **asdict(metric),
                    "method": DISPLAY_NAME[method],
                    "trajectory": trajectory_index,
                    "controller_seed": controller_seed,
                }
            )
            if record_actions:
                for decision_index, count in enumerate(trace["action_count"]):
                    actions.append(
                        {
                            "method": DISPLAY_NAME[method],
                            "trajectory": trajectory_index,
                            "controller_seed": controller_seed,
                            "decision": decision_index,
                            "commanded_count": int(count),
                        }
                    )
        mean_values = {
            field: float(np.mean([getattr(metric, field) for metric in repetitions]))
            for field in (
                "energy_j",
                "delivered_mbit",
                "energy_efficiency_mbit_j",
                "outage_probability",
                "mean_shortfall",
                "p95_shortfall",
                "active_ratio",
                "transitions_per_s",
            )
        }
        averaged.append(
            EpisodeMetrics(
                method=DISPLAY_NAME[method],
                speed_mps=speed,
                wake_delay=delay,
                switch_scale=1.0,
                seed=test_seed,
                **mean_values,
            )
        )
    return raw, averaged, actions


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _paired_summary(metrics: list[EpisodeMetrics]) -> list[dict[str, object]]:
    by_key = {
        (row.method, row.speed_mps, row.wake_delay, row.seed): row for row in metrics
    }
    rows: list[dict[str, object]] = []
    for baseline in ("CSI-CRL", "F-CRL"):
        scenarios = sorted({(row.speed_mps, row.wake_delay) for row in metrics})
        for speed, delay in scenarios:
            energy_delta: list[float] = []
            outage_delta: list[float] = []
            for seed in sorted({row.seed for row in metrics}):
                base = by_key.get((baseline, speed, delay, seed))
                proposed = by_key.get(("RA-CRL", speed, delay, seed))
                if base is None or proposed is None:
                    continue
                energy_delta.append(base.energy_j - proposed.energy_j)
                outage_delta.append(base.outage_probability - proposed.outage_probability)
            item: dict[str, object] = {
                "baseline": baseline,
                "speed_mps": speed,
                "wake_delay": delay,
                "n": len(energy_delta),
            }
            for name, values in (("energy_saved_j", energy_delta), ("outage_reduction", outage_delta)):
                array = np.asarray(values, dtype=float)
                item[name] = float(array.mean())
                item[name + "_ci95"] = (
                    float(1.96 * array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
                )
            rows.append(item)
    return rows


def run_learned_study(cfg: Config, output: Path, workers: int = 8) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    (output / "config.json").write_text(
        json.dumps(cfg.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    training_rows: list[dict[str, object]] = []
    jobs = [
        (method, seed, _checkpoint_path(checkpoint_dir, method, seed))
        for method in VARIANTS
        for seed in cfg.controller_seeds
    ]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(_train_job, cfg, method, seed, checkpoint): (method, seed)
            for method, seed, checkpoint in jobs
        }
        for future in as_completed(futures):
            training_rows.extend(future.result())
    training_rows.sort(key=lambda row: (str(row["method"]), int(row["controller_seed"]), int(row["episode"])))
    _write_rows(output / "training_history.csv", training_rows)

    scenarios = {(float(speed), cfg.wake_delay_slots) for speed in cfg.speeds_mps}
    central_speed = float(cfg.speeds_mps[len(cfg.speeds_mps) // 2])
    scenarios.update((central_speed, int(delay)) for delay in cfg.wake_delays)
    specs = [(method, seed, str(checkpoint)) for method, seed, checkpoint in jobs]
    raw_rows: list[dict[str, object]] = []
    averaged_metrics: list[EpisodeMetrics] = []
    action_rows: list[dict[str, object]] = []
    eval_jobs = [
        (speed, delay, trajectory_index)
        for speed, delay in sorted(scenarios)
        for trajectory_index in range(cfg.test_trajectories)
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_eval_worker,
        initargs=(cfg, specs),
    ) as executor:
        futures = [
            executor.submit(_evaluate_job, cfg, speed, delay, trajectory_index)
            for speed, delay, trajectory_index in eval_jobs
        ]
        for future in as_completed(futures):
            raw, averaged, actions = future.result()
            raw_rows.extend(raw)
            averaged_metrics.extend(averaged)
            action_rows.extend(actions)

    raw_rows.sort(key=lambda row: (float(row["speed_mps"]), int(row["wake_delay"]),
                                    int(row["trajectory"]), str(row["method"]),
                                    int(row["controller_seed"])))
    averaged_metrics.sort(key=lambda row: (row.speed_mps, row.wake_delay, row.seed, row.method))
    action_rows.sort(key=lambda row: (str(row["method"]), int(row["controller_seed"]),
                                      int(row["trajectory"]), int(row["decision"])))
    _write_rows(output / "raw_metrics.csv", raw_rows)
    _write_rows(output / "trajectory_metrics.csv", [asdict(row) for row in averaged_metrics])
    _write_rows(output / "summary.csv", confidence_summary(averaged_metrics))
    _write_rows(output / "paired_summary.csv", _paired_summary(averaged_metrics))
    _write_rows(output / "action_counts.csv", action_rows)
    metadata = {
        "variants": [DISPLAY_NAME[item] for item in VARIANTS],
        "controller_seeds": list(cfg.controller_seeds),
        "training_episodes": cfg.train_episodes,
        "test_trajectories": cfg.test_trajectories,
        "workers": workers,
        "paired_unit": "trajectory after averaging controller seeds",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
