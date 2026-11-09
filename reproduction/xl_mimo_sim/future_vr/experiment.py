from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .belief import VisibilityBelief
from .channel import Trajectory, generate_trajectory
from .config import Config
from .controllers import (
    ConstrainedDQN,
    Transition,
    build_state,
    readiness_aligned_value,
    retention_count,
    select_top,
    symmetric_guard_mask,
)
from .system import HardwareState, XLSystem


METHODS = ("Full array", "Static HRNP", "Current-CSI", "Symmetric guard", "RA-FVV")
ABLATIONS = ("RA-FVV",)
MATCHED_METHODS = ("Current-CSI matched", "Future-only matched", "RA-FVV matched", "Oracle matched")


@dataclass
class EpisodeMetrics:
    method: str
    speed_mps: float
    wake_delay: int
    switch_scale: float
    seed: int
    energy_j: float
    delivered_mbit: float
    energy_efficiency_mbit_j: float
    outage_probability: float
    mean_shortfall: float
    p95_shortfall: float
    active_ratio: float
    transitions_per_s: float


def _prediction(cfg: Config, tracker: VisibilityBelief, hardware: HardwareState, requirements: np.ndarray,
                *, future: bool = True, readiness: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if future:
        gain, probability, uncertainty = tracker.predict(cfg.horizon)
    else:
        gain = tracker.gain_db[None].copy()
        probability = tracker.probability[None].copy()
        uncertainty = tracker.uncertainty[None].copy()
    score = readiness_aligned_value(
        cfg, gain, probability, uncertainty, hardware.readiness(), requirements,
        future=future, align_readiness=readiness,
    )
    return score, gain


def _variant_flags(method: str) -> tuple[bool, bool]:
    if method == "CSI-CRL":
        return False, False
    if method == "F-CRL":
        return True, False
    if method in {"RA-FVV", "RA-CRL"}:
        return True, True
    if method == "Vanilla symmetric DQN":
        return False, False
    return method != "No future", method != "No readiness"


def _symmetric_count_mask(score: np.ndarray, count: int) -> np.ndarray:
    """Contiguous symmetric selection centered on instantaneous spatial utility."""
    count = int(np.clip(count, 1, score.size))
    indices = np.arange(score.size, dtype=float)
    nonnegative = np.maximum(score - float(np.min(score)), 0.0) + 1e-9
    center = int(np.rint(np.sum(indices * nonnegative) / np.sum(nonnegative)))
    lo = int(np.clip(center - (count - 1) // 2, 0, score.size - count))
    mask = np.zeros(score.size, dtype=bool)
    mask[lo:lo + count] = True
    return mask


def _controller_state(
    method: str,
    cfg: Config,
    score: np.ndarray,
    hardware: HardwareState,
    requirements: np.ndarray,
    velocity: np.ndarray,
    last_outage: float,
) -> np.ndarray:
    if method != "Vanilla symmetric DQN":
        return build_state(
            cfg, score, hardware.active, hardware.command, hardware.readiness(),
            requirements, velocity, last_outage,
        )
    # Same input dimension and DQN capacity, but no forecast, motion, or RF
    # readiness information.  This makes the baseline blind to advance wake-up.
    state = build_state(
        cfg, score, hardware.active, hardware.command, np.zeros(cfg.subarrays), requirements,
        np.zeros(cfg.users), last_outage,
    )
    state[-1] = 0.0  # remove the explicit wake-up-delay feature
    return state


def _true_future_score(
    cfg: Config,
    trajectory: Trajectory,
    time_index: int,
    hardware: HardwareState,
    requirements: np.ndarray,
) -> np.ndarray:
    start = time_index + 1
    stop = min(start + cfg.horizon, cfg.episode_steps)
    gain = trajectory.gain_db[start:stop]
    visibility = trajectory.visibility[start:stop]
    if gain.shape[0] == 0:
        gain = trajectory.gain_db[-1:]
        visibility = trajectory.visibility[-1:]
    if gain.shape[0] < cfg.horizon:
        missing = cfg.horizon - gain.shape[0]
        gain = np.concatenate((gain, np.repeat(gain[-1:], missing, axis=0)))
        visibility = np.concatenate((visibility, np.repeat(visibility[-1:], missing, axis=0)))
    return readiness_aligned_value(
        cfg,
        gain,
        visibility,
        np.zeros_like(visibility),
        hardware.readiness(),
        requirements,
    )


def _method_command(
    method: str,
    cfg: Config,
    tracker: VisibilityBelief,
    hardware: HardwareState,
    requirements: np.ndarray,
    controller: ConstrainedDQN | None,
    trajectory: Trajectory,
    time_index: int,
    static_mask: np.ndarray,
    switch_scale: float,
    last_outage: float,
    forced_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None, int]:
    score, predicted_gain = _prediction(cfg, tracker, hardware, requirements)
    if method == "Predictive rate search":
        from .rate_search import rate_search_command
        command = rate_search_command(cfg, tracker, hardware, requirements, score)
        return command, score, None, int(command.sum())
    if method == "Full array":
        return np.ones(cfg.subarrays, dtype=bool), score, None, cfg.subarrays
    if method == "Static HRNP":
        return static_mask.copy(), score, None, int(static_mask.sum())
    if method == "Current-CSI":
        current_score, current_gain = _prediction(cfg, tracker, hardware, requirements, future=False)
        count = retention_count(current_gain[0], current_score, cfg.users, target=0.90)
        return select_top(current_score, count), current_score, None, count
    if method == "Symmetric guard":
        mask = symmetric_guard_mask(cfg, tracker.probability, tracker.velocity)
        return mask, score, None, int(mask.sum())
    if method in MATCHED_METHODS:
        if forced_count is None:
            raise ValueError(f"{method} requires a forced cardinality")
        count = int(forced_count)
        if method == "Current-CSI matched":
            matched_score, _ = _prediction(cfg, tracker, hardware, requirements, future=False, readiness=False)
        elif method == "Future-only matched":
            matched_score, _ = _prediction(cfg, tracker, hardware, requirements, readiness=False)
        elif method == "Oracle matched":
            matched_score = _true_future_score(cfg, trajectory, time_index, hardware, requirements)
        else:
            matched_score = score
        return select_top(matched_score, count), matched_score, None, count

    future, aligned = _variant_flags(method)
    score, _ = _prediction(cfg, tracker, hardware, requirements, future=future, readiness=aligned)
    state = _controller_state(
        method, cfg, score, hardware, requirements, tracker.velocity, last_outage,
    )
    if controller is None:
        raise ValueError(f"{method} requires a trained controller")
    action, count = controller.choose(state, epsilon=0.0)
    command = _symmetric_count_mask(score, count) if method == "Vanilla symmetric DQN" else select_top(score, count)
    return command, score, action, count


def simulate_episode(
    cfg: Config,
    trajectory: Trajectory,
    method: str,
    controller: ConstrainedDQN | None,
    switch_scale: float,
    seed: int,
    *,
    training: bool = False,
    epsilon: float = 0.0,
    forced_counts: np.ndarray | None = None,
    control_period: int | None = None,
    static_mask: np.ndarray | None = None,
) -> tuple[EpisodeMetrics, dict[str, np.ndarray]]:
    system = XLSystem(cfg)
    hardware = HardwareState(cfg)
    tracker = VisibilityBelief(cfg, trajectory.gain_db[0])
    if method == "Predictive rate search":
        tracker.channel_estimate = trajectory.channel[0].copy()
    if method == "Static HRNP" and static_mask is None:
        raise ValueError("Static HRNP requires a mask estimated on disjoint calibration trajectories")
    if static_mask is None:
        static_mask = np.zeros(cfg.subarrays, dtype=bool)
    energy: list[float] = []
    delivered: list[float] = []
    outages: list[float] = []
    shortfalls: list[float] = []
    active_ratios: list[float] = []
    transitions: list[int] = []
    action_counts: list[int] = []
    losses: list[float] = []
    last_outage = 0.0
    pending_transition: tuple[np.ndarray, int, float, float, int] | None = None
    full_step_energy = system.evaluate(
        trajectory.channel[0], np.ones(cfg.subarrays, dtype=bool),
        np.zeros(cfg.subarrays, dtype=bool), 0.0, False,
        trajectory.requirements_mbps[0], observed_count=cfg.subarrays,
    ).energy_j
    decision_period = int(control_period or max(cfg.wake_delay_slots, 1))

    for t in range(cfg.episode_steps):
        scan = False
        observed = hardware.active.copy()
        tracker.update(trajectory.gain_db[t], observed)
        if method == "Predictive rate search":
            antenna_observed = np.repeat(observed, cfg.antennas_per_subarray)
            tracker.channel_estimate[:, antenna_observed] = trajectory.channel[t][:, antenna_observed]
        decision_epoch = t % decision_period == 0
        if training and decision_epoch:
            future, aligned = _variant_flags(method)
            score, _ = _prediction(
                cfg, tracker, hardware, trajectory.requirements_mbps[t],
                future=future, readiness=aligned,
            )
            state = _controller_state(
                method, cfg, score, hardware, trajectory.requirements_mbps[t], tracker.velocity,
                last_outage,
            )
        if training and controller is not None and pending_transition is not None and decision_epoch:
            previous_state, previous_action, energy_cost, outage_cost, slot_count = pending_transition
            controller.remember(Transition(
                previous_state, previous_action, energy_cost / slot_count,
                outage_cost / slot_count, state, False,
            ))
            losses.append(controller.learn())
            pending_transition = None

        if training:
            if controller is None:
                raise ValueError("training requires a controller")
            if decision_epoch:
                action, count = controller.choose(state, epsilon)
                command = (
                    _symmetric_count_mask(score, count)
                    if method == "Vanilla symmetric DQN" else select_top(score, count)
                )
            else:
                action = None
                command = hardware.command.copy()
                count = int(command.sum())
        else:
            if not decision_epoch:
                command = hardware.command.copy()
                action = None
                count = int(command.sum())
            else:
                command, score, action, count = _method_command(
                    method, cfg, tracker, hardware, trajectory.requirements_mbps[t], controller,
                    trajectory, t, static_mask, switch_scale, last_outage,
                    None if forced_counts is None else int(forced_counts[t]),
                )

        available, switching_energy, transition_count = hardware.apply(command, switch_scale)
        result = system.evaluate(
            trajectory.channel[t], available, hardware.countdown > 0, switching_energy, scan,
            trajectory.requirements_mbps[t],
            observed_count=int(observed.sum()),
        )
        hardware.tick()
        energy.append(result.energy_j)
        delivered.append(result.delivered_mbit)
        outages.append(result.outage)
        shortfalls.append(result.shortfall)
        active_ratios.append(float(available.mean()))
        transitions.append(transition_count)
        action_counts.append(count)

        if training and controller is not None:
            energy_cost = float(result.energy_j / max(full_step_energy, 1e-9))
            if action is not None:
                pending_transition = (state, action, energy_cost, float(result.outage), 1)
            elif pending_transition is not None:
                old_state, old_action, old_energy, old_outage, old_slots = pending_transition
                pending_transition = (
                    old_state, old_action, old_energy + energy_cost,
                    old_outage + float(result.outage), old_slots + 1,
                )
        last_outage = result.outage

    if training and controller is not None and pending_transition is not None:
        previous_state, previous_action, energy_cost, outage_cost, slot_count = pending_transition
        controller.remember(Transition(
            previous_state, previous_action, energy_cost / slot_count,
            outage_cost / slot_count, previous_state, True,
        ))
        losses.append(controller.learn())

    total_energy = float(np.sum(energy))
    total_delivered = float(np.sum(delivered))
    metric = EpisodeMetrics(
        method=method,
        speed_mps=float(np.mean(np.linalg.norm(trajectory.velocity_xy, axis=2))),
        wake_delay=cfg.wake_delay_slots,
        switch_scale=float(switch_scale),
        seed=seed,
        energy_j=total_energy,
        delivered_mbit=total_delivered,
        energy_efficiency_mbit_j=total_delivered / max(total_energy, 1e-9),
        outage_probability=float(np.mean(outages)),
        mean_shortfall=float(np.mean(shortfalls)),
        p95_shortfall=float(np.quantile(shortfalls, 0.95)),
        active_ratio=float(np.mean(active_ratios)),
        transitions_per_s=float(np.sum(transitions) / (cfg.episode_steps * cfg.slot_s)),
    )
    traces = {
        "energy_j": np.asarray(energy), "outage": np.asarray(outages), "shortfall": np.asarray(shortfalls),
        "active_ratio": np.asarray(active_ratios), "transitions": np.asarray(transitions),
        "action_count": np.asarray(action_counts), "loss": np.asarray(losses),
    }
    return metric, traces


def train_controller(
    cfg: Config,
    seed: int,
    method: str = "RA-FVV",
    *,
    dual_mode: str = "adaptive",
    dual_value: float | None = None,
) -> tuple[ConstrainedDQN, list[dict[str, float]]]:
    """Train a cardinality controller under a specified constraint treatment.

    ``adaptive`` performs the proposed episodic dual update. ``fixed`` keeps
    the supplied multiplier unchanged, and ``none`` removes both the outage
    multiplier unchanged, and ``none`` removes the outage multiplier.  The
    latter two modes are used only as controlled learning baselines.
    """
    if dual_mode not in {"adaptive", "fixed", "none"}:
        raise ValueError(f"unknown dual mode: {dual_mode}")
    dummy_dim = 4 * cfg.subarrays + 2 * cfg.users + 2
    controller = ConstrainedDQN(cfg, dummy_dim, seed)
    if dual_mode == "none":
        controller.dual = 0.0
    elif dual_value is not None:
        controller.dual = float(dual_value)
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed + cfg.seed)
    for episode in range(cfg.train_episodes):
        delay = int(rng.choice(cfg.wake_delays))
        episode_cfg = cfg.with_delay(delay)
        speed = float(rng.choice(cfg.speeds_mps))
        switch_scale = 1.0
        trajectory = generate_trajectory(episode_cfg, seed * 100_000 + episode, speed)
        fraction = episode / max(cfg.train_episodes - 1, 1)
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * np.exp(-4.0 * fraction)
        metric, trace = simulate_episode(
            episode_cfg, trajectory, method, controller, switch_scale, seed,
            training=True, epsilon=epsilon,
        )
        if dual_mode == "adaptive":
            controller.update_dual(metric.outage_probability)
        history.append({
            "episode": episode, "energy_j": metric.energy_j, "outage": metric.outage_probability,
            "dual": controller.dual, "active_ratio": metric.active_ratio,
            "loss": float(np.mean(trace["loss"])) if trace["loss"].size else 0.0,
            "dual_mode": dual_mode,
        })
    return controller, history


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_grid(
    cfg: Config, output: Path
) -> tuple[list[EpisodeMetrics], list[dict[str, float]], dict[str, list[ConstrainedDQN]]]:
    output.mkdir(parents=True, exist_ok=True)
    controllers: dict[str, list[ConstrainedDQN]] = {method: [] for method in ABLATIONS}
    training_history: list[dict[str, float]] = []
    for method in ABLATIONS:
        for controller_seed in cfg.controller_seeds:
            controller, history = train_controller(cfg, controller_seed, method)
            controllers[method].append(controller)
            for row in history:
                training_history.append({"method": method, "controller_seed": controller_seed, **row})

    metrics: list[EpisodeMetrics] = []
    # Factor-at-a-time design resolves mobility and wake-up latency.  The same
    # random stream is reused across factor levels (common random numbers).
    scenarios: set[tuple[float, int, float]] = set()
    for speed in cfg.speeds_mps:
        scenarios.add((speed, cfg.wake_delay_slots, 1.0))
    for delay in cfg.wake_delays:
        scenarios.add((float(cfg.speeds_mps[len(cfg.speeds_mps) // 2]), delay, 1.0))

    calibration_masks: dict[float, np.ndarray] = {}
    for speed in cfg.speeds_mps:
        calibration_gain = []
        for index in range(cfg.test_trajectories):
            calibration = generate_trajectory(cfg, cfg.seed * 90_000 + index, float(speed))
            calibration_gain.append(calibration.gain_db.mean(axis=(0, 2)))
        mean_gain = np.mean(calibration_gain, axis=0)
        calibration_masks[float(speed)] = select_top(
            mean_gain, max(cfg.users + 1, int(np.ceil(0.65 * cfg.subarrays)))
        )

    for speed, delay, scale in sorted(scenarios):
        local_cfg = cfg.with_delay(delay)
        for trajectory_index in range(cfg.test_trajectories):
            test_seed = cfg.seed * 100_000 + trajectory_index
            trajectory = generate_trajectory(local_cfg, test_seed, speed)
            ra_counts: np.ndarray | None = None
            for method in METHODS:
                method_controllers = controllers["RA-FVV"]
                if method == "RA-FVV":
                    repetitions: list[EpisodeMetrics] = []
                    count_traces: list[np.ndarray] = []
                    for controller in method_controllers:
                        metric, trace = simulate_episode(
                            local_cfg, trajectory, method, controller, scale, test_seed
                        )
                        repetitions.append(metric)
                        count_traces.append(trace["action_count"])
                    averaged = {
                        field: float(np.mean([getattr(item, field) for item in repetitions]))
                        for field in (
                            "energy_j", "delivered_mbit", "energy_efficiency_mbit_j",
                            "outage_probability", "mean_shortfall", "p95_shortfall",
                            "active_ratio", "transitions_per_s",
                        )
                    }
                    metric = EpisodeMetrics(
                        method=method, speed_mps=speed, wake_delay=delay,
                        switch_scale=scale, seed=test_seed, **averaged,
                    )
                    ra_counts = np.rint(np.median(np.stack(count_traces), axis=0)).astype(int)
                else:
                    metric, _ = simulate_episode(
                        local_cfg, trajectory, method, None, scale, test_seed,
                        static_mask=calibration_masks[speed] if method == "Static HRNP" else None,
                    )
                    metric.speed_mps = speed
                metrics.append(metric)
            if scale == 1.0 and ra_counts is not None:
                for method in MATCHED_METHODS:
                    metric, _ = simulate_episode(
                        local_cfg, trajectory, method, None, scale, test_seed, forced_counts=ra_counts
                    )
                    metric.speed_mps = speed
                    metrics.append(metric)
            # Ablations are required only at the central paper operating point.
            central_speed = float(cfg.speeds_mps[len(cfg.speeds_mps) // 2])
            if speed == central_speed and scale == 1.0:
                for method in ABLATIONS[1:]:
                    method_controllers = controllers[method]
                    controller = method_controllers[trajectory_index % len(method_controllers)]
                    metric, _ = simulate_episode(local_cfg, trajectory, method, controller, scale, test_seed)
                    metric.speed_mps = speed
                    metrics.append(metric)

    _write_rows(output / "episode_metrics.csv", [asdict(row) for row in metrics])
    _write_rows(output / "training_history.csv", training_history)
    return metrics, training_history, controllers


def run_fixed_budget_study(cfg: Config, output: Path, commanded_count: int | None = None) -> list[EpisodeMetrics]:
    """Isolate spatial ranking under the same strict subarray budget.

    The study varies one condition at a time: speed at the largest wake-up
    delay and wake-up delay at 3 m/s.  No learned controller is used, so the
    comparison measures ranking rather than cardinality policy differences.
    """
    count = commanded_count or min(cfg.subarrays, cfg.users + 3)
    speeds = tuple(dict.fromkeys((*cfg.speeds_mps, 6.0)))
    speed_for_delay = 3.0
    scenarios = {(float(speed), int(max(cfg.wake_delays))) for speed in speeds}
    scenarios.update((speed_for_delay, int(delay)) for delay in cfg.wake_delays)
    metrics: list[EpisodeMetrics] = []
    for speed, delay in sorted(scenarios):
        local_cfg = cfg.with_delay(delay)
        forced_counts = np.full(local_cfg.episode_steps, count, dtype=int)
        for trajectory_index in range(cfg.test_trajectories):
            test_seed = cfg.seed * 200_000 + trajectory_index
            trajectory = generate_trajectory(local_cfg, test_seed, speed)
            for method in MATCHED_METHODS:
                metric, _ = simulate_episode(
                    local_cfg, trajectory, method, None, 1.0, test_seed, forced_counts=forced_counts
                )
                metric.speed_mps = speed
                metrics.append(metric)
    _write_rows(output / "fixed_budget_metrics.csv", [asdict(row) for row in metrics])
    _write_rows(output / "fixed_budget_summary.csv", confidence_summary(metrics))
    return metrics


def confidence_summary(metrics: list[EpisodeMetrics]) -> list[dict[str, object]]:
    keys = ("method", "speed_mps", "wake_delay", "switch_scale")
    grouped: dict[tuple[object, ...], list[EpisodeMetrics]] = {}
    for metric in metrics:
        grouped.setdefault(tuple(getattr(metric, key) for key in keys), []).append(metric)
    fields = ("energy_j", "energy_efficiency_mbit_j", "outage_probability", "active_ratio", "transitions_per_s")
    rows: list[dict[str, object]] = []
    for key, values in grouped.items():
        row: dict[str, object] = dict(zip(keys, key))
        row["n"] = len(values)
        for field in fields:
            data = np.asarray([getattr(value, field) for value in values])
            row[field] = float(data.mean())
            row[field + "_ci95"] = float(1.96 * data.std(ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0
        rows.append(row)
    return rows


def prediction_diagnostics(cfg: Config, seed: int, speed: float) -> dict[str, np.ndarray]:
    trajectory = generate_trajectory(cfg, seed, speed)
    tracker = VisibilityBelief(cfg, trajectory.gain_db[0])
    hardware = HardwareState(cfg)
    rmse_future = np.zeros(cfg.horizon)
    rmse_hold = np.zeros(cfg.horizon)
    brier_future = np.zeros(cfg.horizon)
    count = np.zeros(cfg.horizon)
    snapshot: dict[str, np.ndarray] = {}
    for t in range(cfg.episode_steps - cfg.horizon):
        observed = hardware.active.copy()
        tracker.update(trajectory.gain_db[t], observed)
        gain, probability, uncertainty = tracker.predict(cfg.horizon)
        for h in range(cfg.horizon):
            true_gain = trajectory.gain_db[t + h + 1]
            rmse_future[h] += np.mean((gain[h] - true_gain) ** 2)
            rmse_hold[h] += np.mean((tracker.gain_db - true_gain) ** 2)
            true_visible = (trajectory.visibility[t + h + 1] >= 0.35).astype(float)
            brier_future[h] += np.mean((probability[h] - true_visible) ** 2)
            count[h] += 1
        if t == min(12, cfg.episode_steps - cfg.horizon - 1):
            readiness = hardware.readiness()
            linear = 10.0 ** ((gain - gain.max(axis=1, keepdims=True)) / 10.0)
            demand_weight = trajectory.requirements_mbps[t] / max(
                float(np.mean(trajectory.requirements_mbps[t])), 1e-9
            )
            conservative_probability = np.maximum(
                probability - cfg.uncertainty_penalty * uncertainty, 0.0
            )
            discount = cfg.visibility_discount ** np.arange(cfg.horizon)
            slot_utility = np.sum(
                conservative_probability * linear * demand_weight[None, None, :], axis=2
            ) * discount[:, None]
            readiness_mask = np.zeros_like(slot_utility, dtype=bool)
            for q in range(cfg.subarrays):
                first_index = min(max(int(readiness[q]) - 1, 0), cfg.horizon - 1)
                readiness_mask[first_index:, q] = True
            aligned_utility = np.where(readiness_mask, slot_utility, np.nan)
            scale = max(float(np.nanmax(aligned_utility)), 1e-9)
            snapshot = {
                "true_gain": trajectory.gain_db[:, :, 0], "prediction": gain[:, :, 0],
                "snapshot_time": np.array([t]), "user_x": trajectory.user_xy[:, 0, 0],
                "slot_utility": slot_utility / scale,
                "aligned_utility": aligned_utility / scale,
                "snapshot_readiness": readiness,
            }
        score, _ = _prediction(
            cfg, tracker, hardware, trajectory.requirements_mbps[t], future=True, readiness=True,
        )
        command = select_top(score, min(cfg.subarrays, cfg.users + 3))
        hardware.apply(command, 1.0)
        hardware.tick()
    return {
        "rmse_future": np.sqrt(rmse_future / np.maximum(count, 1)),
        "rmse_hold": np.sqrt(rmse_hold / np.maximum(count, 1)),
        "brier_future": brier_future / np.maximum(count, 1),
        **snapshot,
    }
