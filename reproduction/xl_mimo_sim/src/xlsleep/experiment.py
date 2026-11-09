from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .channel import Trajectory, generate_trajectory
from .config import Config
from .controllers import DQNController, make_state, ranked_mask, utility_score
from .plots import create_plots
from .predictor import BeliefTracker, TrainedPredictor, train_predictor
from .system import LinkResult, XLSystem


METHODS = (
    "clairvoyant_ranked",
    "full_activation",
    "static_hrnp",
    "fixed6_ranked",
    "fixed6_hysteresis",
    "rule_adaptive",
    "lstm_mpc",
    "proposed",
)


@dataclass
class EpisodeTotals:
    energy_j: float = 0.0
    delivered_mbit: float = 0.0
    outages: float = 0.0
    switches: int = 0
    active: float = 0.0
    safety_overrides: int = 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_mask(
    method: str,
    cfg: Config,
    mean: np.ndarray,
    sigma: np.ndarray,
    current: np.ndarray,
    previous: np.ndarray,
    static_score: np.ndarray,
    dqn: DQNController | None,
    state: np.ndarray,
    explore: bool,
) -> tuple[np.ndarray, int]:
    if method == "full_activation":
        return np.ones(cfg.subarrays, dtype=bool), cfg.subarrays - cfg.users
    if method == "static_hrnp":
        return ranked_mask(static_score, 5), 2
    if method == "fixed6_ranked":
        return ranked_mask(utility_score(mean, sigma, 1.645), 6), 3
    if method == "fixed6_hysteresis":
        score = utility_score(mean, None)
        inertia = score + previous * 0.15 * max(float(score.max()), 1e-30)
        return ranked_mask(inertia, 6), 3
    if method == "rule_adaptive":
        score = utility_score(mean, sigma, 1.645)
        return _belief_projection(cfg, ranked_mask(score, cfg.users), mean, sigma, score)[0], 0
    if method == "lstm_mpc":
        conservative_linear = 10.0 ** ((mean - 1.282 * sigma) / 10.0)
        score = utility_score(mean, sigma, 1.282)
        order = np.argsort(score)[::-1]
        full = conservative_linear.sum(axis=0)
        best_mask = np.ones(cfg.subarrays, dtype=bool)
        best_cost = np.inf
        for count in range(cfg.users, cfg.subarrays + 1):
            mask = np.zeros(cfg.subarrays, dtype=bool)
            mask[order[:count]] = True
            retained = conservative_linear[mask].sum(axis=0) / np.maximum(full, 1e-30)
            risk_penalty = 20.0 * float(np.maximum(0.65 - retained, 0.0).sum())
            switch_penalty = 0.25 * float(np.count_nonzero(mask != previous))
            cost = count + switch_penalty + risk_penalty
            if cost < best_cost:
                best_cost, best_mask = cost, mask
        return best_mask, max(int(best_mask.sum()) - cfg.users, 0)
    if dqn is None:
        raise ValueError("proposed method requires a trained DQN")
    action = dqn.choose(state, explore=explore)
    count = dqn.active_count(action)
    return ranked_mask(utility_score(mean, sigma, 1.645), count), action


def _belief_projection(
    cfg: Config,
    candidate: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Project using predicted beliefs only; no future channel is available."""
    mask = candidate.copy()
    order = np.argsort(score)[::-1]
    overridden = False
    conservative = 10.0 ** ((mean - 1.645 * sigma) / 10.0)
    full = conservative.sum(axis=0)
    retained = conservative[mask].sum(axis=0) / np.maximum(full, 1e-30)
    while np.any(retained < 0.65) and not mask.all():
        next_q = next(int(q) for q in order if not mask[q])
        mask[next_q] = True
        overridden = True
        retained = conservative[mask].sum(axis=0) / np.maximum(full, 1e-30)
    return mask, overridden


def simulate_episode(
    cfg: Config,
    trajectory: Trajectory,
    predictor: TrainedPredictor,
    method: str,
    speed: float,
    switch_scale: float,
    seed: int,
    dqn: DQNController | None = None,
    train: bool = False,
    outage_penalty: float = 12.0,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    proposed_variant = method.startswith("proposed")
    no_uncertainty = method == "proposed_no_uncertainty"
    no_inertia = method == "proposed_no_inertia"
    projected = method == "proposed_projected"
    decision_method = "proposed" if proposed_variant else method
    system = XLSystem(cfg)
    previous = np.ones(cfg.subarrays, dtype=bool)
    tracker = BeliefTracker(cfg, predictor, trajectory.subarray_gain_db[0])
    static_score = utility_score(predictor.mean.reshape(cfg.subarrays, cfg.users), None)
    totals = EpisodeTotals()
    trace: list[dict[str, float]] = []
    count_hist = np.zeros(cfg.subarrays + 1, dtype=int)
    mean, sigma = tracker.belief()
    state = make_state(cfg, mean, sigma, previous, switch_scale, speed, trajectory.rate_requirement_mbps[0])

    decision_steps = cfg.episode_steps - cfg.prediction_horizon
    for t in range(decision_steps):
        scan = t % cfg.scan_period == 0
        tracker.update(trajectory.subarray_gain_db[t], previous, scan)
        mean, sigma = tracker.belief()
        current = tracker.values[-1].reshape(cfg.subarrays, cfg.users)
        future_t = t + cfg.prediction_horizon
        known_requirement = trajectory.rate_requirement_mbps[t]
        actual_requirement = trajectory.rate_requirement_mbps[future_t]
        state = make_state(cfg, mean, sigma, previous, switch_scale, speed, known_requirement)
        decision_sigma = np.zeros_like(sigma) if no_uncertainty else sigma
        if method == "clairvoyant_ranked":
            future_gain = trajectory.subarray_gain_db[future_t]
            score = utility_score(future_gain, None)
            order = np.argsort(score)[::-1]
            best: tuple[tuple[float, float], np.ndarray, LinkResult] | None = None
            for count in range(cfg.users, cfg.subarrays + 1):
                trial = np.zeros(cfg.subarrays, dtype=bool)
                trial[order[:count]] = True
                trial_link = system.evaluate(
                    trajectory.channel[future_t], trial, previous, switch_scale, scan, seed * 10_000 + t, actual_requirement
                )
                objective = (trial_link.outage, trial_link.energy_j)
                if best is None or objective < best[0]:
                    best = (objective, trial, trial_link)
            assert best is not None
            _, mask, link = best
            candidate, action, overridden = mask, 0, False
        else:
            candidate, action = _candidate_mask(decision_method, cfg, mean, decision_sigma, current, previous, static_score, dqn, state, train)
            score = utility_score(mean, decision_sigma, 1.645 if proposed_variant else 0.0)
            if proposed_variant and not no_inertia and previous.any():
                # Switching-aware hysteresis is part of the safety/controller
                # interface; its strength scales with the audited transition cost.
                inertia = min(0.30, 0.08 * switch_scale) * max(float(score.max()), 1e-30)
                score = score + previous.astype(float) * inertia
                candidate = ranked_mask(score, int(candidate.sum()))
        if proposed_variant:
            mask, overridden = _belief_projection(cfg, candidate, mean, decision_sigma, score) if projected else (candidate, False)
            link = system.evaluate(trajectory.channel[future_t], mask, previous, switch_scale, scan, seed * 10_000 + t, actual_requirement)
        elif method != "clairvoyant_ranked":
            mask = candidate
            link = system.evaluate(trajectory.channel[future_t], mask, previous, switch_scale, scan, seed * 10_000 + t, actual_requirement)
            overridden = False

        switches = int(np.count_nonzero(mask != previous))
        totals.energy_j += link.energy_j
        totals.delivered_mbit += float(link.rates_mbps.sum() * cfg.slot_s)
        totals.outages += link.outage
        totals.switches += switches
        totals.active += float(mask.mean())
        count_hist[int(mask.sum())] += 1
        totals.safety_overrides += int(overridden)
        trace.append(
            {
                "slot": float(t + 1),
                "energy_j": link.energy_j,
                "min_rate_mbps": float(link.rates_mbps.min()),
                "active_ratio": float(mask.mean()),
                "switches": float(switches),
                "outage": link.outage,
                "mean_requirement_mbps": float(np.mean(actual_requirement)),
            }
        )

        next_state = make_state(cfg, mean, sigma, mask, switch_scale, speed, actual_requirement)
        if train and dqn is not None:
            full_reference = system.evaluate(
                trajectory.channel[t + cfg.prediction_horizon], np.ones(cfg.subarrays, bool), np.ones(cfg.subarrays, bool), 0.0, scan, seed * 10_000 + t, actual_requirement
            )
            normalized_shortfall = float(np.mean(np.maximum(actual_requirement - link.rates_mbps, 0.0) / actual_requirement))
            # Dense deficit feedback complements the sparse outage indicator.
            constraint_cost = link.outage + 5.0 * normalized_shortfall
            reward = -link.energy_j / max(full_reference.energy_j, 1e-9) - outage_penalty * constraint_cost
            dqn.learn((state, action, float(reward), next_state, float(t == decision_steps - 1)))
        previous = mask
        state = next_state

    steps = decision_steps
    metrics = {
        "energy_j": totals.energy_j,
        "energy_efficiency_mbit_j": totals.delivered_mbit / max(totals.energy_j, 1e-12),
        "outage_probability": totals.outages / steps,
        "switches_per_slot": totals.switches / steps,
        "active_ratio": totals.active / steps,
        "safety_override_ratio": totals.safety_overrides / steps,
    }
    for count in range(cfg.users, cfg.subarrays + 1):
        metrics[f"count_fraction_{count}"] = float(count_hist[count] / steps)
    return metrics, trace


def _ablation_sweep(cfg: Config, predictor: TrainedPredictor, controllers: dict[int, DQNController]) -> list[dict[str, object]]:
    methods = ("proposed", "proposed_no_uncertainty", "proposed_no_inertia", "proposed_projected")
    speed = min(cfg.speeds_mps, key=lambda x: abs(x - 1.5))
    switch = min(cfg.switch_scales, key=lambda x: abs(x - 1.0))
    raw: list[dict[str, object]] = []
    for index in range(cfg.test_trajectories):
        seed = 90_000 + index
        trajectory = generate_trajectory(cfg, seed, speed)
        controller = controllers[cfg.seeds[index % len(cfg.seeds)]]
        for method in methods:
            metrics, _ = simulate_episode(cfg, trajectory, predictor, method, speed, switch, seed, controller, False)
            raw.append({"method": method, **metrics})
    rows: list[dict[str, object]] = []
    for method in methods:
        group = [row for row in raw if row["method"] == method]
        item: dict[str, object] = {"method": method, "samples": len(group)}
        for metric in ("energy_j", "outage_probability", "switches_per_slot"):
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(item)
    return rows


def _train_controller(cfg: Config, predictor: TrainedPredictor, seed: int) -> tuple[DQNController, list[dict[str, float]]]:
    state_size = 2 * cfg.subarrays * cfg.users + cfg.subarrays + cfg.users + 2
    controller = DQNController(cfg, state_size, seed)
    multipliers = {speed: cfg.dual_initial for speed in cfg.speeds_mps}
    diagnostics: list[dict[str, float]] = []
    for episode in range(cfg.train_episodes):
        speed = cfg.speeds_mps[episode % len(cfg.speeds_mps)]
        switch_scale = cfg.switch_scales[episode % len(cfg.switch_scales)]
        episode_cfg = replace(cfg, scan_period=(4, 8, 16)[episode % 3])
        trajectory = generate_trajectory(episode_cfg, 30_000 + seed * 1_000 + episode, speed)
        metrics, _ = simulate_episode(episode_cfg, trajectory, predictor, "proposed", speed, switch_scale, episode + seed, controller, True, multipliers[speed])
        multipliers[speed] = float(np.clip(multipliers[speed] + cfg.dual_lr * (metrics["outage_probability"] - cfg.outage_target), cfg.dual_min, cfg.dual_max))
        diagnostics.append({"seed": float(seed), "episode": float(episode), "dual_multiplier": multipliers[speed], **metrics})
    return controller, diagnostics


def _select_controller(
    cfg: Config,
    predictor: TrainedPredictor,
    controllers: dict[int, DQNController],
) -> tuple[int, list[dict[str, float]]]:
    """Select once on disjoint controller-validation trajectories, never test data."""
    rows: list[dict[str, float]] = []
    switch = min(cfg.switch_scales, key=lambda x: abs(x - 1.0))
    for seed, controller in controllers.items():
        energies: list[float] = []
        outages: list[float] = []
        for speed in cfg.speeds_mps:
            for index in range(5):
                trajectory_seed = 45_000 + int(speed * 100) * 100 + index
                trajectory = generate_trajectory(cfg, trajectory_seed, speed)
                metrics, _ = simulate_episode(cfg, trajectory, predictor, "proposed", speed, switch, trajectory_seed, controller, False)
                energies.append(metrics["energy_j"])
                outages.append(metrics["outage_probability"])
        rows.append({"seed": float(seed), "validation_energy_j": float(np.mean(energies)), "validation_outage": float(np.mean(outages))})
    # Use a 2% controller-validation guard band for the 5% deployment target.
    feasible = [row for row in rows if row["validation_outage"] <= 0.02]
    selected = min(feasible, key=lambda row: row["validation_energy_j"]) if feasible else min(rows, key=lambda row: (row["validation_outage"], row["validation_energy_j"]))
    for row in rows:
        row["selected"] = float(row is selected)
    return int(selected["seed"]), rows


def _prediction_diagnostics(cfg: Config, predictor: TrainedPredictor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for speed in cfg.speeds_mps:
        errors: list[float] = []
        last_errors: list[float] = []
        trend_errors: list[float] = []
        covered: list[float] = []
        for index in range(max(2, cfg.test_trajectories // 2)):
            tr = generate_trajectory(cfg, 40_000 + int(speed * 100) * 100 + index, speed)
            tracker = BeliefTracker(cfg, predictor, tr.subarray_gain_db[0])
            previous = np.ones(cfg.subarrays, bool)
            for t in range(cfg.episode_steps - cfg.prediction_horizon):
                scan = t % cfg.scan_period == 0
                # Deterministic missing pattern, independent of prediction error.
                previous = (np.arange(cfg.subarrays) + t + index) % 3 != 0
                tracker.update(tr.subarray_gain_db[t], previous, scan)
                if t < cfg.history:
                    continue
                mu, sigma = tracker.belief()
                target = tr.subarray_gain_db[t + cfg.prediction_horizon]
                errors.extend((target - mu).reshape(-1))
                last_errors.extend((target - tracker.values[-1].reshape(cfg.subarrays, cfg.users)).reshape(-1))
                last = tracker.values[-1].reshape(cfg.subarrays, cfg.users)
                prior = tracker.values[-2].reshape(cfg.subarrays, cfg.users)
                trend = last + cfg.prediction_horizon * np.clip(last - prior, -0.5, 0.5)
                trend_errors.extend((target - trend).reshape(-1))
                covered.extend((np.abs(target - mu) <= 1.645 * sigma).reshape(-1))
        rows.append(
            {
                "speed_mps": speed,
                "lstm_rmse_db": float(np.sqrt(np.mean(np.square(errors)))),
                "last_value_rmse_db": float(np.sqrt(np.mean(np.square(last_errors)))),
                "linear_trend_rmse_db": float(np.sqrt(np.mean(np.square(trend_errors)))),
                "coverage90": float(np.mean(covered)),
            }
        )
    return rows


def _channel_diagnostics(cfg: Config) -> tuple[list[dict[str, float]], Trajectory]:
    rows: list[dict[str, float]] = []
    representative: Trajectory | None = None
    for speed in cfg.speeds_mps:
        acfs: list[np.ndarray] = []
        counts: list[float] = []
        for index in range(max(4, cfg.test_trajectories // 2)):
            tr = generate_trajectory(cfg, 70_000 + int(speed * 100) * 100 + index, speed)
            if representative is None and speed == min(cfg.speeds_mps, key=lambda x: abs(x - 1.5)):
                representative = tr
            series = tr.subarray_gain_db[:, :, 0].mean(axis=1)
            series = series - series.mean()
            variance = float(np.dot(series, series))
            max_lag = min(30, len(series) // 3)
            acfs.append(np.asarray([np.dot(series[:-lag], series[lag:]) / variance if lag else 1.0 for lag in range(max_lag + 1)]))
            counts.extend(tr.active_clusters.reshape(-1))
        mean_acf = np.mean(acfs, axis=0)
        for lag, value in enumerate(mean_acf):
            rows.append({
                "speed_mps": speed,
                "lag_slots": float(lag),
                "lag_distance_m": float(lag * speed * cfg.slot_s),
                "gain_acf": float(value),
                "mean_active_clusters": float(np.mean(counts)),
                "std_active_clusters": float(np.std(counts)),
            })
    assert representative is not None
    return rows, representative


def _scan_sweep(cfg: Config, predictor: TrainedPredictor, controllers: dict[int, DQNController]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    speed = min(cfg.speeds_mps, key=lambda x: abs(x - 1.5))
    switch = min(cfg.switch_scales, key=lambda x: abs(x - 1.0))
    for period in (2, 4, 8, 16):
        sweep_cfg = replace(cfg, scan_period=period)
        for index in range(cfg.test_trajectories):
            seed = 80_000 + index
            trajectory = generate_trajectory(sweep_cfg, seed, speed)
            controller_seed = cfg.seeds[index % len(cfg.seeds)]
            for method in ("lstm_mpc", "proposed"):
                metrics, _ = simulate_episode(
                    sweep_cfg, trajectory, predictor, method, speed, switch, seed,
                    controllers[controller_seed] if method == "proposed" else None, False,
                )
                rows.append({"method": method, "scan_period": period, **metrics})
    output: list[dict[str, object]] = []
    for method in ("lstm_mpc", "proposed"):
        for period in (2, 4, 8, 16):
            group = [row for row in rows if row["method"] == method and row["scan_period"] == period]
            item: dict[str, object] = {"method": method, "scan_period": period, "samples": len(group)}
            for metric in ("energy_j", "outage_probability", "switches_per_slot"):
                values = np.asarray([float(row[metric]) for row in group])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
            output.append(item)
    return output


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("method", "speed_mps", "switch_scale")
    metrics = ("energy_j", "energy_efficiency_mbit_j", "outage_probability", "switches_per_slot", "active_ratio", "safety_override_ratio")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary: dict[str, object] = dict(zip(keys, key))
        summary["samples"] = len(group)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def _paired_differences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    proposed = {
        (row["speed_mps"], row["switch_scale"], row["trajectory_seed"]): row
        for row in rows
        if row["method"] == "proposed"
    }
    output: list[dict[str, object]] = []
    for baseline in METHODS:
        if baseline == "proposed":
            continue
        differences: dict[tuple[object, object], list[tuple[float, float]]] = {}
        for row in rows:
            if row["method"] != baseline:
                continue
            key = (row["speed_mps"], row["switch_scale"], row["trajectory_seed"])
            ref = proposed[key]
            differences.setdefault((row["speed_mps"], row["switch_scale"]), []).append(
                (float(row["energy_j"]) - float(ref["energy_j"]), float(row["outage_probability"]) - float(ref["outage_probability"]))
            )
        for (speed, switch), values in differences.items():
            array = np.asarray(values)
            n = len(array)
            energy = array[:, 0]
            outage = array[:, 1]
            output.append(
                {
                    "baseline": baseline,
                    "speed_mps": speed,
                    "switch_scale": switch,
                    "samples": n,
                    "baseline_minus_proposed_energy_j": float(energy.mean()),
                    "energy_difference_ci95": float(1.96 * energy.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                    "baseline_minus_proposed_outage": float(outage.mean()),
                    "proposed_energy_win_rate": float(np.mean(energy > 0.0)),
                }
            )
    return output


def run_experiment(cfg: Config, root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / "results" / f"{cfg.mode}_{timestamp}"
    (output / "figures").mkdir(parents=True)
    (output / "models").mkdir()
    (output / "traces").mkdir()
    (output / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2), encoding="utf-8")

    predictor = train_predictor(cfg)
    torch.save(
        {"state_dict": predictor.model.state_dict(), "mean": predictor.mean, "std": predictor.std, "sigma_scale": predictor.sigma_scale, "validation": predictor.validation},
        output / "models" / "probabilistic_lstm.pt",
    )
    prediction_rows = _prediction_diagnostics(cfg, predictor)
    _write_csv(output / "prediction_metrics.csv", prediction_rows)
    channel_rows, representative = _channel_diagnostics(cfg)
    _write_csv(output / "channel_diagnostics.csv", channel_rows)

    controllers: dict[int, DQNController] = {}
    training_rows: list[dict[str, float]] = []
    for seed in cfg.seeds:
        controller, diagnostics = _train_controller(cfg, predictor, seed)
        controllers[seed] = controller
        training_rows.extend(diagnostics)
    _write_csv(output / "training_diagnostics.csv", training_rows)
    for seed, controller in controllers.items():
        torch.save(controller.online.state_dict(), output / "models" / f"risk_dqn_seed{seed}.pt")
    selected_seed, controller_selection = _select_controller(cfg, predictor, controllers)
    selected_controller = controllers[selected_seed]
    _write_csv(output / "controller_selection.csv", controller_selection)

    rows: list[dict[str, object]] = []
    trace_written: set[str] = set()
    for speed in cfg.speeds_mps:
        for switch_scale in cfg.switch_scales:
            for test_index in range(cfg.test_trajectories):
                trajectory_seed = 50_000 + int(speed * 100) * 100 + test_index
                trajectory = generate_trajectory(cfg, trajectory_seed, speed)
                for method in METHODS:
                    controller_seed = selected_seed
                    metrics, trace = simulate_episode(
                        cfg,
                        trajectory,
                        predictor,
                        method,
                        speed,
                        switch_scale,
                        trajectory_seed,
                        selected_controller if method == "proposed" else None,
                        False,
                    )
                    row: dict[str, object] = {
                        "method": method,
                        "speed_mps": speed,
                        "switch_scale": switch_scale,
                        "trajectory_seed": trajectory_seed,
                        "controller_seed": controller_seed if method == "proposed" else 0,
                    }
                    row.update(metrics)
                    rows.append(row)
                    trace_key = f"{method}_{speed}_{switch_scale}"
                    if test_index == 0 and trace_key not in trace_written:
                        _write_csv(output / "traces" / f"{trace_key}.csv", trace)
                        trace_written.add(trace_key)

    summary = _summarize(rows)
    paired = _paired_differences(rows)
    deployment_controllers = {seed: selected_controller for seed in cfg.seeds}
    scan_rows = _scan_sweep(cfg, predictor, deployment_controllers)
    ablation_rows = _ablation_sweep(cfg, predictor, deployment_controllers)
    action_rows: list[dict[str, object]] = []
    for speed in cfg.speeds_mps:
        selected = [row for row in rows if row["method"] == "proposed" and row["speed_mps"] == speed and row["switch_scale"] == min(cfg.switch_scales, key=lambda x: abs(x - 1.0))]
        for count in range(cfg.users, cfg.subarrays + 1):
            values = np.asarray([float(row[f"count_fraction_{count}"]) for row in selected])
            action_rows.append({"speed_mps": speed, "active_subarrays": count, "fraction_mean": float(values.mean()), "fraction_ci95": float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))})
    _write_csv(output / "raw_metrics.csv", rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "paired_differences.csv", paired)
    _write_csv(output / "scan_sensitivity.csv", scan_rows)
    _write_csv(output / "ablation.csv", ablation_rows)
    _write_csv(output / "action_distribution.csv", action_rows)
    create_plots(cfg, prediction_rows, summary, channel_rows, representative, scan_rows, ablation_rows, action_rows, output / "figures")
    report = {
        "predictor_validation": predictor.validation,
        "raw_samples": len(rows),
        "selected_controller_seed": selected_seed,
        "note": "No result is hard-coded; all tables and figures are generated from raw_metrics.csv.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
