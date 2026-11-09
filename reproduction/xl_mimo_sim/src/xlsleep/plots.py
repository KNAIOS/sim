from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .channel import Trajectory
from .config import Config


LABELS = {
    "clairvoyant_ranked": "Clairvoyant ranked",
    "full_activation": "Full activation",
    "static_hrnp": "Static HRNP",
    "fixed6_ranked": "Fixed-6 ranked",
    "fixed6_hysteresis": "Fixed-6 + hysteresis",
    "rule_adaptive": "Rule-adaptive",
    "lstm_mpc": "LSTM + MPC",
    "proposed": "Proposed",
}
MARKERS = ("*", "o", "s", "^", "D", "v", "P", "X")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output: Path, name: str, top: float = 0.98) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top), pad=0.6)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _select(summary: list[dict[str, object]], method: str, switch_scale: float) -> list[dict[str, object]]:
    return sorted(
        [row for row in summary if row["method"] == method and float(row["switch_scale"]) == switch_scale],
        key=lambda row: float(row["speed_mps"]),
    )


def create_plots(
    cfg: Config,
    prediction: list[dict[str, float]],
    summary: list[dict[str, object]],
    channel_rows: list[dict[str, float]],
    representative: Trajectory,
    scan_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    output: Path,
) -> None:
    _style()
    # Fig. 1: the simulated channel itself, before any controller is evaluated.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    heat = representative.subarray_gain_db[:, :, 0].T
    image = axes[0].imshow(heat, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set(xlabel="Slot", ylabel="Subarray index")
    fig.colorbar(image, ax=axes[0], label="Large-scale gain (dB)", pad=0.02)
    for speed in cfg.speeds_mps:
        selected = [row for row in channel_rows if row["speed_mps"] == speed]
        axes[1].plot([row["lag_slots"] * cfg.slot_s for row in selected], [row["gain_acf"] for row in selected], label=f"{speed:g} m/s")
    axes[1].set(xlabel="Time lag (s)", ylabel="Gain autocorrelation", ylim=(-0.15, 1.03))
    axes[1].legend(frameon=False)
    _save(fig, output, "fig_channel_validation")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    speed = np.asarray([row["speed_mps"] for row in prediction])
    axes[0].plot(speed, [row["last_value_rmse_db"] for row in prediction], "--s", label="Last value")
    axes[0].plot(speed, [row["linear_trend_rmse_db"] for row in prediction], ":^", label="Linear trend")
    axes[0].plot(speed, [row["lstm_rmse_db"] for row in prediction], "-o", label="Probabilistic LSTM")
    axes[0].set(xlabel="User speed (m/s)", ylabel="Prediction RMSE (dB)")
    axes[0].legend(frameon=False)
    axes[1].plot(speed, [row["coverage90"] for row in prediction], "-o", label="Empirical")
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1.0, label="Nominal 90%")
    axes[1].set(xlabel="User speed (m/s)", ylabel="Interval coverage", ylim=(0.0, 1.02))
    axes[1].legend(frameon=False)
    bottoms = np.zeros(len(speed))
    for count in range(cfg.users, cfg.subarrays + 1):
        fraction = [next(float(row["fraction_mean"]) for row in action_rows if row["speed_mps"] == value and row["active_subarrays"] == count) for value in speed]
        axes[2].bar(speed, fraction, bottom=bottoms, width=0.35, label=str(count))
        bottoms += np.asarray(fraction)
    axes[2].set(xlabel="User speed (m/s)", ylabel="Action fraction", ylim=(0.0, 1.0))
    axes[2].legend(title="Active Q", frameon=False, ncol=2, fontsize=6)
    _save(fig, output, "fig_prediction")

    headline_switch = min(cfg.switch_scales, key=lambda x: abs(x - 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65))
    handles = []
    labels = []
    for i, method in enumerate(LABELS):
        selected = _select(summary, method, headline_switch)
        x = [float(row["speed_mps"]) for row in selected]
        handle = axes[0].errorbar(
            x,
            [float(row["energy_j_mean"]) for row in selected],
            yerr=[float(row["energy_j_ci95"]) for row in selected],
            marker=MARKERS[i],
            linewidth=1.1,
            markersize=4,
            capsize=2,
        )
        handles.append(handle)
        labels.append(LABELS[method])
        axes[1].errorbar(
            x,
            [float(row["energy_efficiency_mbit_j_mean"]) for row in selected],
            yerr=[float(row["energy_efficiency_mbit_j_ci95"]) for row in selected],
            marker=MARKERS[i], linewidth=1.0, markersize=3.5, capsize=1.5,
        )
        axes[2].plot(x, [float(row["outage_probability_mean"]) for row in selected], marker=MARKERS[i], linewidth=1.1, markersize=4)
    axes[0].set(xlabel="User speed (m/s)", ylabel="Episode energy (J)")
    axes[1].set(xlabel="User speed (m/s)", ylabel="Energy efficiency (Mbit/J)")
    axes[2].axhline(cfg.outage_target, color="black", linestyle="--", linewidth=0.9)
    axes[2].set(xlabel="User speed (m/s)", ylabel="Rate-outage probability", ylim=(-0.003, 0.10))
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    _save(fig, output, "fig_energy_outage_vs_speed", top=0.78)

    headline_speed = min(cfg.speeds_mps, key=lambda x: abs(x - 1.5))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.95))
    handles = []
    labels = []
    transition_methods = ("clairvoyant_ranked", "full_activation", "fixed6_hysteresis", "lstm_mpc", "proposed")
    for i, method in enumerate(transition_methods):
        selected = sorted(
            [row for row in summary if row["method"] == method and float(row["speed_mps"]) == headline_speed],
            key=lambda row: float(row["switch_scale"]),
        )
        x = [float(row["switch_scale"]) for row in selected]
        handle = axes[0].plot(x, [float(row["energy_j_mean"]) for row in selected], marker=MARKERS[i], linewidth=1.1, markersize=4)[0]
        handles.append(handle)
        labels.append(LABELS[method])
        axes[1].plot(x, [float(row["switches_per_slot_mean"]) for row in selected], marker=MARKERS[i], linewidth=1.1, markersize=4)
    axes[0].set(xlabel="Transition cost scale", ylabel="Episode energy (J)")
    axes[1].set(xlabel="Transition cost scale", ylabel="Switches per slot")
    axes[0].set_xticks(cfg.switch_scales)
    axes[1].set_xticks(cfg.switch_scales)
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    _save(fig, output, "fig_transition_sensitivity", top=0.78)

    # Fig. 5: robustness to the cost/latency of full-array sounding.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for i, method in enumerate(("lstm_mpc", "proposed")):
        selected = sorted([row for row in scan_rows if row["method"] == method], key=lambda row: int(row["scan_period"]))
        x = [int(row["scan_period"]) for row in selected]
        axes[0].errorbar(x, [float(row["energy_j_mean"]) for row in selected], yerr=[float(row["energy_j_ci95"]) for row in selected], marker=MARKERS[i], capsize=2, label=LABELS[method])
        axes[1].plot(x, [float(row["outage_probability_mean"]) for row in selected], marker=MARKERS[i], label=LABELS[method])
    axes[0].set(xlabel="Full-scan period (slots)", ylabel="Episode energy (J)")
    axes[1].axhline(cfg.outage_target, color="black", linestyle="--", linewidth=0.9)
    axes[1].set(xlabel="Full-scan period (slots)", ylabel="Rate-outage probability")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    ablation_labels = {
        "proposed": "Full",
        "proposed_no_uncertainty": "No uncertainty",
        "proposed_no_inertia": "No inertia",
        "proposed_projected": "With projection",
    }
    x = np.arange(len(ablation_rows))
    energy = [float(row["energy_j_mean"]) for row in ablation_rows]
    outage = [100.0 * float(row["outage_probability_mean"]) for row in ablation_rows]
    axes[2].bar(x - 0.18, energy, width=0.36, color="tab:blue", label="Energy (J)")
    risk_axis = axes[2].twinx()
    risk_axis.bar(x + 0.18, outage, width=0.36, color="tab:red", label="Outage (%)")
    risk_axis.axhline(100.0 * cfg.outage_target, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_xticks(x, [ablation_labels[str(row["method"])] for row in ablation_rows], rotation=25, ha="right")
    axes[2].set_ylabel("Energy (J)", color="tab:blue")
    risk_axis.set_ylabel("Outage (%)", color="tab:red")
    axes[2].grid(False)
    risk_axis.grid(False)
    _save(fig, output, "fig_scan_sensitivity")
