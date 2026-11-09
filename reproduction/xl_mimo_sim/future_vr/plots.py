from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, PercentFormatter


COLORS = {
    "Full array": "#3C5488", "Static HRNP": "#8491B4", "Current-CSI": "#00A087",
    "Symmetric guard": "#F39B7F", "Oracle future": "#7E6148", "RA-FVV": "#E64B35",
    "No future": "#4DBBD5", "No readiness": "#91D1C2",
    "Current-CSI matched": "#00A087", "Future-only matched": "#4DBBD5",
    "RA-FVV matched": "#E64B35", "Oracle matched": "#7E6148",
    "RA-CRL": "#E64B35", "Vanilla symmetric DQN": "#3C5488", "Energy-only DQN": "#00A087",
}
MARKERS = {"Full array": "o", "Static HRNP": "s", "Current-CSI": "^", "Symmetric guard": "D",
           "Oracle future": "P", "RA-FVV": "*", "Current-CSI matched": "^",
           "Future-only matched": "D", "RA-FVV matched": "*", "Oracle matched": "P"}
LINES = {"Full array": "--", "Static HRNP": "-.", "Current-CSI": ":", "Symmetric guard": "--",
         "Oracle future": "-.", "RA-FVV": "-", "Current-CSI matched": ":",
         "Future-only matched": "--", "RA-FVV matched": "-", "Oracle matched": "-."}


def _display_name(method: str) -> str:
    return "RA-CRL" if method == "RA-FVV" else method


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "legend.fontsize": 7.4, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.35, "lines.markersize": 4.5,
        "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    numeric = {"speed_mps", "wake_delay", "switch_scale", "energy_j", "energy_j_ci95",
               "energy_efficiency_mbit_j", "energy_efficiency_mbit_j_ci95", "outage_probability",
               "outage_probability_ci95", "active_ratio", "active_ratio_ci95", "transitions_per_s",
               "transitions_per_s_ci95", "n", "episode", "dual", "loss", "outage"}
    for row in raw:
        for key in numeric & row.keys():
            row[key] = float(row[key])
    return raw


def _finish(fig: plt.Figure, output: Path, stem: str) -> None:
    for axis in fig.axes:
        axis.tick_params(direction="in", which="both", top=True, right=True, width=0.7)
        axis.grid(True, which="major", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.30, alpha=0.55)
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def _trend_with_ci(axis: plt.Axes, x: np.ndarray, y: np.ndarray, ci: np.ndarray,
                   method: str, label: str | None = None) -> None:
    """Draw a mean trend with a restrained 95% confidence band."""
    axis.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.11, linewidth=0)
    axis.plot(x, y, color=COLORS[method], marker=MARKERS[method],
              linestyle=LINES[method], label=label or _display_name(method))


def _select(rows: list[dict[str, object]], **filters: object) -> list[dict[str, object]]:
    return [row for row in rows if all(np.isclose(row[k], v) if isinstance(v, (float, int)) else row[k] == v
                                       for k, v in filters.items())]


def plot_prediction(output: Path) -> None:
    data = np.load(output / "prediction_diagnostics.npz")
    stability_path = output / "selection_stability.npz"
    stability = np.load(stability_path) if stability_path.exists() else None
    prediction = data["prediction"]
    snapshot = int(data["snapshot_time"][0])
    horizon = np.arange(1, len(data["rmse_future"]) + 1)
    if stability is None:
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.28), constrained_layout=True)
    else:
        fig, grid = plt.subplots(2, 2, figsize=(7.16, 4.25), constrained_layout=True)
        axes = grid.ravel()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis("off")
    boxes = (
        (0.72, "CSI, traffic, RF state,\nand readiness", "#DCE6F1"),
        (0.50, "Future-VR belief\nand RA-FVV scores", "#E2F0D9"),
        (0.28, "DQN chooses count $M_t$;\nranker chooses locations", "#FCE4D6"),
        (0.06, r"RF command $\rightarrow$ rate," + "\noutage, and energy feedback", "#EDEDED"),
    )
    for index, (y, label, color) in enumerate(boxes):
        patch = FancyBboxPatch((0.08, y), 0.84, 0.15,
                               boxstyle="round,pad=0.018,rounding_size=0.025",
                               linewidth=0.75, edgecolor="#555555", facecolor=color)
        axes[0].add_patch(patch)
        axes[0].text(0.50, y + 0.075, label, ha="center", va="center", fontsize=7.0)
        if index < len(boxes) - 1:
            axes[0].add_patch(FancyArrowPatch((0.50, y - 0.005), (0.50, y - 0.065),
                                              arrowstyle="-|>", mutation_scale=8,
                                              linewidth=0.75, color="#555555"))
    axes[0].set_title("(a) WA-RL decision loop")
    aligned_utility = data["aligned_utility"] if "aligned_utility" in data.files else None
    if aligned_utility is None:
        aligned_utility = 10.0 ** ((prediction - prediction.max(axis=1, keepdims=True)) / 10.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#D9D9D9")
    image1 = axes[1].imshow(aligned_utility.T, aspect="auto", origin="lower", cmap=cmap,
                            extent=(0.5, aligned_utility.shape[0] + 0.5,
                                    0.5, aligned_utility.shape[1] + 0.5),
                            vmin=0.0, vmax=1.0)
    readiness = data["snapshot_readiness"] if "snapshot_readiness" in data.files else np.zeros(
        aligned_utility.shape[1]
    )
    for q, delay in enumerate(readiness):
        if delay > 1:
            axes[1].plot([0.5, delay - 0.5], [q + 1, q + 1], color="#666666",
                         linewidth=0.35, alpha=0.8)
    axes[1].set(xlabel="Future slot $h$", ylabel="Subarray index $q$",
                title="(b) Post-readiness utility")
    axes[1].set_xticks(np.arange(1, aligned_utility.shape[0] + 1))
    axes[1].set_yticks(np.arange(1, aligned_utility.shape[1] + 1, 3))
    colorbar = fig.colorbar(image1, ax=axes[1], location="right", shrink=0.78, pad=0.025)
    colorbar.ax.set_title("norm.", fontsize=7.2, pad=3)
    axes[1].text(1.9, 15.3, "excluded\n(not ready)", ha="center", va="center",
                 fontsize=6.3, color="#555555")
    axes[2].plot(horizon, data["rmse_hold"], "--s", color=COLORS["Static HRNP"], label="Persistence")
    axes[2].plot(horizon, data["rmse_future"], "-o", color=COLORS["RA-FVV"], label="Future-VR belief")
    axes[2].set(xlabel="Prediction horizon (slot)", ylabel="Gain RMSE (dB)", title="(c) Prediction error")
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].legend(frameon=False, loc="upper left")
    if stability is not None:
        x = stability["bin_center"]
        axes[3].plot(x, stability["bin_overlap"], "-o", color=COLORS["RA-FVV"],
                     label="RA-FVV overlap")
        axes[3].plot(x, stability["bin_exact"], "--^", color="#3C5488",
                     label="Exact-set rate")
        axes[3].plot(x, stability["bin_current_overlap"], ":s", color=COLORS["Current-CSI"],
                     label="Current-CSI overlap")
        axes[3].set(xlabel="Predicted boundary-value margin", ylabel="Selection agreement",
                    title="(d) Top-$M$ selection stability", ylim=(0.30, 1.03))
        axes[3].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[3].legend(frameon=False, loc="lower right")
    _finish(fig, output, "fig1_future_vr_prediction")


def plot_speed(rows: list[dict[str, object]], output: Path, central_delay: int) -> None:
    methods = ("Full array", "Static HRNP", "Current-CSI", "Symmetric guard", "RA-FVV")
    panels = (("energy_j", "energy_j_ci95", "Energy per episode (J)"),
              ("energy_efficiency_mbit_j", "energy_efficiency_mbit_j_ci95", "Energy efficiency (Mbit/J)"),
              ("outage_probability", "outage_probability_ci95", "Rate-outage probability"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.52))
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.20, top=0.70, wspace=0.43)
    for method in methods:
        values = sorted(_select(rows, method=method, wake_delay=central_delay, switch_scale=1.0),
                        key=lambda row: row["speed_mps"])
        if not values:
            continue
        x = np.asarray([row["speed_mps"] for row in values])
        for axis, (field, ci, ylabel) in zip(axes, panels):
            y = np.asarray([row[field] for row in values], dtype=float)
            spread = np.asarray([row[ci] for row in values], dtype=float)
            _trend_with_ci(axis, x, y, spread, method)
            axis.set(xlabel="UE speed (m/s)", ylabel=ylabel)
    axes[0].set_title("(a) Consumed energy")
    axes[1].set_title("(b) Delivered bits per joule")
    axes[2].set_title("(c) Rate-outage probability")
    axes[2].axhline(0.05, color="black", linestyle=":", linewidth=0.9, label="5% constraint")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_ylim(bottom=0.0)
    for axis in axes:
        axis.set_xticks(sorted(set(row["speed_mps"] for row in rows if row["method"] == "RA-FVV"
                                    and row["wake_delay"] == central_delay)))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    _finish(fig, output, "fig2_energy_reliability_vs_speed")


def plot_delay(rows: list[dict[str, object]], output: Path, central_speed: float) -> None:
    methods = ("Current-CSI", "Symmetric guard", "RA-FVV")
    panels = (("energy_j", "energy_j_ci95", "Energy per episode (J)"),
              ("outage_probability", "outage_probability_ci95", "Rate-outage probability"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.42))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.71, wspace=0.31)
    for method in methods:
        values = sorted(_select(rows, method=method, speed_mps=central_speed, switch_scale=1.0),
                        key=lambda row: row["wake_delay"])
        for axis, (field, ci, ylabel) in zip(axes, panels):
            x = np.asarray([row["wake_delay"] for row in values], dtype=float)
            y = np.asarray([row[field] for row in values], dtype=float)
            spread = np.asarray([row[ci] for row in values], dtype=float)
            _trend_with_ci(axis, x, y, spread, method)
            axis.set(xlabel="Wake-up delay (slots)", ylabel=ylabel)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title("(a) Delay-dependent energy")
    axes[1].set_title("(b) Delay-induced outage")
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(bottom=0.0)
    delays = sorted(set(row["wake_delay"] for row in rows if row["method"] == "RA-FVV"
                        and np.isclose(row["speed_mps"], central_speed)))
    for axis in axes:
        axis.set_xticks(delays)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.98))
    _finish(fig, output, "fig3_wakeup_delay_robustness")


def plot_matched_cardinality(rows: list[dict[str, object]], output: Path, central_speed: float) -> None:
    methods = ("Current-CSI matched", "Future-only matched", "RA-FVV matched", "Oracle matched")
    panels = (("energy_j", "energy_j_ci95", "Energy per episode (J)"),
              ("outage_probability", "outage_probability_ci95", "Rate-outage probability"),
              ("transitions_per_s", "transitions_per_s_ci95", "Subarray transitions/s"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.52))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.70, wspace=0.40)
    for method in methods:
        values = sorted(_select(rows, method=method, speed_mps=central_speed, switch_scale=1.0),
                        key=lambda row: row["wake_delay"])
        for axis, (field, ci, ylabel) in zip(axes, panels):
            axis.errorbar([row["wake_delay"] for row in values], [row[field] for row in values],
                          yerr=[row[ci] for row in values], color=COLORS[method],
                          marker=MARKERS[method], linestyle=LINES[method], capsize=2, label=method)
            axis.set(xlabel="Wake-up delay (slot)", ylabel=ylabel)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title("(a) Matched-count energy")
    axes[1].set_title("(b) Matched-count reliability")
    axes[2].set_title("(c) Switching activity")
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.99))
    _finish(fig, output, "fig4_matched_cardinality")


def plot_fixed_budget(rows: list[dict[str, object]], output: Path, commanded_count: int = 6) -> None:
    methods = ("Current-CSI matched", "Future-only matched", "RA-FVV matched", "Oracle matched")
    largest_delay = int(max(row["wake_delay"] for row in rows))
    fig, axis = plt.subplots(figsize=(3.48, 2.35), constrained_layout=True)
    for method in methods:
        by_speed = sorted(_select(rows, method=method, wake_delay=largest_delay, switch_scale=1.0),
                          key=lambda row: row["speed_mps"])
        x = np.asarray([row["speed_mps"] for row in by_speed], dtype=float)
        y = np.asarray([row["outage_probability"] for row in by_speed], dtype=float)
        spread = np.asarray([row["outage_probability_ci95"] for row in by_speed], dtype=float)
        _trend_with_ci(axis, x, y, spread, method, method.replace(" matched", ""))
    axis.set(xlabel="UE speed (m/s)", ylabel="Rate-outage probability")
    axis.axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left",
                title=f"Fixed: {commanded_count}/16, delay={largest_delay} slots",
                title_fontsize=6.5)
    _finish(fig, output, "fig4_fixed_budget_mechanism")


def plot_ablation(rows: list[dict[str, object]], history_path: Path, output: Path,
                  central_speed: float, central_delay: int) -> None:
    methods = ("RA-FVV", "No future", "No readiness")
    values = {method: _select(rows, method=method, speed_mps=central_speed,
                              wake_delay=central_delay, switch_scale=1.0)[0] for method in methods}
    fig, axes = plt.subplots(1, 2, figsize=(3.48, 2.25), constrained_layout=True)
    x = np.arange(len(methods))
    axes[0].bar(x, [values[m]["energy_j"] for m in methods], color=[COLORS[m] for m in methods], width=0.68)
    axes[0].errorbar(x, [values[m]["energy_j"] for m in methods],
                     yerr=[values[m]["energy_j_ci95"] for m in methods], fmt="none", color="black", capsize=2)
    axes[0].set(ylabel="Energy per episode (J)", title="(a) Component ablation")
    axes[1].bar(x, [values[m]["outage_probability"] for m in methods], color=[COLORS[m] for m in methods], width=0.68)
    axes[1].errorbar(x, [values[m]["outage_probability"] for m in methods],
                     yerr=[values[m]["outage_probability_ci95"] for m in methods],
                     fmt="none", color="black", capsize=2)
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[1].set(ylabel="Rate-outage probability", title="(b) Reliability effect")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.set_xticks(x, ["RA-CRL", "w/o future", "w/o readiness"], rotation=16, ha="right")
    _finish(fig, output, "fig5_component_ablation")


def _moving_average(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_constraint_study(output: Path) -> None:
    summary = _rows(output / "constraint_summary.csv")
    metadata = json.loads((output / "constraint_metadata.json").read_text(encoding="utf-8"))
    ordered = ("RA-CRL", "Vanilla symmetric DQN", "Energy-only DQN")
    rows_by_name = {str(row["method"]): row for row in summary}
    fixed_name = next(name for name in rows_by_name if name.startswith("QoS-matched fixed"))
    names = (*ordered, fixed_name)
    labels = ("RA-CRL", "Symmetric DQN", "Energy-only DQN",
              f"Fixed $M={metadata['selected_fixed_count']}$")
    markers = ("*", "s", "^", "D")
    colors = [COLORS.get(name, "#7E6148") for name in names]
    colors[-1] = "#7E6148"
    fig, axis = plt.subplots(figsize=(3.48, 2.35), constrained_layout=True)
    for name, label, marker, color in zip(names, labels, markers, colors):
        row = rows_by_name[name]
        axis.errorbar(
            float(row["energy_j"]), float(row["outage_probability"]),
            xerr=float(row["energy_j_ci95"]), yerr=float(row["outage_probability_ci95"]),
            marker=marker, markersize=6.2 if name == "RA-CRL" else 4.8,
            color=color, ecolor=color, elinewidth=0.75, alpha=0.88,
            linestyle="none", capsize=1.5, label=label,
        )
    axis.axhline(float(metadata["outage_target"]), color="black", linestyle=":", linewidth=0.9)
    axis.set(xlabel="Energy per episode (J)", ylabel="Rate-outage probability")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="upper right", fontsize=6.7)
    _finish(fig, output, "fig5_constraint_learning")


def make_all(output: Path) -> None:
    _style()
    rows = _rows(output / "summary.csv")
    central_delay = int(np.median([row["wake_delay"] for row in rows if row["method"] == "RA-FVV"]))
    speeds = sorted(set(row["speed_mps"] for row in rows if row["method"] == "RA-FVV"))
    central_speed = speeds[len(speeds) // 2]
    plot_prediction(output)
    plot_speed(rows, output, central_delay)
    plot_delay(rows, output, central_speed)
    fixed_summary = output / "fixed_budget_summary.csv"
    if fixed_summary.exists():
        plot_fixed_budget(_rows(fixed_summary), output)
    else:
        plot_matched_cardinality(rows, output, central_speed)
    if (output / "constraint_summary.csv").exists():
        plot_constraint_study(output)
    elif all(any(row["method"] == method for row in rows)
             for method in ("RA-FVV", "No future", "No readiness")):
        plot_ablation(rows, output / "training_history.csv", output, central_speed, central_delay)
