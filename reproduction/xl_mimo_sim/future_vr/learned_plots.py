from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


COLORS = {"CSI-CRL": "#0F766E", "F-CRL": "#6B7280", "RA-CRL": "#0B5CAD"}
MARKERS = {"CSI-CRL": "^", "F-CRL": "s", "RA-CRL": "o"}
LINES = {"CSI-CRL": ":", "F-CRL": "--", "RA-CRL": "-"}
METHODS = ("CSI-CRL", "F-CRL", "RA-CRL")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _finish(fig: plt.Figure, output: Path, stem: str) -> None:
    for axis in fig.axes:
        axis.grid(True, color="#D7DEE8", linewidth=0.45, alpha=0.75)
        axis.tick_params(direction="in", top=True, right=True, width=0.7)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _trend(axis: plt.Axes, x: np.ndarray, mean: np.ndarray, ci: np.ndarray, method: str) -> None:
    axis.fill_between(x, mean - ci, mean + ci, color=COLORS[method], alpha=0.12, linewidth=0)
    axis.plot(x, mean, color=COLORS[method], marker=MARKERS[method],
              linestyle=LINES[method], label=method)


def plot_learned_speed(output: Path) -> None:
    rows = _read(output / "summary.csv")
    central_delay = max(
        set(int(row["wake_delay"]) for row in rows),
        key=lambda value: sum(int(row["wake_delay"]) == value for row in rows),
    )
    panels = (
        ("energy_j", "energy_j_ci95", "Episode energy (J)"),
        ("energy_efficiency_mbit_j", "energy_efficiency_mbit_j_ci95", "Energy efficiency (Mbit/J)"),
        ("outage_probability", "outage_probability_ci95", "Rate-outage probability"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.74, wspace=0.42)
    for method in METHODS:
        values = sorted(
            (row for row in rows if row["method"] == method and int(row["wake_delay"]) == central_delay),
            key=lambda row: float(row["speed_mps"]),
        )
        x = np.asarray([float(row["speed_mps"]) for row in values])
        for axis, (field, ci_field, ylabel) in zip(axes, panels):
            _trend(
                axis,
                x,
                np.asarray([float(row[field]) for row in values]),
                np.asarray([float(row[ci_field]) for row in values]),
                method,
            )
            axis.set(xlabel="UE speed (m/s)", ylabel=ylabel)
    axes[0].set_title("(a) Energy")
    axes[1].set_title("(b) Delivered bits per joule")
    axes[2].set_title("(c) Reliability")
    axes[2].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_ylim(bottom=0.0)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.99))
    _finish(fig, output, "fig_learned_speed")


def plot_delay_mechanism(output: Path) -> None:
    summary = _read(output / "summary.csv")
    paired = _read(output / "paired_summary.csv")
    central_speed = max(
        set(float(row["speed_mps"]) for row in summary),
        key=lambda value: sum(np.isclose(float(row["speed_mps"]), value) for row in summary),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.74, wspace=0.43)
    for method in METHODS:
        values = sorted(
            (row for row in summary if row["method"] == method
             and np.isclose(float(row["speed_mps"]), central_speed)),
            key=lambda row: int(row["wake_delay"]),
        )
        x = np.asarray([int(row["wake_delay"]) for row in values])
        _trend(axes[0], x, np.asarray([float(row["energy_j"]) for row in values]),
               np.asarray([float(row["energy_j_ci95"]) for row in values]), method)
        _trend(axes[1], x, np.asarray([float(row["outage_probability"]) for row in values]),
               np.asarray([float(row["outage_probability_ci95"]) for row in values]), method)
    for baseline in ("CSI-CRL", "F-CRL"):
        values = sorted(
            (row for row in paired if row["baseline"] == baseline
             and np.isclose(float(row["speed_mps"]), central_speed)),
            key=lambda row: int(row["wake_delay"]),
        )
        x = np.asarray([int(row["wake_delay"]) for row in values])
        y = np.asarray([float(row["outage_reduction"]) for row in values])
        ci = np.asarray([float(row["outage_reduction_ci95"]) for row in values])
        label = f"RA-CRL vs {baseline}"
        color = COLORS[baseline]
        axes[2].fill_between(x, y - ci, y + ci, color=color, alpha=0.12, linewidth=0)
        axes[2].plot(x, y, color=color, marker=MARKERS[baseline],
                     linestyle=LINES[baseline], label=label)
    axes[0].set(xlabel="Wake-up delay (slots)", ylabel="Episode energy (J)", title="(a) Energy")
    axes[1].set(xlabel="Wake-up delay (slots)", ylabel="Rate-outage probability",
                title="(b) Reliability")
    axes[2].set(xlabel="Wake-up delay (slots)", ylabel="Paired outage reduction",
                title="(c) Readiness-alignment gain")
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[2].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.43, 0.99))
    axes[2].legend(frameon=False, fontsize=6.5, loc="best")
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    _finish(fig, output, "fig_learned_delay")


def plot_training_and_actions(output: Path) -> None:
    history = _read(output / "training_history.csv")
    actions = _read(output / "action_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.78, wspace=0.30)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in history:
        grouped[(row["method"], int(row["episode"]))].append(float(row["outage"]))
    for method in METHODS:
        episodes = sorted({episode for name, episode in grouped if name == method})
        mean = np.asarray([np.mean(grouped[(method, episode)]) for episode in episodes])
        if mean.size >= 15:
            kernel = np.ones(15) / 15
            mean = np.convolve(np.pad(mean, (7, 7), mode="edge"), kernel, mode="valid")
        axes[0].plot(episodes, mean, color=COLORS[method], linestyle=LINES[method], label=method)
    action_values = [
        [int(row["commanded_count"]) for row in actions if row["method"] == method]
        for method in METHODS
    ]
    box = axes[1].boxplot(action_values, tick_labels=METHODS, showfliers=False, patch_artist=True,
                          widths=0.55, medianprops={"color": "black", "linewidth": 1.0})
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[method])
    axes[0].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[0].set(xlabel="Training episode", ylabel="Mean training outage",
                title="(a) Training behavior")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1].set(ylabel="Commanded subarray count $M_t$",
                title="(b) Test-time action distribution")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    _finish(fig, output, "fig_training_actions")


def make_learned_figures(output: Path) -> None:
    _style()
    plot_learned_speed(output)
    plot_delay_mechanism(output)
    plot_training_and_actions(output)
