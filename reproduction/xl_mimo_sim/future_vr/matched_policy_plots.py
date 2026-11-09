from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


METHODS = (
    "CSI + shared DQN count",
    "Future + shared DQN count",
    "RA-CRL",
    "Oracle + shared DQN count",
)
DISPLAY = {
    METHODS[0]: "Current-CSI ranking",
    METHODS[1]: "Unaligned-future ranking",
    METHODS[2]: "RA-FVV ranking",
    METHODS[3]: "Noncausal oracle",
}
COLORS = {
    METHODS[0]: "#0F766E", METHODS[1]: "#6B7280",
    METHODS[2]: "#0B5CAD", METHODS[3]: "#8B5E3C",
}
MARKERS = {METHODS[0]: "^", METHODS[1]: "s", METHODS[2]: "o", METHODS[3]: "D"}
LINES = {METHODS[0]: ":", METHODS[1]: "--", METHODS[2]: "-", METHODS[3]: "-."}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.8,
        "legend.fontsize": 6.8, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _trend(axis: plt.Axes, rows: list[dict[str, str]], xfield: str,
           field: str, method: str) -> None:
    values = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row[xfield]))
    x = np.asarray([float(row[xfield]) for row in values])
    mean = np.asarray([float(row[field]) for row in values])
    low = np.asarray([float(row[field + "_ci95_low"]) for row in values])
    high = np.asarray([float(row[field + "_ci95_high"]) for row in values])
    axis.fill_between(x, low, high, color=COLORS[method], alpha=0.11, linewidth=0)
    axis.plot(x, mean, color=COLORS[method], linestyle=LINES[method],
              marker=MARKERS[method], label=DISPLAY[method])


def _finish(fig: plt.Figure, output: Path, stem: str) -> None:
    for axis in fig.axes:
        axis.grid(True, color="#D7DEE8", linewidth=0.45, alpha=0.75)
        axis.tick_params(direction="in", top=True, right=True, width=0.7)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_delay(output: Path) -> None:
    summary = _read(output / "summary.csv")
    paired = _read(output / "paired_summary.csv")
    central_speed = max(
        {float(row["speed_mps"]) for row in summary},
        key=lambda speed: sum(np.isclose(float(row["speed_mps"]), speed) for row in summary),
    )
    subset = [row for row in summary if np.isclose(float(row["speed_mps"]), central_speed)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.73, wspace=0.43)
    for method in METHODS:
        _trend(axes[0], subset, "wake_delay", "outage_probability", method)
        _trend(axes[1], subset, "wake_delay", "energy_j", method)
    for baseline, color, marker, linestyle in (
        (METHODS[0], COLORS[METHODS[0]], "^", ":"),
        (METHODS[1], COLORS[METHODS[1]], "s", "--"),
    ):
        values = sorted(
            (row for row in paired if row["baseline"] == baseline
             and np.isclose(float(row["speed_mps"]), central_speed)),
            key=lambda row: int(row["wake_delay"]),
        )
        x = np.asarray([int(row["wake_delay"]) for row in values])
        mean = np.asarray([float(row["outage_probability_reduction"]) for row in values])
        low = np.asarray([float(row["outage_probability_reduction_ci95_low"]) for row in values])
        high = np.asarray([float(row["outage_probability_reduction_ci95_high"]) for row in values])
        axes[2].fill_between(x, low, high, color=color, alpha=0.11, linewidth=0)
        axes[2].plot(x, mean, color=color, marker=marker, linestyle=linestyle,
                     label=("over current CSI" if baseline == METHODS[0]
                            else "over unaligned future"))
    axes[0].set(xlabel="Wake-up delay (slots)", ylabel="Rate-outage probability", title="(a) Reliability")
    axes[1].set(xlabel="Wake-up delay (slots)", ylabel="Episode energy (J)", title="(b) Energy under shared count")
    axes[2].set(xlabel="Wake-up delay (slots)", ylabel="Paired outage reduction", title="(c) RA-FVV alignment gain")
    axes[0].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[2].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.99))
    axes[2].legend(frameon=False, loc="upper left", handlelength=2.2)
    _finish(fig, output, "fig_shared_count_delay")


def plot_speed(output: Path) -> None:
    summary = _read(output / "summary.csv")
    central_delay = max(
        {int(row["wake_delay"]) for row in summary},
        key=lambda delay: sum(int(row["wake_delay"]) == delay for row in summary),
    )
    subset = [row for row in summary if int(row["wake_delay"]) == central_delay]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.73, wspace=0.30)
    for method in METHODS:
        _trend(axes[0], subset, "speed_mps", "outage_probability", method)
        _trend(axes[1], subset, "speed_mps", "energy_efficiency_mbit_j", method)
    axes[0].set(xlabel="UE speed (m/s)", ylabel="Rate-outage probability", title="(a) Mobility robustness")
    axes[1].set(xlabel="UE speed (m/s)", ylabel="Energy efficiency (Mbit/J)", title="(b) Delivered bits per joule")
    axes[0].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.99))
    _finish(fig, output, "fig_shared_count_speed")


def make_matched_policy_figures(output: Path) -> None:
    _style()
    plot_delay(output)
    plot_speed(output)
