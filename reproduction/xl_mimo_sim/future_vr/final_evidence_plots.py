from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ABS_METHODS = ("Full array", "Static HRNP", "Current-CSI", "Symmetric guard", "RA-CRL")
COLORS = {
    "Full array": "#3C5488", "Static HRNP": "#8491B4", "Current-CSI": "#0F766E",
    "Symmetric guard": "#D97745", "RA-CRL": "#0B5CAD", "Energy-only DQN": "#A23B72",
}
MARKERS = {
    "Full array": "o", "Static HRNP": "s", "Current-CSI": "^",
    "Symmetric guard": "D", "RA-CRL": "o", "Energy-only DQN": "X",
}
LINES = {
    "Full array": "--", "Static HRNP": "-.", "Current-CSI": ":",
    "Symmetric guard": "--", "RA-CRL": "-", "Energy-only DQN": "--",
}


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


def _finish(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for axis in fig.axes:
        axis.grid(True, color="#D7DEE8", linewidth=0.45, alpha=0.75)
        axis.tick_params(direction="in", top=True, right=True, width=0.7)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _trend(axis: plt.Axes, rows: list[dict[str, str]], xfield: str,
           field: str, method: str) -> None:
    selected = sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: float(row[xfield]),
    )
    x = np.asarray([float(row[xfield]) for row in selected])
    mean = np.asarray([float(row[field]) for row in selected])
    low = np.asarray([float(row[field + "_ci95_low"]) for row in selected])
    high = np.asarray([float(row[field + "_ci95_high"]) for row in selected])
    axis.fill_between(x, low, high, color=COLORS[method], alpha=0.10, linewidth=0)
    display = "WA-RL" if method == "RA-CRL" else method
    axis.plot(x, mean, color=COLORS[method], marker=MARKERS[method],
              linestyle=LINES[method], label=display)


def plot_absolute_baselines(input_dir: Path, output: Path) -> None:
    rows = _read(input_dir / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.20, top=0.72, wspace=0.43)
    for method in ABS_METHODS:
        _trend(axes[0], rows, "speed_mps", "energy_j", method)
        _trend(axes[1], rows, "speed_mps", "energy_efficiency_mbit_j", method)
        _trend(axes[2], rows, "speed_mps", "outage_probability", method)
    axes[0].set(xlabel="UE speed (m/s)", ylabel="Episode energy (J)", title="(a) Net energy")
    axes[1].set(xlabel="UE speed (m/s)", ylabel="Energy efficiency (Mbit/J)", title="(b) Delivered bits per joule")
    axes[2].set(xlabel="UE speed (m/s)", ylabel="Rate-outage probability", title="(c) Reliability cost")
    axes[2].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    speeds = sorted({float(row["speed_mps"]) for row in rows})
    for axis in axes:
        axis.set_xticks(speeds)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.99))
    _finish(fig, output, "fig_absolute_energy_speed")


def plot_constraint_ablation(input_dir: Path, output: Path) -> None:
    rows = _read(input_dir / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.15))
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.23, top=0.72, wspace=0.48)
    for method in ("RA-CRL", "Energy-only DQN"):
        _trend(axes[0], rows, "load_scale", "energy_j", method)
        _trend(axes[1], rows, "load_scale", "outage_probability", method)
    axes[0].set(xlabel="Demand scale", ylabel="Episode energy (J)", title="(a) Energy")
    axes[1].set(xlabel="Demand scale", ylabel="Rate-outage probability", title="(b) Reliability")
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=0.9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.55, 0.99))
    _finish(fig, output, "fig_constraint_ablation")


def make_final_evidence_figures(
    absolute_dir: Path, constraint_dir: Path, output: Path
) -> None:
    _style()
    plot_absolute_baselines(absolute_dir, output)
    plot_constraint_ablation(constraint_dir, output)
