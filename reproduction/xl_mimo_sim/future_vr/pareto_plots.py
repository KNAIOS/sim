from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


METHODS = ("Current CSI", "Unaligned future", "Readiness aligned", "Noncausal oracle")
COLORS = {
    "Current CSI": "#B64040",
    "Unaligned future": "#0F766E",
    "Readiness aligned": "#0B5CAD",
    "Noncausal oracle": "#6B7280",
}
MARKERS = {"Current CSI": "^", "Unaligned future": "s", "Readiness aligned": "o", "Noncausal oracle": "D"}
LINES = {"Current CSI": ":", "Unaligned future": "--", "Readiness aligned": "-", "Noncausal oracle": "-."}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_pareto_figure(input_dir: Path, output_dir: Path) -> None:
    _style()
    rows = _read(input_dir / "summary.csv")
    target = _read(input_dir / "confidence_target_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.76, wspace=0.34)
    for method in METHODS:
        selected = sorted(
            (
                row for row in rows
                if row["method"] == method and int(row["commanded_count"]) >= 3
            ),
            key=lambda row: int(row["commanded_count"]),
        )
        outage = np.asarray([float(row["outage_probability"]) for row in selected])
        energy = np.asarray([float(row["energy_j"]) for row in selected])
        axes[0].plot(
            outage,
            energy,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINES[method],
            label=method,
        )
        if method == "Readiness aligned":
            for row, x_value, y_value in zip(selected, outage, energy):
                count = int(row["commanded_count"])
                if count <= 6:
                    axes[0].annotate(
                        f"$M={count}$",
                        (x_value, y_value),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=6.5,
                        color=COLORS[method],
                    )

    target_by_method = {row["method"]: row for row in target}
    baseline_methods = ("Current CSI", "Unaligned future", "Readiness aligned")
    x = np.arange(len(baseline_methods))
    energy = np.asarray([float(target_by_method[method]["energy_j"]) for method in baseline_methods])
    low = np.asarray([float(target_by_method[method]["energy_j_ci95_low"]) for method in baseline_methods])
    high = np.asarray([float(target_by_method[method]["energy_j_ci95_high"]) for method in baseline_methods])
    axes[1].bar(
        x,
        energy,
        width=0.62,
        color=[COLORS[method] for method in baseline_methods],
        edgecolor="#172033",
        linewidth=0.7,
        yerr=np.vstack((energy - low, high - energy)),
        capsize=3,
    )
    axes[1].set_xticks(x, ("Current\nCSI", "Unaligned\nfuture", "Readiness\naligned"))
    for index, method in enumerate(baseline_methods):
        count = float(target_by_method[method]["commanded_count"])
        axes[1].text(index, energy[index] + max(high - low) * 0.10, f"$M={count:.0f}$", ha="center", fontsize=7)

    axes[0].axvline(0.05, color="black", linewidth=0.9, linestyle=(0, (2, 2)))
    axes[0].text(0.0515, 188.0, "5% target", rotation=90, va="top", fontsize=6.8)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_xlim(0.0, 0.17)
    axes[0].set(
        xlabel="Rate-outage probability (lower is better)",
        ylabel="Episode energy (J; lower is better)",
        title="(a) Energy--outage frontier near the target",
    )
    axes[1].set(
        ylabel="Energy at 95%-UCB target (J)",
        title="(b) Confidence-qualified operating point",
    )
    axes[1].set_ylim(0, max(high) * 1.20)
    for axis in axes:
        axis.grid(True, color="#D7DEE8", linewidth=0.45, alpha=0.8, axis="y")
        axis.tick_params(direction="in", top=True, right=True, width=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})):
        fig.savefig(output_dir / f"fig_energy_outage_pareto.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def make_combined_evidence_figure(
    pareto_dir: Path, delay_dir: Path, output_dir: Path
) -> None:
    _style()
    holdout_summary = pareto_dir / "test_summary.csv"
    holdout_target = pareto_dir / "holdout_target_summary.csv"
    rows = _read(holdout_summary if holdout_summary.exists() else pareto_dir / "summary.csv")
    target = _read(holdout_target if holdout_target.exists() else pareto_dir / "confidence_target_summary.csv")
    delay_rows = _read(delay_dir / "paired_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.73, wspace=0.42)

    for method in METHODS:
        selected = sorted(
            (
                row for row in rows
                if row["method"] == method and 3 <= int(row["commanded_count"]) <= 6
            ),
            key=lambda row: int(row["commanded_count"]),
        )
        outage = np.asarray([float(row["outage_probability"]) for row in selected])
        energy = np.asarray([float(row["energy_j"]) for row in selected])
        axes[0].plot(
            outage, energy, color=COLORS[method], marker=MARKERS[method],
            linestyle=LINES[method], label=method,
        )
    axes[0].axvline(0.05, color="black", linewidth=0.9, linestyle=(0, (2, 2)))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_xlim(0.0, 0.17)
    axes[0].set(
        xlabel="Rate outage", ylabel="Episode energy (J)",
        title="(a) Energy--outage frontier",
    )

    target_by_method = {row["method"]: row for row in target}
    bar_methods = ("Current CSI", "Unaligned future", "Readiness aligned")
    x = np.arange(len(bar_methods))
    energy_key = "test_energy_j" if "test_energy_j" in target[0] else "energy_j"
    low_key = energy_key + "_ci95_low"
    high_key = energy_key + "_ci95_high"
    energy = np.asarray([float(target_by_method[method][energy_key]) for method in bar_methods])
    low = np.asarray([float(target_by_method[method][low_key]) for method in bar_methods])
    high = np.asarray([float(target_by_method[method][high_key]) for method in bar_methods])
    axes[1].bar(
        x, energy, width=0.64, color=[COLORS[method] for method in bar_methods],
        edgecolor="#172033", linewidth=0.7,
        yerr=np.vstack((energy - low, high - energy)), capsize=2.5,
    )
    axes[1].set_xticks(x, ("Current\nCSI", "Unaligned\nfuture", "Readiness\naligned"))
    for index, method in enumerate(bar_methods):
        count = int(target_by_method[method]["commanded_count"])
        axes[1].text(index, energy[index] + 3.2, f"$M={count}$", ha="center", fontsize=6.8)
    axes[1].text(2, energy[2] - 16.0, "7.9%", color="white", ha="center", fontweight="bold", fontsize=7.2)
    axes[1].set_ylim(0, 180)
    axes[1].set(
        ylabel="Independent-test energy (J)",
        title="(b) Calibrate then test",
    )

    for baseline in ("Current CSI", "Unaligned future"):
        selected = sorted(
            (row for row in delay_rows if row["baseline"] == baseline),
            key=lambda row: int(row["wake_delay"]),
        )
        delay = np.asarray([int(row["wake_delay"]) for row in selected])
        mean = np.asarray([float(row["outage_probability_reduction"]) for row in selected])
        low = np.asarray([float(row["outage_probability_reduction_ci95_low"]) for row in selected])
        high = np.asarray([float(row["outage_probability_reduction_ci95_high"]) for row in selected])
        axes[2].fill_between(delay, low, high, color=COLORS[baseline], alpha=0.12, linewidth=0)
        axes[2].plot(
            delay, mean, color=COLORS[baseline], marker=MARKERS[baseline],
            linestyle=LINES[baseline], label=baseline,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle=(0, (2, 2)))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set(
        xlabel="Wake-up delay (control slots)",
        ylabel="Paired outage reduction",
        title="(c) Readiness-alignment gain",
    )

    for axis in axes:
        axis.grid(True, color="#D7DEE8", linewidth=0.45, alpha=0.8)
        axis.tick_params(direction="in", top=True, right=True, width=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})):
        fig.savefig(output_dir / f"fig_reliability_matched_energy.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
