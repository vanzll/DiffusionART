"""Render README panels from frozen paper curve exports.

Run: uv run --no-project --with matplotlib python assets/readme/render_efficiency.py
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
METHODS = {
    "DiffusionART": ("#E54763", "o", "DiffusionART (ours)"),
    "Flow-GRPO-Fast": ("#2685A7", "s", "Flow-GRPO-Fast"),
    "DiffusionNFT (EMA)": ("#7775B5", "D", "DiffusionNFT"),
}
PAIRWISE = {
    "Pairwise, K=2": ("#E54763", "o", "K=2"),
    "Pairwise, K=8": ("#2685A7", "s", "Pairwise, K=8"),
}


def smooth(values, half_life=8):
    decay = 2 ** (-1 / half_life)
    result = [values[0]]
    for value in values[1:]:
        result.append(decay * result[-1] + (1 - decay) * value)
    return result


def load(task):
    path = ROOT / f"{task}-curves.csv"
    groups = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["method"]
            if name == "DiffusionNFT":
                name = "DiffusionNFT (EMA)"
            groups.setdefault(name, []).append(row)
    series = {}
    for name, rows in groups.items():
        steps = [int(row["optimizer_step"]) for row in rows]
        assert steps == list(range(1, steps[-1] + 1))
        assert len({row["run"] for row in rows}) == 1
        hours = [float(row["elapsed_gpu_hours" if task == "t2v" else "gpu_hours"])
                 for row in rows]
        if task == "t2v":
            rewards = [float(row["reward_raw"]) for row in rows]
        elif task == "pairwise":
            rewards = [float(row["VQ"]) for row in rows]
        else:
            rewards = [(float(row["VQ"]) + float(row["MQ"])) / 2 for row in rows]
        assert all(math.isfinite(value) for value in hours + rewards)
        assert all(left <= right for left, right in zip(hours, hours[1:]))
        display = smooth(rewards, 12 if task == "pairwise" else 8)
        if task == "t2v":
            assert all(math.isclose(value, float(row["reward_ema8"]), abs_tol=1e-10)
                       for value, row in zip(display, rows))
        series[name] = (hours, display)
    return series, {
        "source": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runs": {name: {"id": rows[0]["run"], "observed_updates": len(rows),
                         "last_gpu_hours": series[name][0][-1]}
                 for name, rows in groups.items()},
    }


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.labelweight": "bold",
        "axes.titleweight": "bold", "axes.edgecolor": "#9BA5AF",
        "axes.linewidth": 0.8, "text.color": "#26323D",
        "xtick.color": "#596570", "ytick.color": "#596570",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def plot_curves(ax, series, task, title):
    displayed = {}
    for name, (color, marker, label) in (PAIRWISE if task == "pairwise" else METHODS).items():
        if name in series:
            hours, values = series[name]
            # Display observed vertices at a regular GPU-hour rhythm; no interpolated rewards.
            targets = [hours[0] + (hours[-1] - hours[0]) * i / 25 for i in range(26)]
            marks = sorted({min(range(len(hours)), key=lambda j: abs(hours[j] - x)) for x in targets})
            displayed[name] = [i + 1 for i in marks]
            ax.plot([hours[i] for i in marks], [values[i] for i in marks],
                    color=color, linestyle="-", solid_capstyle="round",
                    linewidth=1.35, marker=marker, markersize=3.8,
                    markeredgecolor="white", markeredgewidth=.5, label=label.replace("Pairwise, ", ""),
                    zorder=4 if name == "DiffusionART" else 3)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("GPU hours", labelpad=7)
    ax.set_ylabel("VQ train reward" if task == "pairwise" else "Mean train reward", labelpad=7)
    names = PAIRWISE if task == "pairwise" else METHODS
    ax.set_xlim(0, max(series[name][0][-1] for name in names) * 1.03)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)
    ax.grid(color="#E9EDF1", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3.5, width=.8)
    legend = ax.legend(loc="lower right", fontsize=8, handlelength=1.8,
                       labelspacing=.45, borderaxespad=.8, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    return displayed


def annotate_ratios(ax, series):
    hours, rewards = series["DiffusionART"]
    evidence = {}
    for name in ("Flow-GRPO-Fast", "DiffusionNFT (EMA)"):
        baseline_hours, baseline_rewards = series[name]
        threshold = baseline_rewards[-1]
        index = next(i for i, value in enumerate(rewards) if value >= threshold)
        ratio = baseline_hours[-1] / hours[index]
        color = METHODS[name][0]
        ax.hlines(threshold, hours[index], baseline_hours[-1], color=color,
                  linestyles=(0, (3, 3)), linewidth=.7, alpha=.65, zorder=2)
        label_x = hours[index] + .80 * (baseline_hours[-1] - hours[index])
        offset = -15 if name == "Flow-GRPO-Fast" and threshold >= 8 else 9
        if name == "DiffusionNFT (EMA)" and threshold < 8:
            offset = -15
        ax.annotate(f"{ratio:.1f}×", (label_x, threshold),
                    xytext=(0, offset),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=color, weight="bold")
        evidence[name] = {"reward_threshold": threshold, "art_first_step": index + 1,
                          "art_gpu_hours": hours[index], "baseline_gpu_hours": baseline_hours[-1],
                          "ratio": ratio}
    return evidence


def render_comparisons():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.89, wspace=.26)
    result = {}
    for ax, task, title in zip(axes, ("t2v", "i2v"), ("Text-to-video", "Image-to-video")):
        series, evidence = load(task)
        evidence["displayed_steps"] = plot_curves(ax, series, task, title)
        evidence["matched_reward_ratios"] = annotate_ratios(ax, series)
        result[task] = evidence
    fig.savefig(ROOT / "efficiency-comparison.png", dpi=280)
    plt.close(fig)
    return result


def render_pairwise():
    series, evidence = load("pairwise")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.89, wspace=.35)
    evidence["displayed_steps"] = plot_curves(axes[0], series, "pairwise", "Pairwise feedback")
    performance = json.loads((ROOT / "pairwise-performance.json").read_text())
    values = [row["VQ"] for row in performance["rows"]]
    ax = axes[1]
    bars = ax.bar([0, 1], values, width=.55, color=[PAIRWISE[name][0] for name in PAIRWISE])
    ax.set_xticks([0, 1], ["K=2", "K=8"])
    ax.set_ylabel("Final eval VQ", labelpad=7)
    ax.set_title("Final performance", fontsize=12, pad=10)
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EDF1", linewidth=.7)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5, fontsize=10, weight="bold")
    fig.savefig(ROOT / "pairwise-comparison.png", dpi=280)
    plt.close(fig)
    evidence["final_performance"] = performance
    return evidence


if __name__ == "__main__":
    configure_style()
    comparisons = render_comparisons()
    provenance = {
        "comparison_training_weights": "VQ:MQ = 1:1; TA = 0",
        "display_reward": "(raw VQ + raw MQ) / 2",
        "cost": "W&B _runtime * 32 / 3600; includes evaluation and waits",
        "comparison_smoothing": "Common causal EMA half-life 8 optimizer updates",
        "selection": "All observed rows in each frozen export; no extrapolation or stitching",
        "t2v_raw_conversion": {
            "VQ": {"scale": 2.2476, "offset": 3.6757},
            "MQ": {"scale": 1.3811, "offset": 1.1646},
            "note": "The source export recovers raw scores from fixed normalized scorer calibration.",
        },
        **comparisons,
        "pairwise": {
            **render_pairwise(),
            "training_weights": "VQ only; MQ = TA = 0",
            "display_reward": "Raw VQ mean",
            "smoothing": "Common causal EMA half-life 12 optimizer updates",
            "source_export": "wandb_history_raw.json",
            "source_export_sha256": "55a28a09964bd3ee91f66f3b1ac07c711d4b2fc4897fb815f43e0cc1cd486357",
        },
        "presentation": {
            "line": "Thin straight connectors between up to 26 observed marker vertices, evenly spaced in GPU hours; full series retained in source CSVs",
            "legend": "Inside the axes, lower right; no frame",
            "comparison_labels": ["DiffusionART (ours)", "Flow-GRPO-Fast", "DiffusionNFT"],
            "comparison_axes": ["GPU hours", "Mean train reward"],
            "ratio_rule": "Baseline final eight-update-smoothed mean reward; ART first observed crossing using the same smoothing. No interpolation or extrapolation.",
        },
    }
    (ROOT / "efficiency.json").write_text(json.dumps(provenance, indent=2) + "\n")
