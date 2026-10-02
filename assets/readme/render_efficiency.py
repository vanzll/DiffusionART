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
    "DiffusionART": ("#E54763", "-"),
    "Flow-GRPO-Fast": ("#2685A7", "--"),
    "DiffusionNFT (EMA)": ("#7775B5", "-"),
    "DanceGRPO": ("#278B67", "-."),
}


def smooth(values):
    decay = 2 ** (-1 / 8)
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
        rewards = [float(row["reward_raw"]) if task == "t2v" else
                   (float(row["VQ"]) + float(row["MQ"])) / 2 for row in rows]
        assert all(math.isfinite(value) for value in hours + rewards)
        assert all(left <= right for left, right in zip(hours, hours[1:]))
        display = smooth(rewards)
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


def render(task, title):
    series, evidence = load(task)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.labelweight": "bold",
        "axes.titleweight": "bold", "axes.edgecolor": "#9BA5AF",
        "axes.linewidth": 0.8, "text.color": "#26323D",
        "xtick.color": "#596570", "ytick.color": "#596570",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    fig.subplots_adjust(left=.15, right=.97, bottom=.27, top=.88)
    for name, (color, style) in METHODS.items():
        if name in series:
            ax.plot(*series[name], color=color, linestyle=style,
                    linewidth=2.1, label=name, zorder=4 if name == "DiffusionART" else 3)
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Elapsed GPU hours", labelpad=7)
    ax.set_ylabel("Mean raw reward", labelpad=7)
    ax.set_xlim(0, max(hours[-1] for hours, _ in series.values()) * 1.03)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)
    ax.grid(color="#E9EDF1", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3.5, width=.8)
    legend = fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
                        bbox_to_anchor=(.55, .015), ncol=2, fontsize=8.5,
                        handlelength=2.0, columnspacing=1.25, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.savefig(ROOT / f"{task}-gpu-hours.png", dpi=240)
    plt.close(fig)
    return evidence


if __name__ == "__main__":
    provenance = {
        "training_weights": "VQ:MQ = 1:1; TA = 0",
        "display_reward": "(raw VQ + raw MQ) / 2",
        "cost": "W&B _runtime * 32 / 3600; includes evaluation and waits",
        "smoothing": "Causal EMA half-life 8 optimizer updates for every curve",
        "selection": "All observed rows in each frozen export; no extrapolation or stitching",
        "t2v_raw_conversion": {
            "VQ": {"scale": 2.2476, "offset": 3.6757},
            "MQ": {"scale": 1.3811, "offset": 1.1646},
            "note": "The source export recovers raw scores from fixed normalized scorer calibration.",
        },
        "t2v": render("t2v", "Text-to-video"),
        "i2v": render("i2v", "Image-to-video"),
    }
    (ROOT / "efficiency.json").write_text(json.dumps(provenance, indent=2) + "\n")
