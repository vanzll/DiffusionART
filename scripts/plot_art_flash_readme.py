"""Render the two-panel Flash illustration from an explicitly supplied CSV."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="CSV with method, metric, optimizer_updates and value columns")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets/readme/art-flash-comparison")
    args = parser.parse_args()
    with args.data.open() as stream:
        rows = list(csv.DictReader(stream))
    styles = {
        "Original": dict(color="#526779", linestyle="--", marker="o"),
        "DiffusionART-Flash": dict(color="#D55E00", linestyle="-", marker="D"),
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
    for index, metric in enumerate(("VQ", "MQ")):
        ax = axes[index]
        maximum = 0
        for method, style in styles.items():
            points = [r for r in rows if r["method"] == method and r["metric"] == metric]
            if not points:
                raise ValueError(f"Missing {method} {metric} observations")
            x = [float(r["optimizer_updates"]) for r in points]
            y = [float(r["value"]) for r in points]
            if any(not math.isfinite(v) for v in x + y) or any(b <= a for a, b in zip(x, x[1:])):
                raise ValueError("Plot observations must be finite and strictly ordered")
            maximum = max(maximum, x[-1])
            ax.plot(x, y, label=method, **style, linewidth=1.25,
                    markersize=1.8, markeredgewidth=0)
            ax.plot(x[-1], y[-1], marker=style["marker"], markersize=4,
                    color=style["color"], linestyle="none")
        ax.set_xlim(0, math.ceil(maximum / 50) * 50)
        ax.set_ylim(0, 16 if metric == "VQ" else 12)
        ax.set_yticks([0, 4, 8, 12, 16] if metric == "VQ" else [0, 3, 6, 9, 12])
        ax.grid(axis="y", linewidth=.45, alpha=.15)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.text(.025, .93, f"({chr(97 + index)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.51, .005),
               ncol=2, frameon=False, fontsize=9, columnspacing=2, handlelength=2.5,
               markerscale=2)
    fig.subplots_adjust(left=.07, right=.985, top=.95, bottom=.28, wspace=.23)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = args.output.with_suffix("." + extension)
        fig.savefig(path, dpi=600, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
