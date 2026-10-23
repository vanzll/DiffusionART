"""Render the README comparison from frozen, targeted W&B exports; no API access."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "assets/readme/art-flash"
    meta = json.loads((data / "metadata.json").read_text())
    with (data / "curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 4.9), sharey="row")
    labels = {"Original": "Original (9 training timesteps)",
              "DiffusionART-Flash": "DiffusionART-Flash (1 training timestep)"}
    for row_index, metric in enumerate(("VQ", "MQ")):
        for column, (key, xlabel) in enumerate((
                ("optimizer_updates", "Optimizer updates"),
                ("elapsed_gpu_hours", "Elapsed GPU-hours"))):
            ax = axes[row_index, column]
            for method in labels:
                points = [r for r in rows if r["method"] == method and r["metric"] == metric]
                x, y = [float(r[key]) for r in points], [float(r["value"]) for r in points]
                style = meta["style"][method]
                ax.plot(x, y, label=labels[method], color=style["color"],
                        linestyle=style["linestyle"], marker=style["marker"],
                        linewidth=1.25, markersize=1.8, markeredgewidth=0)
                ax.plot(x[-1], y[-1], marker=style["marker"], markersize=4,
                        color=style["color"], linestyle="none")
            ax.set_xlim(0, 410 if column == 0 else 700)
            ax.set_ylim(0, 16 if metric == "VQ" else 12)
            ax.set_xticks([0, 100, 200, 300, 400] if column == 0 else [0, 200, 400, 600])
            ax.set_yticks([0, 4, 8, 12, 16] if metric == "VQ" else [0, 3, 6, 9, 12])
            ax.grid(axis="y", linewidth=.45, alpha=.15)
            ax.set_xlabel(xlabel)
            ax.text(.025, .93, f"({chr(97 + row_index * 2 + column)})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"Raw {metric} score")
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", bbox_to_anchor=(.51, .015),
               ncol=2, frameon=False, fontsize=9, columnspacing=2, handlelength=2.5,
               markerscale=2)
    fig.subplots_adjust(left=.075, right=.985, top=.97, bottom=.17, hspace=.38, wspace=.16)
    output = root / "assets/readme/art-flash-comparison"
    for extension in ("svg", "pdf", "png"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=600, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
