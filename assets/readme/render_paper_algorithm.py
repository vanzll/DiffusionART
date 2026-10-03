"""Retypeset paper Algorithm 1 for a narrow README column; steps are unchanged.

Run: uv run --no-project --with matplotlib python assets/readme/render_paper_algorithm.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def render():
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif"})
    fig = plt.figure(figsize=(5, 5.8), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.05, .955, "Algorithm 1  DiffusionART", fontsize=12, weight="bold")
    ax.text(.05, .915, "One training iteration", fontsize=10, color="#596570")
    ax.plot([.05, .95], [.89, .89], color="#D1D9E0", linewidth=.8)
    ax.text(.05, .85, "Rollout Stage", fontsize=10, weight="bold")
    lines = [
        (.802, "1.", r"Sample with an ODE solver using $v_\theta^{(0)}=v_\theta$."),
        (.763, "", r"For each prompt $c$, collect $K$ endpoints"),
        (.724, "", r"with rewards $R$, states $s_t$, and predictions"),
        (.685, "", r"$v_\theta^{(0)}(s_t)$."),
        (.630, "2.", r"Compute $G(c)$ by Equation 5 for each prompt $c$."),
        (.575, "3.", r"Freeze $v_{\mathrm{tgt}}^{\mathrm{ART}}(s_t)=v_\theta^{(0)}(s_t)+G(c)$"),
        (.530, "", r"for each state $s_t$ of prompt $c$."),
        (.402, "4.", "Split the samples into two equal-sized"),
        (.363, "", r"disjoint minibatches $B_1$, $B_2$."),
        (.308, "5.", r"For each $B\in\{B_1,B_2\}$, recompute $w(s_t)$"),
        (.269, "", r"and update $\theta$ with one step on"),
    ]
    for y, number, line in lines:
        ax.text(.05, y, number, fontsize=10)
        ax.text(.11, y, line, fontsize=9.2)
    ax.plot([.05, .95], [.485, .485], color="#D1D9E0", linewidth=.8)
    ax.text(.05, .450, "Training Stage", fontsize=10, weight="bold")
    ax.text(.11, .195, r"$\mathcal{L}(B;\theta)=\frac{1}{|B|}\sum_{s_t\in B}w(s_t)$", fontsize=11)
    ax.text(.24, .125, r"$\cdot\,\mathrm{MSE}(v_\theta(s_t),v_{\mathrm{tgt}}^{\mathrm{ART}}(s_t)).$", fontsize=11)
    fig.savefig(ROOT / "paper-algorithm.png", dpi=280)
    plt.close(fig)


if __name__ == "__main__":
    render()
