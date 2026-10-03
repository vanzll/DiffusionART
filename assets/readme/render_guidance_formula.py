"""Render the reward-tilt identity without depending on GitHub math support."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig = plt.figure(figsize=(8.8, .9), facecolor="white")
    fig.text(.5, .64,
             r"$p_\eta(x_0\mid c)\propto p_{\mathrm{old}}(x_0\mid c)\exp(\eta R(x_0,c))$",
             ha="center", va="center", fontsize=16, color="#26323D")
    fig.text(.5, .20,
             r"$\left.\frac{\partial}{\partial\eta}\mathbb{E}_{p_\eta}[x_0]\right|_{\eta=0}"
             r"=\operatorname{Cov}_{p_{\mathrm{old}}}(R,x_0)$",
             ha="center", va="center", fontsize=17, color="#26323D")
    fig.savefig(Path(__file__).resolve().parent / "guidance-formula.png", dpi=280)
    plt.close(fig)
