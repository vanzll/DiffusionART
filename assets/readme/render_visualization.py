"""Compose unchanged appendix panels into a single README figure.

Run with the directory containing the paper's selected16/selected26 PDFs:
uv run --no-project --with pymupdf --with matplotlib python \
    assets/readme/render_visualization.py --paper-assets /path/to/paper/figures
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def panel(root, name, box):
    with pymupdf.open(root / name) as document:
        pixels = document[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=pymupdf.Rect(box))
        return np.frombuffer(pixels.samples, dtype=np.uint8).reshape(pixels.height, pixels.width, pixels.n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-assets", type=Path, required=True)
    args = parser.parse_args()
    image = "diffusionart_t2i_selected16_p1.pdf"
    video = "diffusionart_t2v_selected26_p3.pdf"
    rows = [
        ("Text-to-Image", None),
        (None, panel(args.paper_assets, image, (0, 18, 504, 175))),
        (None, panel(args.paper_assets, image, (0, 332, 504, 491))),
        ("Text-to-Video", None),
        (None, panel(args.paper_assets, video, (0, 0, 504, 294))),
    ]
    heights = [24 if pixels is None else pixels.shape[0] / 4 for _, pixels in rows]
    fig = plt.figure(figsize=(7, sum(heights) / 72), facecolor="white")
    grid = fig.add_gridspec(len(rows), 1, height_ratios=heights,
                           left=.01, right=.99, bottom=.005, top=.995, hspace=.03)
    for slot, (title, pixels) in zip(grid, rows):
        ax = fig.add_subplot(slot)
        ax.axis("off")
        if pixels is None:
            ax.text(.005, .3, title, weight="bold", fontsize=14, color="#26323D")
        else:
            ax.imshow(pixels)
    fig.savefig(Path(__file__).resolve().parent / "qualitative-comparison.png", dpi=280)
    plt.close(fig)
