"""Render current stargazers' history from the public release date (UTC)."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REPO = "vanzll/DiffusionART"
START = datetime(2026, 10, 12, tzinfo=timezone.utc)
OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "readme"


def main():
    result = subprocess.run(
        [
            "gh", "api", "--paginate",
            f"repos/{REPO}/stargazers?per_page=100",
            "-H", "Accept: application/vnd.github.star+json",
            "--jq", ".[].starred_at",
        ],
        check=True, capture_output=True, text=True, timeout=180,
    )
    dates = sorted(datetime.fromisoformat(s.replace("Z", "+00:00"))
                   for s in result.stdout.splitlines())
    now = datetime.now(timezone.utc)
    x, y = [START], [sum(d < START for d in dates)]
    for i, date in enumerate(dates, 1):
        if START <= date <= now:
            x.append(date)
            y.append(i)
    x.append(now)
    y.append(len(dates))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Keep the plotted data auditable without publishing stargazer identities.
    (OUTPUT / "star-history.json").write_text(json.dumps({
        "repository": REPO,
        "source": "GitHub stargazers API (currently retained stars)",
        "timezone": "UTC",
        "points": [{"date": d.isoformat(), "stars": n} for d, n in zip(x, y)],
    }, indent=2) + "\n")
    for theme, background, text, grid in (
        ("light", "#ffffff", "#24292f", "#d8dee4"),
        ("dark", "#0d1117", "#e6edf3", "#30363d"),
    ):
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                             "svg.fonttype": "none", "svg.hashsalt": "star-history"}):
            fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
            fig.set_facecolor(background)
            ax.set_facecolor(background)
            ax.step(x, y, where="post", color="#ea535d", linewidth=2, label="DiffusionART")
            ax.set_xlim(START, now)
            ax.set_ylim(0, max(5, len(dates) * 1.12))
            ax.set_title("Star History", loc="left", fontweight="bold", color=text, pad=14)
            ax.set_xlabel("Since October 12, 2026 (UTC)", color=text, labelpad=10)
            ax.set_ylabel("GitHub Stars", color=text)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.tick_params(colors=text, length=0, pad=8)
            ax.xaxis.get_offset_text().set_color(text)
            ax.grid(axis="y", color=grid, linewidth=0.6)
            ax.set_axisbelow(True)
            for side, spine in ax.spines.items():
                spine.set_visible(side in ("left", "bottom"))
                spine.set_color(grid)
            ax.legend(loc="upper left", frameon=False, labelcolor=text)
            fig.savefig(OUTPUT / f"star-history-{theme}.svg", metadata={"Date": None})
            plt.close(fig)
    print(f"Rendered {len(dates)} stars since {START.date()} to {OUTPUT}")


if __name__ == "__main__":
    main()
