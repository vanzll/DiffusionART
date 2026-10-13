"""Render current stargazers' history from the public release date (UTC)."""

import json
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


REPO = "vanzll/DiffusionART"
START = datetime(2026, 10, 12, tzinfo=timezone.utc)
OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "readme"


def fetch_history():
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
    return x, y


def render_history(x, y):
    # OFL-licensed lettering: retain a hand-drawn style without an NC font dependency.
    font_manager.fontManager.addfont(OUTPUT / "fonts" / "PatrickHandSC-Regular.ttf")
    for theme, background, text, color in (
        ("light", "#ffffff", "#000000", "#dd4528"),
        ("dark", "#0d1117", "#ffffff", "#ff6b6b"),
    ):
        with plt.rc_context({"font.family": "Patrick Hand SC", "font.size": 14,
                             "svg.fonttype": "path", "svg.hashsalt": "star-history"}):
            fig, ax = plt.subplots(figsize=(8, 5.33))
            fig.subplots_adjust(left=0.115, right=0.96, bottom=0.17, top=0.86)
            fig.set_facecolor(background)
            ax.set_facecolor(background)
            ax.plot(x, y, color=color, linewidth=2.5, solid_capstyle="round")
            ax.set_xlim(START, x[-1])
            ax.set_ylim(0, max(5, y[-1] * 1.08))
            ax.set_title("Star History", color=text, fontsize=25, pad=22)
            ax.set_xlabel("Date", color=text, labelpad=10)
            ax.set_ylabel("GitHub Stars", color=text, labelpad=10)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.tick_params(colors=text, length=5, width=1.4, pad=8)
            ax.xaxis.get_offset_text().set_color(text)
            for side, spine in ax.spines.items():
                spine.set_visible(side in ("left", "bottom"))
                spine.set_color(text)
                spine.set_linewidth(1.6)
                spine.set_sketch_params(scale=0.5, length=140, randomness=1.5)
            marker = Line2D([], [], marker="s", linestyle="none", color=color, markersize=7)
            legend = ax.legend([marker], [REPO], loc="upper left", frameon=True,
                               facecolor=background, edgecolor=text, framealpha=1,
                               labelcolor=text, borderpad=0.4, handlelength=0.7)
            legend.get_frame().set_linewidth(1.4)
            legend.get_frame().set_sketch_params(scale=0.5, length=140, randomness=1.5)
            path = OUTPUT / f"star-history-{theme}.svg"
            fig.savefig(path, metadata={"Date": None})
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            plt.close(fig)
    print(f"Rendered {y[-1]} stars since {START.date()} to {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Reuse the saved data without fetching")
    args = parser.parse_args()
    if args.render_only:
        points = json.loads((OUTPUT / "star-history.json").read_text())["points"]
        x = [datetime.fromisoformat(p["date"]) for p in points]
        y = [p["stars"] for p in points]
    else:
        x, y = fetch_history()
    render_history(x, y)


if __name__ == "__main__":
    main()
