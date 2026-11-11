"""05 - Figures of the README.

Day-level and stock-level figures are drawn from data/sample/ (00_make_sample.py).
Validation figures use the scores of the full run on 900 stocks
(results/04_pipeline/report.md), typed below, and the step-by-step ablation
on the 120-stock extract (results/04_pipeline_sample/feature_ladder.csv).

    python experiments/05_figures.py        # writes figures/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import calendar_us as CAL
from src.daymodel import DayModel, day_features

OUT = C.ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ---- theme: one dark surface, three validated series colours, text in ink tokens
SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
BLUE, ORANGE, AQUA, MUTED = "#3987e5", "#d95926", "#199e70", "#6f6e69"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 13,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.solid_capstyle": "round",
})


def frame(title, subtitle, figsize=(10, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    h = figsize[1]
    fig.subplots_adjust(top=1 - 1.0 / h, left=0.09, right=0.97, bottom=0.62 / h)
    fig.text(0.02, 1 - 0.18 / h, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.52 / h, subtitle, fontsize=10.5, color=INK2, va="top")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---- scores of the full run (900 stocks), from results/04_pipeline/report.md
ORIGINS = [457, 557, 657]
MSE = {
    "Per-stock mean": [0.6532, 0.6438, 0.6314],
    "Linear on raw inputs": [0.6323, 0.5877, 0.5868],
    "Reference: linear + boosted residuals": [0.5512, 0.5195, 0.5030],
    "Two-part model, no event-day flags": [0.4131, 0.3644, 0.3591],
    "Two-part model (final)": [0.3601, 0.3295, 0.3093],
}
FLOOR = [0.3008, 0.2885, 0.2841]            # stock part with the true day effect
HORIZON = {
    "bins": ["1-50", "51-100", "101-150", "151-347"],
    "Two-part model (final)": [0.3004, 0.3228, 0.3333, 0.3681],
    "Reference": [0.4996, 0.5485, 0.5163, 0.5445],
}
IMPORTANCE = [
    ("Stock level + volume gap, 250 days", 0.2206), ("Stock level + volume gap, 60 days", 0.1762),
    ("Stock level + volume gap, all history", 0.0692), ("Volatility rank within the day", 0.0450),
    ("Stock target level, all history", 0.0358), ("Volatility vs day average", 0.0269),
    ("Volume vs day average", 0.0259), ("Volume rank within the day", 0.0249),
    ("Late-session volume share, 60-day mean", 0.0212), ("Volume centre of mass, 60-day mean", 0.0156),
    ("Volume vs own 60-day mean", 0.0125), ("Volume vs own 20-day mean", 0.0121),
]
PUBLIC_SCORE = 0.3550


def load_sample():
    d = C.DATA_DIR / "sample"
    df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("part*.parquet"))], ignore_index=True)
    daily = pd.read_parquet(d / "daily_agg.parquet").set_index("day")
    return df, daily


def fig_target_over_time(daily, cal):
    tr = daily[daily["target_mean"].notna()]
    ev = day_features(daily, cal)["event"].loc[tr.index].to_numpy()
    level = tr["target_mean"].where(~ev).rolling(41, center=True, min_periods=8).median().interpolate(limit_direction="both")
    fig, ax = frame("The whole market moves together on some days",
                    "Daily mean of the target over the 900 stocks, training period. Spikes are days when every stock trades far more at the close.")
    ax.plot(tr.index, tr["target_mean"], color=BLUE, lw=1.0, label="Daily mean of the target")
    ax.plot(tr.index, level, color=ORANGE, lw=2.2, label="Slow level (ordinary days)")
    ax.set_xlabel("Day index")
    ax.set_ylabel("log(auction / intraday volume)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    save(fig, "01_target_over_time.png")
    return level


def fig_variance_split():
    parts = [("Common to the day", 15, ORANGE), ("Stable level of the stock", 37, BLUE),
             ("Stock-day specific: features + noise", 48, AQUA)]
    fig, ax = frame("Where the variance of the target sits",
                    "Share of the target variance (0.83) by source. Each part is handled by a different piece of the model.", figsize=(10, 2.7))
    fig.subplots_adjust(top=0.66, bottom=0.08, left=0.02, right=0.98)
    left = 0
    for label, v, col in parts:
        ax.barh(0, v - 0.5, left=left + 0.25, color=col, height=0.55)
        ax.text(left + v / 2, 0, f"{v}%", ha="center", va="center", color=INK, fontsize=14, fontweight="bold")
        ax.text(left + v / 2, 0.47, label, ha="center", va="bottom", color=INK2, fontsize=9.5)
        left += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 0.8)
    ax.axis("off")
    save(fig, "02_variance_split.png")


def fig_volume_effect(df):
    tr = df[df["target"].notna()]
    dv = tr["NLV"] - tr.groupby("pid")["NLV"].transform("mean")
    dy = tr["target"] - tr.groupby("pid")["target"].transform("mean")
    bins = pd.qcut(dv, 25)
    g = pd.DataFrame({"x": dv, "y": dy}).groupby(bins, observed=True).mean()
    slope = float((dv * dy).sum() / (dv * dv).sum())
    fig, ax = frame("A heavy intraday day lowers the auction share",
                    "Each dot is 4% of the stock-days of the extract. Both axes are deviations from the stock's own average.")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.plot(g["x"], g["y"], color=BLUE, lw=2)
    ax.scatter(g["x"], g["y"], s=42, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.set_xlabel("NLV of the day minus the stock's average NLV")
    ax.set_ylabel("Target minus the stock's average target")
    ax.grid(axis="y")
    ax.text(0.97, 0.90, f"slope {slope:.2f} per unit of NLV", transform=ax.transAxes, ha="right", color=INK2)
    save(fig, "03_volume_effect.png")


def fig_validation_scheme():
    fig, ax = frame("Validation copies the real task: train on the past, predict the next 347 days",
                    "Three training end points. Scores are computed on the validation window only.", figsize=(10, 3.7))
    fig.subplots_adjust(left=0.12)
    rows = [("Real task", 804, 1151), ("Split C", 657, 804), ("Split B", 557, 804), ("Split A", 457, 804)]
    for i, (name, T, end) in enumerate(rows):
        ax.barh(i, T + 1 - 4, left=2, color=BLUE, height=0.5)
        ax.barh(i, end - T - 4, left=T + 3, color=ORANGE if i else MUTED, height=0.5)
        ax.text(T / 2, i, f"train: days 0-{T}", ha="center", va="center", color=INK, fontsize=9.5)
        lab = f"test: {T + 1}-{end}, no target" if i == 0 else (f"validate: {T + 1}-{end}" if end - T > 200 else f"{T + 1}-{end}")
        ax.text((T + end) / 2, i, lab, ha="center", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 1152)
    ax.set_xlabel("Day index")
    ax.spines["left"].set_visible(False)
    save(fig, "04_validation_scheme.png")


def fig_model_ladder():
    names = list(MSE)
    means = [float(np.mean(MSE[n])) for n in names]
    fig, ax = frame("Each step is measured against the same three validation splits",
                    "Mean squared error on 900 stocks, average of the three splits (lower is better). Dots: the three splits.")
    fig.subplots_adjust(left=0.36)
    y = np.arange(len(names))[::-1]
    cols = [MUTED, MUTED, MUTED, BLUE, BLUE]
    ax.barh(y, means, color=cols, height=0.56)
    for yi, n, m in zip(y, names, means):
        ax.scatter(MSE[n], [yi] * 3, s=22, color=INK, edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.text(max(max(MSE[n]), m) + 0.02, yi, f"{m:.3f}", va="center", color=INK, fontweight="bold")
    ax.axvline(np.mean(FLOOR), color=AQUA, lw=1.5, ls=(0, (4, 3)))
    ax.text(np.mean(FLOOR) + 0.008, y[0] + 0.62, "floor if the day effect were known: 0.291", color=INK2, fontsize=9.5, ha="left", va="center")
    ax.set_ylim(-0.5, len(names) - 0.05)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 0.74)
    ax.set_xlabel("Validation MSE")
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    save(fig, "05_model_ladder.png")


def fig_feature_ladder():
    path = C.RES_DIR / "04_pipeline_sample" / "feature_ladder.csv"
    if not path.exists():
        print("skip feature ladder (no csv)")
        return
    t = pd.read_csv(path, index_col=0)
    names = [s.split(". ", 1)[1] for s in t.index]
    names[0] = "per-stock mean (no model)"
    fig, ax = frame("What each family of features adds to the stock model",
                    "MSE of the stock part (true day effect given), 120-stock extract, mean of three splits. Families are added one at a time.",
                    figsize=(10, 4.6))
    fig.subplots_adjust(left=0.40)
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["mean"], color=[MUTED] + [BLUE] * (len(t) - 1), height=0.56)
    for yi, m in zip(y, t["mean"]):
        ax.text(m + 0.006, yi, f"{m:.3f}", va="center", color=INK, fontweight="bold")
    ax.set_yticks(y, names)
    ax.set_xlim(0.0, t["mean"].max() * 1.12)
    ax.set_xlabel("Validation MSE of the stock part")
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    save(fig, "06_feature_ladder.png")


def fig_level(daily, cal):
    dm = DayModel(daily, cal).fit(804)
    d = daily["target_mean"].loc[40:804]
    base = (d - dm.effects(d.index.to_numpy())).rolling(20, min_periods=5).mean()
    test = np.arange(805, 1152)
    fig, ax = frame("The one thing the training data cannot tell: the level of the test period",
                    "Slow level of the target (event-day effects removed, 20-day mean) and the three extrapolations written as submission files.")
    ax.plot(base.index, base, color=BLUE, lw=2, label="Level in the training period")
    for mult, col, lab in [(2.0, MUTED, "2x the history slope"), (1.0, ORANGE, "history slope (submitted)"), (0.0, MUTED, "flat")]:
        lv = dm.level(test, mult)
        ax.plot(test, lv, color=col, lw=2.2 if mult == 1 else 1.4, ls="-" if mult == 1 else (0, (4, 3)))
        ax.text(1156, lv[-1], lab, color=INK if mult == 1 else INK2, va="center", fontsize=9.5)
    ax.axvline(804.5, color=GRID, lw=1)
    ax.text(810, base.min(), "test period", color=INK2, fontsize=9.5, va="bottom")
    ax.set_xlim(0, 1390)
    ax.set_xticks(range(0, 1200, 200))
    ax.set_xlabel("Day index")
    ax.set_ylabel("Level of the target")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "07_level_extrapolation.png")


def fig_horizon():
    fig, ax = frame("The error grows with the distance to the last training day",
                    "Validation MSE by horizon, 900 stocks, mean of the splits. Stock levels get stale and the level drifts.")
    x = np.arange(4)
    for name, col in [("Reference", MUTED), ("Two-part model (final)", BLUE)]:
        v = HORIZON[name]
        ax.plot(x, v, color=col, lw=2)
        ax.scatter(x, v, s=48, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.017, f"{vi:.3f}", ha="center", color=INK, fontsize=9.5)
        ax.text(3.12, v[-1], name, color=INK2, va="center", fontsize=9.5)
    ax.set_xticks(x, [f"{b} days" for b in HORIZON["bins"]])
    ax.set_xlim(-0.3, 4.1)
    ax.set_ylim(0.2, 0.62)
    ax.set_xlabel("Days after the end of training")
    ax.set_ylabel("Validation MSE")
    ax.grid(axis="y")
    save(fig, "08_error_by_horizon.png")


def fig_importance():
    fig, ax = frame("What the stock model relies on",
                    "Share of the total split gain in the boosted trees, 900 stocks, top 12 of 157 features.", figsize=(10, 4.6))
    fig.subplots_adjust(left=0.38)
    names, vals = zip(*IMPORTANCE)
    y = np.arange(len(names))[::-1]
    ax.barh(y, [100 * v for v in vals], color=BLUE, height=0.56)
    for yi, v in zip(y, vals):
        ax.text(100 * v + 0.4, yi, f"{100 * v:.1f}%", va="center", color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("Share of gain (%)")
    ax.set_xlim(0, 26)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    save(fig, "09_feature_importance.png")


# --------------------------------------------------------------------------- banners
def fig_hero(daily):
    """Top banner of the README: title, author line, four headline numbers,
    and the daily mean of the target as a backdrop."""
    fig = plt.figure(figsize=(12, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    s = daily["target_mean"].dropna()
    bx = fig.add_axes([0.66, 0.42, 0.34, 0.58])
    bx.plot(s.index, s.to_numpy(), color=BLUE, lw=0.9, alpha=0.55)
    bx.set_xlim(s.index.min(), s.index.max())
    bx.axis("off")
    ax.add_patch(plt.Rectangle((0.035, 0.60), 0.006, 0.28, color=ORANGE, lw=0))
    ax.text(0.055, 0.86, "CFM DATA CHALLENGE #60  ·  CHALLENGE DATA ENS", color=INK2, fontsize=10.5, va="center")
    ax.text(0.055, 0.735, "Closing auction volume forecasting", color=INK, fontsize=25, fontweight="bold", va="center")
    ax.text(0.055, 0.625, "Pierre Berthold  ·  MSc Paris-Dauphine & ESILV", color=INK2, fontsize=11.5, va="center")
    tiles = [("0.355", "MSE on the public leaderboard"), ("1st", "public ranking, 9 Nov 2026"),
             ("-37%", "validation error vs reference"), ("10 min", "full run on a laptop")]
    w, gap, x0 = 0.219, 0.018, 0.035
    for i, (big, small) in enumerate(tiles):
        x = x0 + i * (w + gap)
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.09), w, 0.34, boxstyle="round,pad=0,rounding_size=0.012",
                                                       facecolor="#242423", edgecolor=GRID, lw=1))
        ax.text(x + 0.018, 0.305, big, color=INK, fontsize=22, fontweight="bold", va="center")
        ax.text(x + 0.018, 0.165, small, color=INK2, fontsize=9.5, va="center")
    save(fig, "banner.png")


def fig_part_banners():
    parts = [("I", "The problem and the data", "part_1.png"), ("II", "The model", "part_2.png"),
             ("III", "The evidence", "part_3.png"), ("IV", "Run it and vocabulary", "part_4.png")]
    for num, title, name in parts:
        fig = plt.figure(figsize=(12, 0.95))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.035, 0.22), 0.006, 0.56, color=ORANGE, lw=0))
        ax.text(0.055, 0.5, f"PART {num}", color=INK2, fontsize=11, va="center")
        ax.text(0.135, 0.5, title, color=INK, fontsize=17, fontweight="bold", va="center")
        ax.plot([0.60, 0.965], [0.5, 0.5], color=GRID, lw=1)
        save(fig, name)


def main():
    df, daily = load_sample()
    cal = CAL.calendar_features(len(daily))
    fig_hero(daily)
    fig_part_banners()
    fig_target_over_time(daily, cal)
    fig_variance_split()
    fig_volume_effect(df)
    fig_validation_scheme()
    fig_model_ladder()
    fig_feature_ladder()
    fig_level(daily, cal)
    fig_horizon()
    fig_importance()


if __name__ == "__main__":
    main()
