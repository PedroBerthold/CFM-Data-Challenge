"""02 - The CFM benchmark exactly as described (linear + LightGBM on residuals).

No official score exists, so this becomes our own reference. Settings that the
description leaves open are listed in src/models.py.

    QUICK=1 python experiments/02_cfm_benchmark.py
    python experiments/02_cfm_benchmark.py

Evaluation (same splits as 01):
  - outer TimeSeriesSplit(5) on days, OOF over days 135-804
  - one block shaped like the real test: train 0-457, validate the last 347 days
Inside each outer training set the benchmark runs its own inner TimeSeriesSplit
for early stopping, so the outer validation days are never seen.
"""
import contextlib
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.cv import day_tss_splits, last_block_split
from src.data import load_train
from src.logging_utils import log_results
from src.metrics import paired_by_day, scores
from src.models import fit_benchmark, gb_cols, pid_median, predict_benchmark

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

EXP = "02_cfm_benchmark"
TAG = "_quick" if C.QUICK else ""
OUT = C.RES_DIR / EXP
OUT.mkdir(parents=True, exist_ok=True)
C.FIG_DIR.mkdir(exist_ok=True)
MODELS = ["pid_median", "linear", "benchmark"]
S = {}


def run_split(df, tr, va, split, fold, tr_d, va_d):
    t0 = time.time()
    df_tr, df_va = df.loc[tr], df.loc[va]
    bench = fit_benchmark(df_tr)
    preds = {
        "pid_median": pid_median(df_tr, df_va),
        "linear": predict_benchmark(bench, df_va, stage="linear"),
        "benchmark": predict_benchmark(bench, df_va, stage="full"),
    }
    y = df_va["target"].to_numpy()
    rows = [{"experiment": EXP, "model": m, "split": split, "fold": fold,
             "train_days": f"{tr_d.min()}-{tr_d.max()}", "val_days": f"{va_d.min()}-{va_d.max()}",
             "n_train": int(tr.sum()), "n_val": int(va.sum()),
             "best_iters": "/".join(map(str, bench["best_iters"])) if m == "benchmark" else "",
             **scores(y, p)} for m, p in preds.items()]
    print(f"{split} fold {fold}: train {tr_d.min()}-{tr_d.max()}, val {va_d.min()}-{va_d.max()}, "
          f"best iters {bench['best_iters']}, {time.time() - t0:.0f}s")
    return preds, rows, bench


def main():
    df = load_train()
    y, day = df["target"].to_numpy(), df["day"].to_numpy()
    print(f"rows {len(df):,}  pids {df.pid.nunique()}  QUICK={C.QUICK}")

    # ---------------------------------------------------------------- outer TSS5
    oof = {m: np.full(len(df), np.nan) for m in MODELS}
    rows, iters = [], []
    for k, tr, va, tr_d, va_d in day_tss_splits(day, 5):
        preds, r, bench = run_split(df, tr, va, "tss5", k, tr_d, va_d)
        for m in MODELS:
            oof[m][va] = preds[m]
        rows += r
        iters += bench["best_iters"]
    has = ~np.isnan(oof["benchmark"])
    oof_rows = [{"experiment": EXP, "model": m, "split": "tss5", "fold": "oof",
                 "n_val": int(has.sum()), **scores(y[has], oof[m][has])} for m in MODELS]
    folds = pd.DataFrame(rows)
    print(folds[["model", "fold", "mae", "rmse", "r2", "mean_resid", "best_iters"]].round(4).to_string(index=False))

    # ---------------------------------------------------------------- last block
    tr, va, tr_d, va_d = last_block_split(day, C.TEST_HORIZON_DAYS)
    lb_preds, lb_rows, lb_bench = run_split(df, tr, va, f"last{C.TEST_HORIZON_DAYS}", 1, tr_d, va_d)
    iters += lb_bench["best_iters"]
    log_results(rows + oof_rows + lb_rows)

    # ---------------------------------------------------------------- tests
    pairs = [("benchmark", "linear"), ("benchmark", "pid_median")]
    pt = []
    for a, b in pairs:
        for loss in ("mae", "mse"):
            r = paired_by_day(y[has], oof[a][has], oof[b][has], day[has], loss)
            pt.append({"split": "tss5", "A": a, "B": b, **r})
            r = paired_by_day(y[va], lb_preds[a], lb_preds[b], day[va], loss)
            pt.append({"split": "last347", "A": a, "B": b, **r})
    pt = pd.DataFrame(pt)
    print(pt.round(4).to_string(index=False))

    dist = day[va] - tr_d.max()
    bins = pd.cut(dist, [0, 50, 100, 200, 400])
    drift = pd.DataFrame({"bin": bins, **{m: y[va] - lb_preds[m] for m in MODELS}}).groupby(
        "bin", observed=True).mean()
    print(drift.round(4).to_string())

    # ---------------------------------------------------------------- importance
    # gain of the stage-2 models from the last-block fit, summed by group
    gain = np.mean([m.booster_.feature_importance("gain") for m in lb_bench["gbs"]], axis=0)
    imp = pd.Series(gain, index=gb_cols())
    grp = imp.groupby(lambda c: "abs_ret" if c.startswith("abs_ret") else
                      "rel_vol" if c.startswith("rel_vol") else c).sum()
    grp = (grp / grp.sum()).sort_values(ascending=False)
    print("stage-2 gain share by group:\n" + grp.round(3).to_string())
    print("top 10 stage-2 features:\n" + (imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())
    imp.to_csv(OUT / f"stage2_gain{TAG}.csv", header=["gain"])
    folds.to_csv(OUT / f"fold_scores{TAG}.csv", index=False)
    pt.to_csv(OUT / f"paired_tests{TAG}.csv", index=False)

    # ---------------------------------------------------------------- figures
    res = pd.DataFrame({"day": day[va], **{m: y[va] - lb_preds[m] for m in MODELS}}).groupby("day").mean()
    plt.figure(figsize=(9, 3.5))
    for m, c in zip(MODELS, ["#8C8C8C", "#4C72B0", "#C44E52"]):
        plt.plot(res.index, res[m].rolling(20, center=True).mean(), color=c, label=m)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("day"); plt.ylabel("mean residual (20-day avg)"); plt.legend()
    plt.title(f"Bias over the last {C.TEST_HORIZON_DAYS} days (train {tr_d.min()}-{tr_d.max()})")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / f"{EXP}_bias_last_block{TAG}.png", dpi=120); plt.close()

    # ---------------------------------------------------------------- summary data
    S["oof"] = pd.DataFrame(oof_rows).set_index("model")[["mae", "rmse", "r2", "mean_resid"]]
    S["fold_mae"] = folds.pivot(index="model", columns="fold", values="mae")
    S["lb"] = pd.DataFrame(lb_rows).set_index("model")[["mae", "rmse", "r2", "mean_resid"]]
    S["pairs"] = [f"[{r.split}] {r.A} vs {r.B} [{r.loss}]: diff {r.mean_diff:+.4f} t {r.t_stat:+.1f} "
                  f"A wins {r.share_days_A_better:.0%} days" for r in pt.itertuples()]
    S["drift"] = drift
    S["iters"] = (f"best iterations: min {min(iters)}, median {int(np.median(iters))}, max {max(iters)} "
                  f"| at the 100 cap: {np.mean(np.array(iters) >= 100):.0%}")
    S["groups"] = "stage-2 gain share: " + " ".join(f"{k} {v:.2f}" for k, v in grp.items())


def print_summary(log_path):
    f = lambda t: t.round(4).to_string()
    print(f"=== 02 CFM BENCHMARK | QUICK={int(C.QUICK)} | full log: {log_path.name} ===")
    print("[1] OOF tss5 (days 135-804):")
    print(f(S["oof"]))
    print("    MAE by fold:")
    print(f(S["fold_mae"]))
    print(f"[2] last {C.TEST_HORIZON_DAYS} days block:")
    print(f(S["lb"]))
    print("[3] paired tests by day (negative = A better):")
    for p in S["pairs"]:
        print("    " + p)
    print("[4] mean residual by days after train end (last block):")
    print(f(S["drift"]))
    print("[5] " + S["iters"])
    print("    " + S["groups"])


if __name__ == "__main__":
    log_path = OUT / f"log{TAG}.txt"
    with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
        main()
    pd.set_option("display.width", 120)
    print_summary(log_path)
