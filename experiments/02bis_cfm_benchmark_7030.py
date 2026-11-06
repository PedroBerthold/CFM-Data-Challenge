"""02bis - Same models as 02, on a 70/30 time split instead of the 347-day block.

Question: how much does the score depend on the split? 70/30 trains on more days
(0-562) but only validates 242 days ahead, while the real test is 347 days ahead.
Both splits are run here so the comparison is on the same code and seed.

    QUICK=1 python experiments/02bis_cfm_benchmark_7030.py
    python experiments/02bis_cfm_benchmark_7030.py
"""
import contextlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.cv import last_block_split
from src.data import load_train
from src.logging_utils import log_results
from src.metrics import paired_by_day, scores
from src.models import fit_benchmark, pid_median, predict_benchmark

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

EXP = "02bis_cfm_benchmark_7030"
TAG = "_quick" if C.QUICK else ""
OUT = C.RES_DIR / EXP
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["pid_median", "linear", "bench_cfm"]
S = {}


def run(df, horizon, name):
    day, y = df["day"].to_numpy(), df["target"].to_numpy()
    tr, va, tr_d, va_d = last_block_split(day, horizon)
    df_tr, df_va = df.loc[tr], df.loc[va]
    bench = fit_benchmark(df_tr)
    preds = {"pid_median": pid_median(df_tr, df_va),
             "linear": predict_benchmark(bench, df_va, stage="linear"),
             "bench_cfm": predict_benchmark(bench, df_va, stage="full")}
    yv, dv = y[va], day[va]
    rows = [{"experiment": EXP, "model": m, "split": name, "fold": 1,
             "train_days": f"{tr_d.min()}-{tr_d.max()}", "val_days": f"{va_d.min()}-{va_d.max()}",
             "n_train": int(tr.sum()), "n_val": int(va.sum()), **scores(yv, p)} for m, p in preds.items()]
    tab = pd.DataFrame(rows).set_index("model")[["mae", "mse", "rmse", "r2", "mean_resid"]]
    pairs = []
    for a, b in [("bench_cfm", "linear"), ("bench_cfm", "pid_median")]:
        r = paired_by_day(yv, preds[a], preds[b], dv, "mae")
        pairs.append(f"{a} vs {b} [mae]: diff {r['mean_diff']:+.4f} t {r['t_stat']:+.1f} "
                     f"A wins {r['share_days_A_better']:.0%} days")
    bins = pd.cut(dv - tr_d.max(), [0, 50, 100, 200, 400])
    drift = pd.DataFrame({"bin": bins, **{m: yv - p for m, p in preds.items()}}).groupby(
        "bin", observed=True).mean()
    S[name] = {"head": f"{name}: train {tr_d.min()}-{tr_d.max()} ({len(tr_d)} days), "
                       f"val {va_d.min()}-{va_d.max()} ({len(va_d)} days), best iters {bench['best_iters']}",
               "tab": tab, "pairs": pairs, "drift": drift}
    print(S[name]["head"]); print(tab.round(4).to_string()); print("\n".join(pairs)); print(drift.round(4).to_string())
    return rows


def main():
    df = load_train()
    n_days = df["day"].nunique()
    h_7030 = int(round(0.3 * n_days))
    rows = run(df, h_7030, "split_70_30") + run(df, C.TEST_HORIZON_DAYS, f"last{C.TEST_HORIZON_DAYS}")
    log_results(rows)
    pd.DataFrame(rows).to_csv(OUT / f"scores{TAG}.csv", index=False)


def print_summary(log_path):
    print(f"=== 02bis 70/30 vs last {C.TEST_HORIZON_DAYS} | QUICK={int(C.QUICK)} | full log: {log_path.name} ===")
    for name, s in S.items():
        print("[" + s["head"] + "]")
        print(s["tab"].round(4).to_string())
        for p in s["pairs"]:
            print("    " + p)
        print("    mean resid by days after train end: " +
              " | ".join(f"{k}: " + " ".join(f"{v:+.2f}" for v in r.values) for k, r in s["drift"].iterrows()))
    print(f"    (resid order: {', '.join(MODELS)})")


if __name__ == "__main__":
    log_path = OUT / f"log{TAG}.txt"
    with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
        main()
    pd.set_option("display.width", 120)
    print_summary(log_path)
