"""03 - Validation over several training end points.

Question: how much does a validation score depend on where the training period
ends? Each model is trained on days 0..T and validated on every day after T, for
T in config.ORIGINS. Errors are also reported by horizon (days after T), with
bins shared by all origins, so origins can be compared at equal horizon.

    QUICK=1 python experiments/03_multi_origin.py
    python experiments/03_multi_origin.py
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
from src.cv import origin_split
from src.data import load_train
from src.logging_utils import log_results
from src.metrics import paired_by_day, scores
from src.models import fit_benchmark, pid_median, predict_benchmark

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

EXP = "03_multi_origin"
TAG = "_quick" if C.QUICK else ""
OUT = C.RES_DIR / EXP
OUT.mkdir(parents=True, exist_ok=True)
C.FIG_DIR.mkdir(exist_ok=True)
MODELS = ["pid_median", "linear", "bench_cfm"]
S = {}


def main():
    df = load_train()
    y, day = df["target"].to_numpy(), df["day"].to_numpy()
    rows, bin_rows, pair_rows = [], [], []

    for T in C.ORIGINS:
        t0 = time.time()
        tr, va, tr_d, va_d = origin_split(day, T)
        df_tr, df_va = df.loc[tr], df.loc[va]
        bench = fit_benchmark(df_tr)
        preds = {"pid_median": pid_median(df_tr, df_va),
                 "linear": predict_benchmark(bench, df_va, stage="linear"),
                 "bench_cfm": predict_benchmark(bench, df_va, stage="full")}
        yv, dv = y[va], day[va]
        h = pd.cut(dv - T, C.HORIZON_BINS)

        for m, p in preds.items():
            rows.append({"experiment": EXP, "model": m, "split": f"origin{T}", "fold": T,
                         "train_days": f"{tr_d.min()}-{tr_d.max()}", "val_days": f"{va_d.min()}-{va_d.max()}",
                         "n_train": int(tr.sum()), "n_val": int(va.sum()),
                         "best_iters": "/".join(map(str, bench["best_iters"])) if m == "bench_cfm" else "",
                         **scores(yv, p)})
            r = yv - p
            g = pd.DataFrame({"h": h, "abs": np.abs(r), "res": r}).groupby("h", observed=True)
            for b, grp in g:
                bin_rows.append({"origin": T, "model": m, "horizon": str(b),
                                 "mae": grp["abs"].mean(), "bias": grp["res"].mean(), "n": len(grp)})

        for a, b in [("bench_cfm", "linear"), ("bench_cfm", "pid_median"), ("linear", "pid_median")]:
            r = paired_by_day(yv, preds[a], preds[b], dv, "mae")
            pair_rows.append({"origin": T, "A": a, "B": b, **r})

        print(f"origin {T}: train {tr_d.min()}-{tr_d.max()}, val {va_d.min()}-{va_d.max()} "
              f"({len(va_d)} days), best iters {bench['best_iters']}, {time.time() - t0:.0f}s")

    log_results(rows)
    res = pd.DataFrame(rows)
    bins = pd.DataFrame(bin_rows)
    pairs = pd.DataFrame(pair_rows)
    res.to_csv(OUT / f"scores{TAG}.csv", index=False)
    bins.to_csv(OUT / f"by_horizon{TAG}.csv", index=False)
    pairs.to_csv(OUT / f"paired_tests{TAG}.csv", index=False)

    mae = res.pivot(index="fold", columns="model", values="mae")[MODELS].rename_axis("train_end")
    bias = res.pivot(index="fold", columns="model", values="mean_resid")[MODELS].rename_axis("train_end")
    mse = res.pivot(index="fold", columns="model", values="mse")[MODELS].rename_axis("train_end")
    print("MAE by origin:\n" + mae.round(4).to_string())
    print("MSE by origin:\n" + mse.round(4).to_string())
    print("Bias by origin:\n" + bias.round(4).to_string())

    # Ranking stability: rank of each model at each origin (1 = best MAE)
    ranks = mae.rank(axis=1).astype(int)
    print("Rank by origin (1 = best MAE):\n" + ranks.to_string())

    # Spread of the gap between models across origins: the noise floor of our validation
    gaps = pd.DataFrame({"bench_cfm - linear": mae["bench_cfm"] - mae["linear"],
                         "bench_cfm - pid_median": mae["bench_cfm"] - mae["pid_median"],
                         "linear - pid_median": mae["linear"] - mae["pid_median"]})
    print("MAE gaps by origin:\n" + gaps.round(4).to_string())
    print(pairs.round(4).to_string(index=False))

    # Equal-horizon comparison across origins
    hb = bins[bins.model == "bench_cfm"].pivot(index="origin", columns="horizon", values="bias")
    hm = bins[bins.model == "bench_cfm"].pivot(index="origin", columns="horizon", values="mae")
    order = [str(b) for b in pd.cut(pd.Series([25, 75, 125, 200]), C.HORIZON_BINS)]
    hb, hm = hb.reindex(columns=[c for c in order if c in hb]), hm.reindex(columns=[c for c in order if c in hm])
    print("bench_cfm bias by origin x horizon:\n" + hb.round(3).to_string())
    print("bench_cfm MAE by origin x horizon:\n" + hm.round(3).to_string())

    # figure: bench_cfm bias by horizon, one line per origin
    plt.figure(figsize=(7, 4))
    for T in C.ORIGINS:
        d = bins[(bins.model == "bench_cfm") & (bins.origin == T)]
        d = d.set_index("horizon").reindex(order).dropna()
        plt.plot(range(len(d)), d["bias"], "o-", label=f"train end {T}")
    plt.xticks(range(len(order)), order)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("days after the end of training"); plt.ylabel("mean residual")
    plt.title("bench_cfm bias by horizon, one line per training end"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / f"{EXP}_bias_by_origin{TAG}.png", dpi=120); plt.close()

    S.update(mae=mae, mse=mse, bias=bias, ranks=ranks, gaps=gaps, hb=hb, hm=hm,
             pairs=pairs, iters=res[res.model == "bench_cfm"].set_index("fold")["best_iters"])


def print_summary(log_path):
    f = lambda t, n=4: t.round(n).to_string()
    print(f"=== 03 MULTI-ORIGIN | QUICK={int(C.QUICK)} | origins {C.ORIGINS} | full log: {log_path.name} ===")
    m = S["mae"]
    print("[1] MAE by training end (validation = all later days up to 804):")
    print(f(pd.concat([m, m.agg(["mean", "std"])])))
    print("[2] MSE by training end:")
    print(f(S["mse"]))
    print("[3] bias by training end:")
    print(f(S["bias"], 3))
    print("[4] rank (1 = best MAE): " + " | ".join(
        f"{T}: " + ">".join(S["ranks"].loc[T].sort_values().index) for T in S["ranks"].index))
    g = S["gaps"]
    print("[5] MAE gaps across origins (mean, std, min, max):")
    print(f(g.agg(["mean", "std", "min", "max"]).T))
    p = S["pairs"]
    print("[6] paired tests by origin [mae], t-stat: " + " | ".join(
        f"{a} vs {b}: " + " ".join(f"{r.t_stat:+.1f}" for r in p[(p.A == a) & (p.B == b)].itertuples())
        for a, b in [("bench_cfm", "linear"), ("bench_cfm", "pid_median"), ("linear", "pid_median")]))
    print("[7] bench_cfm bias by training end x horizon:")
    print(f(S["hb"], 3))
    print("    bench_cfm MAE by training end x horizon:")
    print(f(S["hm"], 3))
    print("[8] best iters: " + " | ".join(f"{T}: {v}" for T, v in S["iters"].items()))


if __name__ == "__main__":
    log_path = OUT / f"log{TAG}.txt"
    with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
        main()
    pd.set_option("display.width", 130)
    print_summary(log_path)
