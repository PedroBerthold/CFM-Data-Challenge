"""01 - Independent reproduction and audit of the benchmark linear regression.

Scope: linear regression only (no LightGBM yet), NaN -> 0, no pid in the model.
Run from the repo root:
    QUICK=1 python experiments/01_linear_baseline.py   # quick test on 60 stocks
    python experiments/01_linear_baseline.py           # full run

Sections
  A. Data audit
  B. Reproduction: 5-fold TimeSeriesSplit on unique days, comparison with the reported numbers
  C. Checks: naive baselines, linear without `day`, a split shaped like the real test
  D. Residual analysis on the OOF predictions
  E. Coefficients (raw, standardized, by group, conditioning)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config as C
from src.cv import day_tss_splits, last_block_split
from src.data import design_matrix, feature_cols, load_train
from src.logging_utils import log_results
from src.metrics import paired_by_day, scores

EXP = "01_linear_baseline"
TAG = "_quick" if C.QUICK else ""
OUT = C.RES_DIR / EXP
OUT.mkdir(parents=True, exist_ok=True)
C.FIG_DIR.mkdir(exist_ok=True)

# Numbers reported in the first analysis, to compare against
REPORTED = {
    "fold_r2": [0.07356, 0.14889, 0.16734, 0.18549, 0.20704],
    "oof": {"r2": 0.179244, "rmse": 0.812831, "mae": 0.619583,
            "mean_resid": -0.108555, "std_resid": 0.80555},
    "insample_r2": 0.217742,
}

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def title(s):
    print("\n" + "=" * 80 + f"\n{s}\n" + "=" * 80)


def savefig(name):
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"{EXP}_{name}{TAG}.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Models: each one is fitted on the training rows only
# ---------------------------------------------------------------------------
def fit_predict(name, df, tr, va):
    y_tr = df.loc[tr, "target"].to_numpy()
    if name == "global_mean":
        return np.full(va.sum(), y_tr.mean())
    if name == "global_median":
        return np.full(va.sum(), np.median(y_tr))
    if name == "pid_median":
        med = df.loc[tr].groupby("pid")["target"].median()
        return df.loc[va, "pid"].map(med).fillna(np.median(y_tr)).to_numpy()
    if name in ("linear", "linear_no_day"):
        cols = feature_cols(with_day=(name == "linear"))
        m = LinearRegression().fit(design_matrix(df.loc[tr], cols), y_tr)
        return m.predict(design_matrix(df.loc[va], cols))
    raise ValueError(name)


MODELS = ["global_mean", "global_median", "pid_median", "linear", "linear_no_day"]


def main():
    df = load_train()
    y = df["target"].to_numpy()
    day = df["day"].to_numpy()

    # ------------------------------------------------------------------ A
    title("A. DATA AUDIT")
    print(f"rows={len(df):,}  pids={df.pid.nunique()}  days={df.day.nunique()} "
          f"(min {day.min()}, max {day.max()})  QUICK={C.QUICK}")
    dup = df.duplicated(["pid", "day"]).sum()
    print(f"duplicated (pid, day) pairs: {dup}")
    per_day = df.groupby("day").size()
    per_pid = df.groupby("pid").size()
    print(f"rows per day: min {per_day.min()}, median {per_day.median():.0f}, max {per_day.max()}")
    print(f"rows per pid: min {per_pid.min()}, median {per_pid.median():.0f}, max {per_pid.max()}")
    missing_days = np.setdiff1d(np.arange(day.min(), day.max() + 1), np.unique(day))
    print(f"days missing inside the range: {len(missing_days)}")

    nan_ret = df[C.RET_COLS].isna().mean().mean()
    nan_vol = df[C.VOL_COLS].isna().mean().mean()
    nan_other = df[C.OTHER_COLS].isna().mean()
    n_nan_total = int(df[feature_cols()].isna().sum().sum())
    print(f"NaN share: abs_ret {nan_ret:.2%}, rel_vol {nan_vol:.2%}, "
          f"LS {nan_other['LS']:.2%}, NLV {nan_other['NLV']:.2%}  (total NaN cells {n_nan_total:,})")
    same_nan = (df[C.RET_COLS].isna().to_numpy() == df[C.VOL_COLS].isna().to_numpy()).mean()
    print(f"abs_ret_i and rel_vol_i NaN at the same time: {same_nan:.2%} of cells")
    vol_sum = df[C.VOL_COLS].sum(axis=1, min_count=1)
    print("row sum of rel_vol (after skipping NaN):")
    print(vol_sum.describe(percentiles=[0.01, 0.5, 0.99]).round(4).to_string())
    print("target:")
    print(df["target"].describe().round(4).to_string())
    print(f"share target > 0: {np.mean(y > 0):.2%}")

    # Row-level NaN count, reused in section D
    df["n_nan_vol"] = df[C.VOL_COLS].isna().sum(axis=1)

    # ------------------------------------------------------------------ B
    title("B. REPRODUCTION: TimeSeriesSplit(5) ON UNIQUE DAYS")
    oof = {m: np.full(len(df), np.nan) for m in MODELS}
    fold_id = np.zeros(len(df), dtype=int)
    rows = []
    for k, tr, va, tr_d, va_d in day_tss_splits(day, 5):
        fold_id[va] = k
        for m in MODELS:
            p = fit_predict(m, df, tr, va)
            oof[m][va] = p
            s = scores(y[va], p)
            rows.append({"experiment": EXP, "model": m, "split": "tss5", "fold": k,
                         "train_days": f"{tr_d.min()}-{tr_d.max()}",
                         "val_days": f"{va_d.min()}-{va_d.max()}",
                         "n_train": int(tr.sum()), "n_val": int(va.sum()), **s})
    folds = pd.DataFrame(rows)

    lin = folds[folds.model == "linear"].reset_index(drop=True)
    lin["r2_reported"] = REPORTED["fold_r2"]
    lin["r2_diff"] = lin["r2"] - lin["r2_reported"]
    print(lin[["fold", "train_days", "val_days", "n_train", "n_val",
               "r2", "r2_reported", "r2_diff", "rmse", "mae", "mean_resid"]].round(5).to_string(index=False))

    has_oof = fold_id > 0
    oof_rows = []
    for m in MODELS:
        s = scores(y[has_oof], oof[m][has_oof])
        oof_rows.append({"experiment": EXP, "model": m, "split": "tss5", "fold": "oof",
                         "n_val": int(has_oof.sum()), **s})
    oof_df = pd.DataFrame(oof_rows)
    rep = REPORTED["oof"]
    s_lin = oof_df.set_index("model").loc["linear"]
    print("\nOOF linear, mine vs reported:")
    for key in rep:
        print(f"  {key:<11} mine {s_lin[key]: .6f}   reported {rep[key]: .6f}   diff {s_lin[key] - rep[key]: .2e}")

    full = LinearRegression().fit(design_matrix(df, feature_cols()), y)
    r2_in = full.score(design_matrix(df, feature_cols()), y)
    print(f"  in-sample R2 mine {r2_in:.6f}   reported {REPORTED['insample_r2']:.6f}")

    # ------------------------------------------------------------------ C
    title("C. CHECKS")
    print("OOF over the 5 folds (validation days 135-804), all models fitted in-fold:")
    print(oof_df[["model", "mae", "rmse", "mse", "r2", "mean_resid"]].round(4).to_string(index=False))

    print("\nPer-fold MAE (columns = folds):")
    print(folds.pivot(index="model", columns="fold", values="mae").round(4).to_string())

    print("\nPaired tests by day on the OOF rows (A - B, negative = A better):")
    pairs = [("linear", "linear_no_day"), ("pid_median", "linear"), ("linear", "global_median")]
    pt_rows = []
    for a, b in pairs:
        for loss in ("mae", "mse"):
            r = paired_by_day(y[has_oof], oof[a][has_oof], oof[b][has_oof], day[has_oof], loss)
            pt_rows.append({"A": a, "B": b, **r})
    print(pd.DataFrame(pt_rows).round(4).to_string(index=False))

    print(f"\nSplit shaped like the real test: last {C.TEST_HORIZON_DAYS} days in one block")
    tr, va, tr_d, va_d = last_block_split(day, C.TEST_HORIZON_DAYS)
    print(f"train days {tr_d.min()}-{tr_d.max()} ({len(tr_d)}), val days {va_d.min()}-{va_d.max()} ({len(va_d)})")
    lb_rows, lb_pred = [], {}
    for m in MODELS:
        p = fit_predict(m, df, tr, va)
        lb_pred[m] = p
        lb_rows.append({"experiment": EXP, "model": m, "split": f"last{C.TEST_HORIZON_DAYS}",
                        "fold": 1, "train_days": f"{tr_d.min()}-{tr_d.max()}",
                        "val_days": f"{va_d.min()}-{va_d.max()}",
                        "n_train": int(tr.sum()), "n_val": int(va.sum()), **scores(y[va], p)})
    lb = pd.DataFrame(lb_rows)
    print(lb[["model", "mae", "rmse", "r2", "mean_resid"]].round(4).to_string(index=False))
    r = paired_by_day(y[va], lb_pred["linear"], lb_pred["linear_no_day"], day[va], "mae")
    print(f"linear vs linear_no_day (MAE, by day): diff {r['mean_diff']:.4f}, t {r['t_stat']:.2f}, "
          f"A better on {r['share_days_A_better']:.0%} of days")

    # Does the error grow with the distance to the end of the training period?
    dist = day[va] - tr_d.max()
    bins = pd.cut(dist, [0, 50, 100, 200, 400])
    drift = pd.DataFrame({"bin": bins, "lin": y[va] - lb_pred["linear"],
                          "nod": y[va] - lb_pred["linear_no_day"]}).groupby("bin", observed=True)
    print("\nMean residual by days after the end of training (linear / linear_no_day):")
    print(drift.mean().round(4).to_string())

    log_results(rows + oof_rows + lb_rows)

    # ------------------------------------------------------------------ D
    title("D. RESIDUALS (linear, OOF)")
    p = oof["linear"]
    res = y - p
    df["pred"], df["resid"], df["fold"] = p, res, fold_id
    d = df[has_oof]

    print(pd.Series(d.resid).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(4).to_string())

    dec = pd.qcut(d.pred, 10, labels=False)
    print("\nMean target and mean prediction by prediction decile (calibration):")
    print(d.groupby(dec)[["pred", "target", "resid"]].mean().round(3).to_string())

    by_pid = d.groupby("pid")["resid"].agg(["mean", "std", "count"])
    by_pid["t"] = by_pid["mean"] / (by_pid["std"] / np.sqrt(by_pid["count"]))
    by_pid = by_pid.sort_values("mean")
    print(f"\nResidual mean by pid: share of pids with |t| > 3: {(by_pid.t.abs() > 3).mean():.0%}")
    print(f"Variance of y explained by pid means of the residual: "
          f"{d.groupby('pid')['resid'].transform('mean').var() / d.target.var():.1%}")
    print(by_pid.head(5).round(3).to_string())
    print(by_pid.tail(5).round(3).to_string())

    # Is the pid effect stable over time? correlate pid means across folds
    pm = d.pivot_table(index="pid", columns="fold", values="resid", aggfunc="mean")
    print("\nCorrelation of pid mean residuals between folds:")
    print(pm.corr().round(2).to_string())

    by_day = d.groupby("day")["resid"].mean()
    print(f"\nDaily mean residual: std {by_day.std():.3f}, lag-1 autocorrelation {by_day.autocorr(1):.2f}")
    print(f"Share of residual variance that is a common daily shock: "
          f"{d.groupby('day')['resid'].transform('mean').var() / d.resid.var():.1%}")

    print("\nResidual by number of NaN in rel_vol:")
    nb = pd.cut(d.n_nan_vol, [-1, 0, 5, 20, 40, 61])
    print(d.groupby(nb, observed=True)["resid"].agg(["mean", "std", "count"]).round(3).to_string())

    corr = d[feature_cols()].fillna(0).corrwith(d.resid).sort_values(key=np.abs, ascending=False)
    print("\nTop |corr(residual, feature)|:")
    print(corr.head(10).round(4).to_string())
    print("Corr(residual, squared feature), a first hint of non-linearity:")
    sq = (d[["NLV", "LS", "rel_vol0", "rel_vol60"]].fillna(0) ** 2).corrwith(d.resid)
    print(sq.round(4).to_string())

    worst = d.assign(abs_res=d.resid.abs()).nlargest(20, "abs_res")
    print("\n20 largest errors:")
    print(worst[["ID", "pid", "day", "target", "pred", "resid", "n_nan_vol", "LS", "NLV"]].round(3).to_string(index=False))

    by_pid.to_csv(OUT / f"resid_by_pid{TAG}.csv")
    corr.to_csv(OUT / f"resid_feature_corr{TAG}.csv", header=["corr"])
    worst.to_csv(OUT / f"largest_errors{TAG}.csv", index=False)
    folds.to_csv(OUT / f"fold_scores{TAG}.csv", index=False)
    # OOF predictions are large: kept out of git (see .gitignore)
    d[["ID", "pid", "day", "fold", "target", "pred", "resid"]].to_csv(
        OUT / f"oof_linear{TAG}.csv.gz", index=False)

    # ------------------------------------------------------------------ E
    title("E. COEFFICIENTS (model fitted on all training days)")
    X = design_matrix(df, feature_cols())
    sd_x = X.std(axis=0)
    coef = pd.DataFrame({"coef": full.coef_, "sd_x": sd_x}, index=feature_cols())
    coef["effect_1sd"] = coef.coef * coef.sd_x             # target units per 1 sd of x
    coef["beta_std"] = coef.effect_1sd / y.std()           # fully standardized
    print(f"intercept {full.intercept_:.4f}")
    print("Top 10 positive / negative standardized coefficients:")
    s = coef.sort_values("beta_std")
    print(pd.concat([s.tail(10)[::-1], s.head(10)]).round(4).to_string())

    groups = {"day": ["day"], "abs_ret": C.RET_COLS, "rel_vol": C.VOL_COLS, "LS": ["LS"], "NLV": ["NLV"]}
    print("\nContribution by group (std of X_g @ coef_g, in target units):")
    contrib = {g: float((X[:, [feature_cols().index(c) for c in cols]] @ coef.loc[cols, "coef"].to_numpy()).std())
               for g, cols in groups.items()}
    print(pd.Series(contrib).round(4).to_string())
    print("Individual rel_vol coefficients are not interpretable on their own: the 61 shares sum to 1,")
    print("so adding a constant to all of them changes nothing when there is no NaN.")

    Xs = (X - X.mean(0)) / np.where(sd_x > 0, sd_x, 1)
    cond = np.linalg.cond(Xs[np.random.default_rng(0).choice(len(Xs), min(len(Xs), 100_000), replace=False)])
    print(f"Condition number of the standardized design (100k-row sample): {cond:,.0f}")
    coef.to_csv(OUT / f"coefficients{TAG}.csv")

    # ------------------------------------------------------------------ figures
    plt.figure(figsize=(5.5, 5))
    plt.hexbin(d.pred, d.target, gridsize=80, bins="log", cmap="viridis")
    lim = [min(d.pred.min(), d.target.min()), max(d.pred.max(), d.target.max())]
    plt.plot(lim, lim, "w--", lw=1)
    plt.xlabel("OOF prediction"); plt.ylabel("target"); plt.title("Target vs OOF prediction (linear)")
    savefig("target_vs_pred")

    plt.figure(figsize=(6, 4))
    plt.hist(d.resid, bins=150, density=True, color="#4C72B0")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("residual = target - prediction"); plt.title("OOF residual distribution")
    savefig("resid_hist")

    plt.figure(figsize=(6, 4))
    g = d.groupby(dec).agg(pred=("pred", "mean"), resid=("resid", "mean"),
                           q10=("resid", lambda x: x.quantile(0.1)), q90=("resid", lambda x: x.quantile(0.9)))
    plt.fill_between(g.pred, g.q10, g.q90, alpha=0.3, label="10-90% band")
    plt.plot(g.pred, g.resid, "o-", label="mean residual")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("prediction (decile mean)"); plt.ylabel("residual"); plt.legend()
    plt.title("Residuals vs predictions")
    savefig("resid_vs_pred")

    plt.figure(figsize=(9, 3.5))
    plt.plot(by_day.index, by_day.values, lw=0.6, color="#4C72B0")
    plt.plot(by_day.index, by_day.rolling(20, center=True).mean(), color="#C44E52", label="20-day mean")
    for k in range(2, 6):
        plt.axvline(d.loc[d.fold == k, "day"].min(), color="grey", ls=":", lw=0.8)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("day"); plt.ylabel("mean residual"); plt.legend(); plt.title("Daily mean OOF residual (dotted = fold starts)")
    savefig("resid_by_day")

    plt.figure(figsize=(9, 3.5))
    plt.bar(np.arange(len(by_pid)), by_pid["mean"], width=1.0, color="#4C72B0")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("pid (sorted by mean residual)"); plt.ylabel("mean residual")
    plt.title("Mean OOF residual by stock (analysis only, pid is not in the model)")
    savefig("resid_by_pid")

    top = coef.reindex(coef.beta_std.abs().sort_values(ascending=False).index).head(20)[::-1]
    plt.figure(figsize=(6, 6))
    plt.barh(top.index, top.beta_std, color=np.where(top.beta_std > 0, "#4C72B0", "#C44E52"))
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("standardized coefficient"); plt.title("Top 20 standardized coefficients")
    savefig("coefficients")

    plt.figure(figsize=(9, 3.5))
    plt.plot(range(C.N_PERIODS), coef.loc[C.VOL_COLS, "coef"], "o-", ms=3, label="rel_vol_i")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("intraday period i"); plt.ylabel("raw coefficient"); plt.legend()
    plt.title("rel_vol coefficients across the day (read as a profile, not one by one)")
    savefig("relvol_profile")

    print(f"\nFigures in {C.FIG_DIR}, tables in {OUT}, scores appended to {C.RESULTS_CSV}")


if __name__ == "__main__":
    main()
