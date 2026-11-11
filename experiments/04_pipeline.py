"""04 - Full pipeline: validation, final fit, submission files.

Model:   prediction(stock, day) = day effect + stock effect
  day effect    src/daymodel.py    slow level (extrapolated) + calendar effects + market activity
  stock effect  src/stockmodel.py  boosting on stationary features, trained on target - day mean

Validation: train on days <= T, predict the next 347 days, for several T.
References built here from scratch: per-stock mean, and linear + boosted
residuals on the raw inputs (the structure of the official benchmark).

    python experiments/04_pipeline.py              # everything, full data in data/
    SAMPLE=1 python experiments/04_pipeline.py     # on data/sample/ (from 00_make_sample.py)

Options (environment variables):
    CALENDAR=0     do not use the reconstructed calendar (src/calendar_us.py)
    N_SEEDS=3      models averaged in the final fit
    EXTRA=1        also validate a larger-tree variant (report only)
    SKIP_VALID=1 / SKIP_FINAL=1 / SKIP_REF=1
    N_JOBS=4       threads for LightGBM (default: all cores)

Outputs in results/04_pipeline/: report.md, scores.csv, by_horizon.csv,
day_model.csv, importance.csv, submission_main.csv (+ two level variants),
test_components.parquet.
"""
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config as C
from src import calendar_us as CAL
from src.daymodel import DayModel
from src.features import AsOfFeatures, base_features, cross_section_features, rolling_features
from src.logging_utils import log_results
from src.metrics import paired_by_day, scores
from src.stockmodel import fit_predict

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

EXP = "04_pipeline"
SAMPLE = os.environ.get("SAMPLE", "0") == "1"
USE_CAL = os.environ.get("CALENDAR", "1") == "1"
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
EXTRA = os.environ.get("EXTRA", "0") == "1"
SKIP_VALID = os.environ.get("SKIP_VALID", "0") == "1"
SKIP_FINAL = os.environ.get("SKIP_FINAL", "0") == "1"
SKIP_REF = os.environ.get("SKIP_REF", "0") == "1"
OUT = C.RES_DIR / (EXP + ("_sample" if SAMPLE else ""))
OUT.mkdir(parents=True, exist_ok=True)

H = C.TEST_HORIZON_DAYS
ORIGINS = [int(x) for x in os.environ.get("ORIGINS", "457,557,657").split(",")]
STAGES = (300, 600, 900)
TREND_MULTS = {"main": 1.0, "trend0": 0.0, "trend2": 2.0}
INPUTS = C.RET_COLS + C.VOL_COLS + C.OTHER_COLS
T0 = time.time()
REPORT = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    REPORT.append(msg)


def tick(label):
    say(f"[{time.time() - T0:6.0f}s] {label}")


# --------------------------------------------------------------------------- data
def load_all():
    """Train and test rows in one frame sorted by (pid, day). Test rows: target NaN."""
    if SAMPLE:
        d = C.DATA_DIR / "sample"
        df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("part*.parquet"))], ignore_index=True)
        df["order"] = np.arange(len(df))
        daily_all = pd.read_parquet(d / "daily_agg.parquet").set_index("day")
        return df.sort_values(["pid", "day"]).reset_index(drop=True), daily_all
    cache = C.DATA_DIR / "cache" / "all.parquet"
    if cache.exists():
        return pd.read_parquet(cache), None
    dt = {c: np.float32 for c in INPUTS}
    tr = pd.read_csv(C.DATA_DIR / C.X_TRAIN_FILE, dtype=dt)
    y = pd.read_csv(C.DATA_DIR / C.Y_TRAIN_FILE)
    tr = tr.merge(y, on="ID", how="inner", validate="one_to_one")
    assert len(tr) == len(y), "some training IDs have no target"
    tr["order"] = -1
    te = pd.read_csv(C.DATA_DIR / C.X_TEST_FILE, dtype=dt)
    te["target"] = np.nan
    te["order"] = np.arange(len(te))                       # row order of the test file
    assert te["day"].min() > tr["day"].max(), "test days must come after training days"
    df = pd.concat([tr, te], ignore_index=True)
    del tr, te
    df["target"] = df["target"].astype(np.float32)
    assert not df.duplicated(["pid", "day"]).any(), "duplicated (pid, day)"
    df = df.sort_values(["pid", "day"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df, None


def daily_table(F, day, y, n_days, override):
    g = lambda s: pd.Series(s).groupby(day).mean().reindex(range(n_days))
    daily = pd.DataFrame({
        "n_rows": pd.Series(day).groupby(day).size().reindex(range(n_days)).fillna(0),
        "target_mean": g(y),
        "mean_NLV": g(F["NLV"].to_numpy()),
        "nan_share": g(F["n_nan"].to_numpy() / C.N_PERIODS),
        "tot_abs_ret_mean": g(F["r_sum"].to_numpy()),
        "mean_rel_vol0": g(F["v0"].to_numpy()),
    })
    daily.index.name = "day"
    if override is not None:                               # sample mode: day statistics of all stocks
        for c in daily.columns:
            daily[c] = override[c].reindex(daily.index)
    return daily


# --------------------------------------------------------------------------- references
def ref_pid_mean(pid, y, tr, va):
    m = pd.Series(y[tr]).groupby(pid[tr]).mean()
    return pd.Series(pid[va]).map(m).fillna(y[tr].mean()).to_numpy()


def ref_lin_gbm(raw, pid, day, y, tr, va):
    """Linear model on the raw inputs and `day` (stock id excluded), then default
    LightGBM on its residuals with the stock id added. Structure of the official
    benchmark, without its early stopping."""
    def design(mask):
        X = np.nan_to_num(raw[mask]).astype(np.float64)
        return np.column_stack([np.ones(len(X)), day[mask], X])
    Xtr = design(tr)
    beta = np.linalg.lstsq(Xtr.T @ Xtr, Xtr.T @ y[tr].astype(np.float64), rcond=None)[0]
    res = y[tr] - Xtr @ beta
    p_lin = design(va) @ beta
    del Xtr
    G = lambda m: np.column_stack([pid[m], day[m], np.nan_to_num(raw[m])]).astype(np.float32)
    gb = lgb.LGBMRegressor(random_state=0, verbose=-1).fit(G(tr), res)
    return p_lin, p_lin + gb.predict(G(va))


def demean_by_day(p, day):
    return p - pd.Series(p).groupby(day).transform("mean").to_numpy()


# --------------------------------------------------------------------------- main
def main():
    say(f"# 04 pipeline | SAMPLE={int(SAMPLE)} CALENDAR={int(USE_CAL)} N_SEEDS={N_SEEDS} origins={ORIGINS}")
    df, daily_override = load_all()
    pid, day = df["pid"].to_numpy(), df["day"].to_numpy()
    y = df["target"].to_numpy(np.float64)
    is_train = ~np.isnan(y)
    last_train = int(day[is_train].max())
    n_days = int(day.max()) + 1
    tick(f"loaded {len(df):,} rows, {len(np.unique(pid))} stocks, train days 0-{last_train}, test days {last_train + 1}-{n_days - 1}")

    F = base_features(df)
    raw = None if SKIP_REF or SKIP_VALID else df[INPUTS].to_numpy(np.float32)
    ids, order = df["ID"].to_numpy(), df["order"].to_numpy()
    del df
    daily = daily_table(F, day, y, n_days, daily_override)

    # ---- calendar: use it only if the data confirm it
    cal = None
    if USE_CAL:
        try:
            chk = CAL.verify(daily["mean_rel_vol0"], daily["nan_share"])
            say("calendar check:", chk)
            if chk["share_matched"] >= 0.8:
                cal = CAL.calendar_features(n_days)
            else:
                say("WARNING: calendar not confirmed by the data, falling back to the calendar-free model")
        except AssertionError as e:
            say("WARNING: calendar unavailable:", e)
    half_day = (daily["nan_share"] > 0.25).to_numpy()

    # ---- static features (do not depend on the training end)
    R = rolling_features(pid, F)
    XS = cross_section_features(day, F)
    CF = pd.DataFrame({"half": half_day[day].astype(np.float32)}, index=F.index)
    if cal is not None:
        for c in ["witch", "russell", "eom", "eoq", "msci", "dow", "ev", "opex"]:
            CF[c] = cal[c].to_numpy()[day].astype(np.float32)
        event_day = (cal["ev"].to_numpy() >= 1)
    else:
        v0 = daily["mean_rel_vol0"]
        event_day = ((v0 - v0.shift(1).rolling(20, min_periods=5).median()) > 0.15).fillna(False).to_numpy()
    static_names = list(F.columns) + list(R.columns) + list(XS.columns) + list(CF.columns)
    static = np.hstack([F.to_numpy(np.float32), R.to_numpy(np.float32), XS.to_numpy(np.float32), CF.to_numpy(np.float32)])
    del R, XS, CF

    d_true = daily["target_mean"].to_numpy()[day]
    yd = y - d_true                                        # NaN on test rows
    asof = AsOfFeatures(pid, day, yd, F, event_day[day], n_days)
    names = static_names + asof.names()
    tick(f"features ready: {len(names)} columns")

    best_stage = 600
    # ======================================================================= validation
    if not SKIP_VALID:
        rows, hrows, drows, prows = [], [], [], []
        for T in ORIGINS:
            tr = is_train & (day <= T)
            va = is_train & (day > T) & (day <= T + H)
            yv, dv = y[va], day[va]
            vdays = np.unique(dv)
            asof.fit_slope(pid, yd, tr)

            # day model, with and without calendar
            dm = DayModel(daily, cal).fit(T)
            dms = {"cal" if cal is not None else "nocal": dm}
            if cal is not None:
                dms["nocal"] = DayModel(daily, None).fit(T)
            dv_true = daily["target_mean"].loc[vdays].to_numpy()
            w = daily["n_rows"].loc[vdays].to_numpy()
            for k, m in dms.items():
                for name, mult in TREND_MULTS.items():
                    err = dv_true - m.predict(vdays, mult)
                    drows.append({"origin": T, "day_model": k, "level": name,
                                  "mse_day": float(np.average(err ** 2, weights=w)), "bias_day": float(np.average(err, weights=w))})
            drows.append({"origin": T, "day_model": "constant (train mean)", "level": "-",
                          "mse_day": float(np.average((dv_true - np.nanmean(daily["target_mean"].loc[:T])) ** 2, weights=w)),
                          "bias_day": float("nan")})

            # stock model
            res = fit_predict(static, asof, yd, day, tr, va, T, H, seeds=(0,), n_trees=max(STAGES), stages=STAGES, names=names)
            fix = (lambda p: p) if SAMPLE else (lambda p: demean_by_day(p, dv))
            preds = {}
            for s in STAGES:
                for o in ("l2", "l1", "blend"):
                    preds[f"stock_{o}_{s} + true day effect"] = d_true[va] + fix(res[o, s])
            if not SAMPLE:
                preds[f"stock_blend_600 raw (no demeaning) + true day effect"] = d_true[va] + res["blend", 600]
            m1 = fix(res["blend", 600])
            for k, m in dms.items():
                for name, mult in TREND_MULTS.items():
                    preds[f"FULL {k} {name}"] = m.predict(dv, mult) + m1
            if EXTRA:
                big = fit_predict(static, asof, yd, day, tr, va, T, H, seeds=(0,), n_trees=600,
                                  params=dict(num_leaves=127, min_child_samples=200), names=names)
                preds["stock_blend_600 leaves127 + true day effect"] = d_true[va] + fix(big["blend", 600])

            # references
            preds["ref_pid_mean"] = ref_pid_mean(pid, y, tr, va)
            if raw is not None:
                preds["ref_linear"], preds["ref_lin_gbm"] = ref_lin_gbm(raw, pid, day, y, tr, va)

            for name, p in preds.items():
                rows.append({"experiment": EXP, "model": name, "split": f"origin{T}", "fold": T,
                             "train_days": f"0-{T}", "val_days": f"{vdays.min()}-{vdays.max()}",
                             "n_train": int(tr.sum()), "n_val": int(va.sum()), **scores(yv, p)})
            main_name = f"FULL {'cal' if cal is not None else 'nocal'} main"
            hb = pd.cut(dv - T, C.HORIZON_BINS)
            for name in [main_name, "ref_lin_gbm", "ref_pid_mean"]:
                if name in preds:
                    r = yv - preds[name]
                    for b, grp in pd.DataFrame({"h": hb, "sq": r ** 2, "res": r}).groupby("h", observed=True):
                        hrows.append({"origin": T, "model": name, "horizon": str(b), "mse": grp["sq"].mean(), "bias": grp["res"].mean()})
            for ref in ["ref_lin_gbm", "ref_pid_mean"]:
                if ref in preds:
                    for loss in ("mse", "mae"):
                        prows.append({"origin": T, "A": main_name, "B": ref, **paired_by_day(yv, preds[main_name], preds[ref], dv, loss)})
            if T == ORIGINS[-1]:
                imp = pd.Series(res["gain"], names).sort_values(ascending=False)
                (imp / imp.sum()).rename("gain_share").to_csv(OUT / "importance.csv")
                dm.coefficients().rename("effect").to_csv(OUT / "day_model.csv")
            tick(f"origin {T} done ({int(tr.sum()):,} train rows, {int(va.sum()):,} val rows, {len(vdays)} days)")

        res_df, days_df = pd.DataFrame(rows), pd.DataFrame(drows)
        res_df.to_csv(OUT / "scores.csv", index=False)
        pd.DataFrame(hrows).to_csv(OUT / "by_horizon.csv", index=False)
        log_results(rows)

        def table(metric):
            t = res_df.pivot(index="model", columns="fold", values=metric)
            t["mean"] = t.mean(axis=1)
            return t.sort_values("mean")
        say("\n## Validation, MSE by training end (rows sorted by mean)\n" + table("mse").round(4).to_string())
        say("\n## Validation, MAE\n" + table("mae").round(4).to_string())
        say("\n## Validation, mean residual (target - prediction)\n" + table("mean_resid").round(3).to_string())
        dd = days_df.pivot_table(index=["day_model", "level"], columns="origin", values="mse_day")
        dd["mean"] = dd.mean(axis=1)
        say("\n## Day model alone: MSE of the predicted day effect\n" + dd.round(4).to_string())
        db = days_df.pivot_table(index=["day_model", "level"], columns="origin", values="bias_day")
        say("\n## Day model alone: mean error of the level (true - predicted)\n" + db.round(3).to_string())
        say("\n## Paired tests by day, Newey-West (negative = full model better)\n" +
            pd.DataFrame(prows)[["origin", "A", "B", "loss", "mean_diff", "t_stat", "share_days_A_better"]].round(4).to_string(index=False))
        hz = pd.DataFrame(hrows)
        say("\n## MSE by horizon (days after the end of training)\n" +
            hz.pivot_table(index=["model", "origin"], columns="horizon", values="mse", sort=False).round(4).to_string())
        say("\n## Day effects (target units), fitted up to the last origin\n" + dm.coefficients().round(3).to_string())
        say("\n## Top 25 features of the stock model (gain share)\n" + (imp / imp.sum()).head(25).round(4).to_string())

        stage_mse = {s: res_df[res_df.model == f"stock_blend_{s} + true day effect"]["mse"].mean() for s in STAGES}
        best_stage = min(stage_mse, key=stage_mse.get)
        say(f"\nnumber of trees: {stage_mse} -> {best_stage} used for the final fit")

    # ======================================================================= final fit
    if not SKIP_FINAL:
        raw = None
        T = last_train
        tr, te = is_train, ~is_train
        asof.fit_slope(pid, yd, tr)
        dm = DayModel(daily, cal).fit(T)
        say(f"\n## Final day model: anchor {dm.anchor:.3f} at day {dm.anchor_day:.0f}, slope {dm.slope * 100:.4f} per 100 days")
        res = fit_predict(static, asof, yd, day, tr, te, T, H, seeds=tuple(range(N_SEEDS)), n_trees=best_stage, names=names)
        dte = day[te]
        m1 = res["blend", best_stage] if SAMPLE else demean_by_day(res["blend", best_stage], dte)
        eff = dm.effects(dte)
        comp = pd.DataFrame({"ID": ids[te], "pid": pid[te], "day": dte, "order": order[te],
                             "stock_l2": res["l2", best_stage], "stock_l1": res["l1", best_stage], "stock": m1,
                             "day_effects": eff, "level_main": dm.level(dte, 1.0)}).sort_values("order")
        comp.to_parquet(OUT / "test_components.parquet", index=False)
        for name, mult in TREND_MULTS.items():
            sub = pd.DataFrame({"ID": comp["ID"].to_numpy(),
                                "target": dm.level(comp["day"].to_numpy(), mult) + comp["day_effects"].to_numpy() + comp["stock"].to_numpy()})
            assert sub["target"].notna().all() and sub["ID"].is_unique
            sub.to_csv(OUT / f"submission_{name}.csv", index=False)
            say(f"submission_{name}.csv: {len(sub):,} rows, mean {sub['target'].mean():.4f}, std {sub['target'].std():.4f}, "
                f"level first/last test day {dm.level(dte.min(), mult):.3f} / {dm.level(dte.max(), mult):.3f}")
        say(f"training target for comparison: mean {y[tr].mean():.4f}, std {y[tr].std():.4f}, last 60 training days mean {y[tr & (day > T - 60)].mean():.4f}")
        tick("final fit done")

    (OUT / "report.md").write_text("\n".join(REPORT) + "\n")
    print(f"\nreport written to {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
