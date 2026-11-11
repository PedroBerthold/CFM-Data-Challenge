"""Check that no feature of day t depends on anything after day t, and that
target-based features only read targets up to their as-of day.

Method: build the features on the full data, then on a copy where every row
after a cut day is removed (inputs) or has its target replaced by noise, and
require the features of the days up to the cut to be identical.

    python tests/make_fake_data.py /tmp/fake_cfm
    CFM_DATA_DIR=/tmp/fake_cfm python tests/test_no_lookahead.py
    SAMPLE=1 python tests/test_no_lookahead.py        # on data/sample/
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.daymodel import DayModel, day_features
from src.features import AsOfFeatures, base_features, cross_section_features, rolling_features

CUT = 600


def load():
    if os.environ.get("SAMPLE", "0") == "1":
        df = pd.concat([pd.read_parquet(p) for p in sorted((C.DATA_DIR / "sample").glob("part*.parquet"))])
    else:
        x = pd.read_csv(C.DATA_DIR / C.X_TRAIN_FILE)
        df = x.merge(pd.read_csv(C.DATA_DIR / C.Y_TRAIN_FILE), on="ID")
    pids = np.sort(df["pid"].unique())[:40]
    return df[df["pid"].isin(pids)].sort_values(["pid", "day"]).reset_index(drop=True)


def static_block(df):
    F = base_features(df)
    pid, day = df["pid"].to_numpy(), df["day"].to_numpy()
    return F, pd.concat([F, rolling_features(pid, F), cross_section_features(day, F)], axis=1)


def daily_of(df, F, n_days):
    day = df["day"].to_numpy()
    g = lambda s: pd.Series(np.asarray(s, dtype=float)).groupby(day).mean().reindex(range(n_days))
    return pd.DataFrame({"n_rows": pd.Series(day).groupby(day).size().reindex(range(n_days)).fillna(0) + 1000,
                         "target_mean": g(df["target"]), "mean_NLV": g(F["NLV"]),
                         "nan_share": g(F["n_nan"] / C.N_PERIODS), "tot_abs_ret_mean": g(F["r_sum"]),
                         "mean_rel_vol0": g(F["v0"])})


def main():
    df = load()
    n_days = int(df["day"].max()) + 1
    keep = (df["day"] <= CUT).to_numpy()
    past = df[keep].reset_index(drop=True)

    # 1. input features: dropping the future must not change the past
    F_all, S_all = static_block(df)
    F_past, S_past = static_block(past)
    a, b = S_all[keep].to_numpy(), S_past.to_numpy()
    same = np.isclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)
    bad = [c for c, ok in zip(S_all.columns, same.all(0)) if not ok]
    assert not bad, f"input look-ahead in: {bad}"
    print(f"ok: {S_all.shape[1]} input features unchanged when days > {CUT} are removed")

    # 2. as-of features: targets after the as-of day must not matter
    pid, day = df["pid"].to_numpy(), df["day"].to_numpy()
    yd = df["target"].to_numpy(np.float64)
    yd2 = np.where(day > CUT, np.random.default_rng(0).normal(size=len(yd)), yd)
    ev = (day % 21 == 0)
    t0 = np.where(day <= CUT, day - 1 - (day % 7), CUT)
    A1 = AsOfFeatures(pid, day, yd, F_all, ev, n_days).build(t0)
    A2 = AsOfFeatures(pid, day, yd2, F_all, ev, n_days).build(t0)
    assert np.allclose(A1, A2, equal_nan=True), "as-of features read targets after the as-of day"
    print(f"ok: {A1.shape[1]} as-of features ignore targets after the as-of day")

    # 3. day model: features of day t ignore later days; a fit up to CUT ignores later targets
    d_all, d_past = daily_of(df, F_all, n_days), daily_of(past, F_past, CUT + 1)
    X1, X2 = day_features(d_all, None).loc[:CUT], day_features(d_past, None)
    assert np.allclose(X1.to_numpy(float), X2.to_numpy(float), equal_nan=True), "day features look ahead"
    d_noise = d_all.copy()
    d_noise.loc[CUT + 1:, "target_mean"] = 99.0
    days = np.arange(CUT + 1, n_days)
    p1 = DayModel(d_all, None).fit(CUT).predict(days)
    p2 = DayModel(d_noise, None).fit(CUT).predict(days)
    assert np.allclose(p1, p2), "day model reads targets after its training end"
    print("ok: day model uses no information after its training end")


if __name__ == "__main__":
    main()
