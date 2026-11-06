"""Models. Every fit_* function only sees the training rows it is given."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.data import design_matrix, feature_cols

# lightgbm >= 4.6 renames eval_set; keep the old name so older versions also work
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------
def pid_median(df_tr: pd.DataFrame, df_va: pd.DataFrame) -> np.ndarray:
    med = df_tr.groupby("pid")["target"].median()
    return df_va["pid"].map(med).fillna(df_tr["target"].median()).to_numpy()


def linear(df_tr: pd.DataFrame, df_va: pd.DataFrame, with_day: bool = True) -> np.ndarray:
    cols = feature_cols(with_day)
    m = LinearRegression().fit(design_matrix(df_tr, cols), df_tr["target"].to_numpy())
    return m.predict(design_matrix(df_va, cols))


# ---------------------------------------------------------------------------
# CFM benchmark, as described on the challenge page
#   1. NaN -> 0
#   2. linear fit on all columns except pid
#   3. LightGBM on the linear residuals, all columns including pid,
#      early stopping, sklearn TimeSeriesSplit, other settings default
#
# Choices the description leaves open (kept explicit so they can be changed):
#   - pid is a plain numeric column (no categorical declaration: "default settings")
#   - LightGBM defaults: 100 trees max, lr 0.1, 31 leaves, l2 loss
#   - early stopping patience: 10 rounds, on l2
#   - TimeSeriesSplit(5) on the unique training days; each inner fold trains one
#     model with early stopping on the next block; the prediction is the mean of
#     the 5 models
#   - stage 2 learns the in-sample residuals of a linear model fitted on the
#     whole training set (the description does not mention out-of-fold residuals)
# ---------------------------------------------------------------------------
BENCH_LGB_PARAMS = {"random_state": 0, "verbose": -1}
BENCH_ES_ROUNDS = 10
BENCH_INNER_SPLITS = 5


def gb_cols() -> list[str]:
    return ["pid"] + feature_cols(with_day=True)


def fit_benchmark(df_tr: pd.DataFrame) -> dict:
    y = df_tr["target"].to_numpy()
    lin_cols = feature_cols(with_day=True)
    lin = LinearRegression().fit(design_matrix(df_tr, lin_cols), y)
    resid = y - lin.predict(design_matrix(df_tr, lin_cols))

    X = design_matrix(df_tr, gb_cols())
    day = df_tr["day"].to_numpy()
    udays = np.sort(np.unique(day))
    models, best_iters = [], []
    for tr_i, va_i in TimeSeriesSplit(n_splits=BENCH_INNER_SPLITS).split(udays):
        tr, va = np.isin(day, udays[tr_i]), np.isin(day, udays[va_i])
        m = lgb.LGBMRegressor(**BENCH_LGB_PARAMS)
        m.fit(X[tr], resid[tr], eval_set=[(X[va], resid[va])],
              callbacks=[lgb.early_stopping(BENCH_ES_ROUNDS, verbose=False)])
        models.append(m)
        best_iters.append(m.best_iteration_ or m.n_estimators)
    return {"lin": lin, "lin_cols": lin_cols, "gbs": models, "best_iters": best_iters}


def predict_benchmark(model: dict, df: pd.DataFrame, stage: str = "full") -> np.ndarray:
    p = model["lin"].predict(design_matrix(df, model["lin_cols"]))
    if stage == "linear":
        return p
    X = design_matrix(df, gb_cols())
    return p + np.mean([m.predict(X, num_iteration=m.best_iteration_) for m in model["gbs"]], axis=0)
