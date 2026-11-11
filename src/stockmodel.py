"""Stock-level model: target minus the day effect, i.e. where a stock sits
relative to the other stocks of the same day.

Gradient boosting on stationary features only (no `day`, no raw stock id):
same-day summaries, distance to the stock's own past inputs, position in the
day's cross-section, and stock target levels as known at an as-of day.
Two losses are fitted (squared and absolute) and averaged: the absolute loss
is robust to the heavy tails, and the average was better than either on both
MSE and MAE in screening.
"""
import os

import lightgbm as lgb
import numpy as np

LGB_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
    verbose=-1, n_jobs=int(os.environ.get("N_JOBS", os.cpu_count() or 2)),
)
OBJECTIVES = ("l2", "l1")


def draw_asof(day: np.ndarray, train: np.ndarray, T: int, horizon: int, seed: int) -> np.ndarray:
    """As-of day per row: T for rows to predict, day - h for training rows."""
    rng = np.random.default_rng(seed)
    h = rng.integers(1, horizon + 1, size=len(day))
    return np.where(train, day - h, T)


def fit_predict(static: np.ndarray, asof, yd: np.ndarray, day: np.ndarray,
                train: np.ndarray, pred: np.ndarray, T: int, horizon: int,
                seeds=(0,), n_trees: int = 600, stages=None, params: dict | None = None,
                names: list[str] | None = None) -> dict:
    """Train on `train` rows (days <= T), predict `pred` rows.

    Returns {(objective, n_trees_used): prediction averaged over seeds} for every
    stage in `stages` (default: only n_trees), plus the gain importances.
    """
    stages = tuple(stages or (n_trees,))
    out = {(o, s): 0.0 for o in OBJECTIVES for s in stages}
    gain = 0.0
    p = dict(LGB_PARAMS)
    p.update(params or {})
    for seed in seeds:
        dyn = asof.build(draw_asof(day, train, T, horizon, seed))
        Xtr = np.hstack([static[train], dyn[train]])
        Xpr = np.hstack([static[pred], dyn[pred]])
        del dyn
        for obj in OBJECTIVES:
            ds = lgb.Dataset(Xtr, yd[train], feature_name=names or "auto", free_raw_data=False)
            m = lgb.train({**p, "objective": obj, "seed": seed}, ds, max(stages))
            for s in stages:
                out[obj, s] = out[obj, s] + m.predict(Xpr, num_iteration=s) / len(seeds)
            if obj == "l2":
                gain = gain + m.feature_importance("gain") / len(seeds)
        del Xtr, Xpr
    for s in stages:
        out["blend", s] = 0.5 * (out["l2", s] + out["l1", s])
    out["gain"] = gain
    return out
