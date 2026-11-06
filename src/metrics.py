"""Metrics and paired comparisons.

The official metric is not confirmed yet, so MAE, MSE and RMSE are all reported.
"""
import numpy as np
import pandas as pd


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    r = y - p
    mse = float(np.mean(r**2))
    return {
        "mae": float(np.mean(np.abs(r))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - mse / np.var(y)),
        "mean_resid": float(r.mean()),
        "std_resid": float(r.std(ddof=0)),
    }


def _hac_se(d: np.ndarray, lags: int) -> float:
    """Newey-West standard error of the mean of d (Bartlett kernel)."""
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (d[l:] @ d[:-l]) / n
    return float(np.sqrt(max(v, 0.0) / n))


def paired_by_day(y, p_a, p_b, day, loss: str = "mae", lags: int = 10) -> dict:
    """Compare A and B on the same rows.

    Rows from the same day share market shocks, so the unit of comparison is
    the day: d_t = loss_A(t) - loss_B(t). Negative mean => A better.
    Consecutive days are still correlated, hence a Newey-West SE.
    """
    if loss == "mae":
        la, lb = np.abs(y - p_a), np.abs(y - p_b)
    else:
        la, lb = (y - p_a) ** 2, (y - p_b) ** 2
    g = pd.DataFrame({"day": day, "d": la - lb}).groupby("day")["d"].mean()
    d = g.to_numpy()
    se = _hac_se(d, lags)
    mean = float(d.mean())
    return {
        "loss": loss,
        "n_days": len(d),
        "mean_diff": mean,
        "hac_se": se,
        "t_stat": mean / se if se > 0 else np.nan,
        "share_days_A_better": float(np.mean(d < 0)),
    }
