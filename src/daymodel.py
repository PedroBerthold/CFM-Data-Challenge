"""Day-level model: the part of the target shared by all stocks on a day.

    d_t = mean over stocks of target(stock, t)
        = slow level L_t + calendar effects + effect of market-wide activity + noise

Fitted on training days only. In the test period the calendar effects and the
activity effect are known from the date and the same-day inputs; the slow level
is not observable and is extrapolated (anchor on the last training days, slope
from the whole training history, multiplied by `trend_mult`). That
extrapolation is the main source of risk of the whole pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.calendar_us import CAL_COLS

MIN_ROW_SHARE = 0.25    # days with fewer stocks than this share of the median are too noisy to fit on
LEVEL_WINDOW = 41       # centered median used to detrend the training days
ANCHOR_DAYS = 60
HALF_NAN_SHARE = 0.25   # a half-day session leaves about a third of the periods empty


def day_features(daily: pd.DataFrame, cal: pd.DataFrame | None) -> pd.DataFrame:
    """Features known on the day itself: date, and inputs of all stocks up to that day."""
    X = pd.DataFrame(index=daily.index)
    half = (daily["nan_share"] > HALF_NAN_SHARE).astype(int)
    X["half"] = half
    agg = {
        "nlv": daily["mean_NLV"],
        "lr": np.log(daily["tot_abs_ret_mean"].clip(lower=1e-6)),
        "v0": daily["mean_rel_vol0"],
    }
    for k, s in agg.items():
        dev = s - s.shift(1).rolling(20, min_periods=5).median()   # backward window only
        X[f"{k}_d20"] = dev.fillna(0.0).where(half == 0, 0.0)
    if cal is not None:
        for c in CAL_COLS:
            X[c] = cal[c].to_numpy()
        X["eom_half"] = X["eom"] * half
        X["event"] = ((cal["opex"].to_numpy() + cal["russell"].to_numpy() + cal["eom"].to_numpy() + half.to_numpy()) > 0)
        X.attrs["cols"] = CAL_COLS + ["half", "eom_half", "nlv_d20", "lr_d20"]
    else:
        # calendar-free fallback: expiry days show as a spike of the opening volume share
        X["v0_spike"] = (X["v0_d20"] > 0.15).astype(int)
        X["event"] = (half.to_numpy() + (X["v0_d20"].to_numpy() > 0.02)) > 0
        X.attrs["cols"] = ["half", "v0_spike", "v0_d20", "nlv_d20", "lr_d20"]
    return X


class DayModel:
    def __init__(self, daily: pd.DataFrame, cal: pd.DataFrame | None, alpha: float = 1.0):
        self.daily = daily
        self.X = day_features(daily, cal)
        self.cols = self.X.attrs["cols"]
        self.alpha = alpha

    def fit(self, T: int) -> "DayModel":
        d = self.daily["target_mean"].loc[:T]
        n = self.daily["n_rows"].loc[:T]
        ok = (d.notna() & (n >= MIN_ROW_SHARE * n.median())).to_numpy().copy()
        ok[:20] = False                                    # backward windows not filled yet
        plain = d.where(ok & ~self.X["event"].loc[:T].to_numpy())
        level = plain.rolling(LEVEL_WINDOW, center=True, min_periods=8).median().interpolate(limit_direction="both")
        e = (d - level)[ok]

        Xtr = self.X.loc[:T, self.cols].astype(float)[ok]
        self.sd = Xtr.std().replace(0, 1.0)
        self.ridge = Ridge(alpha=self.alpha).fit(Xtr / self.sd, e)
        eff = self.ridge.predict(Xtr / self.sd)

        base = (d[ok] - eff)                               # level series with the effects removed
        days = base.index.to_numpy(dtype=float)
        self.slope = float(np.polyfit(days, base.to_numpy(), 1)[0])
        self.anchor = float(base.iloc[-ANCHOR_DAYS:].mean())
        self.anchor_day = float(days[-ANCHOR_DAYS:].mean())
        self.T = T
        return self

    def effects(self, days: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.X.loc[days, self.cols].astype(float) / self.sd)

    def level(self, days: np.ndarray, trend_mult: float = 1.0) -> np.ndarray:
        return self.anchor + trend_mult * self.slope * (np.asarray(days, dtype=float) - self.anchor_day)

    def predict(self, days: np.ndarray, trend_mult: float = 1.0) -> np.ndarray:
        return self.level(days, trend_mult) + self.effects(days)

    def coefficients(self) -> pd.Series:
        """Effects in target units for a 0 -> 1 change (dummies) or +1 sd (continuous)."""
        coef = pd.Series(self.ridge.coef_, self.cols)
        dummies = [c for c in self.cols if not c.endswith("_d20")]
        coef[dummies] = coef[dummies] / self.sd[dummies]
        return coef
