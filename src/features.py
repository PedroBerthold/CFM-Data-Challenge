"""Feature engineering for the stock-level model.

Rules enforced here (and checked by tests/test_no_lookahead.py):
- a feature of (stock, day t) uses inputs of day t or earlier only;
- same-day aggregates across stocks are allowed (inputs only, never the target);
- anything using the target goes through `Encoder`, which only reads targets
  up to an explicit "as-of" day.

All frames must be sorted by (pid, day).
"""
import numpy as np
import pandas as pd

from src import config as C

KEY = ["NLV", "LS", "log_r_sum", "n_nan"]                     # long-memory stock characteristics
ROLL_COLS = ["NLV", "LS", "n_nan", "log_r_sum", "v_last7", "v0", "v_com", "r_sum"]
XS_COLS = ["NLV", "LS", "log_r_sum", "n_nan", "v_last7", "v0", "v_com"]
SHORT_WINS = (5, 20, 60)
LONG_WINS = (120, 250)
TE_WINS = (20, 60, 250, 2000)
GAP_WINS = (60, 250, 2000)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same-day summaries of the 61 volume shares and absolute returns.

    A missing period is a period without trades: volume share 0, return 0.
    """
    v = df[C.VOL_COLS].to_numpy(np.float32)
    r = df[C.RET_COLS].to_numpy(np.float32)
    miss = np.isnan(v)
    v, r = np.nan_to_num(v), np.nan_to_num(r)
    k = np.arange(C.N_PERIODS, dtype=np.float32)
    f = {}
    f["n_nan"] = miss.sum(1)
    f["nan_first10"] = miss[:, :10].sum(1)
    f["nan_last10"] = miss[:, -10:].sum(1)
    f["v0"] = v[:, 0]
    f["v_first5"] = v[:, :5].sum(1)
    f["v_last1"] = v[:, -1]
    f["v_last3"] = v[:, -3:].sum(1)
    f["v_last7"] = v[:, -7:].sum(1)
    for a, b in [(5, 15), (15, 25), (25, 35), (35, 45), (45, 54)]:
        f[f"v_{a}_{b}"] = v[:, a:b].sum(1)
    f["v_hhi"] = (v ** 2).sum(1)
    f["v_max"] = v.max(1)
    f["v_argmax"] = v.argmax(1)
    f["v_com"] = (v * k).sum(1)
    f["v_ent"] = -(v * np.log(v + 1e-9)).sum(1)
    f["r_sum"] = r.sum(1)
    f["r_sq"] = (r ** 2).sum(1)
    f["r_max"] = r.max(1)
    f["r0"] = r[:, 0]
    f["r_first5"] = r[:, :5].sum(1)
    f["r_last3"] = r[:, -3:].sum(1)
    f["r_last7"] = r[:, -7:].sum(1)
    f["r_mid"] = r[:, 15:45].sum(1)
    f["r_com"] = (r * k).sum(1) / (r.sum(1) + 1e-6)
    f["vr"] = (v * r).sum(1)
    f["log_r_sum"] = np.log(f["r_sum"] + 1e-3)
    f["log_v_last7"] = np.log(f["v_last7"] + 1e-4)
    f["log_v0"] = np.log(f["v0"] + 1e-4)
    f["log_hhi"] = np.log(f["v_hhi"] + 1e-9)
    F = pd.DataFrame(f, index=df.index).astype(np.float32)
    F["LS"] = df["LS"].to_numpy(np.float32)
    F["NLV"] = df["NLV"].to_numpy(np.float32)
    return F


def rolling_features(pid: np.ndarray, F: pd.DataFrame) -> pd.DataFrame:
    """Per-stock backward statistics of inputs: mean over the previous w rows
    (today excluded) and today's distance to it. Windows run across the
    train/test boundary, which is fine: inputs only."""
    out = {}

    def past_mean(s: pd.Series, w: int | None) -> pd.Series:
        prev = s.groupby(pid).shift(1)
        if w is None:
            m = prev.groupby(pid).expanding(5).mean()
        else:
            m = prev.groupby(pid).rolling(w, min_periods=max(2, w // 4)).mean()
        return m.reset_index(level=0, drop=True).reindex(s.index)

    for c in ROLL_COLS:
        s = F[c]
        wins = SHORT_WINS + (LONG_WINS if c in KEY else ())
        for w in wins:
            m = past_mean(s, w)
            out[f"{c}_m{w}"] = m
            out[f"{c}_d{w}"] = s - m
        out[f"{c}_lag1"] = s.groupby(pid).shift(1)
        if c in KEY:
            out[f"{c}_dexp"] = s - past_mean(s, None)
    return pd.DataFrame(out, index=F.index).astype(np.float32)


def cross_section_features(day: np.ndarray, F: pd.DataFrame) -> pd.DataFrame:
    """Position of the stock among all stocks of the same day (inputs only)."""
    out = {}
    for c in XS_COLS:
        gb = F[c].groupby(day)
        out[f"{c}_xd"] = F[c] - gb.transform("mean")
        out[f"{c}_xr"] = gb.rank(pct=True)
    return pd.DataFrame(out, index=F.index).astype(np.float32)


class Encoder:
    """Per-stock means of a quantity over a window ending at an as-of day.

    Built from cumulative sums on a (stock, day) grid, so the mean over days
    (t0 - W, t0] costs O(1) per row and t0 can differ for every row. Rows whose
    value is NaN (test rows for the target) contribute nothing.
    """

    def __init__(self, pid: np.ndarray, day: np.ndarray, val: np.ndarray, n_days: int):
        self.pids = np.unique(pid)
        self.pi = np.searchsorted(self.pids, pid)
        s = np.zeros((len(self.pids), n_days + 1))
        n = np.zeros_like(s)
        ok = ~np.isnan(val)
        s[self.pi[ok], day[ok] + 1] = val[ok]
        n[self.pi[ok], day[ok] + 1] = 1.0
        self.cs, self.cn = s.cumsum(1), n.cumsum(1)
        self.n_days = n_days

    def get(self, t0: np.ndarray, window: int, shrink: float = 5.0):
        """Shrunk mean and count over days (t0 - window, t0]. t0 < 0 gives NaN."""
        t0 = np.asarray(t0)
        hi = np.clip(t0 + 1, 0, self.n_days)
        lo = np.clip(t0 + 1 - window, 0, self.n_days)
        s = self.cs[self.pi, hi] - self.cs[self.pi, lo]
        n = self.cn[self.pi, hi] - self.cn[self.pi, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / (n + shrink)
        bad = (t0 < 0) | (n + shrink == 0)
        m[bad] = np.nan
        n[t0 < 0] = 0
        return m.astype(np.float32), n.astype(np.float32)


class AsOfFeatures:
    """Features that depend on an as-of day t0 per row: stock target levels
    known at t0, and how far today's inputs are from the stock's inputs over
    the same windows.

    At prediction time t0 is the last training day for every row. At training
    time each row gets t0 = day - h with h drawn at random in 1..horizon, so the
    model sees stock levels exactly as stale as they will be in the test.
    """

    def __init__(self, pid, day, yd, F: pd.DataFrame, is_event: np.ndarray, n_days: int):
        self.day = day
        self.enc = Encoder(pid, day, yd, n_days)
        self.enc_ev = Encoder(pid, day, np.where(is_event, yd, np.nan), n_days)
        self.enc_pl = Encoder(pid, day, np.where(~is_event, yd, np.nan), n_days)
        self.key = {k: F[k].to_numpy(np.float32) for k in KEY}
        self.enc_x = {k: Encoder(pid, day, self.key[k].astype(np.float64), n_days) for k in KEY}
        self.slope = 0.0

    def fit_slope(self, pid, yd, train_mask):
        """Within-stock slope of the day-demeaned target on NLV, training rows only."""
        g = pid[train_mask]
        a = pd.Series(self.key["NLV"][train_mask].astype(np.float64))
        b = pd.Series(yd[train_mask])
        a, b = a - a.groupby(g).transform("mean"), b - b.groupby(g).transform("mean")
        self.slope = float((a * b).sum() / (a * a).sum())
        return self.slope

    def names(self) -> list[str]:
        n = [f"te{w}" for w in TE_WINS] + ["te_n", "h", "te_ev", "te_ev_n", "te_ev_diff"]
        for w in GAP_WINS:
            n += [f"{k}_gap{w}" for k in KEY] + [f"lin{w}"]
        return n

    def build(self, t0: np.ndarray) -> np.ndarray:
        cols = []
        te = {}
        for w in TE_WINS:
            te[w], n = self.enc.get(t0, w)
            cols.append(te[w])
        cols += [n, (self.day - t0).astype(np.float32)]
        m_ev, n_ev = self.enc_ev.get(t0, 2000, shrink=3.0)
        m_pl, _ = self.enc_pl.get(t0, 2000, shrink=5.0)
        cols += [m_ev, n_ev, m_ev - m_pl]
        for w in GAP_WINS:
            gap_nlv = None
            for k in KEY:
                m, _ = self.enc_x[k].get(t0, w, shrink=0.0)
                gap = self.key[k] - m
                cols.append(gap)
                if k == "NLV":
                    gap_nlv = gap
            cols.append(te[w] + self.slope * gap_nlv)
        return np.column_stack(cols).astype(np.float32)
