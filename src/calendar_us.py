"""Reconstructed trading calendar for the anonymous `day` index.

Finding (experiment 04, measured on the data): the cross-sectional mean of
`rel_vol0` spikes once every 17 to 25 days, and these spikes fall exactly on
the third Fridays of the US calendar if day 0 is 2014-06-02 and the sessions
are those of the NYSE, with two sessions absent (the last one of 2014 and of
2015). Half-day sessions (a third of the periods without trades) also fall on
the known early closes. The mapping is therefore day -> date, and the test
period runs from 2017-08-14 to 2018-12-28.

This is an inference from the inputs plus public calendar knowledge, not
something given by the organizer. `verify()` re-checks it on the data at every
run, and the pipeline falls back to a calendar-free model if the check fails.
A calendar is known in advance, so using it is not a look-ahead on inputs.
"""
import numpy as np
import pandas as pd

START = "2014-06-02"
HOLIDAYS = """
2014-07-04 2014-09-01 2014-11-27 2014-12-25
2015-01-01 2015-01-19 2015-02-16 2015-04-03 2015-05-25 2015-07-03 2015-09-07 2015-11-26 2015-12-25
2016-01-01 2016-01-18 2016-02-15 2016-03-25 2016-05-30 2016-07-04 2016-09-05 2016-11-24 2016-12-26
2017-01-02 2017-01-16 2017-02-20 2017-04-14 2017-05-29 2017-07-04 2017-09-04 2017-11-23 2017-12-25
2018-01-01 2018-01-15 2018-02-19 2018-03-30 2018-05-28 2018-07-04 2018-09-03 2018-11-22 2018-12-05 2018-12-25
2019-01-01 2019-01-21 2019-02-18 2019-04-19 2019-05-27 2019-07-04
""".split()
# Sessions that exist in the NYSE calendar but not in the data (measured: one
# session is absent between the December and January expiries of both years,
# and no month-end effect shows on the neighbouring days).
ABSENT = ["2014-12-31", "2015-12-31"]

CAL_COLS = ["witch", "russell", "eom", "eoq", "msci", "msci_semi", "opex_m", "first_fri",
            "pre_hol", "post_hol", "dow0", "dow1", "dow2", "dow3"]


def nyse_sessions(end: str = "2019-06-28") -> pd.DatetimeIndex:
    d = pd.bdate_range(START, end)
    return d[~d.isin(pd.to_datetime(HOLIDAYS))]


def day_dates(n_days: int) -> pd.DatetimeIndex:
    d = nyse_sessions()
    d = d[~d.isin(pd.to_datetime(ABSENT))]
    assert n_days <= len(d), "calendar too short"
    return pd.DatetimeIndex(d[:n_days])


def calendar_features(n_days: int) -> pd.DataFrame:
    """One row per day index. Every column is a deterministic function of the date."""
    s = pd.Series(day_dates(n_days))
    full = pd.Series(nyse_sessions())
    ym = full.dt.year * 12 + full.dt.month
    rev = dict(zip(full, full.groupby(ym).cumcount(ascending=False)))

    c = pd.DataFrame(index=pd.RangeIndex(n_days, name="day"))
    c["date"] = s.values
    c["dow"] = s.dt.weekday.values
    third_fri = ((s.dt.weekday == 4) & (s.dt.day >= 15) & (s.dt.day <= 21)).values
    quarter_m = s.dt.month.isin([3, 6, 9, 12]).values
    last = (s.map(rev) == 0).values
    c["opex"] = third_fri.astype(int)
    c["witch"] = (third_fri & quarter_m).astype(int)            # quarterly expiry, index rebalances
    c["opex_m"] = (third_fri & ~quarter_m).astype(int)          # monthly expiry
    c["russell"] = ((s.dt.month == 6) & (s.dt.weekday == 4) & (s.dt.day >= 22) & (s.dt.day <= 28)).values.astype(int)
    c["eom"] = last.astype(int)                                 # last session of the month
    c["eoq"] = (last & quarter_m).astype(int)
    c["msci"] = (last & s.dt.month.isin([2, 5, 8, 11]).values).astype(int)
    c["msci_semi"] = (last & s.dt.month.isin([5, 11]).values).astype(int)
    c["first_fri"] = ((s.dt.weekday == 4) & (s.dt.day <= 7)).values.astype(int)
    gap_next = (s.shift(-1) - s).dt.days.fillna(1).values
    gap_prev = (s - s.shift(1)).dt.days.fillna(1).values
    c["pre_hol"] = ((gap_next > 3) | ((gap_next > 1) & (c["dow"].values != 4))).astype(int)
    c["post_hol"] = ((gap_prev > 3) | ((gap_prev > 1) & (c["dow"].values != 0))).astype(int)
    for k in range(4):
        c[f"dow{k}"] = (c["dow"] == k).astype(int)
    # ordinal event code for the stock-level model
    c["ev"] = np.select([c["russell"] == 1, c["witch"] == 1, c["eom"] == 1], [3, 2, 1], 0)
    return c


def verify(v0_mean: pd.Series, nan_share: pd.Series) -> dict:
    """Check the calendar against the data.

    v0_mean:   cross-sectional mean of rel_vol0 per day (index = day)
    nan_share: cross-sectional mean share of periods without trades per day
    Returns the share of third Fridays that coincide with a local spike of
    v0_mean, and the share of data half-days that fall next to a holiday.
    """
    n = int(v0_mean.index.max()) + 1
    c = calendar_features(n)
    dev = v0_mean - v0_mean.shift(1).rolling(20, min_periods=5).median()
    hits, total = 0, 0
    for d in c.index[(c["opex"] == 1) & (c.index >= 25)]:
        win = dev.reindex(range(d - 3, d + 4)).dropna()
        if len(win) < 5:
            continue
        total += 1
        hits += int(win.idxmax() == d)
    half = nan_share.index[nan_share > 0.25]
    near_hol = c.loc[c.index.isin(half), ["pre_hol", "post_hol"]].max(axis=1)
    return {
        "n_third_fridays": total,
        "share_matched": hits / max(total, 1),
        "n_half_days": int(len(half)),
        "share_half_days_next_to_holiday": float(near_hol.mean()) if len(half) else float("nan"),
        "first_date": str(c["date"].iloc[0].date()),
        "last_date": str(c["date"].iloc[-1].date()),
    }
