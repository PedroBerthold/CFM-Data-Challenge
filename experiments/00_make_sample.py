"""Build a small, shareable extract of the challenge data.

Purpose: let a collaborator develop and debug on real rows without the full files.
Nothing is modelled here. Output goes to data/sample/ (not tracked by git).

    python experiments/00_make_sample.py            # 120 stocks, 3 parts of 40
    N_PIDS=60 python experiments/00_make_sample.py  # smaller extract

Files written:
    part1.parquet ...   all days (train + test) of a subset of stocks, float32.
                        Test rows have target = NaN. Stocks are spread evenly
                        over the range of per-stock mean targets.
    daily_agg.parquet   one row per day, computed on ALL stocks: row count,
                        cross-sectional mean / median of every input, share of
                        missing periods, and target mean / median / std (train days).
    pid_stats.parquet   one row per stock, computed on ALL days: row counts,
                        target level per third of the training period, LS / NLV
                        level and spread, share of missing periods.
    summary.txt         shapes, dtypes, file listing, submission example header.
"""
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

N_PIDS = int(os.environ.get("N_PIDS", 120))
PIDS_PER_PART = int(os.environ.get("PIDS_PER_PART", 40))
OUT = C.DATA_DIR / "sample"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = C.RET_COLS + C.VOL_COLS + C.OTHER_COLS
log = io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=log)


# ---------------------------------------------------------------- load
x_tr = pd.read_csv(C.DATA_DIR / C.X_TRAIN_FILE)
y_tr = pd.read_csv(C.DATA_DIR / C.Y_TRAIN_FILE)
x_te = pd.read_csv(C.DATA_DIR / C.X_TEST_FILE)
say("files in data dir:")
for f in sorted(C.DATA_DIR.iterdir()):
    if f.is_file():
        say(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
say(f"x_train {x_tr.shape}  y_train {y_tr.shape}  x_test {x_te.shape}")
say("y_train columns:", list(y_tr.columns))
say("x_train dtypes:", x_tr.dtypes.value_counts().to_dict())
say("x_train columns == x_test columns:", list(x_tr.columns) == list(x_te.columns))

tr = x_tr.merge(y_tr, on="ID", how="inner", validate="one_to_one")
te = x_te.assign(target=np.nan)
df = pd.concat([tr, te], ignore_index=True)
df["is_test"] = df["day"] > tr["day"].max()
del x_tr, x_te, tr, te

# any other csv in the data dir (e.g. the official random submission example)
for f in sorted(C.DATA_DIR.glob("*.csv")):
    if f.name != C.Y_TRAIN_FILE:
        say(f"\nhead of {f.name}:")
        say(pd.read_csv(f, nrows=5).to_string())

# ---------------------------------------------------------------- daily aggregates (all stocks)
nan_share = df[C.VOL_COLS].isna().mean(axis=1)
filled = df[INPUTS].fillna(0.0)
g = filled.groupby(df["day"])
daily = pd.concat(
    [g.mean().add_prefix("mean_"), g.median().add_prefix("med_")], axis=1
)
daily["n_rows"] = df.groupby("day").size()
daily["nan_share"] = nan_share.groupby(df["day"]).mean()
daily["tot_abs_ret_mean"] = filled[C.RET_COLS].sum(axis=1).groupby(df["day"]).mean()
t = df.groupby("day")["target"]
daily["target_mean"], daily["target_median"], daily["target_std"] = t.mean(), t.median(), t.std()
daily = daily.reset_index().astype({c: "float32" for c in daily.columns if c not in ("day", "n_rows")})
daily.to_parquet(OUT / "daily_agg.parquet", compression="zstd", index=False)

few = daily.loc[daily["n_rows"] < 0.9 * daily["n_rows"].median(), ["day", "n_rows"]]
say(f"\ndays with fewer than 90% of the median row count: {len(few)}")
say(few.to_string(index=False) if len(few) <= 80 else few.describe().to_string())

# ---------------------------------------------------------------- per-stock stats (all days)
train = df[~df["is_test"]]
thirds = pd.qcut(train["day"], 3, labels=["t1", "t2", "t3"])
lvl = train.groupby(["pid", thirds], observed=True)["target"].mean().unstack().add_prefix("target_mean_")
pid_stats = pd.concat(
    [
        lvl,
        train.groupby("pid")["target"].agg(target_mean="mean", target_std="std"),
        df.groupby("pid")["is_test"].agg(n_test="sum", n_all="size"),
        df.groupby(["pid", "is_test"])[["LS", "NLV"]].agg(["mean", "std"]).unstack()
        .pipe(lambda d: d.set_axis(["_".join(map(str, c)) for c in d.columns], axis=1)),
        nan_share.groupby([df["pid"], df["is_test"]]).mean().unstack().add_prefix("nan_share_test_"),
    ],
    axis=1,
).reset_index()
pid_stats.to_parquet(OUT / "pid_stats.parquet", compression="zstd", index=False)

# ---------------------------------------------------------------- stock subset, spread over target levels
order = pid_stats.sort_values("target_mean")["pid"].to_numpy()
pick = order[np.linspace(0, len(order) - 1, min(N_PIDS, len(order))).round().astype(int)]
pick = np.random.default_rng(C.SEED).permutation(np.unique(pick))
say(f"\n{len(pick)} stocks sampled out of {len(order)}")

f32 = {c: "float32" for c in INPUTS + ["target"]}
for k, start in enumerate(range(0, len(pick), PIDS_PER_PART), 1):
    part = df[df["pid"].isin(pick[start:start + PIDS_PER_PART])].drop(columns="is_test")
    part = part.sort_values(["day", "pid"]).astype(f32)
    path = OUT / f"part{k}.parquet"
    part.to_parquet(path, compression="zstd", index=False)
    say(f"  {path.name}: {len(part):,} rows, {path.stat().st_size / 1e6:.1f} MB")

for name in ["daily_agg.parquet", "pid_stats.parquet"]:
    say(f"  {name}: {(OUT / name).stat().st_size / 1e6:.2f} MB")
(OUT / "summary.txt").write_text(log.getvalue())
say(f"\nwritten to {OUT}")
