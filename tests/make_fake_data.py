"""Synthetic data with the same file names and columns as the real challenge.
Only used to check that the scripts run end to end, never for results.

    python tests/make_fake_data.py /tmp/fake_cfm
    CFM_DATA_DIR=/tmp/fake_cfm QUICK=1 python experiments/01_linear_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "fake_data")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
n_pid, n_train_day, n_day, P = 120, 805, 1152, 61

pid = np.repeat(np.arange(n_pid), n_day)
day = np.tile(np.arange(n_day), n_pid)
keep = rng.random(len(pid)) > 0.05
pid, day = pid[keep], day[keep]
n = len(pid)

shape = np.exp(-((np.arange(P) - 30) / 25) ** 2) + 0.5
vol = rng.gamma(2.0, 1.0, (n, P)) * shape
vol /= vol.sum(1, keepdims=True)
ret = np.abs(rng.normal(0, 0.1, (n, P)))
mask = rng.random((n, P)) < 0.08
vol[mask] = np.nan
ret[mask] = np.nan

ls = rng.normal(size=n)
nlv = rng.normal(size=n) + 0.3 * (pid % 7)
pid_eff = rng.normal(0, 0.5, n_pid)[pid]
day_eff = 0.0005 * day + rng.normal(0, 0.2, n_day)[day]
target = -2 + pid_eff + day_eff - 0.2 * nlv + 0.05 * ls + 3 * np.nan_to_num(vol[:, 0]) + rng.normal(0, 0.7, n)

ids = rng.permutation(n)
X = pd.DataFrame({"ID": ids, "pid": pid, "day": day})
X = pd.concat([X, pd.DataFrame(ret, columns=[f"abs_ret{i}" for i in range(P)]),
               pd.DataFrame(vol, columns=[f"rel_vol{i}" for i in range(P)])], axis=1)
X["LS"], X["NLV"] = ls, nlv
is_tr = (X["day"] < n_train_day).to_numpy()
X[is_tr].to_csv(out / "input_training.csv.gz", index=False)
X[~is_tr].sample(frac=1, random_state=1).to_csv(out / "input_test.csv.gz", index=False)
pd.DataFrame({"ID": ids[is_tr], "target": target[is_tr]}).sample(frac=1, random_state=0).to_csv(
    out / "output_training_IxKGwDV.csv", index=False)
print(f"wrote {is_tr.sum()} training rows and {(~is_tr).sum()} test rows to {out}")
