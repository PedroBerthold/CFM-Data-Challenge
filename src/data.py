"""Data loading. Nothing here is fitted on the target."""
import numpy as np
import pandas as pd

from src import config as C


def load_train(quick: bool = C.QUICK) -> pd.DataFrame:
    x = pd.read_csv(C.DATA_DIR / C.X_TRAIN_FILE)
    y = pd.read_csv(C.DATA_DIR / C.Y_TRAIN_FILE)

    assert x["ID"].is_unique and y["ID"].is_unique, "duplicate IDs"
    assert len(x) == len(y), f"X has {len(x)} rows, y has {len(y)}"
    df = x.merge(y, on="ID", how="inner", validate="one_to_one")
    assert len(df) == len(x), "some IDs do not match between X and y"

    if quick:
        rng = np.random.default_rng(C.SEED)
        pids = rng.choice(df["pid"].unique(), size=C.QUICK_N_PIDS, replace=False)
        df = df[df["pid"].isin(pids)]

    return df.sort_values(["day", "pid"]).reset_index(drop=True)


def feature_cols(with_day: bool = True) -> list[str]:
    """Benchmark linear features: everything except ID, pid and target."""
    cols = C.RET_COLS + C.VOL_COLS + C.OTHER_COLS
    return (["day"] + cols) if with_day else cols


def design_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """NaN -> 0, as in the official benchmark. float64 for a stable OLS."""
    return df[cols].fillna(0.0).to_numpy(dtype=np.float64)
