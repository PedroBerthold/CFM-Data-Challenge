"""Append scores to results/results.csv (one row per experiment, model, split, fold)."""
from datetime import datetime

import pandas as pd

from src import config as C


def log_results(rows: list[dict]) -> None:
    C.RES_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))
    df.insert(1, "quick", int(C.QUICK))
    header = not C.RESULTS_CSV.exists()
    df.to_csv(C.RESULTS_CSV, mode="a", header=header, index=False)
