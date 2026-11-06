"""Paths and global settings shared by all experiments."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CFM_DATA_DIR", ROOT / "data"))
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
RESULTS_CSV = RES_DIR / "results.csv"

X_TRAIN_FILE = "input_training.csv.gz"
Y_TRAIN_FILE = "output_training_IxKGwDV.csv"
X_TEST_FILE = "input_test.csv.gz"

# QUICK=1: run on a subset of stocks (all days kept, so the time splits are unchanged)
QUICK = os.environ.get("QUICK", "0") == "1"
QUICK_N_PIDS = 60
SEED = 0

N_PERIODS = 61
RET_COLS = [f"abs_ret{i}" for i in range(N_PERIODS)]
VOL_COLS = [f"rel_vol{i}" for i in range(N_PERIODS)]
OTHER_COLS = ["LS", "NLV"]

# Test days are 805..1151, i.e. 347 days right after the training period
TEST_HORIZON_DAYS = 347
