# CFM Data Challenge #60: closing auction volume prediction

Work in progress. A full write-up (method, results, what did not work, doubts) will come at the end.

## Layout
- `src/`: data loading, time splits, metrics, paired tests, logging
- `experiments/NN_*.py`: one numbered script per experiment
- `figures/`: plots written by the experiments
- `results/results.csv`: one row per experiment, model, split and fold
- `tests/make_fake_data.py`: synthetic data in the same format, only to check that scripts run

## Running
Put the challenge files in `data/` (not tracked by git), then from the repo root:

```bash
pip install -r requirements.txt
QUICK=1 python experiments/01_linear_baseline.py   # 60 stocks, all days
python experiments/01_linear_baseline.py           # full run
```
