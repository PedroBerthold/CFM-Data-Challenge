# Predicting closing auction volumes (CFM Data Challenge #60)

I am working on the [CFM challenge](https://challengedata.ens.fr/challenges/60) on Challenge Data (ENS): predict, for 900 stocks and about 350 future days, how much of the day's trading happens in the closing auction.

This is a work in progress. So far I have built the validation framework, audited the data, reproduced the official benchmark from its description, and measured where it fails. The goal is not only a leaderboard rank: every change I keep has to win in a validation that looks like the real test.

## The problem

**Target.** `log(closing auction volume / volume traded over the 61 intraday periods)`, one value per stock and day. The median auction share is about 14% of the intraday volume.

**Why it is predictable.** The closing auction is the cheapest place to trade and the last chance of the day. Market makers and day traders use it to go flat before the overnight risk; asset managers use it to reach their end-of-day target position, and index funds trade at the close because they are benchmarked to the closing price. These flows are persistent by stock and partly visible in the intraday activity of the same day.

**Data.**
- 684,482 training rows (days 0 to 804) and 311,744 test rows (days 805 to 1151), same 900 stocks.
- Per row: 61 intraday absolute returns, 61 intraday volume shares, and two undisclosed variables (`LS`, `NLV`).
- The test days all come after the training days, so the task is forecasting under drift, not interpolation.

## How I validate

- **Splits by day, never by row.** I use two splits:
  - 5 expanding folds on unique days (`TimeSeriesSplit`), validation on days 135 to 804;
  - one block shaped like the real test: train on days 0 to 457, validate on the last 347 days. This is my main yardstick.
- **The unit of evidence is the day, not the row.** Stocks on the same day share market shocks, so 684k rows are far from 684k independent observations. Models are compared through paired tests on the daily mean loss, with a Newey-West standard error because consecutive days are correlated.
- **Rule for keeping a change.** It must win on most folds and in a significant paired test. Models are chosen in validation, never on the public leaderboard.
- **No leakage.** Anything fitted on the target (medians, the linear stage, the boosted trees, early stopping) only sees training days. The official metric is not stated in the description I have, so I report both MAE and MSE.

## Results so far

| Model | MAE, last 347 days | RMSE, last 347 days | MAE, 5 folds | Mean bias, last 347 days |
|---|---|---|---|---|
| Global median | 0.707 | 0.911 | 0.699 | +0.28 |
| Median per stock | 0.626 | 0.838 | 0.581 | +0.37 |
| Linear regression, no `day` | 0.639 | 0.831 | 0.638 | +0.28 |
| Linear regression (CFM stage 1) | 0.610 | 0.795 | 0.620 | -0.17 |
| **bench_cfm** (linear + LightGBM on residuals) | **0.592** | **0.764** | **0.585** | -0.21 |

`bench_cfm` is my literal reading of the benchmark description: NaN set to 0, a linear fit without the stock id, then LightGBM with default settings on the residuals, stock id included, early stopping with a `TimeSeriesSplit`. The description leaves some choices open (patience, how the fold models are combined), so I call it a literal reading, not an exact reproduction. No official score exists to check it against. Each open choice is written down in [`src/models.py`](src/models.py).

**`bench_cfm` is my reference: MAE 0.592 on the last 347 days.** Every later model is compared to it.

## What I learned

1. **Missing values mean "no trade", so 0 is the right fill for volumes.** The return and the volume of a period are always missing together, and the remaining volume shares still sum to exactly 1. Setting missing volume shares to 0 is therefore correct, not just a convention.
2. **Per-period volume coefficients are meaningless one by one.** The 61 shares sum to 1, so they are exactly collinear with the intercept (condition number around 10^18). Predictions are fine, but I only interpret these coefficients as a group or as an intraday profile.
3. **The choice of validation flips the ranking of models.** On the 5 folds, the per-stock median beats the linear model (MAE 0.581 against 0.620). On the 347-day block, it is the reverse (0.626 against 0.610). Early folds train on only 135 days and reward static models; the block that looks like the test rewards models that follow the trend. I would have picked the wrong baseline with a standard time-series CV.
4. **Auction shares grow over time, and every model gets the level wrong.** Without a time trend, models underpredict more and more (+0.16 to +0.34 by horizon). With a linear trend in `day`, they overpredict (-0.12 to -0.23): the trend is real but slower than a straight line.
5. **Trees cannot extrapolate time.** In `bench_cfm`, `day` takes 28% of the boosting gain, but a tree just reuses the residual level of the last training days for any future day. It makes the bias worse (-0.21 against -0.17 for the linear stage alone).
6. **The stock effect is large but underused.** The mean residual per stock explains 23.5% of the target variance, and its correlation from one fold to another is about 0.8: stable, but drifting. Passed as a plain integer, the stock id only gets 12% of the boosting gain.
7. **A common daily shock is left in the residuals.** 15% of the residual variance is a shock shared by all stocks on the same day, which no model captures yet. My hypothesis is index rebalancing and expiry days. Test-day features of all stocks are available, so cross-sectional daily aggregates of the inputs can be used without leakage.

## Open questions and doubts

- **Metric.** The description I have does not name it. I track MAE and MSE until it is confirmed.
- **Public and private test split.** I do not know yet whether it is by day or by row. By default, the best public submission is used for the private ranking, which rewards overfitting the public test. I will submit rarely, and only models already selected in validation.
- **Unequal days.** Some days have only 16 stocks instead of 900. I have not checked yet which days they are and why.
- **`bench_cfm` may be slightly pessimistic.** A third of its inner models hit the default cap of 100 trees, and averaging them with near-empty models from early folds probably costs some accuracy.

## Next steps

1. Model the time trend outside the trees, and test dampened or recent-window trends.
2. Give the model the level of each stock properly (categorical id, or an encoding fitted on training days only).
3. Cross-sectional daily features to capture the common shock.
4. Revisit missing returns (0 is correct for volumes, less obvious for returns).

## Repository

```
src/            data loading, time splits, metrics and paired tests, models, logging
experiments/    one numbered script per experiment
  01_linear_baseline.py   data audit, linear baseline, naive baselines, residual analysis
  02_cfm_benchmark.py     bench_cfm against the linear and per-stock baselines
  02bis_cfm_benchmark_7030.py  same models on a 70/30 time split, to measure the effect of the split
figures/        plots written by the experiments
results/        results.csv: one row per experiment, model, split and fold
tests/          synthetic data in the challenge format, to check that scripts run
```

Put the challenge files in `data/` (not tracked), then from the repo root:

```bash
pip install -r requirements.txt
QUICK=1 python experiments/01_linear_baseline.py   # 60 stocks, all days, quick check
python experiments/01_linear_baseline.py
python experiments/02_cfm_benchmark.py
python experiments/02bis_cfm_benchmark_7030.py
```

Each script prints a short summary and writes its full output to `results/<experiment>/log.txt`.
