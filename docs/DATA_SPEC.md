# Data specification: CFM Data Challenge #60 (closing auction volume)

Purpose: give a reader with no access to the data everything needed to reason about it.
Every fact is tagged with its source:
- `[DESC]` official challenge description
- `[MEAS]` measured on the real data in this repo (experiments 01 to 02bis)
- `[DERIVED]` computed from other facts, not measured directly
- `[HYP]` hypothesis, not verified
- `[UNKNOWN]` not known yet

---

## 1. Task in one paragraph

For each (stock, day) pair, predict `target = ln(closing auction volume / total volume traded over the 61 intraday periods of that day)` `[DESC]`. Volume means traded value `[DESC]`. 900 stocks, about 800 training days, about 350 test days `[DESC]`. All test days come after all training days, so the task is forecasting under drift `[DESC]`. The organizer states that auction volumes can evolve over time, "for instance by becoming relatively larger and larger" `[DESC]`.

## 2. Files

| File | Shape | Content |
|---|---|---|
| `input_training.csv.gz` | 684,482 x 127 `[MEAS]` | inputs, days 0 to 804 |
| `output_training_IxKGwDV.csv` | 684,482 x 2 `[MEAS]` | `ID`, `target` |
| `input_test.csv.gz` | 311,744 x 127 `[MEAS]` | inputs, days 805 to 1151, no target |

Join: `X.merge(y, on="ID")`, one-to-one, every ID matches `[MEAS]`.

Submission `[DESC]`: CSV with one predicted `target` per test `ID` (311,744 rows), on the same log scale as `y_train`. Exact column names: copy the official random-submission example `[UNKNOWN]` (not checked yet).

## 3. Schema (127 input columns)

| Column | Type | Meaning | Notes |
|---|---|---|---|
| `ID` | int | unique row id | not a feature |
| `pid` | int, 900 values | stock id | same 900 stocks in train and test `[DESC]` |
| `day` | int | chronological day index | train 0 to 804, test 805 to 1151 `[MEAS]`; day 0 before day 1 `[DESC]` |
| `abs_ret0` .. `abs_ret60` | float | absolute return (%) between the last known price (usually at the start of period n) and the end of period n `[DESC]` | 61 equal, non-overlapping periods covering a good part of the session, in time order `[DESC]` |
| `rel_vol0` .. `rel_vol60` | float | share of the day's 61-period volume traded in period n `[DESC]` | sums to 1 over a day `[DESC]`, confirmed `[MEAS]` |
| `LS` | float | undisclosed quantity linked to the day's trades for this stock `[DESC]` | nature `[UNKNOWN]` |
| `NLV` | float | undisclosed quantity linked to the day's trades for this stock `[DESC]` | nature `[UNKNOWN]` |

Panel structure:
- Unit of observation: (pid, day). No duplicated (pid, day) pairs `[MEAS]`.
- Unbalanced panel: 16 to 900 stocks per training day, median close to 900 `[MEAS]`. Which days have few stocks, and why: `[UNKNOWN]`.
- Test: about 898 rows per day on average `[DERIVED]` (311,744 / 347).
- Note: the denominator of the target (the 61 periods) does not include the auction itself `[DESC]`, so the target can be positive.

## 4. Missing values

| Fact | Value | Source |
|---|---|---|
| Share of NaN in `abs_ret*` | 4.6% of cells | `[MEAS]` |
| Share of NaN in `rel_vol*` | 4.6% of cells | `[MEAS]` |
| Share of NaN in `LS`, `NLV` | 0% | `[MEAS]` |
| `abs_ret_n` and `rel_vol_n` missing together | 100% of cases | `[MEAS]` |
| Row sum of non-missing `rel_vol` | 1.000 (1st percentile and median) | `[MEAS]` |
| Rows with no NaN in `rel_vol` | 368,326 rows among days 135 to 804 (share on the full train not computed) | `[MEAS]` |

Interpretation `[HYP]`, consistent with all measures: a NaN means no trade in that period. So the true `rel_vol` is 0, and `abs_ret` = 0 (no new price) is the natural value. Interpolation would invent trades and price moves, so it is ruled out. The count and position of missing periods measure illiquidity and are candidate features.

Consequence: with NaN set to 0, the 61 `rel_vol` columns sum exactly to 1, so they are perfectly collinear with an intercept. OLS design condition number about 1.8e18 `[MEAS]`. Predictions are unaffected, individual `rel_vol` coefficients are not identifiable.

## 5. Target

| Statistic | Value |
|---|---|
| mean | -1.9587 `[MEAS]` |
| median | -1.9488 `[MEAS]` |
| std | 0.9092 `[MEAS]` |
| min / max | -7.1377 / +3.5809 `[MEAS]` |
| share > 0 (auction larger than the 61 periods) | 1.86% `[MEAS]` |
| median auction share, exp(target) | about 14% `[DERIVED]` |

## 6. Known structure (from residual analysis)

| Component | Size | Source |
|---|---|---|
| Time trend | auction share rises over time; `day` has the largest standardized OLS coefficient (+0.197) | `[DESC]` `[MEAS]` |
| Trend shape | a linear trend in `day` overshoots the future, no trend undershoots it; the estimated slope depends on the training window | `[MEAS]` |
| Stock effect | per-stock mean of linear residuals explains 23.5% of target variance; range -2.15 to +0.67; correlation across time folds 0.79 to 0.81 (stable but drifting) | `[MEAS]` |
| Common daily shock | 15% of linear residual variance is shared by all stocks on the same day; lag-1 autocorrelation of the daily mean residual 0.17 | `[MEAS]` |
| Possible cause of daily shocks | index rebalancing, expiry days | `[HYP]` |
| Residual spread by group (heteroscedasticity) | not summarized yet | `[UNKNOWN]` |

Standardized OLS coefficients (all training days, NaN = 0) `[MEAS]`: `day` +0.197, `NLV` -0.196, `rel_vol0` +0.155, `rel_vol1` -0.065, `LS` +0.055.
Contribution by group, standard deviation of X_g @ beta_g in target units `[MEAS]`: `abs_ret` 0.233, `rel_vol` 0.198, `day` 0.179, `NLV` 0.179, `LS` 0.050. Individual coefficients within a group offset each other.

## 7. Market mechanism `[DESC]` `[HYP]`

The closing auction trades each stock at a single price. Trading there is usually cheaper than in the continuous session `[DESC]`. Participants who avoid overnight risk (market makers, day traders) flatten positions there; participants with an end-of-day target position (asset managers) complete their trades there `[DESC]`. Index-tracking funds trade at the close because they are benchmarked to the closing price `[HYP]`. These flows are persistent by stock and partly reflected in same-day intraday activity `[HYP]`.

## 8. Validation facts and reference scores

Splits are always by day, never by row. Rows from the same day are not independent; models are compared with paired tests on daily mean loss, Newey-West standard errors.

| Model | MAE last 347 days (train 0-457) | MSE last 347 days | MAE 70/30 (train 0-562) | Bias last 347 days |
|---|---|---|---|---|
| per-stock median | 0.626 | 0.701 | 0.623 | +0.37 |
| linear OLS, all inputs except `pid` | 0.610 | 0.632 | 0.583 | -0.17 |
| `bench_cfm` (OLS + LightGBM on residuals, defaults) | **0.592** | **0.583** | 0.543 | -0.21 |

All `[MEAS]`. The same model scores 0.592 or 0.543 depending on the split: absolute levels are not comparable across splits, only paired differences on the same split are. Model ranking is stable across splits `[MEAS]`.

## 9. Rules and constraints

- The test has no target: past targets can only be used up to day 804 (static per-stock statistics), never as lags in the test period.
- No look-ahead on inputs: a feature for day t uses inputs from day t or earlier only (project rule, although the test file contains all test days at once).
- Same-day cross-sectional aggregates of inputs (across all stocks) are allowed.
- Anything fitted on the target is fitted on training days only.

## 10. Open unknowns

- Official metric `[UNKNOWN]` (MAE and MSE both tracked).
- Public / private test split: by day or by row `[UNKNOWN]`. The best public submission is used for the private ranking by default `[DESC]`.
- Nature of `LS` and `NLV` `[UNKNOWN]`.
- Days with very few stocks `[UNKNOWN]`.
- Leaderboard: best public score 0.3706, ranks 2 to 4 within 0.0007 (0.3857 to 0.3864), read on the public leaderboard on 2026-11-06. Metric unknown, so not comparable with the validation scores above.
