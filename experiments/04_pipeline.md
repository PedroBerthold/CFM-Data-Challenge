# Experiment 04: day effect + stock effect

Status: developed and screened on a 120-stock extract of the real data (all days).
Not yet run on the 900 stocks. Every number below comes from that extract
(`results/04_pipeline_sample/report.md`) and has to be confirmed by the full run.

## Run

```bash
git pull && git checkout test-claude-autonome
pip install -r requirements.txt
python experiments/04_pipeline.py          # full data in data/, writes results/04_pipeline/
```

The first run reads the CSV files and caches them in `data/cache/all.parquet`.
What to send back: `results/04_pipeline/report.md` (everything printed is in it).

Other entry points:

```bash
SAMPLE=1 python experiments/04_pipeline.py       # on data/sample/ (about 2 minutes)
CALENDAR=0 python experiments/04_pipeline.py     # same pipeline without the reconstructed calendar
SAMPLE=1 python tests/test_no_lookahead.py       # look-ahead checks
```

## What the data say

1. **The day index is the US trading calendar.** The cross-sectional mean of `rel_vol0`
   spikes on every third Friday of the month if day 0 is 2014-06-02 (54 expiries out of 54,
   train and test), and the 11 half-day sessions all fall next to a US market holiday. Two
   sessions are absent (the last one of 2014 and of 2015). Test period: 2017-08-14 to
   2018-12-28. The check is redone on the data at every run (`src/calendar_us.py`).
2. **The common daily shock is mostly a calendar effect.** Average of the target over all
   stocks, relative to the surrounding days: about +1.4 on quarterly expiries, +1.8 on the
   Russell reconstitution, +0.8 on the last session of a month, -0.3 on half days. The test
   period contains 6 quarterly expiries, 1 Russell day and 16 month ends.
3. **`NLV` behaves like a normalized log of the day's intraday volume.** Within a stock, the
   target falls by about 0.8 per unit of `NLV`: a heavy intraday day lowers the ratio. The
   distance of today's `NLV` to the stock's own past is the main same-day signal.
4. **`LS` is floored at ln(0.01)** and is mostly a stock characteristic. Its nature is still
   unknown; it is used as is.
5. **Variance split of the target** (0.83): 15% shared by all stocks of a day, 37% stock
   level, 48% left for stock-day features and noise.

## Model

`prediction = day effect + stock effect`

- **Day effect** (`src/daymodel.py`): ridge on calendar dummies and two market-wide activity
  measures, fitted on the daily mean of the target after removing a slow level. In the test
  period the slow level is extrapolated: anchor on the last 60 training days, slope from the
  whole training history.
- **Stock effect** (`src/stockmodel.py`): LightGBM on `target - day mean`, with stationary
  features only (no `day`, no raw stock id): same-day summaries of the 61 periods, distance
  to the stock's own past inputs (backward windows of 5 to 250 days), position in the day's
  cross-section, and the stock's past target level. Squared and absolute losses are averaged.
  Predictions are re-centred to a zero mean per day.
- **Stale stock levels.** In the test, a stock's target level is known up to day 804 and used
  up to 347 days later. Training rows therefore get their stock level as of `day - h`, with
  `h` drawn in 1..347, and `h` is a feature. This beat levels as of the previous day at all
  three training ends (MSE 0.3094 / 0.2962 / 0.2844 against 0.3110 / 0.3037 / 0.2893).

## Validation on the extract

Train on days 0..T, predict the next 347 days (or up to day 804). MSE:

| Model | T=457 | T=557 | T=657 | mean |
|---|---|---|---|---|
| per-stock mean | 0.623 | 0.626 | 0.614 | 0.621 |
| linear on raw inputs | 0.636 | 0.601 | 0.596 | 0.611 |
| linear + default LightGBM on residuals (reference) | 0.489 | 0.467 | 0.416 | 0.457 |
| full model without calendar | 0.397 | 0.358 | 0.348 | 0.368 |
| **full model** | **0.347** | **0.328** | **0.304** | **0.326** |
| stock effect with the true day effect (floor of this design) | 0.303 | 0.289 | 0.283 | 0.292 |

MAE of the full model: 0.434 / 0.424 / 0.409, against 0.553 / 0.509 / 0.493 for the reference.
Paired tests by day against the reference: t between -5.5 and -16 on both losses.

Things tested and dropped (no gain on the extract): the 122 raw columns next to the
summaries, centered windows and whole-sample stock means (look-ahead on inputs), stock id
as a category, larger trees, more than 300 to 600 trees.

## Limits

- **The level of the test period is not identifiable from the training data.** With the
  history slope, the full model was biased by +0.13 and +0.11 at T=457 and 557 (the rise
  started around day 450) and by +0.02 at T=657. This is the largest remaining error that a better
  choice could remove, worth up to about 0.03 of MSE in validation. Three files are written, with 0x, 1x and 2x the history slope.
- **The official metric is unknown.** MSE and MAE are both reported. The blend of losses is
  the best or close to the best on both.
- **The calendar is an inference.** It is confirmed by the data, but the two absent sessions
  are placed by elimination. `CALENDAR=0` gives the model without it.
- **120 stocks.** Day-level results use statistics of the 900 stocks; stock-level results may
  move with nine times more data (the number of trees is re-selected in the full run).
