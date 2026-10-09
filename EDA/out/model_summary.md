# §H — Demand-rate prediction

*Rendered by `09_demand_model.py`. Figures: **#14** predicted vs actual,
**#15** residual map, **#16** error by hour.*

## H1 · H2 — Target and features

| | |
|---|---|
| Target | hourly **departure count** per (L6 cell, hour, day) |
| Population | top **30** L6 cells by volume — 55,440 cell-hours, zero-filled |
| Non-zero share | 69.9% |
| Departures covered | 599,992 |
| Temporal | `hour`, `dow`, `is_weekend` |
| Lagged | `lag1` (previous hour), `lag24` (same hour yesterday), `lag168` (same hour, same weekday last week), plus a 3-hour mean of `lag1` |
| Spatial | `cell` as a categorical (fixed effect), `cell_prior` (the cell's mean count **over training days only**) |
| Not available | **weather and holidays** — neither is in the dataset. Their absence bounds achievable accuracy and is stated rather than left implicit. Neighbour-cell demand was also left out: at L6 the neighbours of a hotspot are heterogeneous, so the feature is nearly redundant with `cell` |

**Why the restriction to hotspots is stated, not silent.** Dropping 95% of the
cells changes what the model is for. Predicting a rate in a cell with a median
of 2 trips/day is not the operational question, and including those cells would
let noise dominate every aggregate metric.

## H3 — Walk-forward validation

**Random splitting was rejected.** The cleaned window is a time series with a strong
day-of-week cycle; a random row split would put Tuesday 14:00 of week 8 in
training and Tuesday 14:00 of week 6 in validation, with `lag168` carrying the
answer straight across. The design instead is an **expanding window over four
blocks, always validating on days after every training day**:

| block | days | length | role |
|---:|---|---:|---|
| 1 | 2026-03-09 .. 2026-03-27 | 19 | train only |
| 2 | 2026-03-28 .. 2026-04-15 | 19 | validate |
| 3 | 2026-04-16 .. 2026-05-04 | 19 | validate |
| 4 | 2026-05-05 .. 2026-05-24 | 20 | validate |

Blocks 2–4 are the evaluation folds; block 1 is training only. The model is
**refitted per fold**, and `cell_prior` is recomputed from each fold's training
days — a cell mean taken over the whole period would quietly carry the future
into the model and the split would stop being walk-forward.

The lag features are not leakage: to predict hour *t*, an operator knows
everything up to hour *t−1*. `lag1`, `lag24` and `lag168` are all in that set.

## H4 — Models

| # | model | assumptions | software |
|---|---|---|---|
| 1 | **Negative binomial GLM**, log link, cell + hour + dow fixed effects | counts are NB with variance ∝ mean^p; log-linear rate | `statsmodels` 0.14.6, `smf.negativebinomial` |
| 2 | **Gradient-boosted trees, Poisson objective** | no functional form; interactions learnt | `scikit-learn` 1.8.0, `HistGradientBoostingRegressor(loss="poisson")` |
| 3 | **Seasonal naive** (baseline) | the cell's mean count for the same hour and weekday persists | this script |

**Why negative binomial and not Poisson.** §F4 measured the mean–variance slope
at **1.39** where Poisson requires 1.00, and §F1 shows the sample variance
running several times the mean. Fitting a Poisson GLM here returns a Pearson
chi-square over residual degrees of freedom of **1.88** — nearly 1.9× the value
a correct specification would give. The Poisson is fitted alongside the NB as
that evidence, not as a candidate. The fitted NB dispersion parameter is
**alpha ≈ 0.083** (per fold: 0.070, 0.083, 0.097).

## H5 — Results

Pooled over the validation folds:

| model | MAE | RMSE | Poisson deviance | vs baseline | calibration (Σpred/Σobs) |
|---|---:|---:|---:|---:|---:|
| Seasonal naive (baseline) | 3.21 | 8.72 | 3.699 | +0.0% | 0.945 |
| Negative binomial GLM | 3.92 | 16.65 | 2.729 | -26.2% | 1.038 |
| Gradient-boosted trees (Poisson) | 2.95 | 9.24 | 1.899 | -48.7% | 0.997 |

Per-fold Poisson deviance — Negative binomial GLM: 2.18, 2.62, 3.36 ·
Gradient-boosted trees (Poisson): 1.62, 1.80, 2.26 ·
Seasonal naive (baseline): 5.40, 2.61, 3.12.

**Read MAE and Poisson deviance together.** RMSE on counts is dominated by the
highest-volume cells, so a model can look acceptable on RMSE while being wrong
everywhere that matters; deviance is the proper scoring rule for counts and is
the column the model choice rests on. **Calibration matters more than either**:
a model can have low average error and still be systematically low, which is
exactly the failure that empties a station. It is reported as Σpred/Σobs, and
1.000 is perfect.

### The one result that needs explaining rather than presenting

**The negative binomial GLM's RMSE is 4.2 times its MAE, while the
boosting model's is 3.1 times its own.** A gap that wide means a
handful of enormous errors rather than uniformly worse predictions, and the
predictions confirm it: the GLM's 99.9th-percentile prediction is
**547** departures in one cell-hour and its largest is
**1,050**, against a maximum of 495 ever observed.

The mechanism is the log link. An NB GLM is multiplicative, so when `lag1` and
`lag168` are both at a daily peak the fitted coefficients compound instead of
averaging, and the linear predictor extrapolates past anything in the training
data. It is a well-known weakness of log-link GLMs on heavy-tailed lagged
regressors, and it is why a metric designed for counts — Poisson deviance —
still ranks the GLM above the naive baseline while RMSE does not.

**What this means for the deck:** the GLM is the *interpretable* model and the
boosting model is the one to actually dispatch against. Reporting both, with
this gap named, is more useful than reporting whichever one looks better.

## H6 — Outlier handling

All models use `trips_clean_keepzerodur_keepanomday.csv.gz` — the **wider cleaning policy**, which releases
`qc_zero_dur` and `qc_anomaly_day` from the drop set; everything else in the §A3
taxonomy still applies. That retains 99.81% of rows on a 0.19% footprint (1,246
rows dropped), and keeping 2026-05-17 means the window is a contiguous 84 days,
so no 24 h / 168 h lag straddles a hole. The per-run counts are in
`cleaning_report_keepzerodur_keepanomday.json`. **No robustness re-run across
datasets.**

## Limitations carried into the deck

1. **These are predictions of realized departures, not of latent demand** (§G).
   A user who wanted a bike and found none produced no row. The predictions are
   therefore a **floor**, and the gap is widest at peak hours in exactly the
   hotspot cells §E identifies as most imbalanced.
2. **No weather or holiday features.**
3. **`lag1` assumes a one-hour-ahead decision.** A day-ahead operational plan
   could not use it; the `lag24`/`lag168` pair is the part that survives at
   longer horizons.
4. Errors are spatially structured where they are structured at all — see
   figure #15, which is the point of drawing the residual as a map rather than
   reporting one number.
