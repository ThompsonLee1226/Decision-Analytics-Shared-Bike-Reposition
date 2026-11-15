# HW7 GP / Arc half-hourly demand models

Two panel-level XGBoost models for the 30 Zhongguancun gathering points and the
151 frozen service Arcs, at half-hour resolution over 57 working days
(06:30–10:30). Each panel has one booster per channel (departures, arrivals).

Source package and training code: `xgb_hw7/` (same course workspace), built by

```bash
PYTHONPATH=src python -m hw7.two_models --per-channel --preset calendar \
    --trend-in-offset --tune --trials 25 --out runs/gp_arc_models
```

## What is in the box

| Path | Contents |
|---|---|
| `gp/model.json` | 2 boosters (departures/arrivals), 480 offsets, trend parameters, hourly rain series, training-loss curves |
| `arc/model.json` | 2 boosters, 2,416 offsets, same metadata |
| `gp/predictions.csv.gz`, `arc/predictions.csv.gz` | out-of-fold predictions (unit, date, slot, channel, y, pred, offset, fold) |
| `gp/summary.json`, `arc/summary.json` | metrics, per-fold scores, tuned parameters |
| `optuna/` | 4 studies (`gp`/`arc` × channel): trial tables and best parameters |
| `figures/` | training loss, predicted-vs-actual hexbin, error by half-hour, deviance by day, worst GPs, Arc error vs volume |
| `config.json` | feature lists, caps, hyperparameters |
| `predict.py` | loads a `model.json` and predicts without the training code |
| `viz/` | interactive map (actual / predicted / residual) and six diagnostic figures |

Total size ≈ 3 MB. No raw data is included.

## Model

```
log mu = log(offset) + log(clip(trend, 1, cap)) + f_tree(slot, dow, is_raining)
```

- **offset** — each (unit, half-hour) training-window mean, floored at 1e-3.
  It is the exposure: the tree learns a multiplicative deviation from it.
- **trend** — a log-linear OLS on the network's daily totals, fitted on the
  training days only, normalised to 1 at the training mean, then capped and
  folded into the offset. The tree never sees the trend as a feature.
  Caps chosen by three-fold absolute deviance: GP departures 1.20, GP arrivals
  1.15, Arc 1.20 / 1.20.
- **features** — `slot` (0–7), `dow` (make-up Saturday encoded as Friday),
  `is_raining` (rain in the half-hour *before* the slot starts — a forecast at
  08:00 cannot know 08:00–08:30 weather). No geometry: an Arc's distance is
  constant per Arc and collinear with its offset (removing it changed deviance
  by <0.2%).

## Out-of-fold results (3 expanding folds, 42 validation days)

| Panel | Channel | Poisson deviance | Offset-only baseline | MAE | Calibration |
|---|---|---:|---:|---:|---:|
| GP | departures | 1.5716 | 1.6033 | 1.917 | 1.005 |
| GP | arrivals | 1.3252 | 1.3352 | 1.885 | 0.977 |
| Arc | departures | 1.3579 | 1.3633 | 1.370 | 0.970 |
| Arc | arrivals | 1.1879 | 1.2001 | 1.301 | 0.973 |

The baseline is the offset alone (each unit's own history), so the improvement
is what the trees add on top of "predict the historical average".

## Visualisation

`viz/index.html` is a single-file, light-theme map: 42 out-of-fold days x 8
half-hours, toggle between actual, predicted and residual (pred - actual),
for both panels and both channels. `viz/figures/` holds the residual
distribution, calibration curve, error by half-hour, daily totals and the
largest per-unit biases. See `viz/README.md`.

## Loading

```bash
pip install numpy pandas xgboost
python predict.py --verify          # loads both panels and scores 500 rows
```

Or in Python:

```python
import json, numpy as np, xgboost as xgb
payload = json.load(open("gp/model.json"))
booster = xgb.Booster()
booster.load_model(bytearray(json.dumps(payload["booster"]).encode()))
# build X = [slot, dow, is_raining] in the order of payload["features"],
# offset from payload["offset"] (unit, slot, channel), trend factor from
# payload["volume_index"] and payload["trend_caps"], then
d = xgb.DMatrix(X); d.set_base_margin(np.log(offset)); mu = booster.predict(d)
```

## Caveats

- **Arc labels are not additive.** 275 of 329 service patches belong to ≥2 Arcs
  (up to 14), because each Arc's label sums its own patch union. Never sum Arc
  predictions as network demand. Arc 44 has no patch and is structurally zero.
- **The trend is a point estimate.** It is fitted on the training window and
  extrapolated; the validation index leaves the training range in every fold.
  `volume_index_se` (median log-SE ≈ 0.04) is stored but not propagated into
  prediction intervals.
- **Hyperparameters and features were selected on the same three folds** that
  the metrics come from, so the numbers above are selection targets, slightly
  optimistic. The frozen-parameter comparisons in the source report are the
  cleaner evidence.
- **Rain uses observed weather.** In deployment, replace it with a forecast;
  the flag is already causal (previous half-hour).
