# xgb_bikeflow

Poisson XGBoost for bike-share arrivals and departures on a geohash grid.

The model predicts **how many bikes arrive at, and leave from, a spatial unit in a
given time slot**. It is a gradient-boosted Poisson regressor over a long panel of
`(unit, date, slot)` counts, with calendar, geometry, weather and fitted
demand-profile features. Everything is driven by one YAML file: where the orders
come from, the horizon, the spatial grid, which rows are trained and scored, how
the loss weights gathering points against everything else, and the
hyperparameters.

```bash
pip install -r requirements.txt
pip install -e .                       # or: export PYTHONPATH=src
xgb-bikeflow check    --config configs/network_v1.yaml
xgb-bikeflow build    --config configs/network_v1.yaml
xgb-bikeflow run      --config configs/network_v1.yaml
xgb-bikeflow rollup   --config configs/network_v1.yaml
xgb-bikeflow inspect  --model runs/network_v1/arrivals/models
xgb-bikeflow evaluate --model runs/network_v1/arrivals/models \
                      --panel runs/network_v1/panel.csv.gz
```

The package lives in `src/xgb_bikeflow/`, so either install it or put `src/` on
the path — running `python -m xgb_bikeflow.cli` from a bare clone will not find it,
which is the point of the src layout. `pytest` is already configured to look in
`src/` (see `pythonpath` in `pyproject.toml`).

`build` writes the panel without training — the step that turns 666k order rows
into counts, and the one whose errors are invisible later. `check` validates a
config against that panel without training either. `run` trains and writes
`runs/<name>/<target>/`. `rollup` turns a run's cell-level predictions into
per-entity totals for a network of gathering points and arcs. `evaluate` loads a
**saved** bundle and scores it on the deployment fold's own out-of-sample window —
the check that what was saved still behaves like what was trained. `inspect`
prints a bundle's contents and verifies its fingerprint.

**v1.0 is `configs/network_v1.yaml`**: 96 gathering points and 381 arcs, every one
of them forecast from one model. See [docs/network_v1.md](docs/network_v1.md) for
where those two halves came from, how they were reconciled, and why an arc's
catchment total must not be added to another's.

## The problem it is built for

Input: every bike trip in the study area, as raw orders
(`date, start_time, start_geohash, end_time, end_geohash`).

Focus: one morning horizon — **06:30–10:30** for the network config, 07:30–10:30
for the peak-only one — and a set of **gathering points** (the depots or anchor
stations the network is organised around).

Output: for **every cell** on the grid, the predicted arrival and departure rate
in each slot of that horizon. For the network config, additionally the totals per
gathering point and per arc.

The interesting part is the loss. Not every cell matters equally:

```
loss = gamma * mean(loss over gathering points)
     + (1 - gamma) * mean(loss over every other cell)
```

`gamma = 0` is "only the ordinary cells matter", `gamma = 1` is "only the
gathering points matter", and `0.5` weights the two groups equally. Note that it
weights the two **means**, not the rows — see [The loss has a gamma](#the-loss-has-a-gamma).

## Using it on your own data

Three steps: build the panel, write a config, run. The example below was executed
as written; the numbers are what came out.

### 1. Point the config at your orders

The input is a table of trips, one row per trip. Nothing else is required —
calendar, geometry, trend and profile features are all derived.

```yaml
source:
  kind: orders
  path: ../data/orders_2026.csv.gz
  date_column: date
  departures_cell: start_geohash
  departures_time: start_time
  arrivals_cell: end_geohash
  arrivals_time: end_time
  time_format: "%Y-%m-%d %H:%M:%S"
  days: workday                 # all | workday | weekend | labelled | [dates]
  holidays: ["2026-04-04", "2026-04-05", "2026-04-06", ...]
  makeup:   ["2026-05-09"]      # 调休: a Saturday that runs on weekday rhythm
  cells: active_in_window       # all | active_in_window | [list] | {file: ...}
  min_events: 2                 # with active_in_window: drop cells below this
```

Counts are built by grouping trips into `(cell, date, slot)` buckets and
zero-filling the result into a dense grid — one row per unit-day-slot, including
the zeros. Zeros must be present as rows: dropping quiet slots teaches the model
that quiet slots do not exist, which is a different model.

#### Which of the two shipped order files to point at

`data/` holds two tables of the same trips:

| File | Rows | Columns | What it is |
|---|---|---|---|
| `dataset_train_20260921_lsz.csv.gz` | 667,821 | `action_dt`, `order_id`, `bike_id`, `start_time`, `end_time`, `start_geohash`, `end_geohash` | the vendor export, untouched |
| `orders_2026.csv.gz` | 666,575 | `start_time`, `end_time`, `start_geohash`, `end_geohash`, `date` | the same trips after a cleaning pass |

The 1,246 missing rows are not a parse failure. They are dropped by an explicit
list of quality flags — unparseable timestamps, null required fields, duplicate
order ids, malformed geohashes, non-positive durations, durations over three
hours, implausible speeds, trips crossing the study-area boundary, and trips on
days flagged as anomalous. The cleaned file is what every result in this README
was produced from, so it is what the shipped configs name.

The raw export drives the pipeline too — it needs one line changed, because its
date column is called `action_dt` rather than `date`:

```yaml
source:
  kind: orders
  path: ../data/dataset_train_20260921_lsz.csv.gz
  date_column: action_dt
```

Built from the same config otherwise, the two produce the same panel shape and
the same keys, differing in 125 of 8,160 cells and by 0.19% of horizon events.
That is what "the cleaning removed 1,246 trips" looks like at the cell level: a
difference, not an error.

`dataset_train_20260921_lsz.csv.gz` is the gzip of the original file at level 9
(34.0 MB from 83.9 MB); round-tripped through pandas it is identical to the
source, null mask and values both. It ships compressed only because the
uncompressed file is 84 MB and git stores that forever.

### 2. Write the rest of the config

The whole interface is one YAML file. Minimum viable version:

```yaml
run:   {name: my_run, outdir: runs}

slots: {start: "07:30", span_minutes: 180, width_minutes: 30}

unit:  {level: 8, shape: [1, 1], aligned: true}

rows:  {train_units: all}

loss:
  gathering: {file: data/gathering_points.json}
  mode: group_mean
  gamma: 0.5
  fit_weight: 0.0

features:
  blocks: [calendar, geometry, trend, profile, weather]
  options:
    weather: {path: data/weather_hourly.csv}

fold:  {scheme: week_forward, n_blocks: 4}

model:
  params: {objective: count:poisson, n_estimators: 700, max_depth: 8,
           learning_rate: 0.04, min_child_weight: 8.0,
           subsample: 0.8, colsample_bytree: 0.8,
           n_jobs: 1, verbosity: 0}

save:  {models: true, dir: models, predictions: true}
```

Paths inside the config resolve relative to **the config file**, not the working
directory, so a config and its data can be moved together.

### 3. Check, then run

```bash
xgb-bikeflow check --config configs/horizon_am.yaml   # 40 s, no training
xgb-bikeflow run   --config configs/horizon_am.yaml   # 42 min, both targets
```

`check` prints what the pipeline understood — order counts, resolved grid, horizon,
fold boundaries, and the feature columns each block will emit — before spending time
on training. On the shipped data:

```
source   : .../data/orders_2026.csv.gz
           666,575 orders -> 143,286 departures / 143,597 arrivals inside 07:30-10:30/30min
           days 57 (workday)   cells 9,594 (active_in_window)
rows     : 3,281,148  units 9594  days 57  slots 6  2026-03-02..2026-05-22
horizon  : 07:30-10:30/30min  (6 x 30 min)
grid     : L8/cell  (partition), not checkable from ids alone
zeros    : arrivals 97.2%  departures 97.5%
loss     : {'gathering': 'file:../data/gathering_points.json', 'n_gathering': 801, ...}
           gathering rows 273,942 of 3,281,148  (8.35%)
fold 1  valid 2026-03-23..2026-04-10  train 15d/863,460 rows  valid 14d/805,896 rows
fold 2  valid 2026-04-13..2026-05-09  train 29d/1,669,356 rows  valid 18d/1,036,152 rows
fold 3  valid 2026-05-11..2026-05-22  train 47d/2,705,508 rows  valid 10d/575,640 rows
features : 33 columns from 5 blocks
check    : OK
```

### 4. Predict

`run` writes `runs/<name>/<target>/` with predictions, a summary, and the model:

```
arrivals/models/model.json               the booster
arrivals/models/feature_state.json       fitted trend/profile/weather state
arrivals/models/features.json            column names, in order
arrivals/models/schema.json              what predict() needs from you
arrivals/models/meta.json                params, folds, horizon, gamma loss, fingerprint
arrivals/predictions.csv.gz              out-of-sample predictions
arrivals/summary.json                    metrics per fold
```

On the shipped config, out of sample over all three folds:

| | arrivals | departures |
|---|---|---|
| raw deviance | 0.2102 | 0.2432 |
| calibrated deviance | 0.2100 | 0.2430 |
| MAE | 0.0500 | 0.0490 |
| gamma loss (γ=0.5) | 0.3547 | 0.4246 |
| — gathering points | 0.5284 | 0.6426 |
| — all other cells | 0.1810 | 0.2066 |

The gathering points are **2.9x** harder than everything else for arrivals and
**3.1x** harder for departures. That is expected and is the reason the loss has a
gamma at all: those are the cells where being wrong matters, and they are the ones
carrying real volume.

Per fold, calibrated: arrivals 0.2255 / 0.2074 / 0.1929, departures 0.2555 /
0.2429 / 0.2258, on 14 / 18 / 10 validation days. The last fold is the *best* for
both targets. Under equal-count blocks it was the worst (0.1958 and 0.2316 raw,
against 0.2313 and 0.2607 firsts). Two things changed at once and both push the same
way: the last fold now validates ten working days instead of fifteen, and those ten
are two whole weeks rather than two and a half. Fewer, more homogeneous days are
easier. **This is a change in the split, not in the model**, and the honest reading
is that the previous "the last fold is worst because of the May ramp" story was
partly the geometry talking.

The gathering points are **3.0x** harder than everything else for arrivals and
**3.2x** harder for departures. That is expected and is the reason the loss has a
gamma at all: those are the cells where being wrong matters, and they are the ones
carrying real volume.

To predict, load the bundle and hand it a frame with the columns `schema.json`
lists. Labels are not needed and are ignored if present:

```python
import json
import numpy as np
import pandas as pd
import xgb_bikeflow.model as M

model_dir = "runs/my_run/arrivals/models"
bundle = M.Bundle.load(model_dir)
schema = json.loads(open(f"{model_dir}/schema.json", encoding="utf-8").read())
print(schema["required_input_columns"])
# ['block_id', 'date', 'slot', 'lat', 'lon']

units = sorted(panel["block_id"].unique())
future = pd.DataFrame({
    "block_id": np.repeat(units, 6),
    "date": "2026-05-25",
    "slot": np.tile(np.arange(6), len(units)),      # 0..5 = 07:30..10:00
}).merge(panel[["block_id", "lat", "lon"]].drop_duplicates("block_id"),
         on="block_id", how="left")

mu = bundle.predict(future)      # expected arrivals per row, already calibrated
```

`slot` is an index into the horizon, not a clock hour: `0` means 07:30 here
because the config says the horizon starts at 07:30 with 30-minute slots. The
bundle carries that mapping, so `predict` needs no config.

That is the only interface: a frame in, a rate per row out. `Bundle.load` verifies
the fingerprint, so a booster paired with a feature state it was not trained with
raises instead of quietly predicting something else.

To score a saved bundle against labelled data the CLI way:

```bash
xgb-bikeflow evaluate --model runs/my_run/arrivals/models \
                                    --panel runs/my_run/panel.csv.gz
```

## The loss has a gamma

```yaml
loss:
  gathering: {file: data/gathering_points.json}   # or: none | all | [ids]
  mode: group_mean          # group_mean | row_weighted | off
  gamma: 0.5
  fit_weight: 0.0           # optional: reweight the *fit* too
```

The written formula is `gamma * L(gathering) + (1 - gamma) * L(everything else)`,
and the obvious reading is a per-row weight: give each gathering-point row weight
`gamma` and each other row `1 - gamma`. That reading is wrong at this scale. On
the shipped data there are 801 gathering points and 8,793 other cells — a ratio of
about 1:11 — so at `gamma = 0.5` the gathering points would receive 8% of the
gradient and `gamma` would stop meaning what it says. `mode: group_mean` therefore
computes each group's **mean** deviance first and mixes those two numbers.
`mode: row_weighted` is kept for the literal reading, so the difference can be
measured rather than argued.

This cannot be done with `sample_weight`. Deviance is `sum_i w_i l_i`; to make it
a mean per group, `w_i` would have to depend on the group's size, and XGBoost
normalises weights away. So the gamma loss is computed **outside** the booster:
training is always plain `count:poisson`, and gamma only decides how the result is
scored. One bundle can therefore be judged at any gamma without retraining.

`fit_weight` is separate and optional: above zero it balances the two groups
*inside the fit* as well, so the booster spends its capacity where the loss looks.

It is off by default because it was measured and did not help. Sweeping
`fit_weight` over 0, 0.25, 0.5, 0.75 on a 1,201-unit sample (every gathering
point plus 400 others, the shipped hyperparameters and folds) moves the gamma
loss by at most **0.53%**, with no consistent sign and no monotone trend:

| fit_weight | 0 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|
| arrivals, γ loss | 0.345254 | 0.347087 (+0.53%) | 0.345731 (+0.14%) | 0.344737 (−0.15%) |
| departures, γ loss | 0.413984 | 0.413337 (−0.16%) | 0.412786 (−0.29%) | 0.414157 (+0.04%) |

The two groups' own losses barely move either (gathering 0.5122 → 0.5114 on
arrivals, 0.6371 → 0.6386 on departures). The reason is visible in the ratio:
`count:poisson` already fits the conditional mean, and reweighting changes *which
rows dominate the gradient*, not *what the model is being asked to predict*. The
metric is a blend of two means; making the fit care more about one half does not
make that half more predictable. Anyone with a reason to try it can, but the
default should not pretend there is a gain.

## The horizon is a slot grid

```yaml
slots:
  start: "07:30"        # or minutes past midnight: 450
  span_minutes: 180     # 07:30 -> 10:30
  width_minutes: 30     # 6 slots, half-open: 10:30:00 belongs to the next horizon
```

This replaces the integer `hour` column of earlier versions. `slot` is an index
`0..n_slots-1` within the horizon, and the horizon is half-open: a trip at exactly
10:30:00 falls outside. `width_minutes` must divide `span_minutes`; `180/40` is
refused rather than rounded.

A horizon that starts on the hour with 60-minute slots (`start: 0,
width_minutes: 60`) is exactly the old behaviour, and `source.kind: panel` still
reads a pre-aggregated panel in that case. A panel cannot express 07:30–10:30 —
its hours are integers — and the pipeline says so rather than silently shifting
the target.

Weather is hourly, so on a sub-hourly horizon each slot takes the value of the
hour it starts in (floor, not interpolation). Averaging two hours of rainfall would
invent a rate that was never measured, and `is_raining` is a flag that cannot be
averaged at all.

## Why this exists

It is a rewrite of a research pipeline that had grown one script per experiment.
The parts that mattered were spread across three directories and eight copies of
the same deviance function, and adding a fourth variant meant copying a script
and editing its constants.

Five axes explain every variant that ever existed:

| Axis | Question | Config key |
|---|---|---|
| **Source** | where do the counts come from? | `source` |
| **Horizon** | which part of the day, in what slots? | `slots` |
| **Grid** | what is a spatial unit? | `unit` |
| **Rows** | which rows train, which are scored? | `rows` |
| **Loss** | gathering points vs everything else | `loss` |

Each axis is a value, so a new experiment is a new YAML file rather than a new
directory.

## The spatial grid is not baked in

The unit is a **level**, a **shape** and an **alignment**, not a name:

```yaml
unit:
  level: 8        # geohash characters per cell: 6 | 7 | 8
  shape: [3, 3]   # cells per unit; [1, 1] is a single cell
  aligned: true   # false = a window centred on the cell, which OVERLAPS
```

`configs/horizon_am.yaml` and `configs/tile3x3_l8.yaml` differ in one line and
produce different units. Anything expressible in those three numbers works
without touching code — a 2x2 tile, an L7 cell, a 5x5 window.

`aligned: false` produces overlapping units. `grid.GridSpec.partitioned` reports
this, and the pipeline refuses to sum totals across a non-partition, because
overlapping units double-count.

## Features are five independent families

Each block declares the columns it emits and its own input requirements. Order is
preserved, and later blocks may read columns earlier ones emitted.

| Block | Columns | Fitted | Reads |
|---|---|---|---|
| `calendar` | 12 — slot, day type, holiday adjacency, window index | no | date, slot |
| `geometry` | 5 — x/y, distance and bearing from the centroid | no | lat, lon |
| `trend` | 1 — a log-linear scale per day type | **yes** | arrivals, date |
| `profile` | 6 — per (unit, slot) level, spread and share | **yes** | arrivals, departures |
| `weather` | 9 — precipitation, temperature, wind, cloud | no | date, slot |

Turn any of them off by removing the line from `features.blocks`.

The two **fitted** blocks are the reason a model cannot be saved as just a
booster: `trend_scale` comes from a regression on daily totals and `arr_slot_mean`
comes from a group-by over training rows. Both are stored in the bundle.

There used to be a sixth block, `panel_context`, carrying a column
`n_cells_touched`. No shipped config turns it on now, and the reason is worth
keeping: that column was defined as the number of distinct cells inside a unit that
saw a trip that day, which is computed from the same trips as the label. At unit
shape `(1, 1)` it evaluates to exactly `(arrivals + departures) > 0` — the model was
told, for every row of a day, whether that day had any activity at all. It is a
plausible-looking static attribute and it is the label wearing a different name.
`tests/test_golden.py::test_no_feature_is_derived_from_the_label` now fails if any
two-valued panel column lines up with the label's own support.

The block itself remains in the registry, because a genuine static attribute — a
dock count, a station capacity — belongs there and needs no new code. It emits
nothing unless the panel carries such a column.

## Day types are not weekdays

Two corrections, both measured and both applied *before* the split so neither can
leak:

- **A 调休 make-up workday** runs on the working-day rhythm even when it falls on
  a Saturday (`makeup_dow`).
- **A statutory holiday** runs on weekend rhythms whatever weekday it lands on
  (`holiday_dow`).

Without them, a single make-up Saturday scored a deviance of 3.68 against 1.20 for
ordinary days, because no training fold contained one and the tree had nothing to
learn from. With them, 1.17.

What the day-type filter implies for the trend block
---------------------------------------------------
`source.days: workday` drops every weekend row, so the trend block never sees a
weekend and cannot fit a weekend slope. It **borrows** the weekday slope instead
of leaving a hole, and then says so:

```
note    : trend: no weekend row in the training window, so its slope and
          intercept were borrowed from the other day type.  Predictions for
          those days carry that assumption.
```

That note is printed by `check` and, once per fold, by `run`. It is the same
choice as everywhere else in this package: the behaviour is a defensible guess
rather than a crash, but a user who is not told will read the output as if it
rested on data. `FeatureBlock.notes()` is the hook — a block that fires a fallback
or guesses at a missing class returns a sentence, and `check` prints all of them.

## Saving a model

A bundle is five files (listed under [Predict](#4-predict) above), and it refuses
to load if any is missing.

What makes it more than "save the booster" is `feature_state.json` — the fitted
trend slopes, profile tables, holiday list and weather table. A booster on its own
predicts garbage on new data if the profile table it was trained against has
changed, and XGBoost cannot detect that: it only sees numbers, and the numbers
still look plausible. The fingerprint in `meta.json` is a hash of that state, so a
mismatch raises instead of returning a quietly wrong answer.

The bundle also stores the horizon, because a slot index is meaningless without
it: `slot 3` is 09:00 on a horizon starting at 07:30, and 03:00 on a horizon
starting at midnight. Rebuilding the pipeline from the config instead would make
the bundle stop working when the config is edited.

That is also why `Bundle.load` re-checks the booster's `feature_names` against
`features.json`. XGBoost catches a wrong column *order* only when it was fitted on
a DataFrame — fit on `.to_numpy()` and it accepts a permuted frame silently. The
explicit check makes that structural rather than incidental.

## Results

### v1.0 — `configs/network_v1.yaml`

96 gathering points and 381 arcs, 06:30–10:30, 65 labelled days, 4,243,200 rows.

| | raw deviance | calibrated | MAE |
|---|---|---|---|
| arrivals | 0.1391 | **0.1388** | 0.0366 |
| departures | 0.1721 | **0.1721** | 0.0380 |

Rolled up to one morning total per entity per day:

| entity | deviance | calibration | vs its own average | median APE |
|---|---|---|---|---|
| GP, arrivals | 2.7717 | 0.947 | **+40.1%** | 23.9% |
| GP, departures | 2.8191 | 0.973 | **+40.5%** | 25.5% |
| arc, arrivals | 2.2340 | 0.939 | **+34.5%** | 28.5% |
| arc, departures | 1.5982 | 0.984 | **+18.9%** | 29.8% |

The deviance difference between the two tables is aggregation, not decay: the same
forecast scores about 0.14 per cell and 2.8 per dock-day. `docs/network_v1.md`
has the error decomposition — 13% of the entity error is one fold that has never
seen a holiday, 13% more is one dock that grew 22% in a month while the shared
trend slope could not follow it.

A caution that matters more than either number: **all 96 docks get a forecast, but
only 50 of the 90 HW6 docks get an arc.** The repro network drew its docks at 12.5
cells each against HW6's 62.6, so most HW6 docks contain no repro dock and can
therefore be nobody's arc endpoint. The 40 unconnected docks move 21.8% of the
morning's departures, and two of them — `H6GP052` at 91.9/day and `H6GP011` at
51.7/day — are busier than the median connected one. `docs/network_v1.md` breaks
the 40 down and explains why a looser matching rule moves the problem instead of
solving it.

### `configs/horizon_am.yaml` — the peak-only study

07:30–10:30, 10,005 cells, 65 labelled days, 3,901,950 rows.

| | calibrated deviance | MAE | gamma loss (0.5) |
|---|---|---|---|
| arrivals | 0.1877 | 0.0453 | 0.3301 |
| departures | 0.2202 | 0.0446 | 0.3968 |

Against the EDA report's XGBoost on the identical rows (40,320 validation rows,
departures, same deviance function): **1.7854 → 1.4765, a 17.3% improvement**.

## Evaluation

Three expanding-window folds, never a random split: a random split would fit the
per-(unit, slot) profile on future days and score well for the wrong reason.

### Where the block edges fall

Two rules, `fold.scheme`:

| scheme | edges | what a fold means |
|---|---|---|
| `week_forward` (default) | on Mondays | trains on everything before a week, validates whole weeks |
| `walk_forward` | equal day counts | same shape, but edges can land mid-week |

`week_forward` is the default because `walk_forward` has a trap on any panel that
is not every calendar day. `source.days: workday` removes every Saturday and
Sunday, so the shipped panel's 57 working days are spread over 12 calendar weeks,
and 57/4 = 14.25 lands the edges on a Thursday and a Friday:

```
walk_forward   fold 1: 03-20 (Fri) .. 04-09 (Thu)   14d
               fold 2: 04-10 (Fri) .. 04-29 (Wed)   14d
               fold 3: 04-30 (Thu) .. 05-22 (Fri)   15d
week_forward   fold 1: 03-23 (Mon) .. 04-10 (Fri)   14d
               fold 2: 04-13 (Mon) .. 05-09 (Sat)   18d
               fold 3: 05-11 (Mon) .. 05-22 (Fri)   10d
```

Nothing leaks either way — no feature uses a lag — but under `walk_forward` a
validation block that starts on a Friday shares its Monday-Thursday with the
training set, so "predict next week" is not what the fold tested. Under
`week_forward` it is.

Two honest caveats about the week-aligned version on this panel:

- The week counts come out **3 / 4 / 2**, not 3 / 3 / 3. 04-06 and 05-04 are
  Mondays that fall on statutory holidays, so neither is in the panel and neither
  can carry a cut; the nearest Monday takes over. Forcing an even 3 / 3 / 3 would
  mean cutting inside a week, which is the thing the scheme exists to avoid.
- `week_forward` **refuses** rather than falling back: if the panel has fewer
  usable Mondays than it needs interior cuts, it raises and tells you to use
  `walk_forward` or fewer blocks. A silent fallback would produce a number
  labelled "week_forward" that came from a different geometry.

`walk_forward` stays for reproducing published figures.
`configs/compat/cell_l8_published.yaml` uses it — and on that panel it happens to
be week-aligned anyway, because 84 consecutive days starting Monday 2026-03-02
divide by four into 21-day blocks. Its YAML says so, and
`tests/test_golden.py::test_equal_count_blocks_do_cut_mid_week` keeps the trap
from quietly disappearing.

### The holidays are kept, not deleted

`source.days` has four values:

| value | keeps |
|---|---|
| `all` | every day |
| `workday` | weekdays minus statutory holidays, plus make-up Saturdays |
| `weekend` | the complement of `workday` |
| `labelled` | `workday` **plus the statutory holidays** |

`labelled` is what you want when the holiday rate is part of the deliverable. The
eight holidays are in the order file — 2026-04-06 is a Monday with 4,530 orders,
2026-05-04 a Monday with 5,081 — and inside the 07:30-10:30 horizon they run at
**0.34x a working day**. Dropping them is a modelling choice, not data cleaning.

It matters because a model trained without them has `is_holiday == 0` on every
training row, so that column cannot split anything and a holiday is answered by
the trend extrapolation:

| | holiday actual | holiday pred | calibration | holiday dev | workday dev |
|---|---|---|---|---|---|
| `workday` (shipped) | 0.0139 | 0.0425 | **3.05x** | 0.1449 | 0.1874 |
| `labelled` | 0.0139 | 0.0133 | **0.95x** | **0.1022** | 0.1868 |

(departures: 2.86x → 0.96x, dev 0.1647 → 0.1251, workday dev −1.14%)

So keeping them cuts holiday deviance by 24–29% and costs the working days 0.3–1.1%.
`docs/holiday_and_gamma.md` has the full measurement. `horizon_am.yaml` stays on
`workday` because that is the operational scope the peak-only brief describes;
`network_v1.yaml` uses `labelled`, because v1.0's deliverable includes the holiday
rate. Switching is a one-line change.

**The one place this still bites** is fold 1, whose 15-day training window
contains no holiday at all while its validation window has three. It prices them
at 2.7–2.9x their true volume. On the entity rollup that is 13% of the total
absolute error. See `docs/network_v1.md`; it is reported, not engineered away.

### `gamma` is a policy weight, not a searched one

`gamma` mixes the two group *means*, and it never enters the booster — so one
fitted model can be rescored at any gamma. Sweeping it on fold 3 gives a
**monotone** curve (arrivals 0.16824 at 0.0 rising to 0.46660 at 1.0), because the
two group losses are fixed and only their mixture moves. The minimiser is therefore
always an endpoint, i.e. "ignore one group entirely", which is not what the brief
asks for. That is the reason it is set in the YAML rather than tuned.

The knob that *would* change the fit is `loss.fit_weight`; it was swept
(0 / 0.25 / 0.5 / 0.75) and moved the loss by at most 0.53% with no consistent sign,
so it ships at `0.0`. See `docs/holiday_and_gamma.md` §3.

Reported as **Poisson deviance** rather than RMSE, because the target is a count
with a long tail — RMSE rewards getting the big slots right and ignores the many
small ones. Note that deviance is **not comparable across aggregation levels**: the
same predictions score about 0.2 on 30-minute cells and 4.6 on cell-days. Smaller
counts make each one relatively easier to be wrong about.

`calibration` is a forward-only multiplicative level correction: each fold sets
the factor for the next one from its own out-of-sample ratio. Fold 1 is
uncalibrated by construction. Two settings are worth knowing about, and both are
commented in the configs with the numbers they produce:

- `window_origin: train` vs `frame` — whether the day-window offset is counted
  from the training origin. `train` is the default; `frame` is kept because the
  published figures used it.
- `calibration.compound` — whether the correction feeds its own output forward.
  `false` is the default.

`configs/compat/cell_l8_published.yaml` pins the legacy pair of both, so the old
pipeline's behaviour stays reproducible on demand.

## Layout

The repository keeps code and assets apart, which is what a reader expects of a
Python project and what makes the package installable:

```
xgb_bikeflow/
  src/xgb_bikeflow/     the importable package
    cli.py              build | run | check | rollup | evaluate | inspect
    spec.py             YAML -> RunSpec; every error names its YAML key
    sources.py          orders -> collapsed, zero-filled panel
    slots.py            SlotSpec: horizon start, span, slot width
    groups.py           gathering-point membership and the gamma loss
    grid.py             GridSpec: level, shape, alignment
    rollup.py           cell forecasts -> gathering points and arcs
    data.py             panel contract and validation
    rows.py             which units train / are scored / are down-weighted
    split.py            week-aligned or equal-count folds, forward calibrator
    metrics.py          poisson deviance, per row and pooled
    model.py            training loop and the five-file Bundle
    features/           one module per family
  scripts/              every script that produces an artefact
    build_network_v1.py    reconcile HW6 GPs with the repro arcs -> network_v1.json
    build_network_cells.py network_v1.json -> the two flat cell lists
    measure/               the measurements behind docs/ -- see measure/README.md
  configs/              network_v1.yaml (v1.0), horizon_am.yaml, tile3x3_l8.yaml,
                        eda_comparison.yaml, compat/
  data/                 the vendor export (gzip, 34 MB) and the cleaned extract,
                        the network and the five files it was built from, the
                        EDA panel, weather
  docs/                 network_v1.md, fold_geometry.md, holiday_and_gamma.md,
                        eda_comparison/ (measurement output)
  tests/                counting, gamma, rollup additivity, end-to-end anchors
  pyproject.toml        src layout, `pythonpath = ["src"]` for pytest
```

Nothing at the top level is importable except through `src/xgb_bikeflow/`, so
`import xgb_bikeflow` cannot accidentally resolve to the repository root. Config
paths stay relative to the config file, which is why `configs/network_v1.yaml`
reaches its data as `../data/...` whether you run it from the repo or from
elsewhere.

`docs/network_v1.md` explains where the 96 gathering points and 381 arcs came
from, how the two source studies were reconciled, and why an arc's catchment
total is not additive. `docs/fold_geometry.md` explains why the fold edges are cut
on Mondays and what that changed in the numbers; `docs/eda_comparison/` holds the
JSON from the comparison against the EDA report, so the figures quoted above can
be checked without re-running anything.

## Data contract

For `source.kind: orders`, one row per trip:

| Column | Notes |
|---|---|
| `date` | `YYYY-MM-DD`, or fold into the departure timestamp |
| `start_time`, `end_time` | departure and arrival timestamps |
| `start_geohash`, `end_geohash` | 8-character geohash of each end |

Any column may be renamed: `date_column`, `departures_cell`, `departures_time`,
`arrivals_cell` and `arrivals_time` name the five fields, and `time_format` says
how to parse the timestamps. That is how the vendor export is read without
being edited first.

For `source.kind: panel`, a long table with one row per `(unit, date, hour)`:

| Column | Required | Notes |
|---|---|---|
| `block_id` | yes | the unit id |
| `date` | yes | `YYYY-MM-DD` |
| `hour` or `slot` | yes | `hour` is accepted and renamed |
| `arrivals` | yes | non-negative count |
| `departures` | yes | non-negative count |
| `lat`, `lon` | if geometry is on | unit centroid |

Counts must be counts, not rates. Zeros must be present as rows — dropping them
teaches the model that quiet slots do not exist.

## Notes on reproducibility

`n_jobs` is a required config key, not an environment detail: XGBoost is not
bit-reproducible across thread counts. Measured on this package's earlier panel
(arrivals, pooled raw deviance): 1.2133 at one thread, 1.2255 at two, 1.2315 at
four, 1.2201 at eight, 1.2159 at all 32. That is a spread wider than most of the
differences worth reporting, and a config left on `-1` produces different numbers
on a 4-core laptop and a 32-core workstation with nothing in the output saying so.
The shipped configs therefore pin `n_jobs: 1` and pay roughly 3x wall clock.

Row order is preserved from the source. XGBoost's `subsample` draws by row
position, so re-sorting the same rows changes the model. `source.sort_rows: true`
is available and is a modelling decision, not a housekeeping one.

## License

MIT — see `LICENSE`.
