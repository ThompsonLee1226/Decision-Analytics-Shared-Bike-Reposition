# v1.0 — forecasting the network, not a list of cells

The package has been able to forecast a cell for a while. This is the version
that forecasts **the entities the network is described in**: 96 gathering points
and 381 directed arcs, every one of them, from one model.

    xgb-bikeflow check  --config configs/network_v1.yaml
    xgb-bikeflow run    --config configs/network_v1.yaml
    xgb-bikeflow rollup --config configs/network_v1.yaml

## Where the two halves came from

| | source | what it decides |
|---|---|---|
| gathering points | `HW6_A_结果总览.xlsx`, sheet `GP主表` | **where** bikes are worth watching: 90 docks, 5,630 member cells |
| arcs, patches, rates | `GP_arc_patch_repro_20261106/expected_current/network.json` | **how** bikes move between them: 67 GPs, 401 arcs, 811 patches |

Neither file is in this repository — they are course artefacts, and the second is
122 MB zipped. They sit beside the checkout under `数据/` and `外部依赖/`; see the
repository root `README.md` for that layout. Five CSV copies of the parts
`build_network_v1.py` actually reads are in `data/`: `hw6_gp_master.csv`,
`hw6_grid_ownership.csv`, `repro_network.json`, `repro_cell_mapping.csv`,
`repro_cells.csv`. So the build runs from a clone, and its inputs are small enough
to read.

They are two different studies of the same city and they disagree about what a
gathering point is. Reconciling them is the work `build_network_v1.py` does, and
it is the only place in the package where a judgement call was made rather than a
number computed. It is worth reading what it decided.

### The reconciliation, in one table

| | HW6 | repro | merged |
|---|---|---|---|
| gathering points | 90 | 67 | **96** (90 + 6 repro-only) |
| member cells | 5,630 | 835 | **5,678** |
| arcs | — | 401 | **381** |
| arc–cell memberships | — | 28,686 | **28,686** |
| distinct cells | 5,630 | 8,333 | **8,160** |
| ordered arc endpoint pairs | — | 401 | **343** |

The rule: **a repro GP claims the HW6 GP that owns most of its cells**. 61 of the
67 repro GPs could vote, 28 of them unanimously, mean purity 0.73; six shared an
HW6 GP with another repro GP (up to three claimants); six matched nothing and
became their own endpoints, named `R<id>`.

Twenty of the 401 arcs then collapsed: both endpoints landed on the same HW6 GP,
so the arc became a self-loop. A self-loop is not a route, so those twenty are
dropped and counted rather than kept as a zero-length arc. The 381 that remain
cover 343 distinct ordered pairs — 235 pairs appear once, 28 twice, 2 three times
— because two repro GPs on the same HW6 pair still make two different routes.

### The two cell sets do not nest, and that matters

    GP-only cells     3,100      a dock's cells no arc reaches
    arc-only cells    2,482      a route's catchment no dock owns
    in both           2,578
    ------------------------------
    union             8,160

An arc's service cells and a GP's cells overlap but neither contains the other.
This is not a defect in either study: a dock is a place people walk to, a route's
catchment is the area its vehicles can reach, and the two areas were drawn for
different purposes. It does mean the panel has to carry both.

**The panel universe is the union, 8,160 cells**, which is not the same as
`active_in_window`. 1,557 of those cells have no event at all in the horizon and
4,043 have fewer than 2. They stay in anyway: a dock with no observed departures
is a real dock, and dropping it would turn the GP's total into the sum over
whichever of its cells happened to be busy that morning. The model predicts ~0
for them, which is the right answer.

### The forty docks no arc reaches

Every one of the 96 docks gets a forecast, because every dock is a set of cells
and the model predicts cells. Only **50 of the 90 HW6 docks** get an arc, and the
reason is geometry that was fixed before any mapping rule ran.

| | HW6 | repro |
|---|---|---|
| docks | 90 | 67 |
| member cells | 5,630 | 835 |
| cells per dock | **62.6** | **12.5** |

The repro study drew docks two orders of magnitude sparser: its 835 cells are
**14.8%** of HW6's 5,630. So most HW6 docks contain no repro dock at all, and an
arc needs *two* repro docks as endpoints. Splitting the 40 unconnected docks by
cause:

| cause | docks | their departures/day | share |
|---|---|---|---|
| the repro study never touched them | 15 | 70 | 3.9% |
| touched, but lost a contested vote | 25 | 328 | **18.0%** |
| **total unconnected** | **40** | **398** | **21.8%** |

The 15 untouched are the small ones (`H6GP084` is the busiest at 15.1/day). The 25
that lost a vote are not: `H6GP052` moves **91.9 departures/day** from two cells
and lost to `H6GP050`; `H6GP011` moves **51.7/day** from 40 cells and lost to
`H6GP016`. Those two alone are 7.9% of the city's morning departures, sitting
inside a dock that has no route out of it.

The mechanism is visible in one number: **33 of the 61 voting repro docks straddle
more than one HW6 dock**. Together they touch 107 HW6 docks but can only be
delivered to 61, so 46 dock claims are discarded by whichever rule breaks the tie.
A softer rule — every HW6 dock that shares any cell gets the repro dock — lifts
coverage from 50 to **75 of 90**, but it gives 27 docks two owners and lets one
repro dock be an arc endpoint in several docks at once. That moves the
arbitrariness, it does not remove it.

Two smaller mismatches belong to the same story:

- **Six repro docks share no cell with any HW6 dock**, so they survive as their own
  endpoints `R41 R53 R55 R57 R58 R61` (48 cells, 0.8% of the dock cells). Their
  nearest HW6 cell is 8–323 m away: they are neighbours that were drawn on a
  different lattice, not different places.
- **Twenty of the 401 arcs became self-loops** and were dropped (401 → 381), because
  both endpoints voted into the same HW6 dock. The longest is 1,780 m across
  17 patches. Eleven of the 811 patches are served only by such an arc, so they end
  up with no route at all.

None of this is a modelling error, and none of it is fixed by predicting better.
It is the cost of grafting a 67-dock network onto a 90-dock one. A v2 should
either re-derive the arcs on HW6's own docks, or accept that the arcs answer a
coarser question than the docks do — but the forecast layer is already correct for
all 96.

## The horizon is 06:30–10:30, not 07:30–10:30

The network file states the window it was designed on:

    "period": "06:30-10:30"

`horizon_am.yaml` uses 07:30 because that is when the peak bites for a
rebalancing crew. Scoring a network designed for 06:30 against a 07:30 horizon
would answer a different question than the one it was built for, so v1.0 moves to
06:30 and takes the extra hour.

That hour is not free: 06:30–07:30 carries **40,142 endpoints, 13.6% of the
07:30–10:30 volume**, and it is the steepest part of the curve — the hour when
bikes leave home docks and nothing has come back yet. It is also the hour a
morning shift actually starts.

Eight half-hour slots: `06:30 07:00 07:30 08:00 08:30 09:00 09:30 10:00`.

## What the bigger panel costs

| | horizon_am.yaml | network_v1.yaml |
|---|---|---|
| horizon | 07:30–10:30 (6 slots) | 06:30–10:30 (8 slots) |
| cells | 10,005 (active) | 8,160 (the network) |
| days | 65 (labelled) | 65 (labelled) |
| rows | 3,901,950 | **4,243,200** |
| gathering points | 809 | **5,678** |
| gathering share of rows | 8.1% | **69.6%** |
| zero share (arr / dep) | 97.5% / 97.8% | 97.9% / 98.1% |

The gathering share is the difference that changes the meaning of `gamma`. On
`horizon_am` the two groups are 809 and 9,196 cells, so a per-row weight of 0.5
would put 8% of the gradient on the docks — that is why the loss averages within
groups instead. On `network_v1` the groups are 5,678 and 2,482 cells, and
`gamma: 0.5` is close to what an unweighted fit would do anyway. The setting is
the same; what it means is not.

## Rolling up: why arcs do not add up

A gathering point's cells **partition** its area — 5,678 cells, each in exactly
one GP — so a GP's forecast is their sum and the GP totals add up to the total
over all dock cells.

An arc's catchment **does not partition anything**: 5,060 distinct cells carry
28,686 arc memberships, a mean multiplicity of **5.67**, because a cell belongs to
every arc whose patches can reach it. An arc's catchment total is a real number
("how many bikes become available within reach of this route") but adding two
arcs' totals double counts the cells they share.

`rollup.py` reports both, keeps them apart, and gives `pred_per_cell` so a 2-cell
arc and a 252-cell arc can be compared at all. Measured on the shipped network:

    sum over GP totals     83,603      partitions the dock area
    sum over arc totals   234,026      = 2.80x the GP sum, overlapping
    sum over all cells    111,863      the panel total

The 2.80 is not an error to be fixed, and it is worth knowing where it comes from
because the obvious explanation is the wrong one. Splitting it:

    arc cells, counted once       36,236 over 5,060 cells
    arc cells, by membership     234,026
    effective multiplicity         6.46      (volume-weighted)

    arc volume / GP volume = (36,236 / 83,603) x 6.46 = 0.433 x 6.46 = 2.80

So the arcs sum to 2.8x the docks for **two** reasons that push in opposite
directions. Arc cell sets cover less ground than the docks do — 36,236 against
83,603, because only 74.8% of the arc cells sit inside a dock at all — which
pulls the ratio *down*. Overlap pulls it *up*, and by more than the raw 5.67
suggests: the effective multiplicity is **6.46**, because the cells many arcs
share are the busy ones (mean forecast 9.27 in cells reached by four or more arcs
against 4.01 in cells reached by one). A cell at the middle of the network is
inside six catchments because it is a busy place, not despite it.

That is the same reason a per-cell score cannot be read as a per-arc one: the
units that overlap most are where most of the demand is.

## What a v1.0 run produces

    runs/network_v1/
      panel.csv.gz                 4.24M rows, the panel the run used
      arrivals/
        predictions.csv.gz         cell-level, calibrated, with is_gathering
        summary.json               metrics, folds, fingerprints
        models/                    the bundle (booster + feature state + meta)
      departures/
        ...
      entities/
        arrivals.csv.gz            gp and arc rows, with actual totals
        departures.csv.gz

The entity tables are keyed `(kind, id, date, slot)`:

| column | meaning |
|---|---|
| `kind` | `gp` or `arc` |
| `id` | the GP id (`H6GP001`) or arc id (`170`) |
| `pred_total` | summed forecast over the entity's member cells |
| `actual_total` | summed observation over the same cells |
| `n_cells` | how many member cells were present in the predictions |
| `missing_cells` | GP only — member cells absent from the panel |
| `pred_per_cell` | `pred_total / n_cells`, comparable across entity sizes |
| `from_gp`, `to_gp` | arcs only — the endpoints |

## The measured results

Cell level, three folds, 4.24M rows, 6.5% of which are scored out of sample:

| | raw deviance | calibrated | MAE |
|---|---|---|---|
| arrivals | 0.1391 | **0.1388** | 0.0366 |
| departures | 0.1721 | **0.1721** | 0.0380 |

The per-fold story is the same one the whole package tells: fold 1 (15 training
days) is worst at 0.1535 / 0.1867, fold 3 (49 training days) best at 0.1299 /
0.1617. Training time 662 s and 673 s per target at one thread.

### What the entities are worth

Summing the eight slots into one morning total, per entity per day — the number a
dispatcher would actually read:

| entity | rows | deviance | calib | same-entity mean | improvement | median APE |
|---|---|---|---|---|---|---|
| GP, arrivals | 4,800 | 2.7717 | 0.947 | 4.6283 | **+40.1%** | 23.9% |
| GP, departures | 4,800 | 2.8191 | 0.973 | 4.7399 | **+40.5%** | 25.5% |
| arc, arrivals | 19,050 | 2.2340 | 0.939 | 3.4086 | **+34.5%** | 28.5% |
| arc, departures | 19,050 | 1.5982 | 0.984 | 1.9706 | **+18.9%** | 29.8% |

"Same-entity mean" is the honest floor: predict each entity's own historical
average and nothing else. The model beats it by 19–41%, which is the claim worth
making — that a dock's *day* is forecast better than the dock's own average.

Calibration is the thing to watch, not the deviance: every entity level sits at
0.94–0.98, i.e. the model under-predicts the morning total by 2–6% once summed.
That is why the raw and calibrated columns converge here and diverge at cell
level — summing eight slots averages the per-slot bias out.

### Where the error actually is

Two causes account for a third of the entity error, and both are understood.

**1. The first fold has never seen a holiday.** The panel keeps eight statutory
holidays, but fold 1 trains on 15 days that contain none of them. Its validation
window does contain three (04-04, 04-05, 04-06), and it prices them as ordinary
weekdays:

| | fold 1 holidays | fold 2 holidays | fold 3 holidays |
|---|---|---|---|
| arrivals pred/actual | **2.73x** | 0.93x | 1.07x |
| departures pred/actual | **2.90x** | 0.94x | 1.05x |

Fold 2 and 3 have holidays in their training windows and answer them correctly.
The harm is confined to the fold that could not have known:

    share of total entity |gap|     arrivals   departures
      fold-1 holidays                  13.0%       13.5%
      all holidays                     18.2%       17.9%
      non-holiday dock-days            81.8%       82.1%

Dropping the fold-1 holidays takes GP deviance from 2.7717 to **2.0936**
(arrivals) and 2.8191 to **2.1223** (departures). That is not a claim that the
model is that good — it is the price of a fold whose training window is three
weeks long. It is reported rather than engineered away, because the alternative
(folding the holidays into the trend fit) would be fitting on the test window.

**2. One dock is 13% of the departure error.** `H6GP016` is a 19-cell dock with
408 departures/day, four times the median. It is 12.7% of the total departure
|gap| and 17% of the per-dock MAE share by itself. The cause is visible in its
weekly ratio and has nothing to do with holidays:

| week | pred | actual | ratio |
|---|---|---|---|
| 03-23 | 2,118 | 2,212 | 1.00 |
| 03-30 | 2,029 | 2,460 | 0.82 |
| 04-13 | 2,002 | 2,680 | **0.75** |
| 04-20 | 2,167 | 2,605 | 0.83 |
| 05-04 | 1,979 | 1,827 | 1.08 |
| 05-18 | 2,268 | 2,065 | 1.10 |

The dock's volume rose 22% through April and the model tracked the level but not
the rise — its `trend` block fits one slope per day type across all docks, and
this dock grew faster than the city. Then in May the dock fell back and the model
over-predicted. Every other dock is within ±15% of its own average; this one
reaches 2.16 in one direction and 0.74 in the other.

Dropping it as well takes departures to **1.9615**. Both numbers describe the
same model; which one is "the" accuracy depends on whether a 4x-median dock that
grew 22% in a month is the operation's problem or the model's. It is worth
flagging separately rather than averaging away.

### Direction is not the problem

On dock-days with more than half a bike of observed volume, the model gets the
sign of the net movement right **100% of the time**, both targets. The errors are
in magnitude, concentrated in the biggest docks, and mostly traceable to the two
causes above.

## What to read the numbers as

The cell-level deviance on this panel is **not** comparable to the numbers in the
README, and not because the model got worse. Deviance rises steeply with the
aggregation level: the same forecast scores about 0.2 on 30-minute cells, 4.6 on
cell-days, and 23.8 on city-days. `network_v1` is 30-minute cells over a wider
horizon and a different cell set, so its deviance answers a different question
from `horizon_am`'s.

What *is* comparable across the two configs is the shape of the error, and what is
comparable within one config is one entity against another.

The rollup adds a second layer of that trap: fitting and scoring per cell is not
the same as fitting and scoring per GP. A model that is good per cell can still
be biased at the GP level, because errors that cancel over 5,678 cells do not
cancel within one dock's 62. The measured numbers above are 2.77 at GP-day level
against 0.139 at cell level — a factor of twenty, from aggregation alone. That is
why `entities/*.csv.gz` exists: the per-entity error has to be looked at directly
instead of inferred from the cell score.
