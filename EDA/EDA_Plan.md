# EDA Plan — Free-Floating Bike-Sharing Trip Dataset

**Course:** Decision Analytics (Fall 2026) — Homework 3
**Deliverable this plan serves:** `hw_03_team <N>.pptx`
**Input:** `Dataset/dataset_train_20260921_lsz.csv` → cleaned by `EDA/01_data_cleaning.py`
**Boundary input:** `data_select_configure_zgc_20260909.xlsx` (repo root) → `EDA/region.py` → `EDA/02_region_map.py`, `EDA/03_region_flow.py`
**Working files:** `EDA/out/trips_clean.csv.gz`, `EDA/out/cell_summary.csv`, `EDA/out/cleaning_summary.md`, `EDA/out/cleaning_report.json`, `EDA/out/region_map.html`, `EDA/out/region_flow_summary.md` + `region_flow.json`
**Figures:** all sixteen live in `EDA/out/` as `figNN_<slug>.png`, numbered per §I1
**Date written:** 2026-10-08 · **Revised:** 2026-10-08 · **Deadline:** 13:30, 2026-10-12

> **Revision note.** This plan was cut down to what advances the argument. Item IDs are
> **preserved from the previous version** so that changes stay traceable — gaps in the
> numbering (B2, B3, B5, B6, C3, C4, C5, D4, D5, E3, E4, E6, F2, F3, F5, G2) are
> deliberate removals, not omissions. **H6 survives as a one-sentence stub** rather than a
> deleted item, because HW §2 requires an explicit outlier-handling statement. Nothing that
> feeds the two graded sections was dropped.
> **23 figures → 13 · 16 slides → 13.**
>
> **Added 2026-10-08: §E0 — the study area and its boundary exchange.** This is a *new*
> item, not a revival of a deleted one; **E0 was never used**, so no ID is recycled. It
> defines the service area the rest of §1 refers to, and it is the only analysis here that
> uses a boundary rather than the whole dataset. It rests on a boundary polygon supplied as
> `data_select_configure_zgc_20260909.xlsx` and on `03_region_flow.py`. **13 figures → 16 ·
> 13 slides → 15.**
>
> **Built 2026-10-08: every figure and table below now exists.** `04`–`09` were added to
> render §B–§H — sixteen figures and the nine tables of §I2, all from `out/trips_clean.csv.gz`
> so any number on a slide can be regenerated from the cleaned data alone. Output names
> follow §I1 (`figNN_<slug>.png`), including the three E0 figures, which were renamed from
> their earlier descriptive filenames. Where a build turned up a number that disagreed with
> this document, **it was the document that was wrong**; the corrections are marked *(corrected
> at build)* below and the reasons are in the summary `.md` files.
>
> Three of those corrections are worth knowing before reading further, because they change
> what a reader would infer: an L6 cell is **937 m × 607 m**, not 1.22 km × 0.61 km (the
> earlier figure dropped the `cos(lat)` term from the longitude conversion); daily volume is
> **~8,684 weekday / ~5,969 weekend trips a day**, not ~100k/day (that row had been read off
> the wrong scale); and the cleaned window holds **23** weekend days, not 24 — 2026-05-17 was
> a Sunday and §A5 removed it.

---

## Context

HW3 asks for two things: **(§1)** an exploratory analysis of the bike-trip data that identifies **spatiotemporal imbalance in demand and supply**, framed as the evidence base for a bike-repositioning problem; and **(§2)** at least two statistical/ML methods that **predict demand rates**, with explicit cross-validation design and outlier handling.

**Related work.** Luo et al. (2022), *Transportation Science* 56(4):799–826, studies dynamic intra-cell repositioning in free-floating bike-sharing systems. We use it **only for problem framing** — the notion of partitioning a service area into *cells*, the idea that a mover rebalances bikes between popular *gathering points*, and the observation that **spatiotemporal imbalance is the phenomenon of interest**. Its dataset (Beijing, Mobike 2017, seven weekdays, eleven hand-selected gathering points), its spatial unit, and its method (MDP + PFA + OCBA simulation) are **not** adopted here. Where this plan computes a statistic resembling one in that paper, it is our own operationalisation for a different dataset, to be labelled as such rather than compared numerically.

### What the data actually is

| | |
|---|---|
| Rows | 667,821 raw → **658,304 cleaned** (98.57% retained) |
| Columns | 7 raw: `action_dt, order_id, bike_id, start_time, end_time, start_geohash, end_geohash` |
| Period | 2026-03-02 (Mon) → 2026-05-24 (Sun) = **84 days = 12 complete weeks**, 0 missing; **83 days after cleaning** |
| **Study area** | the polygon in `data_select_configure_zgc_20260909.xlsx` — 中关村, **6.47 km²**, 北四环 along its whole northern edge and 北三环 along its southern one, with diagonal cuts at the SW and NE corners. **99.64% of trips touch it** (§E0) |
| Spatial unit | 8-char geohash ≈ **38 m × 19 m** (street-level); 32,664 distinct origins |
| Area | 99.96% of trips fall in two 4-char prefixes, `wx4e` and `wx4g` (Beijing) |
| **Datum** | **GCJ-02 throughout** — both the boundary and the trip coordinates. Not a guess: §E0 states the two measurements that settle it. A WGS-84 reading shifts everything ~500 m |
| Trip volume | **8,684/day** on weekdays · **5,969/day** at weekends; daily mean 7,931, **sample variance 4.22×10⁶** *(corrected at build: this row previously read ≈99k–110k/day, an order of magnitude too high — §B1's table is the authority)* |
| Duration | median 7 min, p95 35, p99 71, max 180; mean 11.1, **std 12.9 — CV = 1.15** *(corrected at build)* |
| Distance | crow-fly median 0.71 km, p95 2.98 km, p99 6.02 km *(corrected at build)* |
| Implied speed | median 7.0 km/h, p95 11.4 km/h |

### Four facts that should shape every decision below

1. **This is a service-area dataset, not a city dataset.** 99.64% of trips have at least one endpoint inside the 6.47 km² study area, and the 0.36% that do not sit within 60 m of its edge — the extract is *"trips touching the region"*, most likely keyed on 8-char geohash cells. The good news is that inflow, outflow and internal trips are all **complete**; the bad news is that **nothing here supports a statement about trips that never touch the area**, including any city-wide comparison. §E0 is built on this.
2. **The data is extraordinarily spatially concentrated.** 11 of 563 six-character cells account for **80% of all trips**. A "uniform grid of cells" mental model is wrong here — this is a handful of hotspots plus a long thin tail.
3. **The panel is extremely sparse.** At the 6-char level (563 cells × 84 days × 24 hours = 1,135,008 cell-hours), only **5.72%** of cell-hours contain any trip at all.
4. **Realized trips ≠ demand.** A trip can only be observed if a bike was available. The assignment explicitly flags this as a limitation. §G treats it as a named limitation rather than a footnote.

---

## 0. Conventions and definitions

Fix these before any analysis so the deck is internally consistent.

| Term | Definition used here | Column |
|---|---|---|
| **Cell** | A spatial unit = truncated geohash. Primary level **L6**. | `start_cell6` / `end_cell6` |
| **Departure** | A trip *starting* in a cell = an observed supply event (a bike leaves). | `start_cell6` |
| **Arrival / return** | A trip *ending* in a cell = an observed supply replenishment. | `end_cell6` |
| **Net flow** | `departures − arrivals` in a cell over a period. Positive = the cell drains bikes (deficit risk); negative = the cell accumulates bikes (surplus risk). | computed |
| **Study area** | The polygon in `data_select_configure_zgc_20260909.xlsx`, used exactly as given (GCJ-02). 6.47 km², 88 ring vertices. Fixed once and reused by §E0 and `region.py`. | `region.py` |
| **Internal trip** | Both endpoints inside the study area. Never leaves the service area. | computed |
| **Outflow** | Starts inside, ends outside — a bike leaves the service area. | computed |
| **Inflow** | Starts outside, ends inside — a bike arrives. | computed |
| **External trip** | Neither endpoint inside. 0.36% of rows; a boundary artefact of the extract, excluded from every flow statistic. | computed |
| **Net exchange** | `inflow − outflow` for the study area as a whole. Registered at the *region* level, in contrast to **net flow**, which is per cell. Positive = the area accumulates bikes. | computed |
| **Demand rate** | *Latent* rate of users wanting to start a trip. **Not observable** — see §G. | — |
| **Realized departure rate** | Observable count of departures per cell per period. The practical proxy. | computed |
| **Time bin** | Primary = 1 hour (`slot_60`); 15 min (`slot_15`) available. | `slot_60` |

**Why L6.** An L6 geohash tile is **937 m × 607 m** (0.0110° lon × 0.0055° lat — the earlier draft of this plan quoted 1.22 km × 0.61 km, which used 111,320 m/deg for *longitude* without the `cos(39.97°)` factor; the arithmetic is in `cells.py`), the closest available approximation of a realistic bike-share operating unit — so **L6 is fixed as the analysis cell for everything downstream**, and no cell-size sensitivity study is run. (L8, the native resolution, yields 32,664 distinct origins with a median of 2 trips each — far too sparse to estimate anything; L5 collapses the area into a handful of cells, too coarse to see imbalance *between* neighbourhoods. Report this as a one-line justification, not an analysis.)

---

## A. Data quality

*Purpose: satisfy HW §2's "report how you handle the outliers" requirement. All numbers come from `01_data_cleaning.py`; `cleaning_summary.md` is the one-page rendering and is the artifact to lift onto the slide.*

### A1 — Structural completeness table
**What.** Null counts, duplicate key counts, timestamp parse failures, malformed geohashes, ragged rows, encoding.
**How.** Already computed — `cleaning_summary.md`, or `cleaning_report.json → structural_checks`.
**Present.** One table with a verdict column.
**Finding.** Every structural check returns **zero**: 0 nulls, 0 duplicate `order_id`s (a true primary key), 0 unparseable timestamps, 0 invalid geohashes, 0 ragged rows. **The cleaning task here is anomaly triage, not repair** — say this explicitly, it buys credibility for the more interpretive claims later.

### A3 — Anomaly taxonomy table
**What.** What was flagged, how much of it, and what each rule costs in sample size.
**How.** `cleaning_summary.md → Flag audit` (already rendered as a table with tier, flag, rows, %, disposition).
**Present.** This table, unchanged, is *the* outlier-handling slide — it shows `rows_in / rows_out / pct` per rule.

| Tier | Flag | Rows | % | Disposition |
|---|---|---:|---:|---|
| 0 | `qc_null_required`, `qc_dup_order`, `qc_unparsed`, `qc_bad_geohash` | 0 | 0.000 | dropped |
| 1 | `qc_neg_dur` | 0 | 0.000 | dropped |
| 1 | `qc_zero_dur` | 5,835 | 0.874 | dropped |
| 1 | `qc_speed_gt_30` | 329 | 0.049 | dropped |
| 1 | `qc_cross_region` | 953 | 0.143 | dropped |
| 2 | `qc_dur_gt_180` | 49 | 0.007 | dropped |
| 2 | `qc_anomaly_day` | 2,463 | 0.369 | dropped |
| — | `qc_dur_gt_60` | 11,116 | 1.665 | kept |
| — | `qc_same_geohash` | 4,736 | 0.709 | kept |
| | **dropped, unique** | **9,517** | **1.425** | |

Note the overlap: flag counts sum to more than 9,517 because 1,921 rows carry two or more flags. Report the *unique* figure, or a reviewer will catch the double count.

### A2 · A4 · A5 · A6 — Micro-diagnostics: **text only, no figures**

These four findings are worth one or two lines each on the quality slide. None of them justifies a dedicated chart, and none of them changes any downstream decision.

| Item | The line to put on the slide |
|---|---|
| **A2** `action_dt` | Redundant with the date part of `start_time` (0 mismatches in 667,821 rows) — dropped. |
| **A4** Zero-minute trips | **5,835 zero-minute trips (0.87%); 4,530 of them (77.6%) cross geohash cells**, which is impossible at 8-char (~38 m) precision — evidence the cause is minute-level timestamp rounding collapsing sub-minute rides, not user error. |
| **A5** Anomalous day | **2026-05-17 collapsed to 2,463 trips = 22.6% of its 10,876 local baseline**; removed in cleaning. |
| **A6** `bike_id` | 177,624 distinct bikes over 667,821 trips (median 3, max 12/day) — **`bike_id` is a session identifier, not a physical fleet ID**; no vehicle-level analysis is supported. |

A4 is the one worth dwelling on for a sentence: it converts "we deleted 5,835 weird rows" into "we identified a systematic measurement artefact and quantified its footprint." That is the difference between a cleaning slide that reads as arbitrary and one that reads as analysis.

**No figures.** No zero-duration diagnostic plot, no anomaly-day line chart.

---

## B. Univariate distributions

*HW §1 explicitly asks for **sample variances and sample quantiles**, not just point estimates. Report a quantile row alongside every distribution figure.*

### B1 — Trip profile: duration and distance (**absorbs B2**)
**What.** Shape, centre, spread, tails of `duration_min` and `crowfly_km` — the two variables that define what a "trip" is.
**How.** `describe()` plus `quantile([.01,.05,.25,.5,.75,.95,.99])` on each. Duration bins: 1-min up to 10, then 5-min to 60, then 30-min. Distance bins: 0.25 km.
**Present.** **One two-panel figure** — (a) duration histogram on a **log-scale x-axis**, (b) crow-fly distance histogram — with a shared quantile table beneath.
**Report.** Duration: median 7 min, p95 36, p99 71. **std ≈ 12.9 min against a mean of 11.1 gives a CV of 1.15 — the standard deviation exceeds the mean**, the signature of heavy right-skew and a warning against assuming normality on raw duration. Distance: median 0.72 km, p95 2.85 km, consistent with published bike-share profiles (majority of trips within 2–3 km).
**Why distance is kept at all.** It is a **validation** check on the pipeline: if distances looked wrong, something would be broken in the geohash decoding or unit conversion. Cheap insurance; fold it into the duration panel rather than giving it a slide.

### B4 — Spatial concentration of trips
**What.** How concentrated is activity across cells? The single most decision-relevant distributional fact in the dataset.
**How.** Rank L6 cells by departure count. Plot **cumulative share of trips vs cumulative share of cells** → a **Lorenz curve**; report the **Gini coefficient** and the counts-covering-80% figure directly.
**Present.** Lorenz curve with the 45° equality line and the Gini annotated, plus one compact line of text: **"563 L6 cells; 11 of them (2.0%) carry 80% of all trips; Gini = …"**
**Why this is the highest-value item in §B.** It justifies narrowing the analysis to a hotspot subset, explains why city-wide heat maps will look like a few dots, and is the empirical basis for the *gathering-point* concept — hotspots are exactly where bikes pile up and where repositioning effort should concentrate.

---

## C. Temporal patterns

### C1 — Temporal overview (**absorbs B3**)
**What.** Both time scales that matter: how volume moves across the 84 days, and how it moves across the 24 hours.
**How.** (a) `groupby("date").size()` with a 7-day centred rolling mean overlaid; (b) `groupby(["hour","is_weekend"]).size()`.
**Present.** **One two-panel figure.** (a) daily volume bars + rolling mean line, weekends marked in a separate colour. (b) hour-of-day, weekday solid and weekend dashed **on the same axes** — the point is the contrast, and overlay makes it immediate.
**Report.** (a) March runs lighter (~7,321/day) than April–May (~8,300/day) — a ramp-up, worth naming so nobody mistakes it for a demand trend the model must extrapolate. Report the sample variance of daily volume. (b) Double peak, roughly 08:00 and 18:00, with an overnight trough near 03:00; the morning peak flattens at weekends.
**Read it as.** The morning peak is a commute-driven *supply* event (bikes leave residential areas) and the evening peak is its mirror. That asymmetry is the engine of the imbalance in §E.

### C2 — Day-of-week × hour heat map
**What.** How the intraday shape differs by day of week.
**How.** Pivot `dow` (rows, 0=Mon) × `hour` (cols), values = trip count, **normalised per row** so each weekday's shape is comparable.
**Present.** Heat map, one colormap, annotated colourbar. **State which normalisation you used** — an unlabelled normalisation is the most common heat-map error.

---

## D. Spatial patterns

**Prerequisite.** Use `cell_summary.csv` — it already provides centroid lat/lon plus `n_departures`, `n_arrivals`, and `net_flow` per cell. Do not re-aggregate the trip file.

### D1 — Departure / arrival map
**What.** Where does activity physically happen?
**How.** Scatter `cell_summary` centroids, colour by `n_departures` / `n_arrivals`. **Log colour scale** — with 11 cells holding 80% of activity, a linear scale renders everything else invisible.
**Present.** Two panels side by side, **sharing one colour scale**. Do not let matplotlib autoscale each panel independently — that is the classic way to make two different maps look identical.

### D2 — Net-flow map
**What.** Where do bikes drain away, and where do they pile up?
**How.** `net_flow = n_departures − n_arrivals` per cell.
**Present.** **Diverging colour map centred at zero** (blue = accumulation, red = depletion) with **symmetric limits** — if the scale runs −500 to +1200 the reader cannot see which side dominates.
**Read it as.** This is the figure that *motivates the repositioning problem*: cells at the red extreme are where users will find no bike; blue cells are where bikes accumulate and parking pressure builds.

### D3 — Hotspot labelling — **downgraded**
**What.** Name the cells that matter, so "hotspots exist" becomes "here are the locations."
**How.** Take the top L6 cells by `n_departures` from `cell_summary.csv`.
**Present.** **Annotate them directly on the D1 and D2 maps with numbered markers** plus a compact **top-5** list in the caption. **No standalone 20-row table and no separate slide** — the information belongs on the map, not beside it.

---

## E. Spatiotemporal joint analysis — **the core of §1**

*This is where the assignment's actual question lives: "spatiotemporal imbalance … which will provide the basis for addressing the bike repositioning problem." Budget the most slides here. E3, E4, and E6 from the previous plan are removed — their content is covered by E1 or cost more effort than it returns.*

### E0 — The study area and its boundary exchange — **new, and it leads**

*Runs `03_region_flow.py` → `out/region_flow_summary.md`. This fixes the spatial scope every later item inherits, and it is the only analysis here that uses a boundary at all.*

**What.** Two steps. First **fix the service area**: the boundary polygon supplied as `data_select_configure_zgc_20260909.xlsx` — 中关村, 6.47 km², 88 ring vertices. Its northern edge sits on 北四环 (median 15 m from the OSM centreline, p90 63 m) and its southern edge on 北三环 (median 155 m), with steeper diagonal cuts closing the SW and NE corners that do **not** correspond to any major road — so the area is road-bounded on two sides and hand-drawn on the other two. Say that plainly; it is better than implying all four sides follow roads. Then **measure what crosses it**: classify every trip by which side of the ring each endpoint falls on — internal / outflow / inflow / external — and describe that exchange by hour, by day and by location.

**Why this is not D2 over again.** D2 computes `net_flow` per cell across the whole dataset: where bikes drain or pile up *inside* the grid. E0 asks the region-level question — **what does an operator have to move across the edge of the service area?** Different quantity, different decision: D2 motivates repositioning *within* the area, E0 sizes the *boundary* demand that a truck fleet is actually dispatched against.

**Two things had to be settled before any number below is trustworthy.**

*Which of the two scanline tables defines the ring.* The workbook holds two 0.001° descriptions of the same polygon and they **disagree on 163 of 885 cells (18.4%)**. The disagreement is structural, not noise: a per-latitude scanline stores one `[min_lon, max_lon]` per row, so it cannot express a notch in a horizontal edge — a row crossing such a notch reports the outer extremes and silently fills the gap. The per-longitude table's south edge rises from lat 39.963 at lon 116.309 to 39.968, which is 北三环 sloping the way the real 3rd Ring Road slopes; the per-latitude table instead claims full lon width at lat 39.963. **Take the per-longitude table** (6.47 km², against 7.90 km² for the other). Independent check: the dataset is an extract of trips touching the area, and the per-longitude ring reproduces that footprint far better — **0.363%** of trips fall outside it, against **1.091%** for the other. Report the 18.4% disagreement on the slide rather than hiding it; it is the honest answer to "why this ring and not the other".

*What datum the coordinates are in.* Amap tiles are **GCJ-02**; geohash decoding conventionally yields **WGS-84**, and in Beijing the two differ by ~500 m — larger than any feature here. Two independent measurements say **everything in this project is GCJ-02 and nothing should be converted**:

| Test | Result |
|---|---|
| 2-D translation fit of the ring onto the OSM (WGS-84) centrelines of 北三环 / 北四环 | best shift **Δlon −0.0056° (−490 m), Δlat −0.0019° (−210 m)** — the ring needs the GCJ-02 correction to reach the roads, so it already *is* GCJ-02 |
| Share of geohash cells landing on water pixels, over a 32,519-cell Amap mosaic (4.1% of which is water) — nobody starts a bike trip in 昆明湖 | **as given 0.30%** · +GCJ-02 0.98% · −GCJ-02 1.40%; a 2-D scan over ±300 m lat / ±800 m lon puts the global minimum **exactly at (0, 0)** |
| Ring corners vs the OSM road-intersection polygon | **+0.005° lon, +0.002° lat** at all four |

**This is almost certainly why the earlier attempts to define the region failed** — assuming WGS-84 and converting puts everything ~540 m off. `02_region_map.py` draws both readings on an Amap basemap so the discrepancy is visible rather than asserted.

**How.** Each endpoint is tested inside/outside by even-odd ray casting against the ring (`region.py`). The vectorised test was verified point-for-point against `matplotlib.path` on all 658,304 trips — zero disagreements. Nothing is converted; all coordinates are GCJ-02.

**Present.** **Fig #7** — the study area drawn on the Amap basemap with the geohash grid under it and the ring on top (`out/fig07_study_area.png`) — this is the slide-8 image that shows *where* the area is. **Fig #8** — 2 panels: (a) crossings by hour, weekday and weekend, inflow vs outflow; (b) daily net exchange with a 7-day mean. **Fig #9** — the ring as ~300 m arcs coloured by crossings per km, one panel per direction, **shared colour scale**. Tables: classification counts · hourly × weekday/weekend · weekly net · top-8 gateways · top-8 boundary arcs · side totals.

**Findings.**

| Class | Trips | Share |
|---|---:|---:|
| internal | 466,075 | 70.80% |
| outflow | 96,340 | 14.63% |
| inflow | 93,497 | 14.20% |
| external | 2,392 | 0.36% |

- **The area is a net source of bikes: −2,843 over 83 days (−34.3/day).** Not a few odd days — the daily net gives **t = −5.00, p ≈ 3×10⁻⁶**, and **12 of 12 weeks are negative**. Left alone the area loses ~237 bikes a week and would drain.
- **The exchange runs on a commute clock.** Both directions peak in the evening (inflow 17:00 at ~129 trips/h, outflow 18:00 at ~141), but the *balance* flips sign: on an average weekday the area **gains 92 bikes between 06:00 and 12:00** and **loses 122 between 17:00 and 23:00**. That is the signature of a workplace — bikes arrive with the morning commute and drain away in the evening — and the evening drain slightly outruns the morning fill. **That gap, not either flow alone, is the daily rebalancing requirement.** *(Corrected at build: these were previously quoted as window totals — 5,516 and 7,325 — which is the same fact at 60× the scale, but it made fig #8's hourly panel look like it disagreed with its own daily panel. Everything in §E0 is now per day.)*
- **Weekends are nearly balanced** (net **+13/weekend day** against **−52/weekday**; +300 across all 23, at roughly half the weekday volume), so the imbalance is a working-day phenomenon rather than a constant leak.
- **Boundary trips are long.** Median crow-fly 1.42 km (inflow), 1.50 km (outflow), against 0.60 km for internal trips — genuine cross-district travel, not edge-hopping along the perimeter.
- **The perimeter is not uniformly leaky.** The northern edge — 北四环, where the area meets 北京大学 / 清华 / 五道口 — carries **40.0% of all crossings on 27.7% of the boundary** (1.45× over-represented), and the single busiest ~300 m arc (北四环 west of 中关村大街) takes **8.9% of the entire exchange on its own**. Fixed capacity belongs on a handful of arcs, not spread around the ring.

**Read it as.** E0 is the coarsest and most directly actionable statement in §1: the service area leaks bikes at one edge, on a clock, through a few gates. E1 then shows where the same imbalance bites at cell resolution *inside* the area.

**Two caveats to state, not bury.**
1. **Inflow and outflow are complete; the exterior is not.** Since the extract is "trips touching the area", the outside endpoints are a census of *crossings*, not a sample of the surrounding city. Nothing here generalises to trips that never touch the area — including any city-wide comparison.
2. **Crossing locations are inferred, not observed.** The data gives two geohash cells, not a path, so each crossing is computed from the straight line between cell centroids. That quantises position to the cell grid — on the busiest ring edge the 7,287 crossings fall on only 808 distinct longitudes — which is why the boundary is reported in ~300 m arcs rather than at the native 85 m edge. The **gateway table needs no such assumption** (it reads the outside endpoint straight off the data) and is the more robust of the two views.

### E1 — Cell × hour net-flow heat map
**What.** The central figure of the whole analysis.
**How.** Restrict to the top ~30 L6 cells by volume. Pivot: rows = cell (ordered by total volume), columns = hour 0–23, values = `departures − arrivals`, **averaged across the 84 days, weekdays only**, normalised per row.
**Present.** Heat map, diverging colormap centred at zero, rows labelled by their D3 hotspot number.
**Reading it.** Red bands = cells that drain at that hour (users will find no bike); blue bands = cells that fill. Expect a red band on residential cells at 08:00 with a matching blue band at 18:00, and the reverse on office cells — **a mirror-image pair**. Annotate an arrow linking the two bands: that single annotation is the clearest statement of the imbalance problem the deck can make.
**Why this subsumes E4 (cell typology).** Cells whose row is red at 08:00 and blue at 18:00 *are* the source/sink classification — read straight off the heat map. A separate typology table adds a slide but no information.

**As built.** The predicted pattern is there, and it is led by the two busiest cells in the dataset, which are **adjacent**: `wx4eqt` drains **+86 trips/day** in the morning window and fills **−101** in the evening, while `wx4eqw` does the exact reverse (**−160 / +133**). Those are the two largest opposing swings in the panel and they sit side by side — the imbalance is not "too few bikes in 中关村", it is bikes on the wrong side of a cell boundary at the wrong hour. **One change from this item as written:** the plan asks for net flow "normalised per row", which is not defined for a signed quantity (a row sum of net flow is ≈0). The figure plots `(departures − arrivals) / (departures + arrivals)` instead — bounded in [−1, +1], literally readable, and comparable across cells of different size. Raw net flow is in `out/table_e1_cell_hour_netflow.csv`. Cell-hours under 2 trips/day of gross flow are **greyed rather than coloured**, because at that volume the sign is a coin flip.

### E2 — Net-flow maps at two anchor hours — **downgraded from four**
**What.** The same imbalance, geographically, at the two hours that dominate.
**How.** For hours **08:00 and 18:00 only**, aggregate net flow per L6 cell across all days.
**Present.** **Two panels side by side, sharing one diverging colour scale** (shared scale, or the panels are not comparable). Noon and 22:00 are dropped — they add axis area without adding signal.
**Why it still earns its place.** E1 says *which* cells and *when*; E2 says *where they are*. Two panels is enough for that.

### E5 — Morning/evening role reversal test
**What.** Do any cells *swap roles* between morning and evening?
**How.** For each cell, sign of mean net flow in 07:00–09:00 vs 17:00–19:00. Cross-tabulate same-role vs reversed-role.
**Present.** A scatter: x = morning net flow, y = evening net flow, quadrants labelled. Points in the top-left and bottom-right quadrants are the reversers.
**Why.** Reversing cells are the strongest justification for *dynamic* rather than static repositioning — a fixed rule would service them wrongly for half the day. It is a cheap scatter and a genuinely non-obvious finding if it appears.

**As built — it appears, and strongly.** **14 of the top 30 cells (47%) swap roles between the morning and evening windows**, counted only where both windows exceed ±1 trip/day so a sign is not being read off noise (a further 9 cells are one-sided and 5 flat, and none of those is called a reversal). The scatter uses **symlog** axes: the two cells that swing ±100 trips/day would otherwise collapse the other 28 onto the origin and hide the finding the figure exists for.

---

## F. Uncertainty: variance and quantiles

*HW §1 requires this explicitly: "Beyond point estimates, you may also want to report sample variances and sample quantiles." F2 (fan charts), F3 (variance decomposition), and F5 (bootstrap) are removed — the table below carries the graded content at a fraction of the effort.*

### F1 — Variance and quantile table for demand rates (**subsumes F2**)
**What.** For each hotspot (cell, hour) pair: how variable is the departure count across the 84 days, and what does its distribution look like?
**How.** Group by `(start_cell6, hour)` → per-day counts → **mean, variance, std, CV, p50, p90** across days. Restrict to cells with a non-trivial mean count.
**Present.** **One table** for the top ~10 cells and their peak hours:

| Cell | Hour | Mean | **Variance** | Std | CV | **p50** | **p90** |
|---|---|---:|---:|---:|---:|---:|---:|
| *(top hotspots, peak hour each)* | | | | | | | |

**Why the quantile columns matter.** This is a deliberate substitution for the deleted fan chart. The assignment asks for **sample variances and sample quantiles** by name; putting `variance`, `p50` and `p90` as columns in the same table satisfies that requirement in one cell rather than one figure. The p90 column is also the operationally relevant number: it is the inventory a cell needs to avoid stockouts on a busy day, whereas the mean would be wrong half the time.
**Feeds.** §H3's split design and §H5's accuracy expectation.

### F4 — Dispersion check
**What.** Is the variance-to-mean ratio of hourly counts ≈ 1 (Poisson) or > 1 (over-dispersed)?
**How.** For each (cell, hour), compute `variance / mean` across days. Plot the **mean vs variance relationship on log-log axes** across all (cell, hour) pairs; draw the `variance = mean` reference line.
**Present.** One log-log scatter with the reference line. Report the share of pairs above it.
**Why.** This is the plan's **model-assumption test**. Deviations above the line mean a Poisson GLM in §H would understate uncertainty and over-fit the zeros — the fix is a **negative binomial** GLM or a quasi-Poisson correction. Finding this *before* modelling is far cheaper than discovering it in the residuals. It also replaces the deleted inter-arrival-time test (C5) at a fraction of the cost: the variance/mean ratio establishes non-Poisson behaviour directly, without fitting an exponential survival curve.

**As built — the assumption fails, and it decides §H4.** Across 572 (cell, hour) pairs with a mean of at least 0.5 trips/day, **87.9% sit above the `variance = mean` line**, the median variance is **1.46×** the mean, and the log-log fit gives **var ∝ mean^1.39** where Poisson requires 1.00. Two robustness checks are reported rather than one, because the panel mixes weekdays with weekends and that mixture is itself a source of variance: weekdays only gives **1.33** — the honest floor for what a model still faces after its day-of-week term is in — and widening to all 563 cells gives **1.36**, so the hotspot subset is not a special case. This is the number §H4 uses to justify fitting a **negative binomial** GLM; the Poisson is fitted alongside it as the evidence, not as a candidate.

---

## G. Censoring: realized trips ≠ demand

*The assignment raises this explicitly: "the realized bike trips do not equal the actual demand, as the data can be censored due to insufficient bike inventory." This is a **writing** task, not an analysis task — no figures, no proxy mining.*

### G1 — State the mechanism
**What.** A user who wants a bike but finds none produces **no row** in this dataset. Observed departures are therefore a **lower bound** on latent demand, and the gap is widest exactly where the §E imbalance is worst — **the cells that most need repositioning are the ones whose demand is most under-recorded.**
**Present.** Two sentences and one simple diagram: latent demand → [inventory availability] → observed departures, with the unmet branch labelled unobservable.
**Note.** We deliberately do **not** attempt to detect stockouts empirically from delayed first-departures or rate discontinuities: those signals are confounded with genuine low-demand periods, and mining them would consume disproportionate effort for an inconclusive result. State the mechanism and move on.

### G3 — State the consequence for §2
**What.** Say explicitly that §H's models predict **realized departure rates**, not latent demand, and that true demand is expected to be higher — especially at peak hours in hotspot cells. Frame the predictions as a **floor**.
**Why.** This is the assignment's stated expectation, and it names the one limitation that no amount of modelling can remove. Engaging with it in two sentences is enough to collect the credit.

---

## H. Feature engineering and modelling design for §2 (demand-rate prediction)

*HW §2 requires: **at least two** statistical/ML methods; explicit **model, assumptions, parameters, algorithms, software**; a stated **train/validation split** with care about cross-validation; explicit **outlier handling**; and **visual illustrations** of performance.*

### H1 — Define the target
**What.** What exactly is being predicted? Fix this before anything else — it determines the whole feature set.
**Recommended: hourly departure count per (L6 cell, hour, day)** — `count(departures)`. It matches the operational decision (how many bikes does this cell need this hour?), integer counts suit count models, and sparsity is manageable at hotspots.
**Restrict to the top ~30–50 hotspot L6 cells** (the cells behind B4's 80% figure). Predicting rates for cells with a median of 2 trips is not a meaningful task and would dominate any aggregate error metric with noise. **State the restriction explicitly** — silently dropping 95% of cells would be a serious methodological omission.
**Rejected alternatives.** 15-min bins (at L6 only 5.7% of *hourly* slots are non-empty, so 15-min slots would be ~1.4% non-empty — far too sparse); city-wide aggregate (loses all spatial structure, defeating the purpose).

### H2 — Candidate features
**What.** What could legitimately predict an hour's demand in a cell?

| Family | Features | Notes |
|---|---|---|
| **Temporal** | `hour`, `dow`, `is_weekend`, hour×dow interaction | The dominant signal, per §C1 and §E1's structure |
| **Lagged demand** | same cell previous hour / same hour previous day / same hour same weekday last week | **Highest-value, highest-leakage-risk** — see H3 |
| **Spatial** | cell identity; cell's total historical volume; neighbour-cell demand | Cell fixed effects capture static popularity (B4) |
| **Calendar** | day-of-month, is-holiday flag | Needs an external holiday calendar — the dataset has raw dates only |
| **Weather** | temperature, precipitation | **Not in the dataset.** Must be sourced externally or declared out of scope. Do not quietly omit it — its absence bounds achievable accuracy |

**State clearly what is missing.** A reviewer will ask about weather and holidays. "Not available in the provided data, and here is how that limits us" is a complete answer; silence is not.

### H3 — Train / validation split — **the highest-risk design decision**
**What.** HW §2 says "be careful when performing cross-validation, and report how you divide your data."
**How.** The 84 days form a time series. A **random row-wise split leaks information catastrophically**: the model would see Tuesday 14:00 of week 8 in training and predict Tuesday 14:00 of week 6 in validation, with the lag features carrying the answer across.
**Recommended protocol — walk-forward (rolling-origin) validation:** split **by day**, never by row; expanding window (train on days 1…k, validate on day k+1, roll forward) or blocked 4-fold, always training on the past and validating on the future. Compute lag features **strictly from data preceding the validation day**. *(As built: four blocks of **19 days** — the 168-hour lag consumes the first week, so the panel runs 76 usable days — with blocks 2–4 as the validation folds and the model refitted per fold.)*
**Present.** A diagram of the split scheme plus the fold sizes. Explicitly state that random splitting was rejected and why. **This slide is worth more than most** — the assignment names cross-validation specifically, and demonstrating that you understood the leakage risk is a direct signal of competence.

### H4 — At least two modelling methods
**What.** HW requires ≥2, and asks for explicit model/assumptions/parameters/software.

| # | Method | Assumptions | Why include it |
|---|---|---|---|
| **1** | **Poisson (or negative binomial) GLM**, log link, cell + hour + dow fixed effects | Counts are Poisson — or NB if F4 shows over-dispersion; log-linear rate structure | Interpretable coefficients; standard baseline; the NB variant answers F4 directly |
| **2** | **Gradient-boosted trees with a Poisson objective** (LightGBM / XGBoost) | No functional form; captures interactions automatically | Strong non-linear benchmark; handles hour×dow×cell interactions a GLM needs hand-specified |
| **3** *(optional)* | **Seasonal naïve baseline**: cell's mean count for the same hour and weekday over prior weeks | Persistence | Sets the floor. If the models cannot beat it, say so — an honest negative result is more credible than an inflated one |

**Recommendation: 1 + 2 as the required pair**, with 3 as the honest baseline — three methods, comfortably above the minimum, without spreading thin.
**Report for each:** the model equation, the assumption on the mean–variance relationship, the link function, regularisation and hyperparameters with how they were selected, and the library + version. HW asks for all of this explicitly.

### H5 — Evaluation metrics and plots
**Metrics.** MAE and RMSE (raw, interpretable); **Poisson deviance** (proper for count data — RMSE on counts is easily dominated by high-volume cells); a **calibration check** (do predicted counts match observed counts when aggregated?).
**Plots:**
- **Predicted vs actual scatter**, pooled across cells, with the 45° line. Log-log axes if the volume range is wide.
- **Residual map** — mean residual per L6 cell on a map with a **diverging scale centred at zero**. This surfaces *spatial* bias (e.g. under-prediction across one whole district) that no aggregate metric reveals.
- **Error by hour** — boxplot of absolute error per hour. Expect peak hours to be hardest; if they are not, be suspicious.
**Why the residual map matters most.** It connects §2 back to §1: if prediction errors are spatially structured, they will cluster in the same cells §E identified as imbalanced — meaning the model fails exactly where the repositioning decision is hardest.

**As built — three models, walk-forward over three validation folds** (block 1 is train-only; ~19 days each), pooled:

| model | MAE | RMSE | Poisson deviance | vs baseline | calibration Σpred/Σobs |
|---|---:|---:|---:|---:|---:|
| **Gradient-boosted trees (Poisson)** | **2.92** | 9.06 | **1.843** | **−49%** | 0.992 |
| Seasonal naïve (baseline) | 3.15 | 8.59 | 3.642 | — | 0.939 |
| Negative binomial GLM | 3.86 | 16.33 | 2.642 | −27% | 1.031 |

Two results need naming rather than presenting.

**The baseline beats both models on RMSE and loses badly on deviance.** RMSE on counts is dominated by the highest-volume cells, and the naïve baseline's predictions never leave the historical range; deviance is the proper scoring rule for counts and it ranks the models correctly. This is exactly why H5 specifies deviance.

**The NB GLM's RMSE is 4.2× its MAE while the boosting model's is 3.1×.** A gap that wide means a handful of enormous errors: the GLM's largest single prediction is **1,017 departures in one cell-hour, against a maximum of 476 ever observed**. The mechanism is the log link — when `lag1` and `lag168` are both at a peak the coefficients compound instead of averaging, and the linear predictor extrapolates past the training data. It is a known weakness of log-link GLMs on heavy-tailed lagged regressors. The honest framing for the deck: **the GLM is the interpretable model, the boosting model is the one to dispatch against**, and reporting both with the gap named is more useful than reporting whichever looks better.

### H6 — Outlier handling: **one sentence, no re-run**
Report that all modelling uses `trips_clean.csv.gz` (all tiers dropped, 98.57% retained) and reference the §A3 taxonomy for the filter criteria. **No robustness re-run across datasets** — the cleaning slide already states the criteria and the 1.4% footprint, and duplicating every model on a second dataset buys a slide the grading does not ask for.

---

## I. Deliverables

### I1 — Chart inventory (**16 figures**)

| # | Figure | Section | Kind |
|---|---|---|---|
| 1 | Trip profile: duration (log x) + crow-fly distance | B1 | 2-panel histogram |
| 2 | Lorenz curve + Gini (cell concentration) | B4 | line |
| 3 | Temporal overview: (a) daily volume + rolling mean, (b) hour-of-day weekday vs weekend | C1 | 2-panel line |
| 4 | Day-of-week × hour heat map | C2 | heat map |
| 5 | Departure / arrival maps (shared scale, hotspots annotated) | D1, D3 | map pair |
| 6 | Net-flow map (diverging, centred at 0, symmetric limits) | D2, D3 | map |
| 7 | **The study area on the Amap basemap (grid + ring)** | **E0** | map — *built* |
| 8 | **Boundary exchange by hour + daily net exchange** | **E0** | 2-panel line — *built* |
| 9 | **Crossings per km along the ring, inflow vs outflow (shared scale)** | **E0** | map pair — *built* |
| 10 | **Cell × hour net-flow heat map** | E1 | heat map — **lead figure** |
| 11 | Net-flow maps at 08:00 and 18:00 (shared scale) | E2 | map pair |
| 12 | Morning vs evening role reversal scatter | E5 | scatter |
| 13 | Mean-vs-variance dispersion (log-log + reference line) | F4 | scatter |
| 14 | Predicted vs actual scatter | H5 | scatter |
| 15 | Residual map (diverging) | H5 | map |
| 16 | Error by hour | H5 | boxplot |

> **All sixteen are built**, and every filename in `out/` is `figNN_<slug>.png` for that number — the E0 three were renamed from their earlier descriptive filenames (`region_boundary.png` → `fig07_study_area.png`, `region_flow_hourly.png` → `fig08_boundary_exchange.png`, `region_flow_map.png` → `fig09_boundary_crossings.png`) so a filename maps onto a slide without a lookup table. Inserting the three E0 figures shifted the old #7–#13 to #10–#16; nothing was dropped. #5, #6, #7, #9, #11 and #15 are rendered on Amap tiles and carry the attribution requirement.
>
> **Kinds, as built.** #1 duration+distance **+ a quantile table rendered inside the image** (slide 4 then needs one file, not a chart and a table that drift apart); #2 Lorenz **+ a bar panel naming the top 12 cells**, because at Gini 0.969 the Lorenz curve alone degenerates into an L-shape and is unreadable; #5, #6, #10, #11, #15 draw L6 cells as the **rectangles they are** rather than centroid dots, and encode extract-completeness with a hatch (§K0); #12 uses **symlog** axes so the two cells that swing ±100 trips/day do not collapse the other 28 onto the origin. Each deviation is argued in the relevant script's docstring.

**Removed from the previous plan:** zero-duration diagnostic (A4), anomaly-day line chart (A5), distance histogram as a standalone (B2, folded into #1), trips-per-bike (B5), same-cell (B6), duration-by-hour boxplots (C4), inter-arrival survival (C5, replaced by #10), cell-size sensitivity (D5), per-cell net-flow small multiples (E3), OD chord diagram (E6), quantile fan charts (F2, folded into the F1 table), variance-decomposition bar (F3), the standalone over-dispersion histogram (the F4 *section* survives as #10; only its second, redundant panel was dropped), censoring proxy panels (G2).

### I2 — Table inventory (9)

| # | Table | Source file (as built) |
|---|---|---|
| 1 | Dataset overview (§0) | `EDA_Plan.md` §0, plus the corrected volume row |
| 2 | Data quality: structural checks + anomaly taxonomy (A1, A3; A2/A4/A5/A6 as text rows) | `out/cleaning_summary.md` |
| 3 | Trip profile: sample variance + quantiles (B1) | `out/table_b1_trip_profile.csv` — **also rendered inside fig #1** |
| 4 | Cell concentration (B4) | `out/table_b4_cell_concentration.csv` |
| 5 | Top-5 hotspots (D3, in the map caption) | `out/table_d3_hotspots.csv` |
| 6 | Region classification + hourly/weekly exchange + top gateways and boundary arcs (E0) | `out/region_flow_summary.md` |
| 7 | Datum evidence (E0 — the 3-row table that justifies not converting) | `EDA_Plan.md` §E0, evidence in `out/datum_evidence_osm_roads.json` |
| 8 | Variance & quantiles for the top cells' peak hours (F1) | `out/table_f1_variance_quantiles.csv` |
| 9 | Model comparison, split scheme, per-fold metrics (H3, H5) | `out/table_h3_folds.csv`, `out/table_h5_metrics.csv` |

Supporting tables written for auditability rather than for the deck — each is the evidence behind a figure, and each is regenerable from `trips_clean.csv.gz`: `table_c1_daily.csv`, `table_c1_hour_profile.csv`, `table_c2_dow_hour_{share,mean}.csv`, `cell6_summary.csv`, `table_d2_netflow_cells.csv`, `table_e1_cell_hour_{balance,netflow}.csv`, `table_e5_cell_roles.csv`, `table_f4_dispersion_pairs.csv`, `table_h1_panel.csv.gz`, `table_h5_metrics_by_fold.csv`.

### I3 — Slide sequence (**15 slides**)

| # | Slide | Source |
|---|---|---|
| 1 | Title, team, contribution statement | — |
| 2 | Data overview: what it is, coverage, units, why L6, **and that it is a service-area extract** | §0 |
| 3 | **Data quality: one table + four one-line diagnostics** | A1, A3, A2/A4/A5/A6 |
| 4 | Trip profile (duration + distance, with quantiles & variance) | B1 |
| 5 | **Spatial concentration: 11 cells = 80% of trips** | B4 |
| 6 | Temporal rhythm: daily trend + hour-of-day + dow×hour | C1, C2 |
| 7 | Spatial distribution: departure/arrival maps + net-flow map | D1, D2, D3 |
| 8 | **The service area: how it was defined, and the net −2,843** | E0 |
| 9 | **Boundary exchange: commute clock + where the perimeter leaks** | E0 |
| 10 | **Cell × hour net-flow heat map** | E1 |
| 11 | Net-flow maps at 08:00 / 18:00 + role reversal | E2, E5 |
| 12 | Uncertainty: variance & quantile table | F1, F4 |
| 13 | **Censoring: realized trips ≠ demand** | G1, G3 |
| 14 | Demand prediction: target, features, split design | H1–H3 |
| 15 | Models, results, residual diagnostics | H4, H5 |

> §E0 earns two slides: one to establish the area and the headline imbalance, one for the temporal and spatial structure of the exchange. If the deck must shrink again, cut from §B/§C (slides 4–6), not from the E0 pair — those carry the argument's most directly actionable claim. **Do not compress §H (slides 14–15): HW §2 is separately graded and needs the room.**

**Contribution statement** is a stated submission requirement — assign owners per section now, not on the last day.

---

## J. Execution order and dependencies

```
01_data_cleaning.py                        [DONE — out/ populated]
        │
        ├─ A1, A3   quality tables          (rendered already — no computation)
        ├─ A2/A4/A5/A6  four text lines     (already in cleaning_summary.md)
        │
        ├─ B1, B4   univariate       ──┐
        ├─ C1, C2   temporal         ──┼─→ independent of each other; parallelisable
        ├─ D1, D2, D3  spatial       ──┘
        │
        ├─ E0      region + boundary  [DONE — 02_region_map.py, 03_region_flow.py]
        │           ← region.py is the single source of truth for the ring
        │
        ├─ E1, E2, E5  spatiotemporal  ← depends on B4 (hotspot list) and §0 (cell = L6)
        │
        ├─ F1, F4   uncertainty        ← depends on E1's (cell, hour) panel
        │
        ├─ G1, G3   censoring          ← writing only; depends on E1 for the peak-hour pattern
        │
        └─ H1–H6   prediction          ← depends on B4 (cell restriction), F4 (Poisson vs NB), §G
```

**Critical path:** cleaning → hotspot list (B4) → spatiotemporal panel (E1) → F1/F4 → §H. Sections B/C/D split across team members immediately; **E blocks everything downstream, so start E early.** E0 is already computed and is independent of B4, so it can go on slides before E1 exists.

**Scripts, in order.** Shared modules first, then one script per section of the plan:

```
region.py          the ring -- single source of truth (§E0)
cells.py           the L6 grid -- geometry, bounds, per-cell aggregation
viz.py             palette, rc, figure numbering (figNN_<slug>.png)
maps.py            cell rectangles, the ring, hotspot markers, the hatch encoding
basemap.py         Amap tiles as a matplotlib background (cached in .tilecache/)

01_data_cleaning.py    §A            -> trips_clean.csv.gz, cell_summary.csv, cleaning_summary.md
02_region_map.py       §E0 (fig 7)   -> region_map.html, fig07_study_area.png   [--png]
03_region_flow.py      §E0 (fig 8-9) -> region_flow_summary.md, fig08, fig09
04_distributions.py    §B  (fig 1-2) -> distributions_summary.md
05_temporal.py         §C  (fig 3-4) -> temporal_summary.md
06_spatial.py          §D  (fig 5-6) -> spatial_summary.md, cell6_summary.csv
07_spatiotemporal.py   §E  (fig 10-12) -> spatiotemporal_summary.md
08_uncertainty.py      §F  (fig 13)  -> uncertainty_summary.md
09_demand_model.py     §H  (fig 14-16) -> model_summary.md
```

`00_build_region.py` and `02_grid_map.py` — which built the region out of OSM road intersections — were superseded by the supplied polygon and have been **deleted**, along with their outputs. `out/datum_evidence_osm_roads.json` is the one deliberate survivor: it is the raw evidence for the datum call and nothing in the pipeline reads it.

**Two rules the set follows.** Every script reads `out/trips_clean.csv.gz` and nothing else — no script depends on another's intermediate output except `cell6_summary.csv`, which `06` writes and `07`/`08`/`09` read for the cell list and geometry. And every script writes both a figure and a `*_summary.md` next to it, so the prose on a slide can be traced to the code that produced the number in it.

---

## K. Known limitations to state in the deck

0. **The dataset is a service-area extract, not a city sample.** 99.64% of trips touch the 6.47 km² study area; the 0.36% that do not lie within 60 m of its edge. Inflow, outflow and internal trips are therefore complete — but the *exterior* is a census of crossings, not a sample of the surrounding city, so **no claim here generalises to trips that never touch the area**. This also means every "city-wide" statistic in §B–§E (concentration, sparsity, peaks) is really a statement about *this service area*, and the deck should say so. In one respect this is a gift: it makes the dataset exactly the object the repositioning problem needs — one operator's service area.
1. **Censoring.** Observed trips are a lower bound on demand (§G). Predictions are of realized departures, not latent demand.
2. **No weather or holiday data.** Both are known drivers of bike-share demand; neither is present. This bounds achievable prediction accuracy and should be stated rather than left implicit.
3. **`bike_id` is not a physical fleet identifier** (A6), so no vehicle-level analysis is possible.
4. **Spatial unit is a geohash grid, not operational cells.** Real operators use irregular cells respecting roads and administrative boundaries. The L6 grid is an approximation; edge effects at cell boundaries are unmodelled.
5. **No inventory data.** The dataset records trips only. Idle bikes are invisible, so cell-level *inventory* — as opposed to flow — cannot be observed. This is precisely why censoring can only be inferred, never measured.
6. **Synthetic-looking future dates.** The data is dated 2026-03 to 2026-05. If simulated rather than observed, that shapes how much external validity the findings carry.
7. **Everything is assumed GCJ-02 (§E0).** Two independent measurements agree, and both the boundary and the trips are consistent with each other under that reading — but if the source tool had in fact used WGS-84, the entire study area would sit ~540 m from where it is drawn. The assumption is stated, tested, and left visible in `02_region_map.py` precisely so a reviewer can overrule it.
8. **Boundary crossing locations are inferred from straight lines** between geohash cell centroids, since no path is recorded (§E0). The *counts* of inflow and outflow are unaffected — they depend only on which endpoint is inside — but the *placement* of each crossing along the perimeter is approximate, which is why the boundary is reported in ~300 m arcs. The gateway view uses no straight-line assumption.
