# Region Flow Summary — study area boundary exchange


| | |
|---|---|
| **Region** | `data_select_configure_zgc_*.xlsx` · 6.47 km² · 88 vertices |
| **Trips** | 658,304 (cleaned) · all coordinates GCJ-02, boundary used as given |
| **Inflow** | 93,497 (14.20%) — outside → inside |
| **Outflow** | 96,340 (14.63%) — inside → outside |
| **Internal** | 466,075 (70.80%) |
| **External** | 2,392 (0.36%) — does not touch the region |
| **Net exchange** | **-2,843** trips (-0.43% of all trips) |

> **Scope.** 99.64% of trips touch the region, so the dataset is an
> *"at least one endpoint in the region"* extract. Inflow and outflow here are
> therefore complete, but the dataset says nothing about trips that never touch
> the area.

## Hourly exchange

Mean trips per hour **per day** of that class (60 weekdays, 23 weekend days).

| hour | in · wk | out · wk | net · wk | in · we | out · we | net · we |
|---:|---:|---:|---:|---:|---:|---:|
| 00 | 4 | 6 | -2 | 5 | 8 | -3 |
| 01 | 2 | 3 | -0 | 3 | 4 | -1 |
| 02 | 1 | 2 | -0 | 2 | 3 | -0 |
| 03 | 1 | 1 | -0 | 1 | 2 | -1 |
| 04 | 2 | 2 | +0 | 1 | 2 | -1 |
| 05 | 6 | 5 | +1 | 3 | 4 | -1 |
| 06 | 28 | 23 | +5 | 17 | 13 | +4 |
| 07 | 88 | 70 | +19 | 34 | 29 | +5 |
| 08 | 119 | 103 | +16 | 51 | 48 | +4 |
| 09 | 80 | 56 | +24 | 60 | 49 | +11 |
| 10 | 54 | 41 | +13 | 62 | 45 | +17 |
| 11 | 64 | 51 | +13 | 70 | 54 | +17 |
| 12 | 64 | 61 | +2 | 73 | 65 | +8 |
| 13 | 59 | 65 | -6 | 66 | 66 | -0 |
| 14 | 50 | 59 | -9 | 56 | 67 | -11 |
| 15 | 52 | 58 | -6 | 57 | 63 | -5 |
| 16 | 70 | 69 | +0 | 74 | 62 | +12 |
| 17 | 129 | 125 | +4 | 100 | 80 | +21 |
| 18 | 110 | 141 | -31 | 83 | 84 | -1 |
| 19 | 68 | 92 | -24 | 54 | 66 | -12 |
| 20 | 50 | 77 | -28 | 40 | 57 | -17 |
| 21 | 43 | 65 | -21 | 35 | 50 | -15 |
| 22 | 24 | 43 | -18 | 24 | 36 | -12 |
| 23 | 11 | 15 | -4 | 12 | 17 | -5 |

> Weekday inflow peaks at **17:00** (129) and outflow at
> **18:00** (141) — per weekday, not summed over the
> window, so these are directly comparable with the daily net below.

## Top gateway cells (6-char, outside the region)

| cell | inflow | outflow | net | centroid (lat, lon) |
|---|---:|---:|---:|---|
| `wx4eqx` | 11,709 | 6,758 | +4,951 | 39.9874, 116.3077 |
| `wx4eqz` | 5,626 | 5,468 | +158 | 39.9872, 116.3165 |
| `wx4eqg` | 5,243 | 4,835 | +408 | 39.9659, 116.3203 |
| `wx4er5` | 5,201 | 5,609 | -408 | 39.9668, 116.3270 |
| `wx4erp` | 4,548 | 5,682 | -1,134 | 39.9875, 116.3285 |
| `wx4eqe` | 4,475 | 4,597 | -122 | 39.9655, 116.3072 |
| `wx4ew8` | 2,777 | 2,024 | +753 | 39.9921, 116.3059 |
| `wx4eqs` | 2,750 | 3,567 | -817 | 39.9705, 116.3040 |

## Weekly net exchange

| week ending | net (inflow − outflow) |
|---|---:|
| 2026-03-08 | -76 |
| 2026-03-15 | -101 |
| 2026-03-22 | -300 |
| 2026-03-29 | -436 |
| 2026-04-05 | -196 |
| 2026-04-12 | -41 |
| 2026-04-19 | -304 |
| 2026-04-26 | -310 |
| 2026-05-03 | -117 |
| 2026-05-10 | -47 |
| 2026-05-17 | -301 |
| 2026-05-24 | -614 |

> The daily net averages **-34.3** with a standard deviation of
> 62.4 (n=83); a one-sample t-test against zero gives
> **t = -5.00, p = 3.2e-06**, and **12 of 12 weeks are negative**. The
> imbalance is a persistent property of the area, not an artefact of a few odd days.

## Which side of the perimeter carries the exchange

| side | crossings | share of crossings | length (km) | share of perimeter | over-represented by |
|---|---:|---:|---:|---:|---:|
| north (北四环) | 75,903 | 40.0% | 3.09 | 27.7% | 1.45× |
| south (北三环) | 59,778 | 31.5% | 3.62 | 32.3% | 0.97× |
| west | 34,308 | 18.1% | 2.59 | 23.2% | 0.78× |
| east | 19,848 | 10.5% | 1.88 | 16.8% | 0.62× |

## Busiest boundary arcs (~300 m)

| arc | midpoint (lat, lon) | length (m) | crossings | share | per km | in | out |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | 39.9850, 116.3175 | 341 | 16,855 | 8.9% | 49,394 | 8,864 | 7,991 |
| 2 | 39.9705, 116.3065 | 948 | 16,268 | 8.6% | 17,161 | 7,459 | 8,809 |
| 25 | 39.9850, 116.3095 | 341 | 12,185 | 6.4% | 35,708 | 6,997 | 5,188 |
| 26 | 39.9850, 116.3055 | 396 | 10,976 | 5.8% | 27,748 | 6,650 | 4,326 |
| 24 | 39.9850, 116.3135 | 341 | 10,671 | 5.6% | 31,271 | 5,813 | 4,858 |
| 16 | 39.9790, 116.3395 | 669 | 10,103 | 5.3% | 15,108 | 4,932 | 5,171 |
| 8 | 39.9680, 116.3245 | 341 | 9,147 | 4.8% | 26,805 | 4,345 | 4,802 |
| 7 | 39.9675, 116.3215 | 310 | 8,386 | 4.4% | 27,030 | 4,392 | 3,994 |

> **How these were located.** The data gives a start and an end geohash, not a
> path, so each crossing is inferred from the straight line between the two
> cell centroids. That quantises the crossing position to the cell grid — on
> the busiest ring edge the 7,287 crossings fall on only 808 distinct
> longitudes — so the boundary is reported in ~300 m arcs rather than at the
> 85 m edge resolution, where grid artefacts would read as structure.
> Crossing points were located for **189,837** of 189,837 crossing trips
> (0.000% unresolved).

## Trip shape by class

| class | n | median duration (min) | mean duration | median crowfly (km) | mean crowfly |
|---|---:|---:|---:|---:|---:|
| internal | 466,075 | 6.0 | 8.6 | 0.60 | 0.67 |
| inflow | 93,497 | 13.0 | 17.0 | 1.42 | 1.88 |
| outflow | 96,340 | 13.0 | 17.9 | 1.50 | 2.02 |

## Findings

- Crossing trips are **28.84%** of the dataset (189,837 trips).
  Net exchange over 83 days is **-2,843** — the region is
  a net source of bikes at **-34.3/day**, and that
  is not noise: the daily net has t = -5.00 against zero (p = 3.2e-06) and
  12 of 12 weeks are negative. Left alone, the area loses roughly
  237 bikes a week and would empty out.
- The exchange runs on a commute clock. Volume in both directions peaks in the
  evening (inflow 17:00, outflow 18:00), but the *balance* flips
  sign over the day: on an average weekday the region **gains 92**
  bikes between 06:00 and 12:00 and **loses 122** between 17:00 and 23:00. That is the signature of a
  workplace — bikes arrive with the morning commute and drain away again in the
  evening — and the evening drain slightly outruns the morning fill. That gap, not
  either flow on its own, is the daily rebalancing requirement.
- Weekend exchange is roughly half the weekday volume and nearly balanced: a net
  of **+13.0 per weekend day** (+300 over all
  23), against -52.4 per weekday. So the imbalance is a working-day
  phenomenon, not a constant leak.
- Trips that cross the boundary are more than twice as long as internal ones
  (median 1.42 km vs
  0.60 km crow-fly), so the boundary exchange is not
  short edge-hopping around the perimeter but genuine cross-district travel.
- The perimeter is not uniformly leaky. The northern edge — 北四环, where the
  area meets 北京大学 / 清华 / 五道口 — carries **40.0% of all crossings on
  27.7% of the boundary**, and the four busiest ~300 m arcs are all on it;
  the single busiest arc (北四环 west of 中关村大街, 341 m) takes 8.9% of the exchange
  on its own. Fixed capacity belongs on a handful of arcs, not spread around the ring.
