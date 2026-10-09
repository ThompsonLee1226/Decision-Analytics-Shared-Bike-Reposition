# Region Flow Summary — study area boundary exchange


| | |
|---|---|
| **Region** | `data_select_configure_zgc_*.xlsx` · 6.47 km² · 88 vertices |
| **Trips** | 666,575 (cleaned) · all coordinates GCJ-02, boundary used as given |
| **Inflow** | 93,940 (14.09%) — outside → inside |
| **Outflow** | 96,778 (14.52%) — inside → outside |
| **Internal** | 473,426 (71.02%) |
| **External** | 2,431 (0.36%) — does not touch the region |
| **Net exchange** | **-2,838** trips (-0.43% of all trips) |

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
| 02 | 1 | 2 | -0 | 2 | 2 | -0 |
| 03 | 1 | 1 | -0 | 1 | 2 | -1 |
| 04 | 2 | 2 | +0 | 1 | 2 | -1 |
| 05 | 6 | 5 | +1 | 3 | 4 | -1 |
| 06 | 28 | 23 | +5 | 17 | 13 | +4 |
| 07 | 88 | 70 | +19 | 33 | 28 | +5 |
| 08 | 119 | 103 | +16 | 50 | 47 | +4 |
| 09 | 80 | 56 | +24 | 59 | 49 | +11 |
| 10 | 54 | 41 | +13 | 61 | 45 | +16 |
| 11 | 64 | 51 | +13 | 70 | 53 | +17 |
| 12 | 64 | 61 | +2 | 74 | 66 | +8 |
| 13 | 60 | 65 | -6 | 64 | 64 | -0 |
| 14 | 50 | 59 | -9 | 54 | 65 | -11 |
| 15 | 52 | 58 | -6 | 55 | 60 | -5 |
| 16 | 70 | 69 | +0 | 71 | 60 | +11 |
| 17 | 129 | 126 | +4 | 97 | 77 | +20 |
| 18 | 110 | 141 | -31 | 80 | 81 | -1 |
| 19 | 68 | 92 | -24 | 52 | 63 | -11 |
| 20 | 50 | 77 | -28 | 38 | 55 | -17 |
| 21 | 43 | 65 | -21 | 34 | 49 | -15 |
| 22 | 24 | 43 | -18 | 23 | 34 | -12 |
| 23 | 12 | 15 | -4 | 12 | 16 | -5 |

> Weekday inflow peaks at **17:00** (129) and outflow at
> **18:00** (141) — per weekday, not summed over the
> window, so these are directly comparable with the daily net below.

## Top gateway cells (6-char, outside the region)

| cell | inflow | outflow | net | centroid (lat, lon) |
|---|---:|---:|---:|---|
| `wx4eqx` | 11,770 | 6,798 | +4,972 | 39.9874, 116.3077 |
| `wx4eqz` | 5,677 | 5,510 | +167 | 39.9872, 116.3165 |
| `wx4eqg` | 5,271 | 4,859 | +412 | 39.9660, 116.3203 |
| `wx4er5` | 5,231 | 5,650 | -419 | 39.9668, 116.3270 |
| `wx4erp` | 4,560 | 5,712 | -1,152 | 39.9875, 116.3285 |
| `wx4eqe` | 4,525 | 4,632 | -107 | 39.9655, 116.3072 |
| `wx4ew8` | 2,791 | 2,029 | +762 | 39.9921, 116.3059 |
| `wx4eqs` | 2,775 | 3,582 | -807 | 39.9705, 116.3040 |

## Weekly net exchange

| week ending | net (inflow − outflow) |
|---|---:|
| 2026-03-08 | -76 |
| 2026-03-15 | -103 |
| 2026-03-22 | -303 |
| 2026-03-29 | -437 |
| 2026-04-05 | -202 |
| 2026-04-12 | -49 |
| 2026-04-19 | -310 |
| 2026-04-26 | -308 |
| 2026-05-03 | -122 |
| 2026-05-10 | -43 |
| 2026-05-17 | -264 |
| 2026-05-24 | -621 |

> The daily net averages **-33.8** with a standard deviation of
> 62.4 (n=84); a one-sample t-test against zero gives
> **t = -4.96, p = 3.7e-06**, and **12 of 12 weeks are negative**. The
> imbalance is a persistent property of the area, not an artefact of a few odd days.

## Which side of the perimeter carries the exchange

| side | crossings | share of crossings | length (km) | share of perimeter | over-represented by |
|---|---:|---:|---:|---:|---:|
| north (北四环) | 76,254 | 40.0% | 3.09 | 27.7% | 1.45× |
| south (北三环) | 60,066 | 31.5% | 3.62 | 32.3% | 0.97× |
| west | 34,485 | 18.1% | 2.59 | 23.2% | 0.78× |
| east | 19,913 | 10.4% | 1.88 | 16.8% | 0.62× |

## Busiest boundary arcs (~300 m)

| arc | midpoint (lat, lon) | length (m) | crossings | share | per km | in | out |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | 39.9850, 116.3175 | 341 | 16,948 | 8.9% | 49,666 | 8,908 | 8,040 |
| 2 | 39.9705, 116.3065 | 948 | 16,384 | 8.6% | 17,283 | 7,527 | 8,857 |
| 25 | 39.9850, 116.3095 | 341 | 12,253 | 6.4% | 35,907 | 7,041 | 5,212 |
| 26 | 39.9850, 116.3055 | 396 | 11,008 | 5.8% | 27,829 | 6,672 | 4,336 |
| 24 | 39.9850, 116.3135 | 341 | 10,733 | 5.6% | 31,453 | 5,854 | 4,879 |
| 16 | 39.9790, 116.3395 | 669 | 10,134 | 5.3% | 15,155 | 4,947 | 5,187 |
| 8 | 39.9680, 116.3245 | 341 | 9,198 | 4.8% | 26,955 | 4,364 | 4,834 |
| 7 | 39.9675, 116.3215 | 310 | 8,428 | 4.4% | 27,165 | 4,413 | 4,015 |

> **How these were located.** The data gives a start and an end geohash, not a
> path, so each crossing is inferred from the straight line between the two
> cell centroids. That quantises the crossing position to the cell grid — on
> the busiest ring edge the 7,287 crossings fall on only 808 distinct
> longitudes — so the boundary is reported in ~300 m arcs rather than at the
> 85 m edge resolution, where grid artefacts would read as structure.
> Crossing points were located for **190,718** of 190,718 crossing trips
> (0.000% unresolved).

## Trip shape by class

| class | n | median duration (min) | mean duration | median crowfly (km) | mean crowfly |
|---|---:|---:|---:|---:|---:|
| internal | 473,426 | 5.0 | 8.5 | 0.59 | 0.66 |
| inflow | 93,940 | 13.0 | 16.9 | 1.42 | 1.88 |
| outflow | 96,778 | 13.0 | 17.9 | 1.50 | 2.02 |

## Findings

- Crossing trips are **28.61%** of the dataset (190,718 trips).
  Net exchange over 84 days is **-2,838** — the region is
  a net source of bikes at **-33.8/day**, and that
  is not noise: the daily net has t = -4.96 against zero (p = 3.7e-06) and
  12 of 12 weeks are negative. Left alone, the area loses roughly
  236 bikes a week and would empty out.
- The exchange runs on a commute clock. Volume in both directions peaks in the
  evening (inflow 17:00, outflow 18:00), but the *balance* flips
  sign over the day: on an average weekday the region **gains 92**
  bikes between 06:00 and 12:00 and **loses 122** between 17:00 and 23:00. That is the signature of a
  workplace — bikes arrive with the morning commute and drain away again in the
  evening — and the evening drain slightly outruns the morning fill. That gap, not
  either flow on its own, is the daily rebalancing requirement.
- Weekend exchange is roughly half the weekday volume and nearly balanced: a net
  of **+13.1 per weekend day** (+314 over all
  24), against -52.5 per weekday. So the imbalance is a working-day
  phenomenon, not a constant leak.
- Trips that cross the boundary are more than twice as long as internal ones
  (median 1.42 km vs
  0.59 km crow-fly), so the boundary exchange is not
  short edge-hopping around the perimeter but genuine cross-district travel.
- The perimeter is not uniformly leaky. The northern edge — 北四环, where the
  area meets 北京大学 / 清华 / 五道口 — carries **40.0% of all crossings on
  27.7% of the boundary**, and the four busiest ~300 m arcs are all on it;
  the single busiest arc (北四环 west of 中关村大街, 341 m) takes 8.9% of the exchange
  on its own. Fixed capacity belongs on a handful of arcs, not spread around the ring.
