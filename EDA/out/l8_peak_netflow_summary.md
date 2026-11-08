# L8 peak-window net inflow maps (fig 17)

*Rendered by `10_l8_peak_maps.py`.*

## What is plotted

The raw geohash is **L8** (8 characters). At this latitude one cell is about
**29 m x 19 m**. The peaks are **three-hour windows**: AM = hours 07–09
(07:00–09:59) and PM = hours 17–19 (17:00–19:59).

The mapped quantity is

    net inflow = arrivals − departures

so **blue means the cell accumulates bikes** (arrivals exceed departures) and
**red means it drains** (departures exceed arrivals). Values are **average net
trips per weekday**, not 60-day totals, so the colour bar is a daily rate.
The companion table keeps both the per-weekday value and the 60-day total.

Only L8 cells whose **centre lies inside the study-area ring** are drawn
(9,759 cells). This is the completeness rule from §K0: the extract
contains every trip touching the area, so an interior cell's counts are
complete; an exterior cell's counts are not, and would make a peak map read
as if demand stopped at the extract boundary.

## Service-wide rates

| peak | net trips/weekday | gross trips/weekday | cells net inflow | cells net outflow | cells balanced |
|---|---:|---:|---:|---:|---:|
| AM peak 07:00–09:59 | +58.6 | 4,308.4 | 2,397 | 4,693 | 2,669 |
| PM peak 17:00–19:59 | -51.1 | 3,869.5 | 2,410 | 4,246 | 3,103 |

The diverging scale is symmetric at **±4.5 trips/weekday**; 98
cell-panels exceed that and saturate. The limit is a high percentile of
|net|, not the maximum: one L8 cell dominates both peaks, and letting it set
the scale would flatten every other cell to white.

## Top 5 cells per direction (single-cell ranking)

### AM peak 07:00–09:59 · net inflow

| # | cell (L8) | centroid (lat, lon) | net trips/weekday |
|---:|---|---|---:|
| 1 | `wx4eqtz0` | 39.9780, 116.3111 | +46.2 |
| 2 | `wx4ernm9` | 39.9809, 116.3310 | +21.9 |
| 3 | `wx4eqtyq` | 39.9790, 116.3100 | +19.5 |
| 4 | `wx4eqwug` | 39.9840, 116.3080 | +18.2 |
| 5 | `wx4eqwzh` | 39.9841, 116.3111 | +16.4 |

### AM peak 07:00–09:59 · net outflow

| # | cell (L8) | centroid (lat, lon) | net trips/weekday |
|---:|---|---|---:|
| 1 | `wx4eqtku` | 39.9759, 116.3080 | -234.8 |
| 2 | `wx4eqv7m` | 39.9761, 116.3169 | -67.3 |
| 3 | `wx4eqtkv` | 39.9761, 116.3080 | -60.7 |
| 4 | `wx4eqydw` | 39.9831, 116.3159 | -49.2 |
| 5 | `wx4eqtkh` | 39.9759, 116.3069 | -20.0 |

### PM peak 17:00–19:59 · net inflow

| # | cell (L8) | centroid (lat, lon) | net trips/weekday |
|---:|---|---|---:|
| 1 | `wx4eqtku` | 39.9759, 116.3080 | +249.9 |
| 2 | `wx4eqt7k` | 39.9759, 116.3059 | +61.8 |
| 3 | `wx4eqt7m` | 39.9761, 116.3059 | +26.8 |
| 4 | `wx4eqydw` | 39.9831, 116.3159 | +23.1 |
| 5 | `wx4eqz49` | 39.9850, 116.3159 | +19.9 |

### PM peak 17:00–19:59 · net outflow

| # | cell (L8) | centroid (lat, lon) | net trips/weekday |
|---:|---|---|---:|
| 1 | `wx4eqw7r` | 39.9819, 116.3059 | -8.4 |
| 2 | `wx4ern5k` | 39.9800, 116.3279 | -6.5 |
| 3 | `wx4ermbw` | 39.9790, 116.3351 | -6.3 |
| 4 | `wx4ernmd` | 39.9811, 116.3310 | -5.9 |
| 5 | `wx4eqwj1` | 39.9795, 116.3083 | -5.7 |

**Read it as.** The two leaders are the same L8 cell, `wx4eqtku`:
it is the largest **net outflow** in the morning
(-234.8 trips/weekday) and the largest **net inflow**
in the evening (+249.9 trips/weekday). That is the
L8-resolution role reversal in one cell. The asymmetry between the two
directions is also informative: the AM drain and PM inflow are an order of
magnitude larger than the largest AM inflow
(+46.2/weekday) and PM outflow
(-8.4/weekday), because both commute peaks
concentrate on the same cell while the opposite directions are spread over
many cells.  Fig 18 turns these single cells into 3 x 3 neighbourhoods.
