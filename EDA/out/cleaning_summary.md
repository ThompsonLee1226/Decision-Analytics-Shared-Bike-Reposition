# Data Cleaning & Annotation Summary

| | |
|---|---|
| **Input** | `dataset_train_20260921_lsz.csv` — 667,821 rows |
| **Output** | `trips_clean.csv.gz` — 658,304 rows (**98.57% retained**) |
| **Removed** | 9,517 rows (1.4251%) |
| **Period** | 2026-03-02 → 2026-05-24 · 84 days · 0 missing · **83 after cleaning** |
| **Spatial scope** | `wx4e`, `wx4g` = 99.96% of trips |

## Flag audit

| Tier | Flag | Rows | % | Disposition |
|:--:|---|---:|---:|---|
| 0 | `qc_unparsed` | 0 | 0.000 | dropped |
| 0 | `qc_null_required` | 0 | 0.000 | dropped |
| 0 | `qc_dup_order` | 0 | 0.000 | dropped |
| 0 | `qc_bad_geohash` | 0 | 0.000 | dropped |
| 1 | `qc_neg_dur` | 0 | 0.000 | dropped |
| 1 | `qc_zero_dur` | 5,835 | 0.874 | dropped |
| 1 | `qc_speed_gt_30` | 329 | 0.049 | dropped |
| 1 | `qc_cross_region` | 953 | 0.143 | dropped |
| 2 | `qc_dur_gt_180` | 49 | 0.007 | dropped |
| 2 | `qc_anomaly_day` | 2,463 | 0.369 | dropped |
| 2 | `qc_outside_window` | 0 | 0.000 | dropped |
| — | `qc_dur_gt_60` | 11,116 | 1.665 | kept |
| — | `qc_same_geohash` | 4,736 | 0.709 | kept |
| | **dropped, unique** | **9,517** | **1.4251** | |

> **Tier key** — 0 parseable · 1 self-contradictory · 2 atypical but plausible. Counts overlap: 1,921 rows carry ≥2 flags; 644,272 rows carry none.

## Anomalous days

| Date | Trips | Local baseline | Ratio |
|---|---:|---:|---:|
| 2026-05-17 | 2,463 | 10,876 | 0.226 |

> Flagged where daily volume < 50% of a centred 7-day rolling median. A low day may be genuine (weather/holiday) rather than a data fault.

## Trip duration (minutes)

| | n | mean | std | CV | p05 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all rows | 667,821 | 11.132 | 13.179 | 1.1839 | 2 | 7 | 36 | 71 | 774 |
| clean rows | 658,304 | 11.145 | 12.864 | 1.1543 | 2 | 7 | 35 | 71 | 180 |

## Findings

- The file is structurally pristine: 0 unparseable timestamps, 0 invalid geohashes, 0 duplicated order_ids, 0 missing values. Cleaning is about anomaly triage, not repair.
- `action_dt` is fully redundant with the date part of `start_time` (0 mismatches).
- 5,835 zero-minute trips (0.8737%). 4,530 of them (77.635% of zero-minute rows) cross geohash cells, which is impossible at 8-char (~38 m) precision -- evidence that the cause is minute-level timestamp rounding collapsing sub-minute rides, not user error.
- Before cleaning, duration is heavily right-skewed: median 7 min, p99 71 min, max 774 min (11,116 trips exceed 60 min). After cleaning the median is unchanged but the max falls to 180 min and the CV drops 1.1839 -> 1.1543.
- 1 candidate anomaly day detected below 50% of its local baseline.
- Cleaning removes 9,517 rows (1.4251%), leaving 658,304 for analysis.
