# Data Cleaning & Annotation Summary

| | |
|---|---|
| **Input** | `dataset_train_20260921_lsz.csv` — 667,821 rows |
| **Output** | `trips_clean_keepzerodur_keepanomday.csv.gz` — 666,575 rows (**99.81% retained**) |
| **Removed** | 1,246 rows (0.1866%) |
| **Period** | 2026-03-02 → 2026-05-24 · 84 days · 0 missing · **84 after cleaning** |
| **Spatial scope** | `wx4e`, `wx4g` — **99.86% of trips stay inside at both ends** (99.93% of endpoints) |

> **Non-default policy** — output tagged `keepzerodur_keepanomday`. Released from the drop set (still flagged and reported, no longer removed): `qc_anomaly_day`, `qc_zero_dur`.

## Flag audit

| Tier | Flag | Rows | % | Disposition |
|:--:|---|---:|---:|---|
| 0 | `qc_unparsed` | 0 | 0.000 | dropped |
| 0 | `qc_null_required` | 0 | 0.000 | dropped |
| 0 | `qc_dup_order` | 0 | 0.000 | dropped |
| 0 | `qc_bad_geohash` | 0 | 0.000 | dropped |
| 1 | `qc_neg_dur` | 0 | 0.000 | dropped |
| 1 | `qc_speed_gt_30` | 329 | 0.049 | dropped |
| 1 | `qc_cross_region` | 953 | 0.143 | dropped |
| 2 | `qc_dur_gt_180` | 49 | 0.007 | dropped |
| 2 | `qc_outside_window` | 0 | 0.000 | dropped |
| — | `qc_dur_gt_60` | 11,116 | 1.665 | kept |
| — | `qc_same_geohash` | 4,736 | 0.709 | kept |
| 2 | `qc_anomaly_day` | 2,463 | 0.369 | **kept (switch)** |
| 1 | `qc_zero_dur` | 5,835 | 0.874 | **kept (switch)** |
| | **dropped, unique** | **1,246** | **0.1866** | |

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
| clean rows | 666,575 | 11.05 | 12.854 | 1.1632 | 2 | 7 | 35 | 71 | 180 |

## Findings

- The file is structurally pristine: 0 unparseable timestamps, 0 invalid geohashes, 0 duplicated order_ids, 0 missing values. Cleaning is about anomaly triage, not repair.
- `action_dt` carries the same calendar day as `start_time` (0 date mismatches), so it is redundant with the derived `date` and is kept only for audit. It is not a text duplicate: `action_dt` is unpadded (`2026/3/2` vs `2026-03-02`), differing on 667,821 rows, so a string join against `date` matches nothing.
- 5,835 zero-minute trips (0.8737%). 4,530 of them (77.635% of zero-minute rows) cross geohash cells, which is impossible at 8-char (~38 m) precision -- evidence that the cause is minute-level timestamp rounding collapsing sub-minute rides, not user error.
- Before cleaning, duration is heavily right-skewed: median 7 min, p99 71 min, max 774 min (11,116 trips exceed 60 min). After cleaning the median is unchanged but the max falls to 180 min and the CV drops 1.1839 -> 1.1632.
- 1 candidate anomaly day detected below 50% of its local baseline.
- Cleaning removes 1,246 rows (0.1866%), leaving 666,575 for analysis.
- Policy override -- `qc_anomaly_day`, `qc_zero_dur` released from the drop set: still computed, still reported, no longer removing rows. 666,575 rows written to trips_clean_keepzerodur_keepanomday.csv.gz.
