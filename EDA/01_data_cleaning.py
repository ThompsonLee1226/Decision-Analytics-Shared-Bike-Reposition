#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_data_cleaning.py
===================
Data cleaning / QC pipeline for the free-floating bike-sharing trip dataset
`Dataset/dataset_train_20260921_lsz.csv`.

Design principle: **flag first, filter second**.

Every quality test is recorded as an independent boolean column (`qc_*`) that
*never* deletes anything.  Filtering is then a composable, reversible decision
built from those flags.  This keeps the audit trail ("how many rows were
removed, for what reason") intact, which is exactly what HW3 s.2 asks us to
report.  The input file is read-only -- no column is overwritten and no value
is edited in place -- so two cleaning policies can be generated side by side
and compared.

Flags are grouped into three tiers by the *nature* of the question they answer:

  Tier 0  Is the record even parseable?      -> I/O usability
  Tier 1  Is the record self-contradictory?  -> record validity
  Tier 2  Is the record representative of    -> research scope
          the phenomenon we model?

Tier 0/1 are judgements about whether a record can be trusted at all.
Tier 2 is a judgement about what we are studying -- a 12.9 h trip is probably
real, it is just not a typical operating-day observation.

`CLEAN_DROP_FLAGS` *is* the policy: a flag listed there removes its rows.  A
flag in `CLEAN_KEEP_FLAGS` is still computed and still written out, it just
never removes anything.  The two switches in CONFIG ("Cleaning-policy
switches") release a flag from the drop set -- the flag is still evaluated and
still recorded, it simply stops filtering.

Outputs (into EDA/out/; `{tag}` is empty for the default policy, so an
unflagged run reproduces the released filenames exactly):
    trips_clean{tag}.csv.gz      main analysis / modelling input
    cell_summary{tag}.csv        geohash8 -> lat/lon + departure/arrival/net counts
    cleaning_report{tag}.json    per-step row counts, flag tallies, before/after stats
    cleaning_summary{tag}.md     one-page rendering of the report, for slides
    trips_flagged{tag}.csv       opt-in (WRITE_FLAGGED / --write-flagged): every
                                 row + derived columns + qc_* flags, nothing dropped

Dependencies: pandas, numpy.  Geohash decoding is implemented here in pure
Python (no `pygeohash` / `geohash` package required).

Usage:
    python 01_data_cleaning.py
    python 01_data_cleaning.py --input <path> --outdir <dir>

    # the default policy -- byte-for-byte the released trips_clean.csv.gz
    python 01_data_cleaning.py

    # a wider dataset: keep the zero-minute trips and the collapsed day
    python 01_data_cleaning.py --keep-zero-dur --keep-anomaly-day
        -> out/trips_clean_keepzerodur_keepanomday.csv.gz

    # one switch at a time, or name the output yourself
    python 01_data_cleaning.py --keep-anomaly-day
    python 01_data_cleaning.py --keep-zero-dur --tag v2

    # also emit the per-row audit trail
    python 01_data_cleaning.py --write-flagged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- the knobs.  Everything the report tells you is driven from here.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "Dataset" / "dataset_train_20260921_lsz.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "out"

# Timestamps are minute-resolution, uniform format, no seconds.
START_FMT = "%Y/%m/%d %H:%M"

# --- Tier 1 thresholds (physically impossible / self-contradictory) ---------
# Crow-fly distance UNDER-estimates real road distance, so an implied crow-fly
# speed is already a lower bound on the true speed.  A crow-fly speed above
# MAX_SPEED_KMH therefore proves the record is wrong -> safe to use as a hard
# threshold rather than an arbitrary cut.
MAX_SPEED_KMH = 30.0

# Geohash prefix length that defines the "operating region".  Prefixes whose
# share of all trips falls below MIN_REGION_SHARE are treated as out-of-region.
REGION_PREFIX_LEN = 4
MIN_REGION_SHARE = 0.001          # 0.1% of all trips

# --- Tier 2 thresholds (atypical but plausible) ----------------------------
LONG_TRIP_MIN = 180.0             # trips longer than this are "holds", not rides
ANOMALY_DAY_REL = 0.50            # a day below 50% of its local baseline is a
                                  # *candidate* anomaly (holiday / outage / rain)

# --- Spatial aggregation levels (geohash truncation) ------------------------
CELL_LEVELS = (4, 5, 6)

# --- Study window (None = use the full observed range) ---------------------
# Set e.g. ("2026-03-02", "2026-05-24") to restrict; rows outside get flagged.
STUDY_START = None
STUDY_END = None

# --- What the clean output drops -------------------------------------------
# This single list IS the cleaning policy.  Edit it to change what survives.
CLEAN_DROP_FLAGS = [
    # Tier 0 -- structural / I/O
    "qc_unparsed", "qc_null_required", "qc_dup_order", "qc_bad_geohash",
    # Tier 1 -- physically impossible
    "qc_neg_dur", "qc_zero_dur", "qc_speed_gt_30", "qc_cross_region",
    # Tier 2 -- atypical but plausible
    "qc_dur_gt_180", "qc_anomaly_day", "qc_outside_window",
]
# Flags that are informational only -- they never remove a row.
CLEAN_KEEP_FLAGS = ["qc_dur_gt_60", "qc_same_geohash"]

# Which question each flag answers (see the module docstring).
TIER_OF_FLAG = {
    "qc_unparsed": 0, "qc_null_required": 0, "qc_dup_order": 0, "qc_bad_geohash": 0,
    "qc_neg_dur": 1, "qc_zero_dur": 1, "qc_speed_gt_30": 1, "qc_cross_region": 1,
    "qc_dur_gt_180": 2, "qc_anomaly_day": 2, "qc_outside_window": 2,
    "qc_dur_gt_60": None, "qc_same_geohash": None,
}

# --- Cleaning-policy switches ----------------------------------------------
# Both default to the policy that produced the released `trips_clean.csv.gz`,
# so an unflagged run reproduces that file byte-for-byte.  Turning either on
# *widens* the output.  Neither edits the input, so both policies can be
# generated side by side and compared -- the output filenames are tagged so a
# wider run can never overwrite the released one.
#
#   KEEP_ZERO_DUR_TRIPS  Minute-level truncation collapses sub-minute rides to
#                        0 minutes (77.6% of them cross an ~38 m geohash cell,
#                        which is impossible).  The trips are real demand; only
#                        their duration is unmeasurable.  `implied_speed_kmh`
#                        stays NaN for them either way, so the speed rule does
#                        not double-count them.
#
#   KEEP_ANOMALY_DAYS    A day at ~23% of its local baseline (2026-05-17) is a
#                        real demand *regime* -- rain, holiday or outage --
#                        not a data fault.  A forecaster usually wants it in.
#                        Dropping it also punches a hole in the calendar, so
#                        any 24 h / 168 h lag built on the cleaned dates is
#                        silently off by one day across the gap.
KEEP_ZERO_DUR_TRIPS = False
KEEP_ANOMALY_DAYS = False

# --- Output shape ----------------------------------------------------------
# The per-row flagged file is ~260 MB and is exactly reproducible by re-running
# this script (the input CSV is never modified), so it is off by default.
# Set WRITE_FLAGGED = True to get the full per-row audit trail.
WRITE_FLAGGED = False
CLEAN_COMPRESSION = "gzip"        # None -> plain .csv

REQUIRED_COLS = [
    "action_dt", "order_id", "bike_id",
    "start_time", "end_time", "start_geohash", "end_geohash",
]

# ---------------------------------------------------------------------------
# Geohash decoding (pure Python, standard base32)
# ---------------------------------------------------------------------------

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_B32_INDEX = {c: i for i, c in enumerate(_B32)}

# Vectorised equivalent of `all(c in _B32 ...) and len == 8`, for use with
# Series.str.fullmatch.  The class excludes the letters the base32 alphabet
# deliberately omits (a, i, l, o).
GEOHASH8_RE = r"[0-9b-hjkmnp-z]{8}"


def is_valid_geohash(gh: str, length: int = 8) -> bool:
    """True iff `gh` is a well-formed base32 geohash of exactly `length` chars."""
    return (
        isinstance(gh, str)
        and len(gh) == length
        and all(c in _B32_INDEX for c in gh)
    )


def geohash_bbox(gh: str) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) of the geohash cell."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True                      # geohash interleaves lon first
    for ch in gh:
        cd = _B32_INDEX[ch]
        for shift in (4, 3, 2, 1, 0):
            bit = (cd >> shift) & 1
            if even:
                mid = 0.5 * (lon_lo + lon_hi)
                if bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = 0.5 * (lat_lo + lat_hi)
                if bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return lat_lo, lat_hi, lon_lo, lon_hi


def geohash_center(gh: str) -> tuple[float, float]:
    """Return (lat, lon) of the geohash cell centroid."""
    lat_lo, lat_hi, lon_lo, lon_hi = geohash_bbox(gh)
    return 0.5 * (lat_lo + lat_hi), 0.5 * (lon_lo + lon_hi)


def haversine_km_vec(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    r = 6371.0088
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def duration_stats(minutes: pd.Series, label: str) -> dict:
    """Quantile + dispersion summary of a duration series (for the report)."""
    s = pd.Series(minutes).dropna()
    if s.empty:
        return {"label": label, "n": 0}
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0])
    return {
        "label": label,
        "n": int(s.size),
        "mean": round(float(s.mean()), 3),
        "std": round(float(s.std(ddof=1)), 3) if s.size > 1 else None,
        "cv": round(float(s.std(ddof=1) / s.mean()), 4) if s.size > 1 and s.mean() else None,
        "q00": round(float(q.loc[0.00]), 2),
        "q01": round(float(q.loc[0.01]), 2),
        "q05": round(float(q.loc[0.05]), 2),
        "q25": round(float(q.loc[0.25]), 2),
        "q50": round(float(q.loc[0.50]), 2),
        "q75": round(float(q.loc[0.75]), 2),
        "q95": round(float(q.loc[0.95]), 2),
        "q99": round(float(q.loc[0.99]), 2),
        "q100": round(float(q.loc[1.00]), 2),
    }


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 4) if total else 0.0


def effective_drop_flags(keep_zero_dur: bool, keep_anomaly_days: bool) -> list[str]:
    """`CLEAN_DROP_FLAGS` minus whichever optional flags a switch releases.

    The released flag is still computed and still written out -- it just stops
    removing rows.  Guarded lookups so this survives someone editing the base
    list in CONFIG.
    """
    drop = list(CLEAN_DROP_FLAGS)
    for flag, released in (("qc_zero_dur", keep_zero_dur),
                           ("qc_anomaly_day", keep_anomaly_days)):
        if released and flag in drop:
            drop.remove(flag)
    return drop


def policy_tag(keep_zero_dur: bool, keep_anomaly_days: bool) -> str:
    """Filename tag for a non-default policy; '' for the default one."""
    parts = []
    if keep_zero_dur:
        parts.append("keepzerodur")
    if keep_anomaly_days:
        parts.append("keepanomday")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """Read as raw strings so that no dtype inference can silently corrupt ids."""
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,     # do not let "NA"/"nan" become NaN in geohashes
        na_values=[],
        encoding="utf-8",
    )


def structural_qc(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Tier 0 checks.  Records the verdict of each test as a flag column."""
    n = len(df)

    # -- column presence / shape -------------------------------------------
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    report["structural_checks"]["missing_columns"] = missing_cols
    if missing_cols:
        raise SystemExit(f"FATAL: input is missing required columns: {missing_cols}")

    # -- empty-string (i.e. missing) values --------------------------------
    blank = df[REQUIRED_COLS].eq("")
    report["structural_checks"]["blank_counts"] = {
        c: int(blank[c].sum()) for c in REQUIRED_COLS
    }
    df["qc_null_required"] = blank.any(axis=1)

    # -- duplicate order_id (the primary key) ------------------------------
    dup = df["order_id"].duplicated(keep=False)
    df["qc_dup_order"] = dup
    report["structural_checks"]["duplicate_order_ids"] = {
        "distinct": int(df["order_id"].nunique()),
        "rows_affected": int(dup.sum()),
    }

    # -- timestamp parseability --------------------------------------------
    start_dt = pd.to_datetime(df["start_time"], format=START_FMT, errors="coerce")
    end_dt = pd.to_datetime(df["end_time"], format=START_FMT, errors="coerce")
    df["qc_unparsed"] = start_dt.isna() | end_dt.isna()
    report["structural_checks"]["unparseable_timestamps"] = int(df["qc_unparsed"].sum())

    # -- geohash well-formedness -------------------------------------------
    valid_start = df["start_geohash"].str.fullmatch(GEOHASH8_RE, na=False)
    valid_end = df["end_geohash"].str.fullmatch(GEOHASH8_RE, na=False)
    df["qc_bad_geohash"] = ~(valid_start & valid_end)
    report["structural_checks"]["invalid_geohashes"] = int(df["qc_bad_geohash"].sum())
    report["structural_checks"]["geohash_lengths"] = {
        str(k): int(v)
        for k, v in Counter(
            list(df["start_geohash"].str.len()) + list(df["end_geohash"].str.len())
        ).items()
    }

    # -- is action_dt redundant with start_time's date? --------------------
    if "action_dt" in df.columns:
        act = pd.to_datetime(df["action_dt"], format="%Y/%m/%d", errors="coerce")
        mismatch = (act.dt.normalize() != start_dt.dt.normalize()).sum()
        # `action_dt` is written unpadded (`2026/3/2`), so it does NOT equal the
        # canonical `%Y/%m/%d` text of the same day -- comparing the two as
        # strings matches nothing, and any join on the text form silently fails.
        unpadded = int(
            (df["action_dt"].str.strip() != start_dt.dt.strftime("%Y/%m/%d")).sum()
        )
        report["structural_checks"]["action_dt_redundancy"] = {
            "mismatches_vs_start_time_date": int(mismatch),
            "rows_whose_text_differs_from_padded_ymd": unpadded,
            "verdict": ("redundant with `date` once parsed (retained -- this script "
                        "drops no column; drop it in feature engineering)"
                        if mismatch == 0 else "NOT redundant"),
        }
    report["structural_checks"]["n_rows"] = n

    df["_start_dt"] = start_dt
    df["_end_dt"] = end_dt
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Time / space / movement features computed once and shared by all outputs."""
    # -- time ---------------------------------------------------------------
    df["duration_min"] = (df["_end_dt"] - df["_start_dt"]).dt.total_seconds() / 60.0
    df["date"] = df["_start_dt"].dt.date.astype("string")
    df["hour"] = df["_start_dt"].dt.hour.astype("Int16")
    df["minute"] = df["_start_dt"].dt.minute.astype("Int16")
    df["dow"] = df["_start_dt"].dt.dayofweek.astype("Int16")          # 0 = Monday
    df["is_weekend"] = df["dow"] >= 5
    # 15-minute slot index 0..95.  Computed through float so an unparseable
    # timestamp yields NaN rather than an integer 0, which would read as
    # midnight.  The column is therefore float-typed, unlike `hour` / `minute` /
    # `slot_60`; harmless (0 such rows today) but worth knowing before a join.
    df["slot_15"] = (df["hour"].astype("float") * 60 + df["minute"]) // 15
    # Alias of `hour` (0..23), kept so every slot_* granularity shares a prefix.
    df["slot_60"] = df["hour"]

    # -- space: geohash prefix truncation = tunable cell size ---------------
    for lvl in CELL_LEVELS:
        df[f"start_cell{lvl}"] = df["start_geohash"].str.slice(0, lvl)
        df[f"end_cell{lvl}"] = df["end_geohash"].str.slice(0, lvl)

    # -- space: decode unique geohashes once, then map (60k decodes, not 1.3M)
    uniq = pd.Index(
        pd.concat([df["start_geohash"], df["end_geohash"]], ignore_index=True).unique()
    )
    centers = {g: geohash_center(g) for g in uniq if is_valid_geohash(g, 8)}
    lat_map = {g: c[0] for g, c in centers.items()}
    lon_map = {g: c[1] for g, c in centers.items()}
    for side in ("start", "end"):
        df[f"{side}_lat"] = df[f"{side}_geohash"].map(lat_map)
        df[f"{side}_lon"] = df[f"{side}_geohash"].map(lon_map)

    # -- movement -----------------------------------------------------------
    df["crowfly_km"] = haversine_km_vec(
        df["start_lat"], df["start_lon"], df["end_lat"], df["end_lon"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = df["crowfly_km"] / (df["duration_min"] / 60.0)
    # Zero / negative duration records carry no meaningful speed; leave them NaN
    # so they trigger exactly ONE flag (`qc_zero_dur`) and are not double-counted
    # against the speed rule.
    df["implied_speed_kmh"] = speed.where(df["duration_min"] > 0, np.nan)

    df["is_same_cell"] = df["start_geohash"] == df["end_geohash"]
    return df


def add_flags(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Tier 1 + Tier 2 flags.  Depends on derived columns."""
    dur = df["duration_min"]

    # -- Tier 1: physically impossible / self-contradictory -----------------
    df["qc_neg_dur"] = dur < 0
    df["qc_zero_dur"] = dur == 0
    df["qc_speed_gt_30"] = df["implied_speed_kmh"] > MAX_SPEED_KMH

    # Operating region = the set of geohash prefixes that actually matter.
    # Share is pooled over *both* endpoints: a prefix that only ever appears as a
    # destination is still part of the operating footprint, and scoring departures
    # alone would silently drop it.  (On this dataset the kept set is unchanged --
    # wx4e/wx4g dominate both ways -- but the definition is now what the
    # threshold claims to mean.)
    pref_all = pd.concat(
        [df["start_geohash"].str.slice(0, REGION_PREFIX_LEN),
         df["end_geohash"].str.slice(0, REGION_PREFIX_LEN)],
        ignore_index=True,
    )
    pref_counts = pref_all.value_counts()
    share = pref_counts / pref_counts.sum()
    keep_mask = share >= MIN_REGION_SHARE
    keep_prefixes = sorted(share[keep_mask].index.tolist())

    in_region = (
        df["start_geohash"].str.slice(0, REGION_PREFIX_LEN).isin(keep_prefixes)
        & df["end_geohash"].str.slice(0, REGION_PREFIX_LEN).isin(keep_prefixes)
    )
    df["qc_cross_region"] = ~in_region
    report["region"] = {
        "prefix_len": REGION_PREFIX_LEN,
        "min_share": MIN_REGION_SHARE,
        "share_basis": "start and end endpoints pooled (2 x n_rows)",
        "kept_prefixes": keep_prefixes,
        # Share of *endpoints* in a kept prefix -- not the share of trips that
        # stay inside.  That is `trips_both_endpoints_inside` below; the two
        # differ whenever a trip leaves the region.
        "kept_endpoint_share": round(float(share[keep_mask].sum()), 6),
        "trips_both_endpoints_inside": round(float(in_region.mean()), 6),
        "dropped_prefix_endpoint_counts": {
            str(k): int(pref_counts[k]) for k in share[~keep_mask].index
        },
    }

    # -- Tier 2: atypical but plausible -------------------------------------
    df["qc_dur_gt_60"] = dur > 60
    df["qc_dur_gt_180"] = dur > LONG_TRIP_MIN
    df["qc_same_geohash"] = df["is_same_cell"]

    # Anomaly days: compare each day's volume to a centred 7-day rolling median
    # of daily volume.  Flagging only -- a low day may be a genuine weather
    # effect rather than a data fault, so the human decides.
    daily = df.groupby("date", observed=True).size().sort_index()
    daily.index = pd.to_datetime(daily.index)
    baseline = daily.rolling(7, center=True, min_periods=3).median()
    baseline = baseline.fillna(daily.median())
    ratio = daily / baseline
    anomaly_dates = ratio[ratio < ANOMALY_DAY_REL].index
    df["qc_anomaly_day"] = pd.to_datetime(df["date"]).isin(anomaly_dates)

    report["anomaly_days"] = {
        "threshold_rel": ANOMALY_DAY_REL,
        "n_flagged": int(len(anomaly_dates)),
        "detail": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "trips": int(daily.loc[d]),
                "local_baseline": round(float(baseline.loc[d]), 1),
                "ratio": round(float(ratio.loc[d]), 3),
            }
            for d in anomaly_dates
        ],
    }

    # Study window (no-op unless STUDY_START / STUDY_END are set)
    d = pd.to_datetime(df["date"])
    outside = pd.Series(False, index=df.index)
    if STUDY_START:
        outside |= d < pd.Timestamp(STUDY_START)
    if STUDY_END:
        outside |= d > pd.Timestamp(STUDY_END)
    df["qc_outside_window"] = outside

    return df


def build_cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """geohash8 -> centroid + departure / arrival / net counts.

    A byproduct of the geohash decode; feeds the spatial maps in the EDA plan.
    """
    dep = df.groupby("start_geohash").size().rename("n_departures")
    arr = df.groupby("end_geohash").size().rename("n_arrivals")
    cells = pd.concat([dep, arr], axis=1).fillna(0).astype(int)
    cells.index.name = "geohash"
    coords = pd.DataFrame(
        [geohash_center(g) for g in cells.index],
        index=cells.index,
        columns=["lat", "lon"],
    )
    cells = cells.join(coords)
    cells["n_trips_touching"] = cells["n_departures"] + cells["n_arrivals"]
    cells["net_flow"] = cells["n_departures"] - cells["n_arrivals"]
    return cells.reset_index()


def render_summary_md(report: dict) -> str:
    """One-page, PPT-ready rendering of the cleaning + annotation results."""
    n = report["input"]["rows"]
    cr = report["cleaning_result"]
    cov = report["coverage"]
    fs = report["flags_summary"]
    reg = report["region"]
    cfg = report["config"]

    drop_flags = cfg["clean_drop_flags"]
    released = sorted(set(cfg["clean_drop_flags_default"]) - set(drop_flags))
    out_name = Path(report["outputs"]["trips_clean"]["path"]).name

    L: list[str] = []
    A = L.append
    A("# Data Cleaning & Annotation Summary")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| **Input** | `{Path(report['input']['path']).name}` — {n:,} rows |")
    A(f"| **Output** | `{out_name}` — {cr['rows_out']:,} rows "
      f"(**{100 - cr['pct_dropped']:.2f}% retained**) |")
    A(f"| **Removed** | {cr['rows_dropped_unique']:,} rows ({cr['pct_dropped']}%) |")
    A(f"| **Period** | {cov['start_time_range'][0][:10]} → "
      f"{cov['start_time_range'][1][:10]} · {cov['distinct_dates']} days · "
      f"{cov['missing_dates']} missing · **{cov['clean_distinct_dates']} after cleaning** |")
    # Two numbers on purpose: the share of endpoints in a kept prefix is NOT the
    # share of trips that stay inside, and only the second one bounds a
    # within-region analysis.
    A(f"| **Spatial scope** | "
      f"{', '.join('`' + p + '`' for p in reg['kept_prefixes'])} — "
      f"**{100 * reg['trips_both_endpoints_inside']:.2f}% of trips stay inside at both ends** "
      f"({100 * reg['kept_endpoint_share']:.2f}% of endpoints) |")
    A("")

    if released:
        A(f"> **Non-default policy** — output tagged `{cfg['output_tag']}`. "
          f"Released from the drop set (still flagged and reported, no longer "
          f"removed): {', '.join('`' + f + '`' for f in released)}.")
        A("")

    A("## Flag audit")
    A("")
    A("| Tier | Flag | Rows | % | Disposition |")
    A("|:--:|---|---:|---:|---|")
    for f in drop_flags:
        d = report["flags"][f]
        A(f"| {TIER_OF_FLAG[f]} | `{f}` | {d['n']:,} | {d['pct']:.3f} | dropped |")
    for f in CLEAN_KEEP_FLAGS:
        d = report["flags"][f]
        A(f"| — | `{f}` | {d['n']:,} | {d['pct']:.3f} | kept |")
    for f in released:
        d = report["flags"][f]
        A(f"| {TIER_OF_FLAG[f]} | `{f}` | {d['n']:,} | {d['pct']:.3f} | **kept (switch)** |")
    A(f"| | **dropped, unique** | **{cr['rows_dropped_unique']:,}** "
      f"| **{cr['pct_dropped']}** | |")
    A("")
    A(f"> **Tier key** — 0 parseable · 1 self-contradictory · 2 atypical but plausible. "
      f"Counts overlap: {fs['rows_with_2plus_flags']:,} rows carry ≥2 flags; "
      f"{fs['rows_clean_of_any_flag']:,} rows carry none.")
    A("")

    ad = report["anomaly_days"]
    if ad["detail"]:
        A("## Anomalous days")
        A("")
        A("| Date | Trips | Local baseline | Ratio |")
        A("|---|---:|---:|---:|")
        for d in ad["detail"]:
            A(f"| {d['date']} | {d['trips']:,} | {d['local_baseline']:,.0f} "
              f"| {d['ratio']:.3f} |")
        A("")
        A(f"> Flagged where daily volume < {ad['threshold_rel']:.0%} of a centred "
          f"7-day rolling median. A low day may be genuine (weather/holiday) "
          f"rather than a data fault.")
        A("")

    A("## Trip duration (minutes)")
    A("")
    A("| | n | mean | std | CV | p05 | p50 | p95 | p99 | max |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in (report["duration_before_cleaning"], report["duration_after_cleaning"]):
        A(f"| {d['label']} | {d['n']:,} | {d['mean']} | {d['std']} | {d['cv']} "
          f"| {d['q05']:.0f} | {d['q50']:.0f} | {d['q95']:.0f} | {d['q99']:.0f} "
          f"| {d['q100']:.0f} |")
    A("")

    A("## Findings")
    A("")
    for line in report["findings"]:
        A(f"- {line}")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Clean the bike-sharing trip dataset.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--keep-zero-dur", action="store_true",
                    help="keep 0-minute trips (default: dropped, they are "
                         "minute-truncation artefacts)")
    ap.add_argument("--keep-anomaly-day", action="store_true",
                    help="keep days below ANOMALY_DAY_REL of their local "
                         "baseline (default: dropped)")
    ap.add_argument("--write-flagged", action="store_true",
                    help="also write the per-row audit trail, nothing dropped")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filenames. Defaults to a tag "
                         "derived from the policy, empty for the default policy, "
                         "so an unflagged run reproduces the released filenames")
    args = ap.parse_args(argv)

    keep_zero = bool(args.keep_zero_dur or KEEP_ZERO_DUR_TRIPS)
    keep_anom = bool(args.keep_anomaly_day or KEEP_ANOMALY_DAYS)
    tag = args.tag if args.tag is not None else policy_tag(keep_zero, keep_anom)
    tag_suffix = f"_{tag}" if tag else ""
    drop_flags = effective_drop_flags(keep_zero, keep_anom)
    write_flagged = bool(args.write_flagged or WRITE_FLAGGED)

    inp: Path = args.input
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"FATAL: input not found: {inp}", file=sys.stderr)
        return 2

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": {"path": str(inp), "sha256": file_sha256(inp)},
        "config": {
            "max_speed_kmh": MAX_SPEED_KMH,
            "long_trip_min": LONG_TRIP_MIN,
            "anomaly_day_rel": ANOMALY_DAY_REL,
            "region_prefix_len": REGION_PREFIX_LEN,
            "min_region_share": MIN_REGION_SHARE,
            "cell_levels": list(CELL_LEVELS),
            "study_window": [STUDY_START, STUDY_END],
            "clean_drop_flags": drop_flags,
            "clean_drop_flags_default": list(CLEAN_DROP_FLAGS),
            "keep_zero_dur_trips": keep_zero,
            "keep_anomaly_days": keep_anom,
            "write_flagged": write_flagged,
            "output_tag": tag,
            "clean_keep_flags": CLEAN_KEEP_FLAGS,
        },
        "structural_checks": {},
        "flags": {},
        "outputs": {},
        "findings": [],
    }

    print(f"[1/6] reading {inp.name} ...")
    df = load_raw(inp)
    report["input"]["rows"] = int(len(df))
    report["input"]["cols"] = int(df.shape[1])
    report["input"]["columns"] = list(df.columns)

    print(f"[2/6] structural QC on {len(df):,} rows ...")
    df = structural_qc(df, report)

    print("[3/6] deriving time / space / movement features ...")
    df = add_derived(df)

    print("[4/6] computing flags ...")
    df = add_flags(df, report)

    # ---- flag tallies ------------------------------------------------------
    flag_cols = [c for c in df.columns if c.startswith("qc_")]
    n = len(df)
    report["flags"] = {
        c: {"n": int(df[c].sum()), "pct": pct(int(df[c].sum()), n)} for c in flag_cols
    }
    multi = df[flag_cols].sum(axis=1)
    report["flags_summary"] = {
        "rows_clean_of_any_flag": int((multi == 0).sum()),
        "rows_with_1_flag": int((multi == 1).sum()),
        "rows_with_2plus_flags": int((multi >= 2).sum()),
    }

    # ---- apply the cleaning policy ----------------------------------------
    drop_mask = df[drop_flags].any(axis=1)
    clean = df.loc[~drop_mask].copy()

    report["duration_before_cleaning"] = duration_stats(df["duration_min"], "all rows")
    report["duration_after_cleaning"] = duration_stats(clean["duration_min"], "clean rows")

    report["filters"] = [
        {
            "flag": f,
            "tier": TIER_OF_FLAG[f],
            "applied": f in drop_flags,
            "rows_matching_flag": int(df[f].sum()),
            "pct_matching_flag": pct(int(df[f].sum()), n),
        }
        for f in CLEAN_DROP_FLAGS
    ]
    report["cleaning_result"] = {
        "rows_in": n,
        "rows_out": int(len(clean)),
        "rows_dropped_unique": int(drop_mask.sum()),
        "pct_dropped": pct(int(drop_mask.sum()), n),
    }

    # ---- time coverage -----------------------------------------------------
    sd = pd.to_datetime(df["date"])
    report["coverage"] = {
        "start_time_range": [str(df["_start_dt"].min()), str(df["_start_dt"].max())],
        "end_time_range": [str(df["_end_dt"].min()), str(df["_end_dt"].max())],
        "distinct_dates": int(df["date"].nunique()),
        "missing_dates": int(
            (pd.date_range(sd.min(), sd.max()).difference(sd.unique())).size
        ),
        "span_days": int((sd.max() - sd.min()).days) + 1,
        "per_weekday_counts": {
            int(k): int(v) for k, v in df["dow"].value_counts().sort_index().items()
        },
        "clean_distinct_dates": int(clean["date"].nunique()),
        "clean_start_time_range": [
            str(clean["_start_dt"].min()), str(clean["_start_dt"].max())
        ],
        "clean_end_time_range": [
            str(clean["_end_dt"].min()), str(clean["_end_dt"].max())
        ],
    }

    # ---- memorable findings for the slide deck ----------------------------
    zd = int(df["qc_zero_dur"].sum())
    zd_cross = int((df["qc_zero_dur"] & ~df["qc_same_geohash"]).sum())
    report["findings"] = [
        f"The file is structurally pristine: {report['structural_checks']['unparseable_timestamps']} "
        f"unparseable timestamps, {report['structural_checks']['invalid_geohashes']} invalid geohashes, "
        f"{report['structural_checks']['duplicate_order_ids']['rows_affected']} duplicated order_ids, "
        f"0 missing values. Cleaning is about anomaly triage, not repair.",
        f"`action_dt` carries the same calendar day as `start_time` "
        f"({report['structural_checks']['action_dt_redundancy']['mismatches_vs_start_time_date']} date mismatches), "
        f"so it is redundant with the derived `date` and is kept only for audit. It is not a "
        f"text duplicate: `action_dt` is unpadded (`2026/3/2` vs `2026-03-02`), differing on "
        f"{report['structural_checks']['action_dt_redundancy']['rows_whose_text_differs_from_padded_ymd']:,} rows, "
        f"so a string join against `date` matches nothing.",
        f"{zd:,} zero-minute trips ({pct(zd, n)}%). {zd_cross:,} of them "
        f"({pct(zd_cross, zd)}% of zero-minute rows) cross geohash cells, which is impossible at "
        f"8-char (~38 m) precision -- evidence that the cause is minute-level timestamp "
        f"rounding collapsing sub-minute rides, not user error.",
        f"Before cleaning, duration is heavily right-skewed: median "
        f"{report['duration_before_cleaning']['q50']:.0f} min, "
        f"p99 {report['duration_before_cleaning']['q99']:.0f} min, "
        f"max {report['duration_before_cleaning']['q100']:.0f} min "
        f"({int(df['qc_dur_gt_60'].sum()):,} trips exceed 60 min). After cleaning the "
        f"median is unchanged but the max falls to "
        f"{report['duration_after_cleaning']['q100']:.0f} min and the CV drops "
        f"{report['duration_before_cleaning']['cv']} -> {report['duration_after_cleaning']['cv']}.",
        f"{len(report['anomaly_days']['detail'])} candidate anomaly "
        f"{'day' if len(report['anomaly_days']['detail']) == 1 else 'days'} detected "
        f"below {ANOMALY_DAY_REL:.0%} of "
        f"{'its' if len(report['anomaly_days']['detail']) == 1 else 'their'} local baseline.",
        f"Cleaning removes {report['cleaning_result']['rows_dropped_unique']:,} rows "
        f"({report['cleaning_result']['pct_dropped']}%), leaving "
        f"{report['cleaning_result']['rows_out']:,} for analysis.",
    ]

    # Policy overrides are stated in the findings rather than left for the
    # reader to infer from the row count.
    released = sorted(set(CLEAN_DROP_FLAGS) - set(drop_flags))
    if released:
        report["findings"].append(
            f"Policy override -- {', '.join('`' + f + '`' for f in released)} "
            f"released from the drop set: still computed, still reported, no "
            f"longer removing rows. {report['cleaning_result']['rows_out']:,} rows "
            f"written to trips_clean{tag_suffix}.csv.gz."
        )
    if not keep_anom and report["anomaly_days"]["n_flagged"]:
        report["findings"].append(
            "Dropping the flagged day(s) leaves the cleaned window "
            "non-contiguous. Anything built on the cleaned date sequence -- a "
            "24 h or 168 h lag, a day-wise train/test split -- is off by one day "
            "across the gap until the panel is reindexed onto the full calendar."
        )

    # ---- write outputs -----------------------------------------------------
    print("[5/6] writing outputs ...")
    drop_cols = ["_start_dt", "_end_dt"]
    clean_out = clean.drop(columns=drop_cols)
    suffix = ".csv.gz" if CLEAN_COMPRESSION else ".csv"

    p_clean = outdir / f"trips_clean{tag_suffix}{suffix}"
    p_cells = outdir / f"cell_summary{tag_suffix}.csv"
    p_report = outdir / f"cleaning_report{tag_suffix}.json"
    p_summary = outdir / f"cleaning_summary{tag_suffix}.md"

    clean_out.to_csv(p_clean, index=False, encoding="utf-8",
                     compression=CLEAN_COMPRESSION)
    cells = build_cell_summary(clean)
    cells.to_csv(p_cells, index=False, encoding="utf-8")

    report["outputs"] = {
        "trips_clean": {"path": str(p_clean), "rows": int(len(clean_out)),
                        "cols": int(clean_out.shape[1])},
        "cell_summary": {"path": str(p_cells), "rows": int(len(cells))},
    }

    if write_flagged:
        p_flagged = outdir / f"trips_flagged{tag_suffix}.csv"
        flagged_out = df.drop(columns=drop_cols)
        flagged_out.to_csv(p_flagged, index=False, encoding="utf-8")
        report["outputs"]["trips_flagged"] = {
            "path": str(p_flagged), "rows": int(len(flagged_out)),
            "cols": int(flagged_out.shape[1]),
        }

    # The summary is rendered from `report`, which including `outputs` is now
    # complete -- so it can be written alongside the JSON and stay in sync with
    # whatever thresholds are configured.
    report["outputs"]["cleaning_summary"] = {"path": str(p_summary)}
    report["outputs"]["cleaning_report"] = {"path": str(p_report)}

    with open(p_summary, "w", encoding="utf-8") as fh:
        fh.write(render_summary_md(report))
    with open(p_report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ---- console summary ---------------------------------------------------
    print("[6/6] done.\n")
    print("=" * 74)
    print(f"POLICY   keep_zero_dur={keep_zero}  keep_anomaly_days={keep_anom}"
          f"   tag='{tag}'")
    print(f"OUTPUT   {p_clean.name}")
    print("=" * 74)
    print("FLAG TALLIES")
    print("=" * 74)
    for c in flag_cols:
        f = report["flags"][c]
        if c in drop_flags:
            mark = "DROP"
        elif c in CLEAN_DROP_FLAGS:
            mark = "KEPT"            # released by a policy switch
        else:
            mark = "keep"
        print(f"  {mark:<6}{c:<20} {f['n']:>9,}  {f['pct']:>7.3f}%")
    if released:
        print("  ('KEPT' = released by a policy switch: flagged, not removed)")
    print("-" * 74)
    print(f"  rows in        : {n:,}")
    print(f"  rows dropped   : {report['cleaning_result']['rows_dropped_unique']:,} "
          f"({report['cleaning_result']['pct_dropped']}%)")
    print(f"  rows out       : {report['cleaning_result']['rows_out']:,}")
    print("=" * 74)
    print("FINDINGS")
    print("=" * 74)
    for line in report["findings"]:
        print(f"  - {line}")
    print("=" * 74)
    for k, v in report["outputs"].items():
        print(f"  wrote {k:<15} -> {v['path']}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
