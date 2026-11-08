#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
l8.py
=====
L8 (8-character geohash) aggregation shared by the two peak-demand scripts.

The cleaned trip table keeps the raw geohash.  At L8 a cell is about
**29 m x 19 m** at 39.97 N, which is the finest spatial unit the extract
supports; the L6 "analysis cell" in `cells.py` is a prefix truncation of the
same string, not a different observation.

Only weekdays are aggregated.  The two peaks are **three-hour windows**:
AM = hours 07, 08 and 09 (07:00–09:59) and PM = hours 17, 18 and 19
(17:00–19:59).  The hour bin is the trip's start hour, so the boundary is
"departed before 10:00" / "departed before 20:00".

Counts are returned in two forms:

    <metric>_<peak>          total over every weekday in the sample
    <metric>_<peak>_per_day  the same total divided by the weekday count

plus the signed net inflow

    net_<peak> = arrivals - departures

so a positive value means the cell **accumulates** bikes (net inflow) and a
negative value means it **drains** (net outflow).  Gross flow
(`gross_<peak> = departures + arrivals`) is kept so a net value can be read
against the traffic that carries it.

The 3 x 3 block helpers expand a centre cell to its nine-cell neighbourhood
for the top-10 hotspot analysis.  `top_net` selects by single-cell net flow;
`block_panel` / `top_blocks` evaluate every active cell as a block centre and
rank the blocks by their aggregate.  Both selectors can require the chosen
neighbourhoods to be pairwise disjoint, so the reported shares do not
double-count overlapping blocks.

Ranking is identical under either total or per-day form; the maps use the
per-weekday value because the request is for a daily flow rate.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import cells as C
import region as R

GEOHASH8 = re.compile(r"[0-9b-hjkmnp-z]{8}\Z")

# Three-hour peaks: hours 07–09 and 17–19, i.e. 07:00–09:59 and 17:00–19:59.
PEAKS = {"am": (7, 8, 9), "pm": (17, 18, 19)}
PEAK_LABELS = {"am": "AM peak 07:00–09:59",
               "pm": "PM peak 17:00–19:59"}

METRICS = {"dep": "start_geohash", "arr": "end_geohash"}
METRIC_LABELS = {"dep": "Departures", "arr": "Arrivals"}

TRIP_COLS = ["start_geohash", "end_geohash", "hour", "is_weekend", "date"]


def load_trips(path: Path) -> pd.DataFrame:
    """Read only the columns the peak aggregation needs, and validate L8."""
    df = pd.read_csv(
        path, usecols=TRIP_COLS,
        dtype={"start_geohash": str, "end_geohash": str, "date": str},
    )
    for col in ("start_geohash", "end_geohash"):
        bad = ~df[col].str.fullmatch(GEOHASH8)
        if bad.any():
            raise ValueError(f"{path.name}: {int(bad.sum()):,} invalid L8 "
                             f"geohashes in {col}")
    if df["hour"].isna().any() or df["is_weekend"].isna().any():
        raise ValueError(f"{path.name}: null hour/is_weekend")
    return df


def aggregate(trips: pd.DataFrame, ring) -> pd.DataFrame:
    """One row per active L8 cell, with AM/PM departures and arrivals.

    A cell is included if it appears as an origin or a destination at any
    weekday hour.  Cells that are inactive in one peak therefore carry a
    genuine zero in that window, not a missing value.
    """
    wk = trips.loc[~trips["is_weekend"]]
    n_days = int(wk["date"].nunique())
    if n_days == 0:
        raise ValueError("no weekday rows")

    cells = pd.Index(sorted(set(wk["start_geohash"]) | set(wk["end_geohash"])),
                     name="cell")
    lat = np.array([C.centre(g)[0] for g in cells], dtype=float)
    lon = np.array([C.centre(g)[1] for g in cells], dtype=float)

    out = pd.DataFrame({"cell": cells, "lat": lat, "lon": lon})
    out["in_region"] = R.points_in_ring(lat, lon, ring)
    out["weekdays"] = n_days

    for peak, hours in PEAKS.items():
        sub = wk.loc[wk["hour"].isin(hours)]
        for metric, col in METRICS.items():
            counts = (sub.groupby(col, sort=False).size()
                      .reindex(cells, fill_value=0))
            total = counts.to_numpy(dtype=int)
            out[f"{metric}_{peak}"] = total
            out[f"{metric}_{peak}_per_day"] = total / n_days

    for peak in PEAKS:
        dep = out[f"dep_{peak}"]
        arr = out[f"arr_{peak}"]
        out[f"gross_{peak}"] = dep + arr
        out[f"net_{peak}"] = arr - dep
        out[f"net_{peak}_per_day"] = out[f"net_{peak}"] / n_days

    return out


def grid_index(agg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Integer (row, column) indices of each L8 cell centre on the global grid."""
    dlat, dlon = C.cell_size(8)
    ii = np.floor((agg["lat"].to_numpy() + 90.0) / dlat).astype(np.int64)
    jj = np.floor((agg["lon"].to_numpy() + 180.0) / dlon).astype(np.int64)
    return ii, jj


def top_net(agg: pd.DataFrame, peak: str, direction: str, n: int = 10,
            inside_only: bool = False, non_overlap: bool = True) -> pd.DataFrame:
    """Top `n` cells by signed net inflow, or by net outflow.

    ``direction="inflow"`` ranks ``net = arrivals - departures`` descending
    (largest accumulation first); ``"outflow"`` ranks it ascending (largest
    drain first, so rank 1 is the most negative value).

    With ``non_overlap=True`` the selected cells must have disjoint 3 x 3
    neighbourhoods: two centres at least three L8 columns or rows apart.  This
    keeps ten *distinct* areas instead of ten views of the same hotspot, so the
    neighbourhood shares do not double-count the same cells.
    """
    if direction not in ("inflow", "outflow"):
        raise ValueError(f"direction must be inflow/outflow, got {direction!r}")
    col = f"net_{peak}"
    sub = agg.loc[agg["in_region"]] if inside_only else agg
    order = sub.sort_values(col, ascending=(direction == "outflow"), kind="stable")

    if non_overlap:
        ii, jj = grid_index(order)
        chosen_pos: list[int] = []
        chosen_idx: list[tuple[int, int]] = []
        for pos, (i, j) in enumerate(zip(ii, jj)):
            if all(abs(int(i) - ki) > 2 or abs(int(j) - kj) > 2
                   for ki, kj in chosen_idx):
                chosen_pos.append(pos)
                chosen_idx.append((int(i), int(j)))
            if len(chosen_pos) == n:
                break
        if len(chosen_pos) < n:
            raise ValueError(f"only {len(chosen_pos)} non-overlapping cells "
                             f"for {peak}/{direction}")
        top = order.iloc[chosen_pos].copy()
    else:
        top = order.head(n).copy()

    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["peak"] = peak
    top["direction"] = direction
    return top


def block_3x3(agg: pd.DataFrame, cell: str, in_region_only: bool = True) -> pd.DataFrame:
    """Rows of `agg` whose L8 centres lie in the 3 x 3 block centred on `cell`.

    L8 cells are a regular lat/lon grid, so expanding the centre cell's bounds
    by one cell width on each side selects exactly the nine cells around it.
    Cells with no activity are absent from `agg` and contribute zero.
    """
    la0, lo0, la1, lo1 = C.bounds(cell)
    dlat, dlon = C.cell_size(8)
    eps = 1e-9
    m = ((agg["lat"] >= la0 - dlat - eps) & (agg["lat"] <= la1 + dlat + eps)
         & (agg["lon"] >= lo0 - dlon - eps) & (agg["lon"] <= lo1 + dlon + eps))
    if in_region_only:
        m &= agg["in_region"].astype(bool)
    return agg.loc[m]


def block_stats(agg: pd.DataFrame, cell: str, peak: str,
                direction: str | None = None) -> dict:
    """Net/departure/arrival totals inside the 3 x 3 block centred on `cell`.

    With `direction`, also name the cell in the block with the largest net
    inflow (or outflow).  This is the *dominant* cell; it can differ from the
    geometric centre when blocks are ranked by their aggregate rather than by
    a single cell.
    """
    sub = block_3x3(agg, cell)
    col = f"net_{peak}_per_day"
    out = {
        "n_active_cells": int(len(sub)),
        "dep": int(sub[f"dep_{peak}"].sum()),
        "arr": int(sub[f"arr_{peak}"].sum()),
        "net": int(sub[f"net_{peak}"].sum()),
        "dep_per_day": float(sub[f"dep_{peak}_per_day"].sum()),
        "arr_per_day": float(sub[f"arr_{peak}_per_day"].sum()),
        "net_per_day": float(sub[col].sum()),
        "dominant_cell": None,
        "dominant_net_per_day": float("nan"),
    }
    if direction in ("inflow", "outflow") and len(sub):
        idx = sub[col].idxmax() if direction == "inflow" else sub[col].idxmin()
        out["dominant_cell"] = sub.loc[idx, "cell"]
        out["dominant_net_per_day"] = float(sub.loc[idx, col])
    return out


def block_panel(agg: pd.DataFrame, peak: str,
                in_region_only: bool = True) -> pd.DataFrame:
    """Every active cell as a 3 x 3 block centre, with the block aggregate.

    This is the candidate table for ranking **blocks** rather than single
    cells: the block net is the sum of the nine cells around each centre
    (missing cells contribute zero).  It is computed with a dictionary over
    the integer grid indices, so all cells are handled in one pass.
    """
    sub = agg.loc[agg["in_region"]] if in_region_only else agg
    ii, jj = grid_index(sub)
    keys = list(zip(ii.astype(int), jj.astype(int)))
    dep = dict(zip(keys, sub[f"dep_{peak}_per_day"].astype(float)))
    arr = dict(zip(keys, sub[f"arr_{peak}_per_day"].astype(float)))
    net = dict(zip(keys, sub[f"net_{peak}_per_day"].astype(float)))
    net_total = dict(zip(keys, sub[f"net_{peak}"].astype(int)))
    active = set(keys)
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 0), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    rows = []
    for (i, j), r in zip(keys, sub.itertuples()):
        nb = [(i + di, j + dj) for di, dj in neighbours]
        rows.append({
            "cell": r.cell,
            "lat": r.lat,
            "lon": r.lon,
            "ii": i,
            "jj": j,
            "block_dep_per_day": sum(dep.get(k, 0.0) for k in nb),
            "block_arr_per_day": sum(arr.get(k, 0.0) for k in nb),
            "block_net_per_day": sum(net.get(k, 0.0) for k in nb),
            "block_net_total": sum(net_total.get(k, 0) for k in nb),
            "n_active_cells": sum(1 for k in nb if k in active),
        })
    return pd.DataFrame(rows)


def top_blocks(agg: pd.DataFrame, peak: str, direction: str, n: int = 10,
               inside_only: bool = True, non_overlap: bool = True) -> pd.DataFrame:
    """Top `n` 3 x 3 blocks, ranked by their block net flow.

    Every active in-region cell is a candidate centre (`block_panel`).  With
    `non_overlap=True`, selected centres must be at least three L8 columns or
    rows apart so their nine-cell blocks do not overlap.
    """
    if direction not in ("inflow", "outflow"):
        raise ValueError(f"direction must be inflow/outflow, got {direction!r}")
    panel = block_panel(agg, peak, in_region_only=inside_only)
    order = panel.sort_values("block_net_per_day",
                              ascending=(direction == "outflow"), kind="stable")

    if non_overlap:
        chosen_pos: list[int] = []
        chosen_idx: list[tuple[int, int]] = []
        for pos, r in enumerate(order.itertuples()):
            i, j = int(r.ii), int(r.jj)
            if all(abs(i - ki) > 2 or abs(j - kj) > 2 for ki, kj in chosen_idx):
                chosen_pos.append(pos)
                chosen_idx.append((i, j))
            if len(chosen_pos) == n:
                break
        if len(chosen_pos) < n:
            raise ValueError(f"only {len(chosen_pos)} non-overlapping blocks "
                             f"for {peak}/{direction}")
        top = order.iloc[chosen_pos].copy()
    else:
        top = order.head(n).copy()

    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["peak"] = peak
    top["direction"] = direction
    return top
