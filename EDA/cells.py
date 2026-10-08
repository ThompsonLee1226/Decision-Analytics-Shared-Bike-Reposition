#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cells.py
========
The analysis cell: the L6 geohash grid, its geometry, and the per-cell
aggregation every spatial figure is built from.

`region.py` is the single source of truth for the *ring*; this is the single
source of truth for the *grid* drawn inside it.  Keeping the grid geometry in
one place matters because the mapping from a geohash to a rectangle is easy to
get subtly wrong, and three figures (06, 07, 09) depend on it agreeing.

Why rectangles and not centroid dots
------------------------------------
A geohash cell **is** a rectangle: geohash encodes latitude and longitude as two
independent uniform grids and interleaves the bits, so at level L every cell has
the same size and the edges fall on fixed multiples of that size from -90/-180.
Drawing the centroid as a dot throws that away and makes the L6 grid look like
scattered points rather than a partition of the map.  `bounds()` recovers the
rectangle exactly, so the maps can show the real spatial unit.

L6 -- the level fixed in plan §0 -- is **0.010986° lon x 0.005493° lat**, which
at 39.97 N is about **937 m x 607 m = 0.57 km²**: the closest geohash level to a
realistic bike-share operating unit.  (The plan text quotes 1.22 x 0.61 km,
which used 111320 m/deg for *longitude* without the cos(lat) factor; the
conclusion -- L6 and no other level -- is unaffected.)

Completeness warning that governs every per-cell number here
------------------------------------------------------------
The dataset is an extract of *trips touching the study area* (§K0).  So:

  * for a cell **inside** the ring, every trip starting or ending in it touches
    the area and is therefore in the extract -- its counts are **complete**;
  * for a cell **outside**, only the trips that also touch the area are present,
    so its counts are **truncated**, and its `net_flow` is not an estimate of
    anything.

`summary()` therefore carries an `in_region` / `inside_frac` pair, and every
figure that maps a per-cell quantity must use them to separate the two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BITS_PER_CHAR = 5


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def cell_size(level: int = 6) -> tuple[float, float]:
    """(lat_deg, lon_deg) size of one cell at `level`."""
    bits = (level * BITS_PER_CHAR) // 2          # each axis gets half the bits
    return 180.0 / 2 ** bits, 360.0 / 2 ** bits


def bounds(g: str) -> tuple[float, float, float, float]:
    """(lat_min, lon_min, lat_max, lon_max) of a geohash cell.

    Walks the same bisection the encoder does: bits alternate longitude first,
    and a set bit takes the *upper* half of the current interval.
    """
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    bit = 0
    for ch in g:
        v = BASE32.index(ch)
        for k in range(BITS_PER_CHAR - 1, -1, -1):
            on = (v >> k) & 1
            if bit % 2 == 0:                     # even bit -> longitude
                mid = (lon[0] + lon[1]) / 2
                lon = [mid, lon[1]] if on else [lon[0], mid]
            else:                                # odd bit -> latitude
                mid = (lat[0] + lat[1]) / 2
                lat = [mid, lat[1]] if on else [lat[0], mid]
            bit += 1
    return lat[0], lon[0], lat[1], lon[1]


def centre(g: str) -> tuple[float, float]:
    """(lat, lon) of the cell centre."""
    la0, lo0, la1, lo1 = bounds(g)
    return (la0 + la1) / 2, (lo0 + lo1) / 2


def rect(g: str) -> np.ndarray:
    """Closed 5-point (lat, lon) ring of the cell, for drawing."""
    la0, lo0, la1, lo1 = bounds(g)
    return np.array([[la0, lo0], [la0, lo1], [la1, lo1], [la1, lo0], [la0, lo0]])


def inside_fraction(g: str, ring, n: int = 16) -> float:
    """Share of the cell's area lying inside `ring`, by brute-force sampling.

    Only used to *label* cells, never to move a boundary: the trip
    classification in `03_region_flow.py` uses exact endpoints.  Sampling an
    n x n grid inside the rectangle is accurate to ~1/n in the worst case and
    is plenty for "is this cell wholly inside or not".
    """
    import region as R
    la0, lo0, la1, lo1 = bounds(g)
    la = la0 + (np.arange(n) + .5) / n * (la1 - la0)
    lo = lo0 + (np.arange(n) + .5) / n * (lo1 - lo0)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    return float(R.points_in_ring(LA.ravel(), LO.ravel(), ring).mean())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summary(trips: pd.DataFrame, ring=None, level: int = 6) -> pd.DataFrame:
    """One row per cell with departures, arrivals, net flow and geometry.

    `trips` needs `start_cell<level>` / `end_cell<level>` columns.  Cells with
    neither a departure nor an arrival are absent rather than zero -- a cell the
    extract never touched is not an observation of "no demand".
    """
    dep = trips.groupby(f"start_cell{level}").size().rename("n_departures")
    arr = trips.groupby(f"end_cell{level}").size().rename("n_arrivals")
    out = pd.concat([dep, arr], axis=1).fillna(0).astype(int)
    out.index.name = "cell"

    geo = pd.DataFrame([centre(g) for g in out.index], index=out.index,
                       columns=["lat", "lon"])
    out = out.join(geo)
    out["n_touching"] = out.n_departures + out.n_arrivals
    out["net_flow"] = out.n_departures - out.n_arrivals
    if ring is not None:
        import region as R
        out["inside_frac"] = [inside_fraction(g, ring) for g in out.index]
        out["in_region"] = R.points_in_ring(out.lat.values, out.lon.values, ring)
    return out.reset_index()


def window_mask(cells: pd.DataFrame, ring, pad_km: float = 3.0,
                m_per_deg_lat: float = 110540.0) -> pd.Series:
    """Cells whose centre is within `pad_km` of the ring's bounding box."""
    import region as R
    la0, lo0, la1, lo1 = R.ring_bbox(ring)
    dlat = pad_km * 1000.0 / m_per_deg_lat
    lat0 = np.radians((la0 + la1) / 2)
    dlon = pad_km * 1000.0 / (111320.0 * np.cos(lat0))
    return ((cells.lat.between(la0 - dlat, la1 + dlat))
            & (cells.lon.between(lo0 - dlon, lo1 + dlon)))


def cell_km(level: int = 6) -> tuple[float, float]:
    """(height_km, width_km) of one cell at the study latitude."""
    dlat, dlon = cell_size(level)
    lat0 = np.radians(39.973)
    return dlat * 110.540, dlon * 111.320 * np.cos(lat0)
