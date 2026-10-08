#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
region.py
=========
The single source of truth for the study region.

Both `02_region_map.py` (which draws it) and `03_region_flow.py` (which uses it
to classify trips) import this, so the ring the map shows and the ring the
analysis tests against cannot drift apart.

Where the boundary comes from
-----------------------------
`data_select_configure_zgc_<date>.xlsx` (repo root) holds two scanline tables:

    sheet "Latitude_range"   indexed by Longitude -> [min_lat, max_lat]
    sheet "Longitude_range"  indexed by Latitude  -> [min_lon, max_lon]

(the sheet names are swapped relative to their content).

The ring is traced from the per-longitude table: south edge west->east along
min_lat, then north edge east->west along max_lat.  That table is preferred
because a per-latitude scanline stores one [min_lon, max_lon] per row and so
cannot express a notch in a horizontal edge -- a row crossing the notch
reports the outer extremes and silently fills it.  Here the notch matters:
the per-longitude table's south edge rises from 39.963 at lon 116.309 to
39.968, which is 北三环 sloping the way the real 3rd Ring Road slopes, while
the per-latitude table claims full lon width at lat 39.963.  `cross_check`
reports how far apart the two readings are.

Datums
------
Everything in this project is **GCJ-02**, including this ring -- established
against OSM road centrelines (WGS-84) and against the Amap tile water mask.
See the `02_region_map.py` docstring for the measurements.  Nothing here
converts anything; the ring is used exactly as the xlsx gives it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

XLSX = ROOT / "data_select_configure_zgc_20260909.xlsx"
GRID_RES = 0.001                     # the boundary tables are sampled at this step


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

def load_boundary(xlsx: Path = XLSX):
    """Return (ring, by_lon, by_lat).

    ring is a list of (lat, lon) vertices, closed only implicitly -- pass
    `ring + [ring[0]]` if you need an explicit closing edge.
    """
    by_lon = pd.read_excel(xlsx, "Latitude_range").sort_values("Longitude")
    by_lat = pd.read_excel(xlsx, "Longitude_range").sort_values("Latitude")

    for df, name, cols in ((by_lon, "Latitude_range",
                            ("Longitude", "Min_latitude", "Max_latitude")),
                           (by_lat, "Longitude_range",
                            ("Latitude", "Min_longitude", "Max_longitude"))):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"{xlsx.name}[{name}]: missing columns {missing}")

    ring = ([(r.Min_latitude, r.Longitude) for r in by_lon.itertuples()]
            + [(r.Max_latitude, r.Longitude) for r in by_lon.itertuples()][::-1])
    return ring, by_lon, by_lat


def _spell_out(df, key_col, lo_col, hi_col, key_is_lat):
    """Expand a scanline table into the set of 0.001-deg (lat, lon) cells it covers."""
    out = set()
    for r in df.itertuples():
        key = round(getattr(r, key_col), 6)
        lo = round(getattr(r, lo_col), 6)
        hi = round(getattr(r, hi_col), 6)
        for k in range(int(round((hi - lo) / GRID_RES)) + 1):
            v = round(lo + k * GRID_RES, 6)
            out.add((key, v) if key_is_lat else (v, key))
    return out


def cross_check(by_lon, by_lat):
    """The two tables describe one polygon; measure how well they agree.

    Returns (from_lat, from_lon, only_lat, only_lon), each normalised to
    (lat, lon) 0.001-deg cells.
    """
    from_lon = _spell_out(by_lon, "Longitude", "Min_latitude", "Max_latitude",
                          key_is_lat=False)
    from_lat = _spell_out(by_lat, "Latitude", "Min_longitude", "Max_longitude",
                          key_is_lat=True)
    return from_lat, from_lon, from_lat - from_lon, from_lon - from_lat


def ring_area_km2(ring) -> float:
    """Spherical-excess area of a lat/lon ring; good to ~0.1% at this size."""
    la = np.radians([p[0] for p in ring] + [ring[0][0]])
    lo = np.radians([p[1] for p in ring] + [ring[0][1]])
    R = 6371.0088
    return abs(float(np.sum((lo[1:] - lo[:-1]) * (2 + np.sin(la[:-1]) + np.sin(la[1:]))))
               * R * R / 2.0)


def ring_bbox(ring, pad_deg: float = 0.0):
    """(lat_min, lon_min, lat_max, lon_max) of the ring."""
    la = [p[0] for p in ring]
    lo = [p[1] for p in ring]
    return (min(la) - pad_deg, min(lo) - pad_deg, max(la) + pad_deg, max(lo) + pad_deg)


# ---------------------------------------------------------------------------
# Point in polygon
# ---------------------------------------------------------------------------

def points_in_ring(lat, lon, ring) -> np.ndarray:
    """Vectorised even-odd crossing test.

    Uses plain arithmetic rather than a ray cast per point so the whole trip
    table can be classified in one pass.  Boundary cases are decided by the
    same strict/non-strict comparisons as the JavaScript in `02_region_map.py`,
    so the map's "界内格点" count and this classification agree.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    inside = np.zeros(lat.shape, dtype=bool)

    n = len(ring)
    for i in range(n):
        y1, x1 = ring[i]
        y2, x2 = ring[(i + 1) % n]
        if y1 == y2:
            continue                     # horizontal edge: no latitude to straddle
        straddles = (y1 > lat) != (y2 > lat)
        if not straddles.any():
            continue
        x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (lon < x_cross)
    return inside


def ring_edges(ring):
    """(N, 4) array of [lat1, lon1, lat2, lon2] for every edge, closed."""
    n = len(ring)
    return np.array([[ring[i][0], ring[i][1], ring[(i + 1) % n][0], ring[(i + 1) % n][1]]
                     for i in range(n)], dtype=float)
