#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gp_amap_map.py
==============
Draw every HW6 gathering point (GP) and its L8 grid ownership on a Gaode
(Amap / 高德) basemap.

What "this map" means here
--------------------------
`xgb_bikeflow/data/hw6_grid_ownership.csv` is the assignment of **all** L8
geohash cells in the study area to a gathering point:

    geohash, gp, lon, lat, out_p, in_p, dist

`gp` is either one of 90 GP ids or `UNASSIGNED` (a cell inside the study area
that no GP owns).  The file therefore contains two things at once:

  * the *region map* -- the 10,473 L8 cells that make up the area;
  * the *ownership map* -- which of those cells belong to which GP.

This script renders both, plus the 90 GP points themselves:

    fig01_gp_points_amap.png         Amap + 90 GP points (the minimal view)
    fig02_grid_assignment_amap.png   Amap + all cells coloured by GP + GP points
    gp_points.csv                    one row per GP (GCJ-02 coordinates)
    gp_assignment_summary.csv        per-GP cell counts and cell bounding boxes
    gp_points_amap.html              interactive Amap + GP markers (optional)

Grid cells are drawn as rectangles, not centroid dots: a geohash cell **is** a
rectangle, and at L8 it is about 29 m x 19 m at this latitude, so the ownership
map is a real partition rather than a scatter of points.

Datum -- do not convert anything
--------------------------------
Every coordinate used here is **GCJ-02**, and Amap serves GCJ-02 tiles:

  * `hw6_grid_ownership.csv` `lon/lat` are GCJ-02 (despite the neutral column
    names; verified by re-encoding the geohash and by matching the repro cells'
    `longitude_original/latitude_original` exactly);
  * `hw6_gp_master.csv` `lon_gcj/lat_gcj` are the GCJ-02 GP centres; its
    `lon/lat` columns are WGS-84 and must **not** be used with Amap tiles.

Converting either side to WGS-84 shifts it by roughly 540 m and would put the
GP dots off their own cells.  The script deliberately has no conversion code.

Tile cache
----------
Amap tiles are cached under `inventory_distribution/.tilecache/` so reruns are
offline and free.  The mosaic code is reused from `../EDA/basemap.py`, which
documents why the axes are Web-Mercator pixels rather than degrees.
"""
from __future__ import annotations

import argparse
import colorsys
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDA = ROOT / "EDA"
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))

import basemap as B  # noqa: E402  (path-dependent)
import cells as C  # noqa: E402

# Keep this project's tile cache inside inventory_distribution.
B.CACHE = HERE / ".tilecache"

UNASSIGNED = "UNASSIGNED"
DPI = 150
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0
MARKER = "#d4380d"
INK = "#1f1f1f"


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ownership + GP master, validate the assignment, return both."""
    own_path = data_dir / "hw6_grid_ownership.csv"
    gp_path = data_dir / "hw6_gp_master.csv"
    for p in (own_path, gp_path):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    own = pd.read_csv(own_path)
    master = pd.read_csv(gp_path)
    own["gp"] = own["gp"].astype(str).str.strip()
    master["gp"] = master["gp"].astype(str).str.strip()

    required = {"geohash", "gp", "lon", "lat"}
    if not required <= set(own.columns):
        raise ValueError(f"{own_path.name} is missing columns {sorted(required - set(own.columns))}")
    if not {"lon_gcj", "lat_gcj"} <= set(master.columns):
        raise ValueError(
            f"{gp_path.name} needs lon_gcj/lat_gcj; its lon/lat columns are "
            "WGS-84 and cannot be drawn under Amap tiles")

    if own.geohash.duplicated().any():
        dup = int(own.geohash.duplicated().sum())
        raise ValueError(f"{own_path.name} has {dup} duplicated geohash cells")

    assigned = own[own.gp.ne(UNASSIGNED)]
    if assigned.empty:
        raise ValueError("no assigned cells in the ownership table")

    # Every assigned cell must have a GP centre; if the master is incomplete,
    # fall back to the mean centre of that GP's cells and say so.
    missing = sorted(set(assigned.gp) - set(master.gp))
    if missing:
        cents = assigned.groupby("gp")[["lat", "lon"]].mean()
        master = master.set_index("gp")
        for gp in missing:
            master.loc[gp, ["lon_gcj", "lat_gcj"]] = [
                float(cents.loc[gp, "lon"]), float(cents.loc[gp, "lat"])]
            print(f"  warning: {gp} has no row in {gp_path.name}; "
                  f"using its cell mean instead")
        master = master.reset_index()

    # Per-GP cell count from the ownership file, checked against the master.
    counts = assigned.groupby("gp").size().rename("n_cells_own")
    gps = master.merge(counts, on="gp", how="left")
    gps["n_cells_own"] = gps["n_cells_own"].fillna(0).astype(int)
    if "n_cells" in gps.columns:
        bad = gps[gps.n_cells.astype(int) != gps.n_cells_own]
        if len(bad):
            print(f"  warning: {len(bad)} GP(s) have n_cells != ownership count; "
                  f"e.g. {bad.iloc[0].gp} master={int(bad.iloc[0].n_cells)} "
                  f"ownership={int(bad.iloc[0].n_cells_own)}")

    return own, gps


def check_datum(assigned: pd.DataFrame, gps: pd.DataFrame) -> None:
    """Fail loudly if the GP centres are not on the GCJ cell grid.

    A WGS-84 centre sits ~540 m from its GCJ cells; the worst true offset here
    is ~250 m, so 350 m is a safe tripwire.
    """
    j = assigned.merge(gps[["gp", "lon_gcj", "lat_gcj"]], on="gp", how="left")
    dlat = (j["lat"] - j["lat_gcj"]) * M_PER_DEG_LAT
    dlon = (j["lon"] - j["lon_gcj"]) * M_PER_DEG_LON * np.cos(np.radians(j["lat_gcj"]))
    d = np.hypot(dlat, dlon)
    worst = float(d.max())
    print(f"  datum check: furthest cell-to-GP distance {worst:.0f} m "
          f"(median {float(d.median()):.0f} m)")
    if worst > 350:
        raise ValueError(
            f"GP centres are {worst:.0f} m from their cells; this looks like a "
            "WGS-84/GCJ-02 mix-up. Use hw6_gp_master.lon_gcj/lat_gcj, not lon/lat")


# ---------------------------------------------------------------------------
# Extent and basemap
# ---------------------------------------------------------------------------

def window(own: pd.DataFrame, gps: pd.DataFrame, pad_m: float) -> tuple[float, float, float, float]:
    """(lat0, lat1, lon0, lon1) covering all cells and GP points, plus padding."""
    lat0 = min(float(own.lat.min()), float(gps.lat_gcj.min()))
    lat1 = max(float(own.lat.max()), float(gps.lat_gcj.max()))
    lon0 = min(float(own.lon.min()), float(gps.lon_gcj.min()))
    lon1 = max(float(own.lon.max()), float(gps.lon_gcj.max()))
    mid = math.radians((lat0 + lat1) / 2)
    dlat = pad_m / M_PER_DEG_LAT
    dlon = pad_m / (M_PER_DEG_LON * math.cos(mid))
    return lat0 - dlat, lat1 + dlat, lon0 - dlon, lon1 + dlon


def make_mosaic(lat0, lat1, lon0, lon1, zoom: int) -> B.AmapMosaic:
    """Mosaic at `zoom`, stepping down once if the tile server under-delivers."""
    m = B.AmapMosaic(zoom, lat0, lat1, lon0, lon1)
    total = m.nx * m.ny
    if m.got < 0.6 * total and zoom > 12:
        print(f"  zoom {zoom}: {m.got}/{total} tiles fetched; retrying at {zoom - 1}")
        m = B.AmapMosaic(zoom - 1, lat0, lat1, lon0, lon1)
    print(f"  basemap: {m!r}")
    return m


def new_figure(m: B.AmapMosaic):
    """A figure whose pixel size equals the mosaic, with no margins."""
    import matplotlib.pyplot as plt
    w, h = m.nx * B.TILE, m.ny * B.TILE
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    m.draw(ax, ticks=False, attribution=True)
    return fig, ax


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def gp_palette(gps: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Deterministic, well-separated fill colours for the 90 ownership blocks.

    Golden-angle hue steps avoid adjacent ids (which are often spatially near)
    getting near-identical colours; alternating saturation/value keeps the
    pastel fills distinguishable from each other and from the red GP dots.
    """
    out = {}
    golden = 0.6180339887498949
    for i, gp in enumerate(sorted(gps.gp)):
        h = (i * golden) % 1.0
        s = 0.55 if i % 2 == 0 else 0.42
        v = 0.92 if i % 3 else 0.80
        out[gp] = colorsys.hsv_to_rgb(h, s, v)
    return out


def draw_ownership(ax, m: B.AmapMosaic, own: pd.DataFrame,
                   palette: dict, show_unassigned: bool = True) -> None:
    """Paint every cell, coloured by the GP that owns it."""
    from matplotlib.collections import PolyCollection

    if show_unassigned:
        un = own[own.gp.eq(UNASSIGNED)]
        polys = []
        for g in un.geohash:
            r = C.rect(g)
            x, y = m.xy(r[:, 0], r[:, 1])
            polys.append(np.column_stack([x, y]))
        if polys:
            ax.add_collection(PolyCollection(
                polys, facecolors="none",
                edgecolors=(0.25, 0.25, 0.25, 0.26),
                linewidths=0.10, zorder=1.5, antialiased=True))

    polys, cols = [], []
    assigned = own[own.gp.ne(UNASSIGNED)]
    for gp, sub in assigned.groupby("gp", sort=False):
        c = palette[gp]
        for g in sub.geohash:
            r = C.rect(g)
            x, y = m.xy(r[:, 0], r[:, 1])
            polys.append(np.column_stack([x, y]))
            cols.append((c[0], c[1], c[2], 0.30))
    ax.add_collection(PolyCollection(
        polys, facecolors=cols,
        edgecolors=(1.0, 1.0, 1.0, 0.55),
        linewidths=0.16, zorder=2, antialiased=True))


def draw_points(ax, m: B.AmapMosaic, gps: pd.DataFrame, labels: bool = True) -> None:
    """The 90 GP dots, with optional ids."""
    import matplotlib.patheffects as pe

    x, y = m.xy(gps.lat_gcj.values, gps.lon_gcj.values)
    ax.scatter(x, y, s=46, c=MARKER, edgecolors="white", linewidths=0.65,
               zorder=10)
    if labels:
        for gp, xx, yy in zip(gps.gp, x, y):
            ax.annotate(
                gp, (xx, yy), xytext=(4, -5), textcoords="offset points",
                fontsize=6.8, color=INK, zorder=11,
                path_effects=[pe.withStroke(linewidth=1.7, foreground="white")])


def add_legend(ax, n: int) -> None:
    from matplotlib.lines import Line2D

    handle = Line2D([], [], marker="o", linestyle="", markerfacecolor=MARKER,
                    markeredgecolor="white", markeredgewidth=0.8,
                    markersize=7, label=f"Gathering point (n = {n})")
    ax.legend(handles=[handle], loc="lower left", frameon=True,
              framealpha=0.90, fontsize=10.5, borderpad=0.6)


def add_title(ax, text: str) -> None:
    ax.text(0.010, 0.990, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=15, weight="semibold", color=INK, zorder=20,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="none",
                      alpha=0.88))


def save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Interactive HTML (optional)
# ---------------------------------------------------------------------------

def write_html(gps: pd.DataFrame, path: Path) -> None:
    import folium

    lats = gps.lat_gcj.astype(float)
    lons = gps.lon_gcj.astype(float)
    m = folium.Map(location=[float(lats.mean()), float(lons.mean())],
                   zoom_start=16, tiles=None, control_scale=True)
    # HTTPS for the interactive file: an HTML page opened from a secure origin
    # would otherwise block Amap's HTTP tiles as mixed content.
    folium.TileLayer(
        tiles=B.TILES.replace("http://", "https://", 1),
        subdomains=B.SUBDOMAINS,
        attr="© 高德地图 AutoNavi", name="高德路网",
        max_zoom=18, max_native_zoom=18,
    ).add_to(m)

    fg = folium.FeatureGroup(name="Gathering points")
    for r in gps.itertuples(index=False):
        folium.CircleMarker(
            location=[float(r.lat_gcj), float(r.lon_gcj)], radius=4,
            color="white", weight=1.0, fill=True, fill_color=MARKER,
            fill_opacity=0.95,
            tooltip=f"{r.gp} · {int(r.n_cells_own)} cells",
            popup=folium.Popup(
                f"<b>{r.gp}</b><br>{int(r.n_cells_own)} L8 grid cells",
                max_width=220),
        ).add_to(fg)
    fg.add_to(m)
    m.fit_bounds([[float(lats.min()), float(lons.min())],
                  [float(lats.max()), float(lons.max())]], padding=(30, 30))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=ROOT / "xgb_bikeflow" / "data")
    p.add_argument("--out-dir", type=Path, default=HERE / "out")
    p.add_argument("--zoom", type=int, default=16,
                   help="Amap tile zoom (16 gives ~1.8 m/px here)")
    p.add_argument("--pad-m", type=float, default=220.0,
                   help="collar around the data window, in metres")
    p.add_argument("--mode", choices=("all", "points", "assignment"),
                   default="all")
    p.add_argument("--no-labels", action="store_true",
                   help="omit the per-GP id labels")
    p.add_argument("--no-unassigned", action="store_true",
                   help="do not outline UNASSIGNED cells")
    p.add_argument("--no-html", action="store_true",
                   help="skip the interactive Folium map")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": DPI,
    })

    print(f"reading {args.data_dir}")
    own, gps = load_tables(args.data_dir)
    assigned = own[own.gp.ne(UNASSIGNED)]
    print(f"  cells: {len(own):,} total, {len(assigned):,} assigned, "
          f"{len(own) - len(assigned):,} UNASSIGNED")
    print(f"  gathering points: {len(gps)}")
    check_datum(assigned, gps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    palette = gp_palette(gps)
    labels = not args.no_labels

    # -- CSV deliverables -------------------------------------------------
    cols = [c for c in ("gp", "cls", "lon_gcj", "lat_gcj", "n_cells",
                        "n_cells_own", "dep_day", "arr_day", "out_p", "in_p")
            if c in gps.columns]
    csv = args.out_dir / "gp_points.csv"
    gps[cols].to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"  wrote {csv} ({len(gps)} rows)")

    bounds = assigned.groupby("gp").agg(
        n_cells=("geohash", "size"),
        lon_min=("lon", "min"), lon_max=("lon", "max"),
        lat_min=("lat", "min"), lat_max=("lat", "max"),
        lon_cell_mean=("lon", "mean"), lat_cell_mean=("lat", "mean"))
    summary = gps.merge(bounds, on="gp", how="left")
    scsv = args.out_dir / "gp_assignment_summary.csv"
    summary.to_csv(scsv, index=False, encoding="utf-8-sig")
    print(f"  wrote {scsv} ({len(summary)} rows)")

    # -- points-only figure ----------------------------------------------
    if args.mode in ("all", "points"):
        lat0, lat1, lon0, lon1 = window(assigned, gps, args.pad_m)
        m = make_mosaic(lat0, lat1, lon0, lon1, args.zoom)
        fig, ax = new_figure(m)
        draw_points(ax, m, gps, labels=labels)
        add_legend(ax, len(gps))
        add_title(ax, f"Gathering points (n = {len(gps)})\n"
                      f"H6 grid ownership · {len(assigned):,} assigned cells")
        save(fig, args.out_dir / "fig01_gp_points_amap.png")

    # -- ownership figure -------------------------------------------------
    if args.mode in ("all", "assignment"):
        lat0, lat1, lon0, lon1 = window(own, gps, args.pad_m)
        m = make_mosaic(lat0, lat1, lon0, lon1, args.zoom)
        fig, ax = new_figure(m)
        draw_ownership(ax, m, own, palette,
                       show_unassigned=not args.no_unassigned)
        draw_points(ax, m, gps, labels=labels)
        add_legend(ax, len(gps))
        add_title(ax, f"H6 L8 grid to GP assignment\n"
                      f"{len(assigned):,} assigned of {len(own):,} cells · "
                      f"{len(gps)} GPs")
        save(fig, args.out_dir / "fig02_grid_assignment_amap.png")

    # -- HTML -------------------------------------------------------------
    if not args.no_html:
        write_html(gps, args.out_dir / "gp_points_amap.html")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
