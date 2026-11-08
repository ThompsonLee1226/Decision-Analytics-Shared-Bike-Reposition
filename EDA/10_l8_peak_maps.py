#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_l8_peak_maps.py
==================
L8 peak-window **net inflow** heat maps.

    fig17  weekday AM (07:00–09:59) and PM (17:00–19:59) net inflow on the
           L8 geohash grid — two panels, diverging scale centred at zero,
           Amap basemap, study-area ring on top.

    net inflow = arrivals - departures

    positive (blue)  the cell accumulates bikes
    negative (red)   the cell drains bikes

The two peaks are the **three-hour** windows: hours 07–09 and 17–19.  The
mapped value is **net trips per weekday**, not the 60-day total, so the
colour bar reads directly as a daily flow rate.

Why L8
------
The raw `start_geohash` / `end_geohash` are 8 characters.  At this latitude
one L8 cell is about **29 m x 19 m**, the finest spatial unit the extract
supports.  The L6 cell used by `cells.py` is a prefix truncation of the same
string, not a different observation.

What is drawn
-------------
Only L8 cells whose **centre lies inside the study-area ring** (`region.py`).
That is what makes the counts complete: the extract contains every trip that
touches the area, so an interior cell cannot have trips cut off by the extract
boundary.  At L8 the cell is small enough that a centre test is effectively an
exact inside/outside test.

Values are **average net trips per weekday** in the three-hour window.  The
companion table keeps both the per-weekday rate and the 60-day total.

Usage:
    python 10_l8_peak_maps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import cells as C
import l8
import maps
import region as R
import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPS = OUT / "trips_clean_keepzerodur_keepanomday.csv.gz"

PAD_KM = 0.4             # only a thin collar: the ring is the subject
ZOOM = 16                # L8 cells are ~10 x 16 screen px at this zoom


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _poly_verts(bm, cells_):
    """One closed L8 rectangle per cell, in mosaic pixel coordinates."""
    return [np.column_stack(bm.xy(*C.rect(g).T)) for g in cells_]


def draw_field(ax, bm, sub, col: str, cmap, norm):
    """Fill every in-region L8 rectangle with its signed net value."""
    from matplotlib.collections import PolyCollection

    ax.add_collection(PolyCollection(
        _poly_verts(bm, sub["cell"]),
        array=sub[col].to_numpy(dtype=float),
        cmap=cmap, norm=norm, edgecolors="none", zorder=2))


def add_colorbar(plt, fig, ax, cmap, norm, lim: tuple[float, float]):
    """Diverging colour bar with symmetric, round per-weekday ticks."""
    ticks = np.unique(np.round(np.linspace(lim[0], lim[1], 5), 1))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=.040, pad=.012, ticks=ticks)
    cb.ax.set_yticklabels([("0" if t == 0 else f"{t:+.1f}") for t in ticks],
                          fontsize=8)
    cb.set_label("net inflow = arrivals − departures\n(trips per weekday)",
                 fontsize=8.2)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_net_heatmaps(plt, bm, sub, ring, n_days: int, lim, n_sat: int):
    """Fig 17 -- AM and PM net inflow, one shared diverging scale."""
    from matplotlib.colors import Normalize

    norm = Normalize(*lim)
    # RdBu: low = red (net outflow / drains), high = blue (net inflow / fills)
    cmap = plt.get_cmap("RdBu")

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.4))
    for ax, peak in zip(axes, ("am", "pm")):
        bm.draw(ax, attribution=True, ticks=True)
        draw_field(ax, bm, sub, f"net_{peak}_per_day", cmap, norm)
        maps.draw_ring(ax, bm, ring)
        ax.set_title(
            f"{l8.PEAK_LABELS[peak]}  ·  Net inflow\n"
            f"{len(sub):,} interior L8 cells  ·  mean of {n_days} weekdays",
            fontsize=11.2)
        add_colorbar(plt, fig, ax, cmap, norm, lim)

    fig.suptitle(
        "Fig 17 — L8 net inflow in the two 3-hour weekday peaks\n"
        "arrivals − departures per weekday: blue = accumulate · red = drain",
        fontsize=13.2, weight="semibold", y=1.01)
    maps.caption(
        fig,
        "Only cells whose centre lies inside the study-area ring are drawn.  "
        f"Diverging scale symmetric at ±{lim[1]:.1f} trips/weekday; {n_sat} "
        "cell-panels saturate.  Values are average net trips per weekday in "
        "the three-hour peak, not 60-day totals.",
        y=-0.005)
    viz.save(fig, 17, "l8_peak_netflow")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(agg, sub, n_days: int, lim, n_sat: int):
    peaks = list(l8.PEAKS)
    net_per_day = {p: float(sub[f"net_{p}_per_day"].sum()) for p in peaks}
    gross_per_day = {p: float(sub[f"gross_{p}"].sum() / n_days) for p in peaks}
    sign_counts = {
        p: {
            "inflow": int((sub[f"net_{p}"] > 0).sum()),
            "outflow": int((sub[f"net_{p}"] < 0).sum()),
            "zero": int((sub[f"net_{p}"] == 0).sum()),
        } for p in peaks
    }

    def top5(peak, direction):
        t = l8.top_net(agg, peak, direction, 5, inside_only=True, non_overlap=False)
        return t[["rank", "cell", "lat", "lon", f"net_{peak}_per_day"]].round(3)\
            .to_dict("records")

    payload = {
        "weekdays": n_days,
        "peak_hours": {p: list(h) for p, h in l8.PEAKS.items()},
        "l8_cells_active": int(len(agg)),
        "l8_cells_inside_ring": int(len(sub)),
        "net_per_weekday_in_region": net_per_day,
        "gross_per_weekday_in_region": gross_per_day,
        "sign_counts_in_region": sign_counts,
        "diverging_limit_per_weekday": float(lim[1]),
        "saturated_cell_panels": int(n_sat),
        "top5_net_inflow": {p: top5(p, "inflow") for p in peaks},
        "top5_net_outflow": {p: top5(p, "outflow") for p in peaks},
    }
    (OUT / "l8_peak_netflow.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")

    def top_rows(peak, direction):
        t = l8.top_net(agg, peak, direction, 5, inside_only=True, non_overlap=False)
        return "\n".join(
            f"| {r.rank} | `{r.cell}` | {r.lat:.4f}, {r.lon:.4f} | "
            f"{getattr(r, f'net_{peak}_per_day'):+.1f} |"
            for r in t.itertuples())

    summary_rows = "\n".join(
        f"| {l8.PEAK_LABELS[p]} | {net_per_day[p]:+.1f} | "
        f"{gross_per_day[p]:,.1f} | {sign_counts[p]['inflow']:,} | "
        f"{sign_counts[p]['outflow']:,} | {sign_counts[p]['zero']:,} |"
        for p in peaks)

    sections = "\n\n".join(
        f"### {l8.PEAK_LABELS[p]} · net {d}\n\n"
        f"| # | cell (L8) | centroid (lat, lon) | net trips/weekday |\n"
        f"|---:|---|---|---:|\n{top_rows(p, d)}"
        for p in peaks for d in ("inflow", "outflow"))

    am_out = l8.top_net(agg, "am", "outflow", 1, inside_only=True, non_overlap=False).iloc[0]
    am_in = l8.top_net(agg, "am", "inflow", 1, inside_only=True, non_overlap=False).iloc[0]
    pm_out = l8.top_net(agg, "pm", "outflow", 1, inside_only=True, non_overlap=False).iloc[0]
    pm_in = l8.top_net(agg, "pm", "inflow", 1, inside_only=True, non_overlap=False).iloc[0]

    md = f"""# L8 peak-window net inflow maps (fig 17)

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
({len(sub):,} cells). This is the completeness rule from §K0: the extract
contains every trip touching the area, so an interior cell's counts are
complete; an exterior cell's counts are not, and would make a peak map read
as if demand stopped at the extract boundary.

## Service-wide rates

| peak | net trips/weekday | gross trips/weekday | cells net inflow | cells net outflow | cells balanced |
|---|---:|---:|---:|---:|---:|
{summary_rows}

The diverging scale is symmetric at **±{lim[1]:.1f} trips/weekday**; {n_sat}
cell-panels exceed that and saturate. The limit is a high percentile of
|net|, not the maximum: one L8 cell dominates both peaks, and letting it set
the scale would flatten every other cell to white.

## Top 5 cells per direction (single-cell ranking)

{sections}

**Read it as.** The two leaders are the same L8 cell, `{am_out['cell']}`:
it is the largest **net outflow** in the morning
({am_out['net_am_per_day']:+.1f} trips/weekday) and the largest **net inflow**
in the evening ({pm_in['net_pm_per_day']:+.1f} trips/weekday). That is the
L8-resolution role reversal in one cell. The asymmetry between the two
directions is also informative: the AM drain and PM inflow are an order of
magnitude larger than the largest AM inflow
({am_in['net_am_per_day']:+.1f}/weekday) and PM outflow
({pm_out['net_pm_per_day']:+.1f}/weekday), because both commute peaks
concentrate on the same cell while the opposite directions are spread over
many cells.  Fig 18 turns these single cells into 3 x 3 neighbourhoods.
"""
    (OUT / "l8_peak_netflow_summary.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    if not TRIPS.exists():
        print(f"FATAL: {TRIPS} not found. Run 01_data_cleaning.py first.",
              file=sys.stderr)
        return 2

    ring, _, _ = R.load_boundary()
    trips = l8.load_trips(TRIPS)
    agg = l8.aggregate(trips, ring)
    sub = agg.loc[agg["in_region"]].copy()
    n_days = int(agg["weekdays"].iloc[0])

    vals = np.concatenate([sub[f"net_{p}_per_day"].to_numpy(dtype=float)
                           for p in l8.PEAKS])
    lim = viz.sym_lim(vals, q=99.5, floor=1.0)
    n_sat = int((np.abs(vals) > lim[1]).sum())

    print(f"weekdays : {n_days}")
    print(f"L8 cells : {len(agg):,} active, {len(sub):,} with centre inside the ring")
    for p in l8.PEAKS:
        print(f"  {p:>2}: net inflow {sub[f'net_{p}_per_day'].sum():+.1f} trips/weekday, "
              f"{int((sub[f'net_{p}'] > 0).sum()):,} cells inflow, "
              f"{int((sub[f'net_{p}'] < 0).sum()):,} cells outflow")
    print(f"diverging: symmetric ±{lim[1]:.1f} trips/weekday "
          f"({n_sat} cell-panels saturate)")

    keep = (["cell", "lat", "lon", "in_region", "weekdays"]
            + [f"{m}_{p}" for p in l8.PEAKS for m in ("dep", "arr", "net", "gross")]
            + [f"dep_{p}_per_day" for p in l8.PEAKS]
            + [f"arr_{p}_per_day" for p in l8.PEAKS]
            + [f"net_{p}_per_day" for p in l8.PEAKS])
    agg[keep].to_csv(OUT / "table_l8_peak_netflow.csv.gz", index=False,
                     compression="gzip", encoding="utf-8")

    bm = maps.window_mosaic(ring, PAD_KM, ZOOM)
    print(f"basemap  : {bm}")

    fig_net_heatmaps(plt, bm, sub, ring, n_days, lim, n_sat)
    write_report(agg, sub, n_days, lim, n_sat)

    print(f"\nwrote {OUT/'l8_peak_netflow_summary.md'}, l8_peak_netflow.json,")
    print("      table_l8_peak_netflow.csv.gz")
    return 0


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
