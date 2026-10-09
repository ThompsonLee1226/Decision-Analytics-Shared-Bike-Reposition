#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_spatial.py
=============
Section D of the plan: where activity physically happens (D1) and where bikes
drain away or pile up (D2/D3).

    fig05  departure and arrival volume, L6 cells, one shared log colour scale
    fig06  net flow per L6 cell, diverging scale centred at zero

Three encodings carry that, and the legend says so:

    solid, full colour   cell wholly inside the ring  -> counts complete
    hatched              cell straddles the ring      -> counts incomplete
    faded                cell outside the ring        -> context only, its
                                                         net flow estimates
                                                         nothing

This is the one place in §D where being careful costs nothing: the alternative
is a net-flow map whose most extreme cells are artefacts of where the extract
was cut.

Window
------
Maps cover the study area plus a **3 km collar**, which holds 99.1% of all trip
endpoints (the extract itself reaches 28 km, but those are the far ends of
cross-district trips, one or two records per cell).  The omitted share is stated
on the figure.

Usage:
    python 06_spatial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import basemap
import cells as C
import maps
import region as R
import viz
from maps import (COMPLETE, COMPLETENESS_NOTE, caption, draw_cells,
                  draw_hotspots, draw_ring)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPS = OUT / "trips_clean_keepzerodur_keepanomday.csv.gz"

PAD_KM = 3.0
ZOOM = 14
TOP_N = 5                 # D3: hotspots named on the map


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# The cell/ring/marker primitives and the completeness encoding live in
# `maps.py`, so figure 11 reads the same way as figure 5.

def fig_volume(plt, bm, win, hot, ring):
    """Fig 5 -- departures and arrivals on one shared log scale."""
    from matplotlib.colors import LogNorm

    vmax = max(win.n_departures.max(), win.n_arrivals.max())
    norm = LogNorm(vmin=1, vmax=max(vmax, 10))
    cmap = plt.get_cmap("YlOrRd")
    # log scale, because 11 cells hold 80% of the activity: on a linear scale
    # everything else renders as one flat colour
    levels = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.6))
    for ax, col, title in ((axes[0], "n_departures", "Departures  (trips starting in the cell)"),
                           (axes[1], "n_arrivals", "Arrivals  (trips ending in the cell)")):
        bm.draw(ax, attribution=True, ticks=True)
        draw_cells(ax, bm, win, win[col].values, cmap=cmap, norm=norm)
        draw_ring(ax, bm, ring)
        draw_hotspots(ax, bm, hot)
        ax.set_title(f"{title}\nL6 cells within {PAD_KM:.0f} km of the study area", fontsize=11.5)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=.037, pad=.012, ticks=levels)
        cb.ax.set_yticklabels([f"{v:,}" for v in levels], fontsize=8)
        cb.set_label("trips", fontsize=9)

    fig.suptitle("Fig 5 — Departure and arrival volume, one shared log colour scale",
                 fontsize=12.5, weight="semibold", y=1.005)
    caption(fig, COMPLETENESS_NOTE + "\n"
            f"1–{TOP_N} mark the top {TOP_N} cells by departures (D3, see the summary table)")
    viz.save(fig, 5, "departure_arrival_maps")


def fig_netflow(plt, bm, win, hot, ring):
    """Fig 6 -- net flow per cell, diverging and centred at zero."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    v = win.net_flow.values.astype(float)
    lim = viz.sym_lim(v[win.inside_frac > 0], q=96, floor=500)
    norm = Normalize(*lim)
    cmap = plt.get_cmap(viz.DIVERGING)

    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    bm.draw(ax)
    draw_cells(ax, bm, win, v, cmap=cmap, norm=norm)
    draw_ring(ax, bm, ring)
    draw_hotspots(ax, bm, hot)

    ax.set_title("Fig 6 — Net flow per L6 cell  (departures − arrivals)", fontsize=12.5)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=.037, pad=.012)
    cb.set_label("net flow  (negative = accumulates · positive = drains)", fontsize=9)

    fig.legend(handles=maps.legend_handles(), loc="upper left",
               bbox_to_anchor=(0.02, 0.055), ncols=3, fontsize=8.6, frameon=False)

    sat = win[np.abs(v) > lim[1]].nlargest(4, "net_flow")
    nsat = int((np.abs(v) > lim[1]).sum())
    key = "  ".join(f"{i} {r.cell} {r.net_flow:+,}" for i, r in enumerate(hot.itertuples(), 1))
    caption(fig,
            f"hotspot key (D3):  {key}\n"
            f"diverging scale is symmetric at ±{lim[1]:,.0f}; "
            f"{nsat} cell{'s' if nsat != 1 else ''} saturate"
            + ("  (" + ", ".join(f"{r.cell} {r.net_flow:+,}" for r in sat.itertuples()) + ")"
               if len(sat) else ""),
            y=-0.005)
    viz.save(fig, 6, "netflow_map")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    if not TRIPS.exists():
        print(f"FATAL: {TRIPS} not found. Run 01_data_cleaning.py first.", file=sys.stderr)
        return 2

    ring, _, _ = R.load_boundary()
    df = pd.read_csv(TRIPS, usecols=["start_cell6", "end_cell6"])
    print(f"building L6 cell summary from {len(df):,} trips ...")
    summ = C.summary(df, ring)
    summ.to_csv(OUT / "cell6_summary.csv", index=False, encoding="utf-8")
    print(f"  {len(summ):,} L6 cells with activity · "
          f"{int(summ.in_region.sum())} with their centre inside the region")

    win = summ[C.window_mask(summ, ring, PAD_KM)].copy()
    kept = 100 * win.n_touching.sum() / summ.n_touching.sum()
    print(f"  window ±{PAD_KM:.0f} km: {len(win):,} cells, {kept:.2f}% of all trip endpoints")
    print(f"    wholly inside the ring : {int((win.inside_frac >= COMPLETE).sum())}")
    print(f"    straddling the boundary: {int(((win.inside_frac > 0) & (win.inside_frac < COMPLETE)).sum())}")
    print(f"    outside (context only) : {int((win.inside_frac == 0).sum())}")

    hot = summ.nlargest(TOP_N, "n_departures")[["cell", "lat", "lon", "n_departures",
                                                "n_arrivals", "net_flow"]]
    print("  top-5 hotspots by departures:")
    for i, r in enumerate(hot.itertuples(), 1):
        print(f"    {i}. {r.cell}  dep {r.n_departures:,}  arr {r.n_arrivals:,}  net {r.net_flow:+,}")

    # the cells whose counts are actually trustworthy, for the table
    trusted = win[win.inside_frac >= COMPLETE].copy()
    trusted["share_pct"] = 100 * trusted.n_departures / summ.n_departures.sum()
    trusted = trusted.sort_values("n_departures", ascending=False)
    trusted.to_csv(OUT / "table_d2_netflow_cells.csv", index=False, encoding="utf-8")
    hot.assign(rank=range(1, len(hot) + 1)).to_csv(
        OUT / "table_d3_hotspots.csv", index=False, encoding="utf-8")

    # the mosaic covers the window, not just the ring: the surrounding cells are
    # the context that shows the study area is a dense island in a sparse field
    bm = maps.window_mosaic(ring, PAD_KM, ZOOM)
    print(f"  basemap: {bm}")

    fig_volume(plt, bm, win, hot, ring)
    fig_netflow(plt, bm, win, hot, ring)

    write_summary(summ, win, trusted, hot, kept)

    (OUT / "spatial.json").write_text(json.dumps({
        "cells_total": int(len(summ)),
        "cells_center_inside_region": int(summ.in_region.sum()),
        "window_pad_km": PAD_KM,
        "window_cells": int(len(win)),
        "window_endpoint_share_pct": round(kept, 3),
        "window_complete": int((win.inside_frac >= COMPLETE).sum()),
        "window_straddling": int(((win.inside_frac > 0) & (win.inside_frac < COMPLETE)).sum()),
        "window_outside": int((win.inside_frac == 0).sum()),
        "net_flow_top": summ.nlargest(5, "net_flow")[["cell", "net_flow"]].to_dict("records"),
        "net_flow_bottom": summ.nsmallest(5, "net_flow")[["cell", "net_flow"]].to_dict("records"),
        "hotspots": hot.to_dict("records"),
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {OUT/'spatial_summary.md'}, spatial.json, cell6_summary.csv,")
    print("      table_d2_netflow_cells.csv, table_d3_hotspots.csv")
    return 0


def write_summary(summ, win, trusted, hot, kept):
    hot_rows = "\n".join(
        f"| {i} | `{r.cell}` | {r.lat:.4f}, {r.lon:.4f} | {r.n_departures:,} | "
        f"{r.n_arrivals:,} | {r.net_flow:+,} |"
        for i, r in enumerate(hot.itertuples(), 1))

    trust_rows = "\n".join(
        f"| `{r.cell}` | {r.n_departures:,} | {r.n_arrivals:,} | {r.net_flow:+,} | "
        f"{r.share_pct:.2f}% | {r.inside_frac:.3f} |"
        for r in trusted.itertuples())

    top_drain = summ.nlargest(5, "net_flow")
    top_pile = summ.nsmallest(5, "net_flow")

    def lst(df_, col="cell"):
        return ", ".join(f"`{c}` ({v:+,})" for c, v in
                         zip(df_["cell"], df_["net_flow"])) if len(df_) else "—"

    md = f"""# §D — Spatial patterns

*Rendered by `06_spatial.py`. Figures: **#5** departure/arrival maps, **#6** net-flow map.*

## What the numbers can and cannot say

The extract is *trips touching the study area* (§K0). So a cell inside the ring
has **complete** counts — every trip starting or ending there touches the area
and is therefore in the extract — while a cell outside it has only the trips
that also touch the area. Three classes follow, and the maps encode the
difference rather than hiding it:

| class | cells in the ±{PAD_KM:.0f} km window | how it is drawn | what it means |
|---|---:|---|---|
| wholly inside the ring | {int((win.inside_frac >= COMPLETE).sum())} | solid fill | counts **complete** |
| straddles the ring | {int(((win.inside_frac > 0) & (win.inside_frac < COMPLETE)).sum())} | hatched | counts **incomplete** — the outside sliver's trips are missing |
| outside the ring | {int((win.inside_frac == 0).sum())} | faded | context only; its net flow estimates nothing |

**{len(win)} L6 cells** fall in the window; they carry **{kept:.2f}% of all trip
endpoints** in the dataset. The remaining {len(summ) - len(win)} cells lie
further out — the far ends of cross-district trips, one or two records each —
and are the subject of §E0, not of these maps.

## D1 — Departure / arrival volume

Both panels share **one log colour scale**. A linear scale is not an option
here: departures across L6 cells have a Gini of 0.969 (§B4), so on a linear ramp
each panel would render as a handful of dark cells on a flat field, and any
difference between them would be invisible.

**The two panels look nearly identical, and that is a finding, not a plotting
failure.** Departures and arrivals per cell agree closely — the gross flow
through every hotspot is close to balanced, and the imbalance is a *small
residual* on top of it. That is exactly why D2 needs a diverging scale centred
at zero (a shared sequential scale would show nothing) and why §E1 has to go to
the (cell, hour) resolution: the residual is real but it is an order of
magnitude smaller than the flow that carries it.

## D2 — Net flow (departures − arrivals)

Diverging scale, **centred at zero with symmetric limits** — with an asymmetric
scale a reader cannot tell which side dominates, which is the entire point of
the figure. Cells at the red extreme are where users will find no bike; blue
cells are where bikes accumulate and parking pressure builds.

Largest **drains** (positive net flow): {lst(top_drain)}

Largest **accumulations** (negative net flow): {lst(top_pile)}

> Note that the extreme values belong to cells that straddle or sit outside the
> ring, which is exactly the artefact the hatching warns about. The cells whose
> counts are complete are the ones in the table below, and it is those that
> should be read as the imbalance inside the service area.

### Cells with complete counts (wholly inside the ring)

| cell | departures | arrivals | net flow | share of all departures | inside fraction |
|---|---:|---:|---:|---:|---:|
{trust_rows}

## D3 — Hotspots

Top {TOP_N} L6 cells by departures, marked 1–{TOP_N} on figures #5 and #6:

| # | cell (L6) | centroid (lat, lon) | departures | arrivals | net flow |
|---:|---|---|---:|---:|---:|
{hot_rows}

**Read it as.** The hotspot list is the "gathering points" of the repositioning
problem, and the net-flow column already shows the asymmetry that §E1 resolves
in time: the busiest origin cell is also one of the largest accumulators, i.e.
its bikes come back faster than they leave across the day as a whole, while the
cell beside it drains. Aggregate net flow cancels opposite-signed hours — which
is precisely why E1 goes to the (cell, hour) resolution.
"""
    (OUT / "spatial_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
