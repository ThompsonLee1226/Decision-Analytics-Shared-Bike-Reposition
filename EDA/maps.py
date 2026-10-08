#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
maps.py
=======
Drawing primitives shared by the map figures (06, 07, 09).

`basemap.py` puts Amap imagery on an axis; this puts *our* geometry on top of
it -- geohash cell rectangles, the study-area ring, numbered hotspot markers --
and fixes the one encoding that has to be identical on every map, because a
reader who learns it on figure 5 will read figure 11 with it:

    solid, full colour   cell wholly inside the ring   -> counts complete
    hatched              cell straddles the ring       -> counts incomplete
    faded                cell outside the ring         -> context only

That encoding is not decorative.  The extract is "trips touching the study
area" (§K0), so a cell outside the ring has only its trips that also touch the
area; its net flow estimates nothing.  A map that colours those cells the same
way as the interior ones is quietly wrong, and the fix costs one hatch pattern.
"""

from __future__ import annotations

import numpy as np

import cells as C
import viz

COMPLETE = 0.999          # inside_frac at or above this -> counts complete

COMPLETENESS_NOTE = ("white outline = study area  ·  hatched cell = straddles the boundary, "
                     "counts incomplete  ·  faded cell = outside the ring, context only")


def completeness_class(inside_frac: float) -> str:
    """'complete' | 'straddle' | 'outside' for one cell."""
    if inside_frac >= COMPLETE:
        return "complete"
    return "straddle" if inside_frac > 0 else "outside"


def draw_cells(ax, bm, sub, values, *, cmap, norm):
    """Paint one rectangle per cell, with completeness encoded in the fill."""
    from matplotlib.patches import Polygon

    drawn = {"complete": 0, "straddle": 0, "outside": 0}
    for (_, r), v in zip(sub.iterrows(), np.asarray(values, dtype=float)):
        if not np.isfinite(v):
            continue
        kind = completeness_class(r.inside_frac)
        drawn[kind] += 1
        xy = C.rect(r.cell)
        pts = np.column_stack(bm.xy(xy[:, 0], xy[:, 1]))
        kw = dict(closed=True, linewidth=0)
        if kind == "complete":
            kw.update(facecolor=cmap(norm(v)), edgecolor="white", linewidth=.4)
        elif kind == "straddle":
            kw.update(facecolor=cmap(norm(v)), edgecolor="white", linewidth=.4,
                      hatch="///")
        else:
            # no part of the cell is inside the ring: colour it, but faded --
            # its counts are the extract's, not the cell's
            kw.update(facecolor=cmap(norm(v)), edgecolor="#b8b8b8",
                      linewidth=.4, alpha=.45)
        ax.add_patch(Polygon(pts, **kw))
    return drawn


def draw_ring(ax, bm, ring, lw: float = 2.6):
    """The study-area boundary: a white halo under a black line, always on top."""
    x, y = bm.xy([p[0] for p in ring], [p[1] for p in ring])
    xs, ys = np.r_[x, x[0]], np.r_[y, y[0]]
    ax.plot(xs, ys, color="white", lw=lw + 1.8, zorder=7, alpha=.6,
            solid_joinstyle="round")
    ax.plot(xs, ys, color=viz.INK, lw=lw, zorder=8, solid_joinstyle="round")


def draw_hotspots(ax, bm, hot, label: bool = True):
    """D3 -- numbered markers on the top cells, so 'hotspots exist' names them."""
    x, y = bm.xy(hot.lat.values, hot.lon.values)
    ax.scatter(x, y, s=190, facecolor="white", edgecolor=viz.INK,
               linewidth=1.3, zorder=9)
    if label:
        for i, (xx, yy) in enumerate(zip(x, y), start=1):
            ax.annotate(str(i), (xx, yy), ha="center", va="center",
                        fontsize=8.6, weight="bold", color=viz.INK, zorder=10)


def legend_handles():
    """The three completeness swatches, for a figure-level legend."""
    from matplotlib.patches import Patch
    return [Patch(facecolor="#d9d9d9", edgecolor="white",
                  label="counts complete (wholly inside the ring)"),
            Patch(facecolor="#d9d9d9", edgecolor="white", hatch="///",
                  label="straddles the ring — counts incomplete"),
            Patch(facecolor="#d9d9d9", edgecolor="#b8b8b8", alpha=.45,
                  label="outside the ring — context only")]


def caption(fig, text: str, y: float = -0.01):
    """One figure-level note under the axes.

    Deliberately not an axes annotation: above the axes it collides with the
    panel titles, and inside the axes it sits on the imagery.
    """
    fig.text(0.5, y, text, ha="center", va="top", fontsize=8.6, color=viz.INK,
             bbox=dict(boxstyle="round,pad=0.38", fc="#f7f7f7", ec="#dddddd"))


def window_mosaic(ring, pad_km: float, zoom: int):
    """An Amap mosaic covering the ring plus a `pad_km` collar."""
    import region as R
    la0, lo0, la1, lo1 = R.ring_bbox(ring)
    dlat = pad_km * 1000 / 110540.0
    dlon = pad_km * 1000 / (111320.0 * np.cos(np.radians((la0 + la1) / 2)))
    import basemap
    return basemap.AmapMosaic(zoom, la0 - dlat, la1 + dlat, lo0 - dlon, lo1 + dlon)
