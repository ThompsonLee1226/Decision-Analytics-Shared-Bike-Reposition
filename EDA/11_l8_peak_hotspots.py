#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_l8_peak_hotspots.py
======================
Top-10 **3 x 3 neighbourhoods** for weekday peak net flow, at L8 resolution.

Two selection modes are produced in one run:

    fig18  centres chosen by **single-cell** net flow
    fig19  centres chosen by **3 x 3 block** net flow, over every active
           in-region cell as a candidate centre

Each figure has two maps (AM, PM); each map shows the top-10 net-inflow
neighbourhoods (blue) and the top-10 net-outflow neighbourhoods (red)
together, on the Amap basemap.

    net inflow = arrivals - departures

    positive (blue)  the neighbourhood accumulates bikes
    negative (red)   the neighbourhood drains bikes

Neighbourhood rule
------------------
Every candidate is expanded to the 3 x 3 block centred on it.  The displayed
value and the share are computed over that nine-cell block, not over the
centre cell alone.  The selection requires the ten 3 x 3 blocks to be
**pairwise disjoint**: adjacent centres would otherwise show ten nearly
identical copies of the same hotspot and the shares would double-count the
same cells.

The block-ranked mode (fig19) answers a different question from the
single-cell mode: which nine-cell areas have the largest aggregate imbalance,
regardless of whether their centre is itself the busiest cell.

Peak windows are three hours: AM = 07–09 (07:00–09:59), PM = 17–19
(17:00–19:59).  All values are **average net trips per weekday** in that
window, not 60-day totals.

Share denominators:

    net inflow  / sum of positive cell net flows in the service area
    net outflow / sum of |negative cell net flows| in the service area

The companion table also reports the share of service-wide **gross** flow
(departures + arrivals) for each neighbourhood.

Usage:
    python 11_l8_peak_hotspots.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cells as C
import l8
import maps
import region as R
import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPS = OUT / "trips_clean_keepzerodur_keepanomday.csv.gz"

TOP_N = 10
NON_OVERLAP = True
PAD_KM = 0.4
ZOOM = 16
CONTEXT = "#8f8f8f"
DIRECTIONS = ("inflow", "outflow")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_context(ax, bm, sub):
    """All active in-region L8 cells as fine dots, so the basemap stays visible."""
    x, y = bm.xy(sub["lat"].to_numpy(), sub["lon"].to_numpy())
    ax.scatter(x, y, s=1.2, c=CONTEXT, alpha=.50, linewidths=0, zorder=1)


def annotate_ranks(ax, x, y, labels, color):
    """Numbered bubbles with simple collision avoidance.

    The axes are in mosaic pixels, so a distance test in data units is a
    reasonable stand-in for screen distance.  Each label is pushed to the
    first candidate offset that is clear of every marker and every label
    already placed; a thin leader line ties it back to its cell.
    """
    pts = [(float(a), float(b)) for a, b in zip(x, y)]
    placed: list[tuple[float, float]] = []

    offsets = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
               for r in (34, 46, 60, 76, 94)
               for a in range(0, 360, 30)]

    for (xi, yi), label in zip(pts, labels):
        chosen = offsets[-1]
        for dx, dy in offsets:
            px, py = xi + dx, yi + dy
            if all((px - qx) ** 2 + (py - qy) ** 2 >= 34 ** 2 for qx, qy in pts):
                if all((px - qx) ** 2 + (py - qy) ** 2 >= 46 ** 2
                       for qx, qy in placed):
                    chosen = (dx, dy)
                    break
        px, py = xi + chosen[0], yi + chosen[1]
        placed.append((px, py))
        ax.annotate(
            label, xy=(xi, yi), xytext=(px, py), textcoords="data",
            ha="center", va="center", fontsize=5.9, weight="bold",
            color="white", zorder=12,
            bbox=dict(boxstyle="circle,pad=0.19", fc=color, ec="white", lw=.8),
            arrowprops=dict(arrowstyle="-", color="#4d4d4d", lw=.6,
                            shrinkA=1, shrinkB=6, zorder=11))


def draw_top(ax, bm, top, peak: str, direction: str, size_max: float):
    """Top-10 markers: size ∝ √(3x3 net), colour encodes the direction."""
    x, y = bm.xy(top["lat"].to_numpy(), top["lon"].to_numpy())
    magnitude = np.abs(top["block_net_per_day"].to_numpy(dtype=float))
    sizes = 60 + 240 * np.sqrt(magnitude / size_max)
    color = viz.BLUE if direction == "inflow" else viz.RED
    prefix = "I" if direction == "inflow" else "O"

    # Outline the nominal 3 x 3 block so the map shows the aggregation unit,
    # not just a point.  At the ring edge the block is clipped for the metric;
    # the outline is still the nominal nine-cell footprint.
    dlat, dlon = C.cell_size(8)
    for r in top.itertuples():
        la0, lo0, la1, lo1 = C.bounds(r.cell)
        lats = np.array([la0 - dlat, la1 + dlat, la1 + dlat,
                         la0 - dlat, la0 - dlat])
        lons = np.array([lo0 - dlon, lo0 - dlon, lo1 + dlon,
                         lo1 + dlon, lo0 - dlon])
        bx, by = bm.xy(lats, lons)
        ax.plot(bx, by, color=color, lw=1.1, alpha=.78, zorder=6)

    ax.scatter(x, y, s=sizes, facecolor=color, edgecolor="white",
               linewidth=1.0, alpha=.95, zorder=9)
    labels = [f"{prefix}{r}" for r in top["rank"]]
    annotate_ranks(ax, x, y, labels, color)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_top10(plt, bm, sub, tops, ring, *, fig_n: int, slug: str,
              rank_by: str):
    """One figure: AM and PM maps, each with both directions."""
    from matplotlib.lines import Line2D

    selection = ("single-cell net flow" if rank_by == "single"
                 else "3×3 block net across all cell centres")
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.8))
    size_max = max(float(t["block_net_per_day"].abs().max())
                   for t in tops.values())

    for ax, peak in zip(axes, ("am", "pm")):
        bm.draw(ax, attribution=True, ticks=True)
        draw_context(ax, bm, sub)
        maps.draw_ring(ax, bm, ring)

        for direction in DIRECTIONS:
            draw_top(ax, bm, tops[(peak, direction)], peak, direction, size_max)

        lin = tops[(peak, "inflow")].iloc[0]
        lout = tops[(peak, "outflow")].iloc[0]
        ax.set_title(
            f"{l8.PEAK_LABELS[peak]}  ·  3×3 neighbourhood net flow\n"
            f"I1 {lin['cell']}  {lin['block_net_per_day']:+.1f}/day  "
            f"({lin['share_net_pct']:.1f}% net, {lin['share_gross_pct']:.1f}% gross)\n"
            f"O1 {lout['cell']}  {lout['block_net_per_day']:+.1f}/day  "
            f"({lout['share_net_pct']:.1f}% net, {lout['share_gross_pct']:.1f}% gross)",
            fontsize=10.4)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=viz.BLUE, markeredgecolor="white",
               label="net inflow top10 (I#), 3×3 block"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=viz.RED, markeredgecolor="white",
               label="net outflow top10 (O#), 3×3 block"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.055),
               ncols=2, fontsize=8.8, frameon=False)

    fig.suptitle(
        f"Fig {fig_n} — Top-10 L8 3×3 net-flow neighbourhoods, weekday peaks\n"
        f"centres selected by {selection}; blue = net inflow · red = net outflow",
        fontsize=13.2, weight="semibold", y=1.005)
    maps.caption(
        fig,
        "Marker = centre L8 cell of a 3×3 neighbourhood; outline = the nominal "
        "nine-cell block (clipped to in-region cells for the metric).  Top10 "
        f"selected by {selection}, with pairwise-disjoint neighbourhoods; "
        "labels rank the 3×3 net.  Marker area ∝ √(3×3 net trips/weekday).\n"
        "Values are average net trips per weekday in the three-hour peak; "
        "% net = share of service-wide net inflow (blue) or net outflow (red), "
        "% gross = share of total service departures + arrivals.",
        y=-0.005)
    viz.save(fig, fig_n, slug)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_tops(agg: pd.DataFrame, rank_by: str = "single") -> tuple[dict, dict]:
    """Top-10 neighbourhoods and the service-wide share denominators.

    `rank_by="single"` chooses candidate centres by single-cell net flow;
    `rank_by="block"` considers every active in-region cell as a centre and
    ranks the 3 x 3 block aggregate itself.
    """
    sub = agg.loc[agg["in_region"]]
    n_days = float(sub["weekdays"].iloc[0])
    service = {}
    for peak in l8.PEAKS:
        net = sub[f"net_{peak}_per_day"]
        service[peak] = {
            "net_in": float(net[net > 0].sum()),
            "net_out": float(-net[net < 0].sum()),
            "gross": float(sub[f"gross_{peak}"].sum() / n_days),
        }

    tops = {}
    for peak in l8.PEAKS:
        net_col = f"net_{peak}_per_day"
        for direction in DIRECTIONS:
            ascending = direction == "outflow"
            order = sub.sort_values(net_col, ascending=ascending, kind="stable")
            single_rank_map = {c: i + 1 for i, c in enumerate(order["cell"])}
            if rank_by == "single":
                ranked = l8.top_net(agg, peak, direction, TOP_N,
                                    inside_only=True, non_overlap=NON_OVERLAP)
            elif rank_by == "block":
                ranked = l8.top_blocks(agg, peak, direction, TOP_N,
                                       inside_only=True, non_overlap=NON_OVERLAP)
            else:
                raise ValueError(f"rank_by must be single/block, got {rank_by!r}")

            rows = []
            center_net = dict(zip(sub["cell"], sub[net_col]))
            for r in ranked.itertuples():
                b = l8.block_stats(agg, r.cell, peak, direction=direction)
                if direction == "inflow":
                    share_net = 100 * b["net_per_day"] / service[peak]["net_in"]
                else:
                    share_net = 100 * (-b["net_per_day"]) / service[peak]["net_out"]
                rows.append({
                    "peak": peak,
                    "direction": direction,
                    "single_rank": single_rank_map.get(r.cell),
                    "cell": r.cell,
                    "lat": r.lat,
                    "lon": r.lon,
                    "center_net_per_day": center_net.get(r.cell, np.nan),
                    "dominant_cell": b["dominant_cell"],
                    "dominant_net_per_day": b["dominant_net_per_day"],
                    "block_dep_per_day": b["dep_per_day"],
                    "block_arr_per_day": b["arr_per_day"],
                    "block_net_per_day": b["net_per_day"],
                    "block_net_total": b["net"],
                    "share_net_pct": share_net,
                    "share_gross_pct": 100 * abs(b["net_per_day"]) / service[peak]["gross"],
                    "n_active_cells": b["n_active_cells"],
                })
            df = pd.DataFrame(rows).sort_values(
                "block_net_per_day", ascending=(direction == "outflow")
            ).reset_index(drop=True)
            df["rank"] = np.arange(1, len(df) + 1)
            tops[(peak, direction)] = df
    return tops, service


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(tops, service, n_days: int, *, prefix: str, rank_by: str):
    fig_no = 18 if rank_by == "single" else 19
    table = pd.concat(tops.values(), ignore_index=True)
    cols = ["peak", "direction", "rank", "single_rank", "cell", "lat", "lon",
            "dominant_cell", "center_net_per_day", "dominant_net_per_day",
            "block_dep_per_day", "block_arr_per_day", "block_net_per_day",
            "block_net_total", "share_net_pct", "share_gross_pct",
            "n_active_cells"]
    table[cols].round(4).to_csv(OUT / f"table_{prefix}.csv",
                                index=False, encoding="utf-8")

    records = {
        f"{peak}_{direction}": top.to_dict("records")
        for (peak, direction), top in tops.items()
    }
    (OUT / f"{prefix}.json").write_text(
        json.dumps({
            "weekdays": n_days,
            "top_n": TOP_N,
            "rank_by": rank_by,
            "topology": "3x3 L8 neighbourhood centred on each candidate cell",
            "non_overlapping": NON_OVERLAP,
            "peak_hours": {p: list(h) for p, h in l8.PEAKS.items()},
            "service_per_weekday": service,
            "top10": records,
        }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    def table_for(peak, direction):
        t = tops[(peak, direction)]
        rows = "\n".join(
            f"| {r.rank} | {r.single_rank} | `{r.cell}` | `{r.dominant_cell}` | "
            f"{r.block_dep_per_day:.1f} | {r.block_arr_per_day:.1f} | "
            f"{r.block_net_per_day:+.1f} | {r.share_net_pct:.1f}% | "
            f"{r.share_gross_pct:.1f}% | {r.n_active_cells} |"
            for r in t.itertuples())
        return (f"| # | single rank | centre cell | dominant cell | "
                f"3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | "
                f"share of gross | active cells |\n"
                f"|---:|---:|---|---|---:|---:|---:|---:|---:|---:|\n{rows}")

    sections = "\n\n".join(
        f"### {l8.PEAK_LABELS[p]} · net {d}\n\n{table_for(p, d)}"
        for p in l8.PEAKS for d in DIRECTIONS)

    lead = {k: tops[k].iloc[0] for k in tops}
    am_out = lead[("am", "outflow")]
    am_in = lead[("am", "inflow")]
    pm_in = lead[("pm", "inflow")]
    pm_out = lead[("pm", "outflow")]

    service_rows = "\n".join(
        f"| {l8.PEAK_LABELS[p]} | {service[p]['net_in']:,.1f} | "
        f"{service[p]['net_out']:,.1f} | {service[p]['gross']:,.1f} |"
        for p in l8.PEAKS)

    lead_rows = "\n".join(
        f"| {l8.PEAK_LABELS[p]} · net {d} | `{lead[(p, d)]['cell']}` | "
        f"`{lead[(p, d)]['dominant_cell']}` | "
        f"{lead[(p, d)]['block_net_per_day']:+.1f} | "
        f"{lead[(p, d)]['share_net_pct']:.1f}% | "
        f"{lead[(p, d)]['share_gross_pct']:.1f}% |"
        for p in l8.PEAKS for d in DIRECTIONS)

    overlap_text = (
        "The ten selected neighbourhoods are **pairwise disjoint**, so their "
        "shares do not double-count the same cells; they still cover only "
        "part of the service area, so the shares need not sum to 100%."
        if NON_OVERLAP else
        "Neighbourhoods may overlap; the shares are individually valid but "
        "are not additive."
    )

    if rank_by == "single":
        rank_step = (
            "rank cells by **single-cell** net flow — descending for inflow, "
            "ascending for outflow — and expand each selected cell to the "
            "**3 × 3 L8 block** centred on it"
        )
        centre_text = (
            "The centre cell is the \"top grid\"; the reported value is the "
            "**net flow of the nine-cell block**, not of the centre alone."
        )
        comparison = (
            "The 3×3 aggregation strengthens the single-cell result: the "
            "surrounding cells belong to the same hotspot, not to its "
            "neighbours."
        )
    else:
        rank_step = (
            "compute the **3 × 3 block net** for **every active in-region L8 "
            "cell as a centre**, then rank those blocks — descending for "
            "inflow, ascending for outflow"
        )
        centre_text = (
            "The centre is the candidate cell that produced the block; the "
            "reported value is the **net flow of the nine-cell block**. "
            "Because the block is ranked directly, its centre need not be its "
            "dominant cell — the `dominant cell` column names the cell inside "
            "the block with the largest inflow or outflow."
        )
        comparison = (
            "The block ranking can promote a centre that is not itself the "
            "busiest cell, because its surrounding cells carry the imbalance. "
            "This is the intended difference from the single-cell ranking."
        )

    narrative = (
        f"The AM net-outflow leader is the block centred on "
        f"`{am_out['cell']}` (dominant cell `{am_out['dominant_cell']}`, "
        f"{am_out['block_net_per_day']:+.1f}/weekday, "
        f"{am_out['share_net_pct']:.1f}% of service net outflow). The PM "
        f"net-inflow leader is the block centred on `{pm_in['cell']}` "
        f"(dominant cell `{pm_in['dominant_cell']}`, "
        f"{pm_in['block_net_per_day']:+.1f}/weekday, "
        f"{pm_in['share_net_pct']:.1f}% of service net inflow)."
    )
    if am_out["dominant_cell"] == pm_in["dominant_cell"]:
        narrative += (
            f" Both blocks have `{am_out['dominant_cell']}` as their dominant "
            "cell, which is the L8-resolution role reversal in one cell."
        )

    md = f"""# L8 peak net-flow neighbourhoods (fig {fig_no})

*Rendered by `11_l8_peak_hotspots.py`.*

## Method

For each three-hour peak (AM 07:00–09:59, PM 17:00–19:59):

1. accumulate departures and arrivals over the {n_days} weekdays and form the
   per-cell signed net inflow `arrivals − departures`, expressed as
   **trips per weekday**;
2. {rank_step};
3. keep ten blocks, requiring them to be pairwise disjoint.

{centre_text} The table's `#` column ranks the selected blocks by that 3×3
net; `single rank` is the cell's position in the single-cell ranking, for
reference. {overlap_text}

`active cells` counts the in-region L8 cells with any weekday activity in the
block (maximum 9). An inactive cell contributes zero, and blocks on the ring
edge are clipped to their in-region part, so this column can be below 9.

## Service denominators (per weekday)

| peak | sum of positive cell net | sum of abs(negative cell net) | gross flow |
|---|---:|---:|---:|
{service_rows}

`share of net` is the block net divided by the matching column (positive net
for inflow blocks, |negative net| for outflow blocks). `share of gross` is
|block net| divided by the service gross flow; it is a smaller, more
conservative number.

## Leaders

| panel | #1 centre | dominant cell | 3×3 net/weekday | share of net | share of gross |
|---|---|---|---:|---:|---:|
{lead_rows}

{narrative}

{comparison}

The non-overlap constraint matters here. `wx4eqtku` and `wx4eqtkv`, for
example, are adjacent cells whose 3×3 blocks cover almost the same ground;
keeping both would report the same core twice. The selected top10 therefore
reaches farther down the candidate ranking to find genuinely distinct areas.

## Top 10, panel by panel

{sections}
"""
    (OUT / f"{prefix}_summary.md").write_text(md, encoding="utf-8")


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

    print(f"weekdays : {n_days}")
    bm = maps.window_mosaic(ring, PAD_KM, ZOOM)
    print(f"basemap  : {bm}")

    modes = [
        ("single", 18, "l8_peak_netflow_top10_3x3"),
        ("block", 19, "l8_block_netflow_top10_3x3"),
    ]
    for rank_by, fig_n, prefix in modes:
        tops, service = build_tops(agg, rank_by=rank_by)
        print(f"\nrank by {rank_by}:")
        for peak in l8.PEAKS:
            for direction in DIRECTIONS:
                t = tops[(peak, direction)]
                lead = t.iloc[0]
                print(f"  {peak:>2} {direction:>7}: #1 centre {lead['cell']} "
                      f"(dominant {lead['dominant_cell']}) "
                      f"3x3 {lead['block_net_per_day']:+.1f}/weekday "
                      f"({lead['share_net_pct']:.1f}% of net, "
                      f"{lead['share_gross_pct']:.1f}% of gross)")

        fig_top10(plt, bm, sub, tops, ring,
                  fig_n=fig_n, slug=prefix, rank_by=rank_by)
        write_report(tops, service, n_days, prefix=prefix, rank_by=rank_by)
        print(f"  wrote {prefix}_summary.md, {prefix}.json,")
        print(f"        table_{prefix}.csv")

    return 0


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
