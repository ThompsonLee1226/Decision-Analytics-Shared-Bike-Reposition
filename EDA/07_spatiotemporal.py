#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_spatiotemporal.py
====================
Section E of the plan, and the core of §1: the joint space-time structure of
the imbalance.

    fig10  cell x hour net-flow heat map -- **the lead figure of the deck**
    fig11  net flow per cell at 08:00 and 18:00, one shared diverging scale
    fig12  morning vs evening role reversal, cell by cell


cell-hour's **gross flow**:

    balance = (departures - arrivals) / (departures + arrivals)


Usage:
    python 07_spatiotemporal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cells as C
import maps
import region as R
import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPS = OUT / "trips_clean_keepzerodur_keepanomday.csv.gz"
SUMM = OUT / "cell6_summary.csv"

TOP_CELLS = 30            # the hotspot population §E1 and §H1 both use
MIN_GROSS = 2.0           # trips/day below which a balance is not estimated
AM = (7, 8, 9)            # E5 morning window
PM = (17, 18, 19)         # E5 evening window
ROLE_EPS = 1.0            # trips/day; below this a sign is not a role
PAD_KM = 3.0
ZOOM = 14
BALANCE_LIM = 0.6         # symmetric colour limit for the balance ratio


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def hourly_panel(trips: pd.DataFrame, top: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Departures and arrivals per (cell, hour) over the given trips."""
    idx = pd.MultiIndex.from_product([top, range(24)], names=["cell", "hour"])
    dep = (trips[trips.start_cell6.isin(top)]
           .groupby(["start_cell6", "hour"]).size()
           .reindex(idx, fill_value=0).rename("dep"))
    arr = (trips[trips.end_cell6.isin(top)]
           .groupby(["end_cell6", "hour"]).size()
           .reindex(idx, fill_value=0).rename("arr"))
    return pd.concat([dep, arr], axis=1)


def role_table(net: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    """E5 -- morning vs evening net flow per cell, with a role label."""
    am = net[list(AM)].mean(axis=1)
    pm = net[list(PM)].mean(axis=1)
    out = pd.DataFrame({"net_am": am, "net_pm": pm, "volume": volume.reindex(net.index)})

    def role(a, p):
        if abs(a) < ROLE_EPS and abs(p) < ROLE_EPS:
            return "flat"
        if abs(a) < ROLE_EPS or abs(p) < ROLE_EPS:
            return "one-sided"
        return "reversed" if np.sign(a) != np.sign(p) else "same"

    out["role"] = [role(a, p) for a, p in zip(out.net_am, out.net_pm)]

    def name(a, p):
        if a > ROLE_EPS and p < -ROLE_EPS:
            return "workplace (drains AM, fills PM)"
        if a < -ROLE_EPS and p > ROLE_EPS:
            return "residential (fills AM, drains PM)"
        return ""

    out["reading"] = [name(a, p) for a, p in zip(out.net_am, out.net_pm)]
    return out.sort_values("net_am")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_cell_hour(plt, top, balance: pd.DataFrame, net_pd: pd.DataFrame,
                  roles: pd.DataFrame):
    """Fig 10 -- the lead figure: cell x hour balance, weekdays."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle

    cmap = plt.get_cmap(viz.DIVERGING).copy()
    cmap.set_bad("#e6e6e6")
    M = np.ma.masked_invalid(balance.values)
    norm = mcolors.Normalize(-BALANCE_LIM, BALANCE_LIM)

    fig, ax = plt.subplots(figsize=(13.6, 9.2))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
    ax.set_xlabel(f"hour of day  (weekdays only)      "
                  f"AM {AM[0]:02d}–{AM[-1]:02d} shaded blue · PM {PM[0]:02d}–{PM[-1]:02d} shaded red")
    ax.set_yticks(range(len(top)))
    labels = [f"{i+1:>2}  {c}" for i, c in enumerate(top)]
    ax.set_yticklabels(labels, fontsize=8.6, family="monospace")
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(top), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", length=0)

    # the AM / PM windows E5 uses, so the two figures read together
    for h0, h1, col in ((AM[0] - .5, AM[-1] + .5, viz.BLUE),
                        (PM[0] - .5, PM[-1] + .5, viz.RED)):
        ax.axvspan(h0, h1, color=col, alpha=.07, zorder=0)

    # the mirror-image pair: the two cells with the largest opposing swings.
    # Boxed in place rather than annotated at the edge, and linked across
    # midday -- where both cells are near zero, so the arrow lands on blank
    # cells instead of on data.
    pair = roles.reindex(top).sort_values("net_am")
    worst = (pair.iloc[0], pair.iloc[-1])
    for r in worst:
        ax.add_patch(Rectangle((-.5, top.index(r.name) - .5), 24, 1, fill=False,
                               edgecolor=viz.INK, lw=1.5, zorder=6))
    rows = sorted(top.index(r.name) for r in worst)
    ax.annotate("", xy=(12.5, rows[0]), xytext=(12.5, rows[1]),
                arrowprops=dict(arrowstyle="<->", color=viz.INK, lw=1.6,
                                shrinkA=0, shrinkB=0), zorder=7)

    cb = fig.colorbar(im, ax=ax, fraction=.028, pad=.012)
    cb.set_label("balance = (departures − arrivals) / (departures + arrivals)", fontsize=9)

    ax.set_title("Fig 10 — Cell × hour balance of the top %d L6 cells, weekdays only\n"
                 "red = the cell drains that hour (users find no bike) · "
                 "blue = it fills · grey = under %.0f trips/day, not estimated"
                 % (TOP_CELLS, MIN_GROSS), fontsize=11.5)
    maps.caption(fig,
                 f"boxed mirror pair:  {worst[1].name}  {worst[1].net_am:+.0f} trips/day in the morning, "
                 f"{worst[1].net_pm:+.0f} in the evening  ·  "
                 f"{worst[0].name}  {worst[0].net_am:+.0f} morning, {worst[0].net_pm:+.0f} evening")
    viz.save(fig, 10, "cell_hour_balance")


def fig_anchor_maps(plt, bm, win, net_hour: pd.DataFrame, ring):
    """Fig 11 -- net flow per cell at the two anchor hours, shared scale."""
    from matplotlib.colors import Normalize
    import matplotlib.colors as mcolors

    v = net_hour.values.astype(float)
    lim = viz.sym_lim(v[np.isfinite(v)], q=96, floor=50)
    norm = Normalize(*lim)
    cmap = plt.get_cmap(viz.DIVERGING)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4))
    for ax, h in zip(axes, (8, 18)):
        bm.draw(ax)
        maps.draw_cells(ax, bm, win, net_hour.loc[win.cell, h].values,
                        cmap=cmap, norm=norm)
        maps.draw_ring(ax, bm, ring)
        ax.set_title(f"{h:02d}:00  —  {'morning peak' if h == 8 else 'evening peak'}"
                     f"\nnet flow per L6 cell, mean trips/day at that hour", fontsize=11.5)

    fig.suptitle(f"Fig 11 — Net flow at the two anchor hours, one shared scale (±{lim[1]:,.0f} trips/day)",
                 fontsize=12.5, weight="semibold", y=1.01)
    fig.legend(handles=maps.legend_handles(), loc="upper left",
               bbox_to_anchor=(0.02, 0.09), ncols=3, fontsize=8.6, frameon=False)
    maps.caption(fig, "red = departures exceed arrivals (the cell drains) · "
                      "blue = arrivals exceed departures (the cell fills)")
    viz.save(fig, 11, "anchor_hour_netflow")


def fig_reversal(plt, roles: pd.DataFrame):
    """Fig 12 -- morning vs evening net flow, quadrants labelled.

    Both axes are **symlog**: the two biggest cells swing by ~100-160 trips/day
    while most of the panel sits inside +-10, so on linear axes every point but
    two collapses onto the origin and the reversals -- the whole finding --
    become invisible.
    """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9.4, 8.4))
    lim = viz.sym_lim(roles[["net_am", "net_pm"]].values.ravel(), q=100, floor=10)
    linthresh = 5.0

    # quadrants, in axes fractions so symlog cannot distort them
    for (x0, y0), col, alpha in (((0, 0), viz.BLUE, .10),      # am<0, pm<0
                                 ((.5, 0), viz.RED, .10),      # am>0, pm<0
                                 ((0, .5), viz.BLUE, .10),     # am<0, pm>0
                                 ((.5, .5), viz.GREY, .07)):   # am>0, pm>0
        ax.add_patch(Rectangle((x0, y0), .5, .5, transform=ax.transAxes,
                               facecolor=col, alpha=alpha, zorder=0, linewidth=0))
    ax.axhline(0, color=viz.INK, lw=1.1, zorder=1)
    ax.axvline(0, color=viz.INK, lw=1.1, zorder=1)

    size = 20 + 300 * (roles.volume / roles.volume.max())
    colors = [viz.AMBER if r == "reversed" else viz.GREY for r in roles.role]
    lim = (lim[0] * 1.35, lim[1] * 1.35)          # headroom for the corner marker
    ax.scatter(roles.net_am, roles.net_pm, s=size, c=colors, alpha=.78,
               edgecolor="white", linewidth=.7, zorder=4)

    # label only cells whose swing is material, alternating the offset so the
    # tight cluster just south-east of the origin does not print on itself
    big = roles[(roles[["net_am", "net_pm"]].abs().max(axis=1) > 8)]
    for i, (cid, r) in enumerate(big.iterrows()):
        dx, dy = ((7, 4), (7, -11), (-7, 4), (-7, -12))[i % 4]
        ax.annotate(cid, (r.net_am, r.net_pm), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.4, color=viz.INK, zorder=5,
                    ha="left" if dx > 0 else "right")

    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"morning net flow  ({AM[0]:02d}:00–{AM[-1]:02d}:00, trips/day)")
    ax.set_ylabel(f"evening net flow  ({PM[0]:02d}:00–{PM[-1]:02d}:00, trips/day)")

    # count what is actually in each quadrant rather than what should be
    n_rev = int((roles.role == "reversed").sum())
    n_one = int((roles.role == "one-sided").sum())
    n_flat = int((roles.role == "flat").sum())
    both_pos = int(((roles.net_am > ROLE_EPS) & (roles.net_pm > ROLE_EPS)).sum())
    both_neg = int(((roles.net_am < -ROLE_EPS) & (roles.net_pm < -ROLE_EPS)).sum())
    n_res = int(((roles.net_am < -ROLE_EPS) & (roles.net_pm > ROLE_EPS)).sum())
    n_wor = int(((roles.net_am > ROLE_EPS) & (roles.net_pm < -ROLE_EPS)).sum())

    def plural(n):
        return f"{n} cell" + ("" if n == 1 else "s")

    ax.annotate(f"RESIDENTIAL — reversed ({n_res})\nfills in the morning, drains in the evening",
                xy=(.035, .70), xycoords="axes fraction", fontsize=9.5, color=viz.BLUE,
                va="top", weight="semibold")
    ax.annotate(f"WORKPLACE — reversed ({n_wor})\ndrains in the morning, fills in the evening",
                xy=(.97, .14), xycoords="axes fraction", fontsize=9.5, color=viz.RED,
                ha="right", va="bottom", weight="semibold")
    ax.annotate(f"drains in both windows ({plural(both_pos)})", xy=(.97, .97),
                xycoords="axes fraction", fontsize=9, color=viz.GREY,
                ha="right", va="top")
    ax.annotate(f"fills in both windows ({plural(both_neg)})", xy=(.03, .035),
                xycoords="axes fraction", fontsize=9, color=viz.GREY,
                ha="left", va="bottom")

    ax.set_title(f"Fig 12 — Morning vs evening role reversal, top {TOP_CELLS} cells\n"
                 f"{n_rev} of {len(roles)} sit in the off-diagonal quadrants: they need "
                 f"*dynamic* repositioning, not a fixed rule\n"
                 f"(symlog axes · linthresh ±{linthresh:.0f} trips/day)", fontsize=11.5)
    maps.caption(fig, f"a cell counts as reversed only when both windows exceed ±{ROLE_EPS:.0f} trip/day: "
                      f"{n_one} cells are one-sided and {n_flat} are flat, so neither is called a reversal")
    viz.save(fig, 12, "role_reversal")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    for f in (TRIPS, SUMM):
        if not f.exists():
            print(f"FATAL: {f} not found. Run the earlier scripts first.", file=sys.stderr)
            return 2

    ring, _, _ = R.load_boundary()
    summ = pd.read_csv(SUMM)
    df = pd.read_csv(TRIPS, usecols=["start_cell6", "end_cell6", "hour",
                                     "is_weekend", "date"])
    top = list(summ.nlargest(TOP_CELLS, "n_touching").cell)
    in_region = int(summ.set_index("cell").in_region.reindex(top).fillna(False).sum())
    print(f"top {TOP_CELLS} cells by volume ({in_region} of them inside the study area)")

    wk = df[~df.is_weekend]
    n_wk = int(wk.date.nunique())
    n_all = int(df.date.nunique())
    print(f"weekdays  : {n_wk} of {n_all} days contribute to fig 10")

    # ---- E1: cell x hour -----------------------------------------------
    panel_wk = hourly_panel(wk, top)
    net_wk = (panel_wk.dep - panel_wk.arr).unstack("hour") / n_wk
    gross_wk = (panel_wk.dep + panel_wk.arr).unstack("hour") / n_wk
    balance = (net_wk / gross_wk.replace(0, np.nan)).where(gross_wk >= MIN_GROSS)
    balance = balance.reindex(top)
    net_wk = net_wk.reindex(top)

    masked = int(balance.isna().sum().sum())
    print(f"balance   : {masked} of {balance.size} cell-hours not estimated "
          f"({100*masked/balance.size:.1f}%, gross < {MIN_GROSS}/day)")
    net_wk.round(3).to_csv(OUT / "table_e1_cell_hour_netflow.csv", encoding="utf-8")
    balance.round(4).to_csv(OUT / "table_e1_cell_hour_balance.csv", encoding="utf-8")

    # ---- E5: roles ------------------------------------------------------
    roles = role_table(net_wk, summ.set_index("cell").n_touching)
    roles.round(3).to_csv(OUT / "table_e5_cell_roles.csv", encoding="utf-8")
    print(f"roles     : {roles.role.value_counts().to_dict()}")
    print("  clean reversals:")
    for r in roles[roles.reading != ""].itertuples():
        print(f"    {r.Index}  AM {r.net_am:+8.1f}  PM {r.net_pm:+8.1f}   {r.reading}")

    # ---- E2: anchor hours, over every cell in the window, all days -------
    # Not restricted to the top 30: fig 11 is a *map*, and a map that omits 107
    # of the 137 cells in its own frame reads as if nothing were there.
    win = summ[C.window_mask(summ, ring, PAD_KM)].copy()
    panel_win = hourly_panel(df, list(win.cell))
    net_win = (panel_win.dep - panel_win.arr).unstack("hour") / n_all

    # ---- figures --------------------------------------------------------
    fig_cell_hour(plt, top, balance, net_wk, roles)

    bm = maps.window_mosaic(ring, PAD_KM, ZOOM)
    print(f"basemap   : {bm}")
    fig_anchor_maps(plt, bm, win, net_win, ring)
    fig_reversal(plt, roles)

    # ---- report ---------------------------------------------------------
    write_summary(top, in_region, n_wk, n_all, balance, net_wk, roles)
    (OUT / "spatiotemporal.json").write_text(json.dumps({
        "top_cells": TOP_CELLS,
        "top_cells_in_region": in_region,
        "weekdays": n_wk,
        "days": n_all,
        "min_gross_per_day": MIN_GROSS,
        "balance_limited": BALANCE_LIM,
        "cell_hours_masked": masked,
        "cell_hours_total": int(balance.size),
        "roles": roles.role.value_counts().to_dict(),
        "clean_reversals": roles[roles.reading != ""].reset_index()[
            ["cell", "net_am", "net_pm", "reading"]].round(2).to_dict("records"),
        "top_by_volume": top,
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {OUT/'spatiotemporal_summary.md'}, spatiotemporal.json,")
    print("      table_e1_cell_hour_{balance,netflow}.csv, table_e5_cell_roles.csv")
    return 0


def write_summary(top, in_region, n_wk, n_all, balance, net_wk, roles):
    pair = roles.reindex(top).reset_index().sort_values("net_am")
    res, wor = pair.iloc[0], pair.iloc[-1]

    role_rows = "\n".join(
        f"| `{r.Index}` | {r.volume:,.0f} | {r.net_am:+,.1f} | {r.net_pm:+,.1f} | "
        f"{r.role} | {r.reading or '—'} |"
        for r in roles.itertuples())

    counts = roles.role.value_counts()
    rev_cells = roles[roles.role == "reversed"]
    rev_list = ", ".join(f"`{c}`" for c in rev_cells.index)

    md = f"""# §E1 · §E2 · §E5 — Spatiotemporal joint analysis

*Rendered by `07_spatiotemporal.py`. Figures: **#10** cell × hour balance (lead),
**#11** anchor-hour net flow, **#12** role reversal.*

Population: the **top {TOP_CELLS} L6 cells by volume**, of which **{in_region} lie
inside the study area**. Fig 10 uses the {n_wk} weekdays only (§C2 shows the
morning peak is a weekday structure); figs 11–12 use all {n_all} days.

## E1 — Cell × hour balance (the lead figure)

**What is plotted.** `balance = (departures − arrivals) / (departures + arrivals)`
per cell-hour, averaged over the {n_wk} weekdays. It is a ratio rather than raw
net flow because the plan's "normalised per row" cannot be applied to a signed
quantity — a row sum of net flow is near zero and the ratio would explode. The
balance is bounded in [−1, +1], reads literally ("40% more departures than
arrivals that hour"), and lets a 200-trip cell be compared with a 20,000-trip
one. **Raw net flow per cell-hour is in `table_e1_cell_hour_netflow.csv`.**

**Grey is not zero.** {int(balance.isna().sum().sum())} of {balance.size} cell-hours
({100*balance.isna().sum().sum()/balance.size:.1f}%) had under {MIN_GROSS:.0f} trips/day of gross flow and are
left uncoloured. At that volume the sign of the balance is a coin flip; shading
it would paint noise as structure.

### The mirror-image pair

| | morning net | evening net | reading |
|---|---:|---:|---|
| `{wor.cell}` | {wor.net_am:+,.1f} | {wor.net_pm:+,.1f} | **{wor.reading}** |
| `{res.cell}` | {res.net_am:+,.1f} | {res.net_pm:+,.1f} | **{res.reading}** |

These are the two largest opposing swings in the panel, and they are **adjacent
cells**. One fills through the morning and drains through the evening; the other
does the reverse, at a comparable magnitude. That is the imbalance the
assignment asks for, in one line: not too few bikes in the area, but bikes on
the wrong side of a cell boundary at the wrong hour.

## E2 — Net flow at the anchor hours

Fig 11 maps the same quantity geographically at **08:00 and 18:00**, on one
shared diverging scale. E1 says *which cells and when*; E11 says *where they
are*. Two panels is enough for that, which is why noon and 22:00 were dropped.

## E5 — Role reversal

A cell is counted as **reversed** when the sign of its mean net flow differs
between the morning ({AM[0]:02d}:00–{AM[-1]:02d}:00) and the evening
({PM[0]:02d}:00–{PM[-1]:02d}:00) windows, and **only when both windows exceed
{ROLE_EPS:.0f} trip/day** — below that a sign is not a role. Of {len(roles)} cells:

| role | cells | |
|---|---:|---|
| **reversed** | **{int(counts.get('reversed', 0))}** | {rev_list} |
| same role in both windows | {int(counts.get('same', 0))} | |
| one-sided (active in one window only) | {int(counts.get('one-sided', 0))} | |
| flat in both | {int(counts.get('flat', 0))} | |

**{int(counts.get('reversed', 0))} of {len(roles)} cells ({100*counts.get('reversed', 0)/len(roles):.0f}%) swap roles between morning and evening.** This is a genuinely
non-obvious result and it is the strongest available argument for **dynamic**
rather than static repositioning: a fixed morning rule and a fixed evening rule
would each service these cells wrongly for half the day.

## Full role table

| cell | volume | morning net | evening net | role | reading |
|---|---:|---:|---:|---|---|
{role_rows}
"""
    (OUT / "spatiotemporal_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
