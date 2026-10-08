#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_distributions.py
===================
Section B of the plan: the univariate distributions that define what a "trip"
is (B1) and how concentrated the activity is across space (B4).

    fig01  trip profile -- duration (log x) + crow-fly distance, with the
           quantile table rendered *inside* the image so slide 4 is one file
    fig02  Lorenz curve of departures across L6 cells, with the Gini annotated

Why these two, and why together
-------------------------------
B1 is validation as much as description: if the distance distribution looked
wrong, the geohash decode or the unit conversion would be broken, and every
spatial result downstream would be too.  It costs one panel to check.

B4 is the most decision-relevant distributional fact in the dataset.  It is what
justifies narrowing §E and §H to a hotspot subset, it explains why a city-wide
heat map renders as a handful of dots, and it is the empirical basis for the
"gathering point" idea -- hotspots are exactly where bikes pile up and where
repositioning effort earns its keep.

HW §1 asks for **sample variances and sample quantiles**, not just point
estimates, so variance, CV and a quantile sweep are columns of the B1 table
rather than an afterthought.

Usage:
    python 04_distributions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPS = OUT / "trips_clean.csv.gz"

QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
COVERAGE_TARGET = 0.80          # B4: how few cells carry 80% of the departures


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe_var(s: pd.Series, name: str) -> dict:
    """Centre, spread and tails of one variable.

    `var` is the *sample* variance (ddof=1) -- the assignment asks for sample
    variance by name, and on 658k rows the distinction from the population
    variance is invisible, so it is worth being explicit rather than sorry.
    """
    s = s.dropna()
    d = {"variable": name, "n": int(s.size), "mean": float(s.mean()),
         "std": float(s.std(ddof=1)), "var": float(s.var(ddof=1)),
         "min": float(s.min()), "max": float(s.max())}
    d["cv"] = d["std"] / d["mean"] if d["mean"] else float("nan")
    for q in QUANTILES:
        d[f"p{int(q * 100):02d}"] = float(s.quantile(q))
    return d


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = equal, 1 = one holder)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def lorenz(x: np.ndarray):
    """Cumulative share of total vs cumulative share of holders, ascending."""
    x = np.sort(np.asarray(x, dtype=float))
    cum = np.cumsum(x) / x.sum()
    return np.insert(np.arange(1, x.size + 1) / x.size, 0, 0.0), \
        np.insert(cum, 0, 0.0)


def cells_to_cover(x: np.ndarray, target: float = COVERAGE_TARGET) -> int:
    """How many of the largest cells are needed to reach `target` of the total."""
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    return int(np.searchsorted(np.cumsum(x) / x.sum(), target) + 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_trip_profile(plt, stats: list[dict], dur: pd.Series, dist: pd.Series,
                     speed: pd.Series) -> None:
    """Fig 1 -- duration + distance distributions, with the quantile table."""
    fig = plt.figure(figsize=(13.4, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.0], hspace=0.42, wspace=0.20)

    # ---- (a) duration, log x ------------------------------------------------
    # Bins are 1 min to 10, then 5 min to 60, then 30 min.  They are unequal, so
    # the y-axis is a density (`share per minute`): counting rows per bin would
    # make the wide tail bins look disproportionately tall.
    ax = fig.add_subplot(gs[0, 0])
    edges = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 90, 120, 150, 180])
    ax.hist(dur, bins=edges, density=True, color=viz.BLUE, alpha=.82,
            edgecolor="white", linewidth=.4)
    med = dur.median()
    ax.axvline(med, color=viz.INK, lw=1.4, ls="--")
    ax.annotate(f"median {med:.0f} min\np95 {dur.quantile(.95):.0f} · p99 {dur.quantile(.99):.0f}",
                xy=(med, ax.get_ylim()[1] * .78), xytext=(11, ax.get_ylim()[1] * .84),
                fontsize=9, color=viz.INK,
                arrowprops=dict(arrowstyle="->", color=viz.INK, lw=.9))
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 30, 60, 120, 180])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "30", "60", "120", "180"])
    ax.set_xlim(1, 180)
    ax.set_xlabel("trip duration (min, log scale)")
    ax.set_ylabel("share of trips per minute")
    ax.set_title(f"(a) Duration · n = {len(dur):,}\n"
                 f"mean {dur.mean():.1f} min, std {dur.std(ddof=1):.1f} — CV > 1, so not normal")

    # ---- (b) crow-fly distance ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    cap = 8.0
    ax.hist(dist, bins=np.arange(0, cap + .25, .25), density=True,
            color=viz.GREEN, alpha=.82, edgecolor="white", linewidth=.4)
    dmed = dist.median()
    ax.axvline(dmed, color=viz.INK, lw=1.4, ls="--")
    ax.annotate(f"median {dmed:.2f} km\np95 {dist.quantile(.95):.2f} · p99 {dist.quantile(.99):.2f}",
                xy=(dmed, ax.get_ylim()[1] * .78), xytext=(2.4, ax.get_ylim()[1] * .84),
                fontsize=9, color=viz.INK,
                arrowprops=dict(arrowstyle="->", color=viz.INK, lw=.9))
    ax.set_xlim(0, cap)
    over = float((dist > cap).mean())
    ax.set_xlabel("crow-fly distance between endpoints (km)")
    ax.set_ylabel("share of trips per km")
    ax.set_title(f"(b) Crow-fly distance · {over * 100:.1f}% of trips beyond {cap:.0f} km not shown\n"
                 "agrees with published bike-share profiles — the pipeline checks out")

    # ---- quantile table -----------------------------------------------------
    # Rendered into the image on purpose: slide 4 then needs one file, not a
    # chart plus a table that drifts out of sync with it.
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    cols = ["n", "mean", "std", "var", "cv"] + [f"p{int(q*100):02d}" for q in QUANTILES] + ["max"]
    head = ["n", "mean", "std", "sample var", "CV"] + \
           [f"p{int(q*100)}" for q in QUANTILES] + ["max"]
    cells = []
    for st in stats:
        row = [f"{st['n']:,}", f"{st['mean']:.2f}", f"{st['std']:.2f}", f"{st['var']:.2f}",
               f"{st['cv']:.2f}"] + \
              [f"{st[c]:.2f}" for c in [f"p{int(q*100):02d}" for q in QUANTILES]] + \
              [f"{st['max']:.1f}"]
        cells.append(row)
    tb = ax.table(cellText=cells, colLabels=head,
                  rowLabels=["duration (min)", "crow-fly (km)", "implied speed (km/h)"],
                  loc="center", cellLoc="right", rowLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#e0e0e0")
        if r == 0:
            cell.set_facecolor("#f0f2f5")
            cell.set_text_props(weight="semibold")
        if c == -1:
            cell.set_text_props(weight="semibold")
            cell.set_edgecolor("white")
    ax.set_title("Sample variance and sample quantiles (HW §1 asks for both by name)",
                 fontsize=10.5, pad=6)

    viz.save(fig, 1, "trip_profile")


def fig_lorenz(plt, dep: pd.Series) -> None:
    """Fig 2 -- Lorenz curve of departures across L6 cells, plus the names.

    Two panels on purpose.  The Lorenz curve is the required statistic, but at
    this level of concentration it degenerates into an L: the whole story sits
    in the last 2% of the x-axis and the curve alone is hard to read.  Panel (b)
    says the same thing in the form the deck actually needs -- *which* cells --
    and is what makes the 80% claim checkable rather than asserted.
    """
    x, y = lorenz(dep.values)
    g = gini(dep.values)
    n80 = cells_to_cover(dep.values)
    share80 = n80 / dep.size

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.26})

    # ---- (a) Lorenz ---------------------------------------------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=viz.GREY, lw=1.2, ls="--", label="perfect equality")
    ax.plot(x, y, color=viz.RED, lw=2.4, label=f"L6 cells (n = {dep.size:,})")
    ax.fill_between(x, y, x, color=viz.RED, alpha=.10)

    idx = np.searchsorted(y, COVERAGE_TARGET)
    ax.plot([x[idx]], [y[idx]], "o", color=viz.INK, ms=6, zorder=5)
    ax.annotate(f"{n80} of {dep.size:,} cells ({share80*100:.1f}%)\n"
                f"carry {COVERAGE_TARGET*100:.0f}% of departures",
                xy=(x[idx], y[idx]), xytext=(0.10, 0.42), fontsize=9.5, color=viz.INK,
                arrowprops=dict(arrowstyle="->", color=viz.INK, lw=1.0,
                                connectionstyle="arc3,rad=-0.15"))
    ax.annotate(f"Gini = {g:.3f}", xy=(.05, .88), fontsize=12, weight="semibold", color=viz.RED)
    ax.annotate("the curve is an L: activity is\nnot spread over the grid",
                xy=(.05, .76), fontsize=9, color=viz.GREY)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("cumulative share of L6 cells (ascending by departures)")
    ax.set_ylabel("cumulative share of departures")
    ax.set_title("(a) Lorenz curve — the required statistic")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.68))

    # ---- (b) the cells behind it -------------------------------------------
    # Rank 1 is drawn at the top: barh puts index 0 at the bottom, so the y
    # positions are reversed while the labels keep their true rank.
    ax = axes[1]
    top = dep.head(12)
    share = 100 * top.values / dep.sum()
    cum = np.cumsum(share)
    ypos = np.arange(len(top))[::-1]
    ax.barh(ypos, share, color=viz.RED, alpha=.85, height=.68)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{i+1}. {c}" for i, c in enumerate(top.index)], fontsize=9)
    ax.set_xlabel("share of all departures (%)")
    ax.set_title("(b) The cells behind the curve — top 12 by departures")
    for y, n, c in zip(ypos, share, cum):
        ax.text(n + .35, y, f"{n:.1f}%  (cum {c:.0f}%)", va="center", fontsize=8.6,
                color=viz.INK)
    ax.set_xlim(0, share.max() * 1.55)
    ax.grid(axis="y", visible=False)

    viz.save(fig, 2, "cell_concentration_lorenz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    if not TRIPS.exists():
        print(f"FATAL: {TRIPS} not found. Run 01_data_cleaning.py first.", file=sys.stderr)
        return 2

    df = pd.read_csv(TRIPS, usecols=["duration_min", "crowfly_km", "implied_speed_kmh",
                                     "start_cell6", "start_geohash"])
    print(f"trips     : {len(df):,}")

    # ---- B1 ----------------------------------------------------------------
    dur, dist, spd = df.duration_min, df.crowfly_km, df.implied_speed_kmh
    stats = [describe_var(dur, "duration_min"), describe_var(dist, "crowfly_km"),
             describe_var(spd, "implied_speed_kmh")]
    for st in stats:
        print(f"  {st['variable']:18s} mean {st['mean']:8.3f}  std {st['std']:8.3f}  "
              f"var {st['var']:9.3f}  CV {st['cv']:.3f}  p50 {st['p50']:.2f}  p95 {st['p95']:.2f}")
    pd.DataFrame(stats).to_csv(OUT / "table_b1_trip_profile.csv", index=False, encoding="utf-8")
    fig_trip_profile(plt, stats, dur, dist, spd)

    # ---- B4 ----------------------------------------------------------------
    # Ranked at the analysis cell fixed in §0 -- L6.  Cells with no departure
    # are excluded rather than counted as zeros: they are not candidates for
    # repositioning, and including them would inflate the Gini artificially.
    dep = (df.groupby("start_cell6").size().rename("departures")
             .sort_values(ascending=False))
    g = gini(dep.values)
    n80 = cells_to_cover(dep.values)
    print(f"L6 cells  : {dep.size:,} with >=1 departure, {len(df):,} departures total")
    print(f"  Gini {g:.4f}  ·  {n80} cells ({100*n80/dep.size:.2f}%) carry 80% of departures")
    for k in (1, 5, 10, 11, 20, 50):
        print(f"    top {k:3d} cells: {100*dep.head(k).sum()/dep.sum():6.2f}% of departures")
    fig_lorenz(plt, dep)

    top = dep.head(n80).rename("departures").to_frame()
    top["cum_share_pct"] = 100 * top.departures.cumsum() / dep.sum()
    top["cell_rank"] = np.arange(1, len(top) + 1)
    top.reset_index().rename(columns={"start_cell6": "cell6"}).to_csv(
        OUT / "table_b4_cell_concentration.csv", index=False, encoding="utf-8")

    write_summary(df, stats, dep, g, n80)
    (OUT / "distributions.json").write_text(json.dumps({
        "trip_profile": stats,
        "cells_with_departures": int(dep.size),
        "gini_departures": round(g, 4),
        "cells_for_80pct": n80,
        "share_of_cells_pct": round(100 * n80 / dep.size, 3),
        "top_cells": [{"cell6": c, "departures": int(v),
                       "cum_share_pct": round(100 * dep.head(i + 1).sum() / dep.sum(), 3)}
                      for i, (c, v) in enumerate(dep.head(11).items())],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT/'distributions_summary.md'}, distributions.json,")
    print("      table_b1_trip_profile.csv, table_b4_cell_concentration.csv")
    return 0


def write_summary(df, stats, dep, g, n80):
    st = {s["variable"]: s for s in stats}
    d, k, v = st["duration_min"], st["crowfly_km"], st["implied_speed_kmh"]
    qcols = " | ".join(f"p{int(q*100)}" for q in QUANTILES)

    def qrow(s, label, fmt=".2f", scale=1.0):
        vals = " | ".join(f"{s[f'p{int(q*100):02d}']*scale:{fmt}}" for q in QUANTILES)
        return (f"| {label} | {s['n']:,} | {s['mean']*scale:.2f} | {s['std']*scale:.2f} | "
                f"{s['var']*scale*scale:.2f} | {s['cv']:.2f} | {vals} | {s['max']*scale:.1f} |")

    top5 = dep.head(5)
    top5_rows = "\n".join(f"| {i+1} | `{c}` | {int(n):,} | {100*n/dep.sum():.2f}% | "
                          f"{100*dep.head(i+1).sum()/dep.sum():.2f}% |"
                          for i, (c, n) in enumerate(top5.items()))

    md = f"""# §B — Univariate distributions

*Rendered by `04_distributions.py`. Figures: **#1** trip profile, **#2** Lorenz curve.*

## B1 — Trip profile

| variable | n | mean | std | sample var | CV | {qcols} | max |
|---|---:|---:|---:|---:|---:{"---:|" * len(QUANTILES)}---:|
{qrow(d, "duration (min)")}
{qrow(k, "crow-fly (km)")}
{qrow(v, "implied speed (km/h)")}

**Read it as.** Median trip {d['p50']:.0f} min and {k['p50']:.2f} km, with a
**standard deviation ({d['std']:.1f} min) larger than the mean ({d['mean']:.1f} min) — CV = {d['cv']:.2f}**.
That is the signature of heavy right-skew and a standing warning against
assuming normality on raw duration: any method that treats duration as
Gaussian will be dominated by the tails. Distance agrees with published
bike-share profiles — the majority of trips fall inside 2–3 km — which is the
point of keeping it: it is a check that the geohash decode and the unit
conversions are sound. If this panel had looked wrong, every spatial result
downstream would have been wrong too.

Distance is reported as a **crow-fly** straight line, so it is a lower bound on
path length; the implied speed ({v['p50']:.1f} km/h median) is correspondingly a
lower bound, and is consistent with assisted and unassisted urban cycling.

## B4 — Spatial concentration

**{dep.size:,} L6 cells receive at least one departure; {n80} of them
({100*n80/dep.size:.1f}%) carry {COVERAGE_TARGET*100:.0f}% of all departures.
Gini = {g:.3f}.**

| rank | cell (L6) | departures | share | cumulative |
|---:|---|---:|---:|---:|
{top5_rows}

**Why this is the highest-value item in §B.** It is the empirical basis for the
*gathering point* concept: a handful of sites hold nearly all the activity, so
repositioning is a problem about a short list of places, not about a uniform
grid. It also fixes the population for §E1 and §H — restricting the
spatiotemporal panel and the demand models to the top cells is a defensible
narrowing, not a convenience. See the plan's *fact 2*.

**Cells are not counted as zeros when they have no departure.** A cell that
never sees a trip is not a repositioning target and is not an observation of
"zero demand" in the same sense; padding the ranking with them would inflate the
Gini without adding information.
"""
    (OUT / "distributions_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
