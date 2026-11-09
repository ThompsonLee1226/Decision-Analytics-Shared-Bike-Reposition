#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise the GP-level flow imbalance produced by `flow_balance_gp.py`.

Reads the CSVs in `out/` (run flow_balance_gp.py first) and writes:

  fig10_gp_boundary_flow_daily.png   GP-cell boundary (GP <-> non-GP) per day
  fig11_gp_boundary_net_map.png      boundary net imbalance per GP, on Amap
  fig12_gp_net_map_day.png           one day's GP-to-GP net, on Amap
  fig13_gp_net_heatmap.png           every GP x every workday net flow
  fig14_gp_od_flow_map.png           top GP-to-GP flows as arrows, on Amap
  fig15_gp_net_ranked.png            ranked mean net outflow, sources/sinks
  fig16_gp_od_matrix.png             90x90 GP-to-GP OD matrix

Colour convention: red = net outflow (source), blue = net inflow (sink).
The blue polygon on the maps is the study-area ring from
`data_select_configure_zgc_20260909.xlsx`, read through EDA/region.py -- the
same ring as EDA/out/fig07_study_area.png.  All coordinates are GCJ-02 and
stay unconverted; Amap tiles are GCJ-02.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDA = ROOT / "EDA"
for p in (HERE, EDA):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import basemap as B  # noqa: E402
import gp_amap_map as G  # noqa: E402
import region as R  # noqa: E402

DPI = G.DPI
RED, BLUE = "#b2182b", "#2166ac"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_outputs(out: Path) -> dict:
    read = lambda f: pd.read_csv(out / f, encoding="utf-8-sig")
    summ = read("gp_flow_balance_summary.csv").set_index("gp")
    bsum = read("gp_boundary_summary.csv").set_index("gp")
    od = pd.read_csv(out / "gp_gp_od_mean.csv", index_col=0, encoding="utf-8-sig")
    od.columns.name = "dest"
    daily = pd.read_csv(out / "workday_morning_gp_flow_daily.csv",
                        parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    net = pd.read_csv(out / "gp_net_flow_daily.csv",
                      parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    bdaily = pd.read_csv(out / "gp_boundary_flow_daily.csv",
                         parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    return dict(summ=summ, bsum=bsum, od=od, daily=daily, net=net, bdaily=bdaily)


# ---------------------------------------------------------------------------
# Map primitives
# ---------------------------------------------------------------------------

def draw_region(ax, m: B.AmapMosaic) -> None:
    """The study-area ring from data_select_configure_zgc_20260909.xlsx.

    Same ring as EDA/out/fig07_study_area.png (EDA/region.py is the single
    source of truth).  GCJ-02, drawn unconverted on GCJ-02 Amap tiles.
    """
    ring, _, _ = R.load_boundary()
    lat = np.array([p[0] for p in ring], float)
    lon = np.array([p[1] for p in ring], float)
    x, y = m.xy(lat, lon)
    ax.add_patch(Polygon(np.column_stack([x, y]), closed=True,
                         facecolor=(0.0, 0.24, 1.0, 0.05),
                         edgecolor=(0.0, 0.24, 1.0, 0.95),
                         linewidth=2.0, zorder=3.5))


def bubble_map(ax, cax, m, gps, values, title, label_top=8, note=""):
    """GP points at GCJ centres, bubble area ~ sqrt(|net|), red = outflow."""
    x, y = m.xy(gps.lat_gcj.values, gps.lon_gcj.values)
    v = np.asarray(values, float)
    vmax = float(np.nanpercentile(np.abs(v), 96)) or 1.0
    sizes = 26 + 300 * np.sqrt(np.clip(np.abs(v), 0, vmax) / vmax)
    sc = ax.scatter(x, y, s=sizes, c=np.clip(v, -vmax, vmax),
                    cmap="RdBu_r", norm=Normalize(-vmax, vmax),
                    edgecolors="white", linewidths=0.6, zorder=10)
    for k in np.argsort(-np.abs(v))[:label_top]:
        ax.annotate(gps.index[k], (x[k], y[k]), xytext=(4, -6),
                    textcoords="offset points", fontsize=6.5, color="#222",
                    zorder=12,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    G.add_title(ax, title + (f"\n{note}" if note else ""))
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("net outflow (bikes/workday)\nred = source, blue = sink",
                 fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return sc


def new_map(own, gps, zoom, pad_m):
    """Mosaic covering both the GP points and the xlsx study-area ring."""
    ring, _, _ = R.load_boundary()
    assigned = own[own.gp.ne("UNASSIGNED")]
    lat0 = min(float(assigned.lat.min()), float(gps.lat_gcj.min()),
               min(p[0] for p in ring))
    lat1 = max(float(assigned.lat.max()), float(gps.lat_gcj.max()),
               max(p[0] for p in ring))
    lon0 = min(float(assigned.lon.min()), float(gps.lon_gcj.min()),
               min(p[1] for p in ring))
    lon1 = max(float(assigned.lon.max()), float(gps.lon_gcj.max()),
               max(p[1] for p in ring))
    mid = math.radians((lat0 + lat1) / 2)
    dlat = pad_m / G.M_PER_DEG_LAT
    dlon = pad_m / (G.M_PER_DEG_LON * math.cos(mid))
    return G.make_mosaic(lat0 - dlat, lat1 + dlat, lon0 - dlon, lon1 + dlon, zoom)


def new_bubble_figure(m: B.AmapMosaic, margin: float = 0.10):
    """Mosaic figure with a reserved right margin for the colourbar."""
    w, h = m.nx * B.TILE, m.ny * B.TILE
    fig = plt.figure(figsize=(w * (1 + margin) / DPI, h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1 / (1 + margin), 1])
    m.draw(ax, ticks=False, attribution=True)
    cax = fig.add_axes([1 / (1 + margin) + 0.030, 0.30, 0.015, 0.42])
    return fig, ax, cax


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_boundary_daily(d, out: Path) -> None:
    b = d["bdaily"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1]})
    ax.plot(b.index, b["out"], color="#d95f02", lw=1.5,
            label="outbound: GP → non-GP")
    ax.plot(b.index, b["in"], color="#1b9e77", lw=1.5,
            label="inbound: non-GP → GP")
    ax.set_ylabel("trips/day")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("GP-cell boundary flow (GP <-> non-GP) · workdays, [06:30, 10:30)")

    net = b.net_in.values
    ax2.bar(b.index, net, width=1.0,
            color=np.where(net >= 0, "#2c7fb8", "#d95f02"), alpha=0.85)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.axhline(net.mean(), color="k", ls="--", lw=1.0,
                label=f"mean {net.mean():+.1f} bikes/day")
    ax2.set_ylabel("net in − out (bikes/day)")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    txt = (f"out {b['out'].sum():,}  ·  in {b['in'].sum():,}  ·  "
           f"in/out {b['in'].sum() / b['out'].sum():.3f}\n"
           f"net in {net.mean():+.1f}/day  ·  days net in > 0: "
           f"{(net > 0).sum()}/{len(net)}")
    ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec="none", alpha=0.88))
    fig.autofmt_xdate()
    fig.tight_layout()
    G.save(fig, out / "fig10_gp_boundary_flow_daily.png")


def fig_boundary_map(d, m, gps, out: Path) -> None:
    fig, ax, cax = new_bubble_figure(m)
    draw_region(ax, m)
    bubble_map(ax, cax, m, gps, d["bsum"].net_out_day.values,
               "GP <-> non-GP net flow (mean workday)",
               note="bubble ∝ sqrt(|net|); colour/size clipped at p96 · "
                    "blue = xlsx study-area boundary")
    G.save(fig, out / "fig11_gp_boundary_net_map.png")


def pick_day(daily: pd.DataFrame) -> pd.Timestamp:
    g = daily.gross_imbalance
    return (g - g.median()).abs().idxmin()


def fig_day_map(d, m, gps, date, out: Path) -> None:
    v = d["net"].loc[date].reindex(gps.index)
    gross = d["daily"].loc[date, "gross_imbalance"]
    trips = int(d["daily"].loc[date, "trips"])
    n_src = int((v > 0).sum())
    n_sink = int((v < 0).sum())
    fig, ax, cax = new_bubble_figure(m)
    draw_region(ax, m)
    bubble_map(ax, cax, m, gps, v.values,
               f"GP-to-GP net flow · {date.date()}",
               note=f"{trips:,} GP trips · {n_src} sources / {n_sink} sinks · "
                    f"gross imbalance {gross:.0f} bikes · "
                    f"blue = xlsx study-area boundary")
    G.save(fig, out / "fig12_gp_net_map_day.png")


def fig_heatmap(d, out: Path) -> None:
    summ, net, daily = d["summ"], d["net"], d["daily"]
    order = summ.sort_values("net_day", ascending=False).index
    mat = net[order].T                       # GP x date
    vmax = float(np.nanmax(np.abs(mat.values)))
    norm = SymLogNorm(linthresh=5, vmin=-vmax, vmax=vmax, base=10)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 7], hspace=0.06)
    ax0 = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax0)

    ax0.plot(daily.index, daily.gross_imbalance, color="#333", lw=1.4)
    ax0.set_ylabel("gross\nimbalance", fontsize=9)
    ax0.grid(axis="y", alpha=0.25)
    ax0.set_title("GP-to-GP net flow dynamics · all workdays "
                  "(rows sorted by mean net outflow)", fontsize=12)

    x = mat.columns
    y = np.arange(len(order))
    im = ax.pcolormesh(x, y, mat.values, cmap="RdBu_r", norm=norm,
                       shading="nearest")
    ax.set_yticks(np.arange(0, len(order), 5))
    ax.set_yticklabels(order[::5], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("date")
    ax.set_ylabel("GP")

    sign = summ.loc[order, "net_day"].values
    for c in np.where(np.diff(np.sign(sign)) != 0)[0]:
        ax.axhline(c + 0.5, color="k", lw=0.9, ls="--")

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.010, shrink=0.85)
    cb.set_label("net outflow (bikes/workday)\nsymlog scale, red = source",
                 fontsize=9)
    fig.autofmt_xdate()
    G.save(fig, out / "fig13_gp_net_heatmap.png")


def fig_od_arrows(d, m, gps, top: int, out: Path) -> None:
    od, summ = d["od"], d["summ"]
    pairs = od.stack().rename("flow").reset_index()
    pairs = pairs[(pairs.origin != pairs.dest) & (pairs.flow > 0)]
    pairs = pairs.sort_values("flow", ascending=False).head(top)

    lat = gps.lat_gcj.to_dict()
    lon = gps.lon_gcj.to_dict()
    fmax = float(pairs.flow.max())

    fig, ax = G.new_figure(m)
    draw_region(ax, m)
    for r in pairs.itertuples(index=False):
        x1, y1 = m.xy(lat[r.origin], lon[r.origin])
        x2, y2 = m.xy(lat[r.dest], lon[r.dest])
        lw = 0.35 + 3.6 * np.sqrt(r.flow / fmax)
        ax.annotate("", xy=(float(x2), float(y2)), xytext=(float(x1), float(y1)),
                    arrowprops=dict(arrowstyle="-|>", color=(0.15, 0.15, 0.20, 0.55),
                                    lw=lw, mutation_scale=8, shrinkA=4, shrinkB=4),
                    zorder=8)
    x, y = m.xy(gps.lat_gcj.values, gps.lon_gcj.values)
    ax.scatter(x, y, s=14, c="#222", edgecolors="white", linewidths=0.4,
               zorder=9)
    top_gps = (summ.net_day.abs().sort_values(ascending=False).head(12).index)
    for gp in top_gps:
        i = gps.index.get_loc(gp)
        ax.annotate(gp, (x[i], y[i]), xytext=(4, -6), textcoords="offset points",
                    fontsize=6.3, color="#222", zorder=12,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    handles = [Line2D([0], [0], color=(0.15, 0.15, 0.20, 0.7),
                      lw=0.35 + 3.6 * np.sqrt(v / fmax), label=f"{v:.0f}")
               for v in (10, 50, 100) if v <= fmax]
    ax.legend(handles=handles, title="bikes/workday", loc="upper right",
              bbox_to_anchor=(0.985, 0.985), frameon=True, fontsize=7.5,
              title_fontsize=7.5, borderaxespad=0.6)
    G.add_title(ax, f"Top {len(pairs)} GP-to-GP flows (mean workday)\n"
                    f"{pairs.flow.sum():,.0f} bikes/day shown · "
                    f"blue = xlsx study-area boundary")
    G.save(fig, out / "fig14_gp_od_flow_map.png")


def fig_ranked(d, out: Path) -> None:
    s = d["summ"].sort_values("net_day")
    b = d["bsum"].reindex(s.index)
    fig, ax = plt.subplots(figsize=(8.5, 12))
    ax.barh(np.arange(len(s)), s.net_day,
            color=np.where(s.net_day >= 0, RED, BLUE), alpha=0.88)
    ax.scatter(b.net_out_day, np.arange(len(s)), s=8, c="k", alpha=0.45,
               zorder=3, label="boundary net out (GP <-> non-GP)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s.index, fontsize=6)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("mean net outflow (bikes/workday, symlog)")
    ax.set_title("GP net flow, ranked · red = source, blue = sink")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.99, 0.01,
            f"{int((s.net_day > 0).sum())} sources · "
            f"{int((s.net_day < 0).sum())} sinks",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    G.save(fig, out / "fig15_gp_net_ranked.png")


def fig_od_matrix(d, out: Path) -> None:
    summ, od = d["summ"], d["od"]
    order = summ.sort_values("net_day", ascending=False).index
    m = od.loc[order, order].to_numpy()
    mm = np.ma.masked_less_equal(m, 0)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mm, cmap=cmap, norm=LogNorm(vmin=0.05, vmax=max(1.0, m.max())),
                   interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(0, len(order), 5))
    ax.set_xticklabels(order[::5], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(0, len(order), 5))
    ax.set_yticklabels(order[::5], fontsize=6)
    ax.set_xlabel("destination GP")
    ax.set_ylabel("origin GP")
    ax.set_title("GP-to-GP OD matrix · mean workday (bikes/day, log scale)",
                 fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("bikes/workday", fontsize=9)
    fig.tight_layout()
    G.save(fig, out / "fig16_gp_od_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "out")
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--pad-m", type=float, default=220.0)
    ap.add_argument("--date", type=str, default=None,
                    help="YYYY-MM-DD for fig12 (default: median-imbalance day)")
    ap.add_argument("--top-flows", type=int, default=40)
    args = ap.parse_args(argv)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": DPI,
    })

    d = load_outputs(args.out_dir)
    own, gps = G.load_tables(ROOT / "xgb_bikeflow" / "data")
    gps = gps.set_index("gp").reindex(d["summ"].index)
    G.check_datum(own[own.gp.ne("UNASSIGNED")], gps.reset_index())

    print("[1/7] boundary daily flow")
    fig_boundary_daily(d, args.out_dir)

    print("[2/7] ranked GPs")
    fig_ranked(d, args.out_dir)

    print("[3/7] OD matrix")
    fig_od_matrix(d, args.out_dir)

    print("[4/7] heatmap")
    fig_heatmap(d, args.out_dir)

    print("[5/7] building basemap")
    m = new_map(own, gps, args.zoom, args.pad_m)

    print("[6/7] boundary map + one-day map")
    fig_boundary_map(d, m, gps, args.out_dir)
    date = pd.Timestamp(args.date) if args.date else pick_day(d["daily"])
    fig_day_map(d, m, gps, date, args.out_dir)

    print("[7/7] OD arrow map")
    fig_od_arrows(d, m, gps, args.top_flows, args.out_dir)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
