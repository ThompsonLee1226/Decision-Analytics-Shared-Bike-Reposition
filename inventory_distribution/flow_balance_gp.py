#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workday morning flow balance *within the GP set*.

Variant of `flow_balance.py` (which is left unchanged).

Scope
    Core is no longer all 10,473 cells; it is only the 5,630 cells assigned
    to the 90 gathering points.  Two complementary views are produced:

    * internal -- GP-to-GP trips only: both endpoints in GP cells and
      origin != destination.  Each trip contributes exactly one departure at
      its origin GP and one arrival at its destination GP, so total
      departures == total arrivals by construction.
    * boundary -- GP <-> non-GP trips crossing the GP-region edge, counted
      event-based exactly like `flow_balance.py`: outbound at `start_time`,
      inbound at `end_time`.  This is the GP-pool analogue of the original
      core-region boundary flow.

Window
    [06:30, 10:30) local time.  Both `start_time` and `end_time` must fall in
    the window; GP trips crossing either edge of the window are excluded and
    reported separately.

Workday
    Identical to `flow_balance.py`: Mon-Fri minus the eight statutory
    holidays, plus the 2026-05-09 Saturday make-up workday.  2026-05-22
    (half day) is excluded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "xgb_bikeflow" / "data"
ORDERS = ROOT / "Dataset" / "dataset_train_20260921_lsz.csv"

HOLIDAYS = pd.to_datetime([
    "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
])
MAKEUP = pd.to_datetime(["2026-05-09"])
START_MIN, END_MIN = 6 * 60 + 30, 10 * 60 + 30     # [06:30, 10:30)


def workday(d: pd.Series) -> pd.Series:
    d = d.dt.normalize()
    return ((d.dt.dayofweek < 5) & ~d.isin(HOLIDAYS)) | d.isin(MAKEUP)


def in_window(t: pd.Series) -> pd.Series:
    m = t.dt.hour * 60 + t.dt.minute
    return (m >= START_MIN) & (m < END_MIN)


def main() -> None:
    own = pd.read_csv(DATA / "hw6_grid_ownership.csv")
    gps = sorted(set(own.gp) - {"UNASSIGNED"})
    gh2gp = dict(zip(own.geohash, own.gp))

    df = pd.read_csv(ORDERS, usecols=[
        "start_time", "end_time", "start_geohash", "end_geohash"])
    for c in ("start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], format="%Y/%m/%d %H:%M")
    df["sgp"] = df.start_geohash.map(gh2gp)
    df["egp"] = df.end_geohash.map(gh2gp)

    wd = workday(df.start_time) & (df.start_time.dt.normalize() != pd.Timestamp("2026-05-22"))
    s_win, e_win = in_window(df.start_time), in_window(df.end_time)
    s_gp, e_gp = df.sgp.isin(gps), df.egp.isin(gps)

    trips = df[wd & s_win & e_win & s_gp & e_gp].copy()
    n_self = int((trips.sgp == trips.egp).sum())
    trips = trips[trips.sgp != trips.egp]
    n_all = int((df[wd & s_win & e_win].shape[0]))

    # GP-region boundary: GP <-> non-GP, event-based like flow_balance.py.
    b_out = df[wd & s_win & s_gp & ~e_gp].copy()
    b_in = df[wd & e_win & ~s_gp & e_gp].copy()
    b_out["date"] = b_out.start_time.dt.normalize()
    b_in["date"] = b_in.end_time.dt.normalize()

    # Per-day totals: every trip is one departure and one arrival, so net = 0.
    trips["date"] = trips.start_time.dt.normalize()
    days = pd.date_range(df.start_time.min().normalize(),
                         df.start_time.max().normalize(), freq="D")
    days = days[workday(pd.Series(days)).to_numpy() & (days != pd.Timestamp("2026-05-22"))]

    dep = (trips.groupby(["date", "sgp"]).size().unstack("sgp", fill_value=0)
                .reindex(index=days, columns=gps, fill_value=0))
    arr = (trips.groupby(["date", "egp"]).size().unstack("egp", fill_value=0)
                .reindex(index=days, columns=gps, fill_value=0))
    net = dep - arr                                   # source > 0, sink < 0
    net.index.name = "date"

    bdaily = pd.DataFrame({
        "out": b_out.groupby("date").size().reindex(days, fill_value=0),
        "in": b_in.groupby("date").size().reindex(days, fill_value=0),
    })
    bdaily["net_in"] = bdaily["in"] - bdaily["out"]
    bdaily["in_over_out"] = bdaily["in"] / bdaily["out"].replace(0, np.nan)

    bsum = pd.DataFrame({
        "out_day": b_out.groupby("sgp").size().reindex(gps, fill_value=0) / len(days),
        "in_day": b_in.groupby("egp").size().reindex(gps, fill_value=0) / len(days),
    })
    bsum["net_out_day"] = bsum.out_day - bsum.in_day
    bsum["in_over_out"] = bsum.in_day / bsum.out_day.replace(0, np.nan)
    bsum.index.name = "gp"

    daily = pd.DataFrame({
        "trips": dep.sum(axis=1),
        "source_gps": (net > 0).sum(axis=1),
        "sink_gps": (net < 0).sum(axis=1),
        "balanced_gps": (net == 0).sum(axis=1),
        "gross_imbalance": net.clip(lower=0).sum(axis=1),
    })
    daily["gross_over_trips"] = daily.gross_imbalance / daily.trips.replace(0, np.nan)

    summary = pd.DataFrame({
        "dep_day": dep.mean(),
        "arr_day": arr.mean(),
        "net_day": net.mean(),
        "sd_net": net.std(ddof=1),
        "in_over_out": arr.sum() / dep.sum().replace(0, np.nan),
        "days_net_out": (net > 0).sum(),
    })
    summary.index.name = "gp"
    summary["role"] = np.where(summary.net_day > 0, "source",
                       np.where(summary.net_day < 0, "sink", "balanced"))

    od = (trips.groupby(["sgp", "egp"]).size().unstack("egp", fill_value=0)
               .reindex(index=gps, columns=gps, fill_value=0) / len(days))
    od.index.name = "origin"

    print(f"workdays: {len(days)} ({days.min().date()} .. {days.max().date()})")
    print(f"GPs: {len(gps)}   GP cells: {int((own.gp != 'UNASSIGNED').sum()):,}")
    print(f"morning trips (start+end in window): {n_all:,}")
    print(f"GP -> GP trips (i != j): {len(trips):,} total, "
          f"{len(trips) / len(days):,.1f}/day, "
          f"{len(trips) / n_all:.1%} of morning trips")
    print(f"same-GP trips excluded: {n_self:,}")
    print(f"balance check: sum departures {dep.to_numpy().sum():,} == "
          f"sum arrivals {arr.to_numpy().sum():,}")
    print()
    print("GP-region boundary (GP <-> non-GP; out by start, in by end):")
    print(f"  out {bdaily['out'].sum():,} ({bdaily['out'].mean():.1f}/day), "
          f"in {bdaily['in'].sum():,} ({bdaily['in'].mean():.1f}/day), "
          f"in/out {bdaily['in'].sum() / bdaily['out'].sum():.3f}, "
          f"net in {bdaily.net_in.mean():+.1f}/day")
    print(f"  days net in > 0: {(bdaily.net_in > 0).sum()}/{len(days)}")
    print()

    n_src, n_sink = (summary.net_day > 0).sum(), (summary.net_day < 0).sum()
    print(f"per average workday: {n_src} source GPs, {n_sink} sink GPs, "
          f"{len(gps) - n_src - n_sink} balanced")
    print(f"gross imbalance: {daily.gross_imbalance.mean():.1f} bikes/day "
          f"({daily.gross_over_trips.mean():.1%} of GP trips), "
          f"sd {daily.gross_imbalance.std(ddof=1):.1f}")
    print(f"|net| per GP: median {summary.net_day.abs().median():.1f}, "
          f"p90 {summary.net_day.abs().quantile(.9):.1f}, "
          f"max {summary.net_day.abs().max():.1f}")

    print("\ntop 5 sources (mean net out/day):")
    print(summary.sort_values("net_day", ascending=False).head(5)
                 [["dep_day", "arr_day", "net_day", "in_over_out", "days_net_out"]]
                 .round(1).to_string())
    print("\ntop 5 sinks (mean net in/day):")
    print(summary.sort_values("net_day").head(5)
                 [["dep_day", "arr_day", "net_day", "in_over_out", "days_net_out"]]
                 .round(1).to_string())

    out = HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "workday_morning_gp_flow_daily.csv",
                 index_label="date", float_format="%.4f", encoding="utf-8-sig")
    net.to_csv(out / "gp_net_flow_daily.csv",
               float_format="%.4f", encoding="utf-8-sig")
    summary.to_csv(out / "gp_flow_balance_summary.csv",
                   float_format="%.4f", encoding="utf-8-sig")
    od.to_csv(out / "gp_gp_od_mean.csv", float_format="%.4f", encoding="utf-8-sig")
    bdaily.to_csv(out / "gp_boundary_flow_daily.csv",
                  index_label="date", float_format="%.4f", encoding="utf-8-sig")
    bsum.to_csv(out / "gp_boundary_summary.csv",
                float_format="%.4f", encoding="utf-8-sig")
    print(f"\nwrote {out / 'workday_morning_gp_flow_daily.csv'}")
    print(f"wrote {out / 'gp_net_flow_daily.csv'}")
    print(f"wrote {out / 'gp_flow_balance_summary.csv'}")
    print(f"wrote {out / 'gp_gp_od_mean.csv'}")
    print(f"wrote {out / 'gp_boundary_flow_daily.csv'}")
    print(f"wrote {out / 'gp_boundary_summary.csv'}")


if __name__ == "__main__":
    main()
