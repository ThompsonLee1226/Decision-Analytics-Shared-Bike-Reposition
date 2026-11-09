#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workday morning boundary-flow balance of the core area.

Core area
    Every geohash in `xgb_bikeflow/data/hw6_grid_ownership.csv` -- all 10,473
    rows, including the 4,843 `UNASSIGNED` cells.  The file is the region, not
    just the GP-owned part.

Flows (boundary crossings only)
    outbound = start in core, end outside, counted at `start_time`
    inbound  = start outside, end in core, counted at `end_time`
    Window: [06:30, 10:30) local time, half-open like the forecasting panel.

Workday
    Mon-Fri, dropping the eight statutory holidays in the period and adding the
    2026-05-09 (Saturday) make-up workday.  This is the same rule as
    `xgb_bikeflow`'s `days: workday`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    d = pd.to_datetime(d)
    return ((d.dt.dayofweek < 5) & ~d.isin(HOLIDAYS)) | d.isin(MAKEUP)


def in_window(t: pd.Series) -> pd.Series:
    m = t.dt.hour * 60 + t.dt.minute
    return (m >= START_MIN) & (m < END_MIN)


def main() -> None:
    core = set(pd.read_csv(DATA / "hw6_grid_ownership.csv", usecols=["geohash"]).geohash)
    df = pd.read_csv(ORDERS, usecols=[
        "start_time", "end_time", "start_geohash", "end_geohash"])
    for c in ("start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], format="%Y/%m/%d %H:%M")

    s_in = df.start_geohash.isin(core)
    e_in = df.end_geohash.isin(core)
    print("endpoint membership (all trips):")
    print(pd.crosstab(s_in, e_in, rownames=["start_in_core"],
                      colnames=["end_in_core"]).to_string())
    print()

    # Boundary crossings, timestamped by the event that crosses the boundary.
    out = df[s_in & ~e_in & in_window(df.start_time)].copy()
    inn = df[~s_in & e_in & in_window(df.end_time)].copy()
    out["date"] = out.start_time.dt.normalize()
    inn["date"] = inn.end_time.dt.normalize()
    out = out[workday(out.date)]
    inn = inn[workday(inn.date)]

    days = pd.date_range(df.start_time.min().normalize(),
                         df.start_time.max().normalize(), freq="D")
    days = days[workday(pd.Series(days)).to_numpy()]
    flow = pd.DataFrame({
        "out": out.groupby("date").size(),
        "in": inn.groupby("date").size(),
    }).reindex(days).fillna(0).astype(int)
    flow["net_in"] = flow["in"] - flow["out"]
    flow["in_over_out"] = flow["in"] / flow["out"].replace(0, np.nan)

    n = len(flow)
    diff = flow.net_in
    ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
    t = stats.ttest_rel(flow["in"], flow["out"])

    print(f"workdays: {n} ({flow.index.min().date()} .. {flow.index.max().date()})")
    print(f"outbound: {flow.out.sum():,}")
    print(f"inbound : {flow['in'].sum():,}")
    print(f"in/out  : {flow['in'].sum() / flow.out.sum():.3f}")
    print(f"net inbound: {diff.sum():,} total, {diff.mean():.1f}/day "
          f"(sd {diff.std(ddof=1):.1f}), 95% CI "
          f"[{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"days inbound > outbound: {(diff > 0).sum()}/{n}")
    print(f"daily in/out: mean {flow.in_over_out.mean():.3f}, "
          f"sd {flow.in_over_out.std(ddof=1):.3f}, "
          f"range {flow.in_over_out.min():.3f}..{flow.in_over_out.max():.3f}")
    print(f"paired t-test: t={t.statistic:.2f}, p={t.pvalue:.2e}")

    out_path = HERE / "out" / "workday_morning_boundary_flow.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flow.to_csv(out_path, index_label="date", encoding="utf-8-sig")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
