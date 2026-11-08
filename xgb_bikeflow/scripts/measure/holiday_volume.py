#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What the holiday rows are worth, and what dropping them costs.

`source.days: workday` removes the eight statutory holidays.  They are not missing
from the data -- they are present and busy.  This measures how busy, inside the
07:30-10:30 horizon, and compares them against a working day.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Running a script straight from a clone: the package lives in src/.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

ORDERS = ROOT / "data" / "orders_2026.csv.gz"
HOL = ["2026-04-04", "2026-04-05", "2026-04-06",
       "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"]
MK = ["2026-05-09"]


def minutes(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    return t[0].astype(float) * 60 + t[1].astype(float)


def main() -> int:
    df = pd.read_csv(ORDERS, usecols=["date", "start_time"])
    df["date"] = pd.to_datetime(df["date"])
    h = minutes(df["start_time"])
    inw = (h >= 450) & (h < 630)
    print(f"orders {len(df):,}   inside 07:30-10:30: {int(inw.sum()):,}"
          f"  ({inw.mean() * 100:.1f}%)")

    x = df[inw]
    per_day = x.groupby("date").size()
    d = pd.to_datetime(per_day.index)
    hol = np.isin(d.values, pd.to_datetime(HOL).values)
    mk = np.isin(d.values, pd.to_datetime(MK).values)
    weekend = (d.dayofweek.values >= 5) & ~hol & ~mk
    workday = ((d.dayofweek.values < 5) & ~hol) | mk

    print("\nvolume inside the horizon, by day type")
    print(f"{'day type':<14}{'days':>6}{'mean':>9}{'median':>9}")
    for name, m in (("workday", workday), ("make-up", mk),
                    ("weekend", weekend), ("holiday", hol)):
        v = per_day.to_numpy()[m]
        if len(v) == 0:
            continue
        print(f"{name:<14}{len(v):>6}{v.mean():>9.0f}{np.median(v):>9.0f}")

    print("\neach holiday against the workday mean")
    wd_mean = float(per_day.to_numpy()[workday].mean())
    rows = []
    for i in np.where(hol)[0]:
        day = d[i]
        v = int(per_day.iloc[i])
        rows.append({"date": str(day.date()), "weekday": day.day_name()[:3],
                     "events": v, "vs_workday": round(v / wd_mean, 3)})
        print(f"  {day.date()}  {day.day_name()[:3]}  {v:>6,}"
              f"  {v / wd_mean:>5.2f}x workday")

    hol_mean = float(per_day.to_numpy()[hol].mean())
    wk_mean = float(per_day.to_numpy()[weekend].mean())
    print(f"\nholiday mean {hol_mean:,.0f} = {hol_mean / wd_mean:.2f}x workday,"
          f" {hol_mean / wk_mean:.2f}x weekend")
    print(f"8 holidays = {int(per_day.to_numpy()[hol].sum()):,} events"
          f" = {per_day.to_numpy()[hol].sum() / per_day.sum() * 100:.1f}%"
          f" of horizon volume")

    out = {"workday_mean": wd_mean, "weekend_mean": wk_mean,
           "holiday_mean": hol_mean, "holidays": rows}
    p = ROOT / "docs" / "eda_comparison" / "holiday_volume.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote : {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
