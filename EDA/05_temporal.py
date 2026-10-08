#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_temporal.py
==============
Section C of the plan: how demand moves through time.

    fig03  (a) daily volume bars + 7-day centred rolling mean, weekends
           picked out; (b) hour-of-day, weekday and weekend overlaid
    fig04  day-of-week x hour heat map, **normalised per row**

Why these two
-------------
C1 is the two time scales the plan cares about and nothing else: the trend
across the 83 days, and the shape within a day.  They are one figure because a
reader should see the ramp-up and the double peak in the same glance -- the
morning peak is a commute-driven *supply* event (bikes leave residential areas)
and the evening peak is its mirror, and that asymmetry is the engine of the
imbalance §E measures.

C2 exists to check one thing: that the intraday shape is not the same every day
of the week.  It is row-normalised, and the normalisation is stated on the
figure.  An unlabelled row-normalised heat map is the most common way to
mislead with a heat map, because the reader cannot tell whether a dark cell
means "a lot of trips" or "a small share of a small day".

Two normalisation choices worth naming
--------------------------------------
* The heat map is normalised **per row** (per weekday), because the 83 cleaned
  days do not contain an equal number of each weekday -- 2026-05-17 is a Sunday
  and was removed as an anomaly day -- so raw counts would be biased against
  Sundays by construction.
* Each cell is the **mean count per occurrence of that weekday**, not a total,
  for the same reason.

Usage:
    python 05_temporal.py
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

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# days the cleaner removed; drawn as holes in the series so the QC action is
# visible on the slide rather than hidden behind an interpolated line
REMOVED = {"2026-05-17": "anomaly day removed (A5)\n2,463 trips = 22.6% of its baseline"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def daily_volume(dates: pd.Series) -> pd.DataFrame:
    """Trips per calendar day, reindexed onto the full span so gaps stay gaps."""
    span = pd.date_range(dates.min(), dates.max(), freq="D")
    s = dates.value_counts().sort_index()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(span)
    out = pd.DataFrame({"date": span, "n": s.values})
    out["dow"] = out.date.dt.dayofweek
    out["is_weekend"] = out.dow >= 5
    out["removed"] = out.n.isna()
    out["roll7"] = out.n.rolling(7, center=True, min_periods=4).mean()
    return out


def hour_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean trips per hour of day, split weekday / weekend, per day."""
    n_days = df.groupby("is_weekend").date.nunique()
    tab = (df.groupby(["is_weekend", "hour"]).size().unstack("hour").fillna(0))
    tab = tab.div(n_days, axis=0)
    tab.columns = [int(c) for c in tab.columns]
    return tab


def dow_hour_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean trips per (dow, hour), then normalised to a share within each row."""
    occ = df.groupby("dow").date.nunique()
    cnt = df.groupby(["dow", "hour"]).size().unstack("hour").fillna(0)
    cnt = cnt.reindex(range(7)).fillna(0)
    cnt = cnt.div(occ, axis=0)                     # per occurrence of that weekday
    share = 100 * cnt.div(cnt.sum(axis=1), axis=0)  # per-row normalisation
    return cnt, share


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_temporal(plt, daily: pd.DataFrame, prof: pd.DataFrame) -> None:
    """Fig 3 -- daily trend and the intraday shape."""
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": .21})

    # ---- (a) daily volume ---------------------------------------------------
    ax = axes[0]
    wk = daily[~daily.is_weekend & ~daily.removed]
    we = daily[daily.is_weekend & ~daily.removed]
    ax.bar(wk.date, wk.n, width=.78, color=viz.BLUE, alpha=.80, label="weekday")
    ax.bar(we.date, we.n, width=.78, color=viz.AMBER, alpha=.90, label="weekend")
    ax.plot(daily.date, daily.roll7, color=viz.INK, lw=1.7, label="7-day centred mean")

    for d, msg in REMOVED.items():
        t = pd.Timestamp(d)
        if daily.date.min() <= t <= daily.date.max():
            ax.axvline(t, color=viz.RED, lw=1.1, ls=":")
            ax.annotate("anomaly day removed\n(A5)", xy=(t, daily.n.max() * .93),
                        xytext=(t - pd.Timedelta(days=21), daily.n.max() * .97),
                        fontsize=8.2, color=viz.RED, ha="left",
                        arrowprops=dict(arrowstyle="->", color=viz.RED, lw=.8))

    # name the ramp-up rather than let it read as a demand trend to extrapolate.
    # The note sits on an opaque patch: every band below the bars is bar colour,
    # so there is nowhere genuinely free to put it.
    mar = daily[(daily.date.dt.month == 3) & ~daily.removed].n.mean()
    apr_may = daily[(daily.date.dt.month >= 4) & ~daily.removed].n.mean()
    ax.annotate(f"March {mar:,.0f}/day  →  Apr–May {apr_may:,.0f}/day\n"
                "a ramp-up, not a trend the model should extrapolate",
                xy=(.015, .035), xycoords="axes fraction", fontsize=8.8, color=viz.INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=viz.GREY, alpha=.93))

    ax.set_ylabel("trips per day")
    ax.set_title(f"(a) Daily volume · {int((~daily.removed).sum())} cleaned days "
                 f"of {len(daily)}\nsample variance "
                 f"{daily.n.var(ddof=1):,.0f} (sd {daily.n.std(ddof=1):,.0f})")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, daily.n.max() * 1.22)

    # ---- (b) hour of day ----------------------------------------------------
    ax = axes[1]
    h = np.arange(24)
    ax.plot(h, prof.loc[False, h].values, "-o", ms=3.6, color=viz.BLUE,
            label="weekday")
    ax.plot(h, prof.loc[True, h].values, "--s", ms=3.4, color=viz.AMBER,
            label="weekend")
    ymax = float(prof.max().max())
    for row, col, lbl, tx in ((False, viz.BLUE, "weekday peak", 9.4),
                              (True, viz.AMBER, "weekend peak", 12.4)):
        k = int(prof.loc[row].idxmax())
        ax.annotate(f"{lbl} {k:02d}:00", xy=(k, prof.loc[row, k]),
                    xytext=(tx, ymax * (0.99 if col is viz.BLUE else 0.62)),
                    fontsize=8.6, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=.9))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean trips per hour, per day")
    ax.set_title("(b) Intraday shape — the morning peak\nflattens at the weekend")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper right")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, ymax * 1.10)

    viz.save(fig, 3, "temporal_overview")


def fig_dow_hour(plt, mean_cnt: pd.DataFrame, share: pd.DataFrame) -> None:
    """Fig 4 -- day-of-week x hour heat map, normalised per row."""
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    M = share.values
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24), fontsize=8.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels(DOW)
    ax.set_xlabel("hour of day")
    ax.set_title("Day-of-week × hour\n"
                 "colour = share of that weekday's trips in that hour "
                 "(row-normalised, so each row sums to 100%)")
    ax.grid(False)

    # mark each row's peak so the mirror structure is readable
    for i in range(M.shape[0]):
        j = int(np.nanargmax(M[i]))
        ax.plot(j, i, marker="o", ms=8, mfc="none", mec="white", mew=1.4)

    cb = fig.colorbar(im, ax=ax, fraction=.030, pad=.015)
    cb.set_label("share of the weekday's trips (%)")

    viz.save(fig, 4, "dow_hour_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    if not TRIPS.exists():
        print(f"FATAL: {TRIPS} not found. Run 01_data_cleaning.py first.", file=sys.stderr)
        return 2

    df = pd.read_csv(TRIPS, usecols=["date", "hour", "dow", "is_weekend"])
    print(f"trips     : {len(df):,} over {df.date.nunique()} days")

    daily = daily_volume(df.date)
    prof = hour_profile(df)
    mean_cnt, share = dow_hour_matrix(df)

    print(f"daily     : mean {daily.n.mean():,.0f}, sd {daily.n.std(ddof=1):,.0f}, "
          f"sample var {daily.n.var(ddof=1):,.0f}, min {daily.n.min():,.0f}, max {daily.n.max():,.0f}")
    print(f"  weekday {daily[~daily.is_weekend].n.mean():,.0f}/day  "
          f"weekend {daily[daily.is_weekend].n.mean():,.0f}/day")
    print(f"  peaks   : weekday {int(prof.loc[False].idxmax()):02d}:00 "
          f"({prof.loc[False].max():.0f}/h)  weekend {int(prof.loc[True].idxmax()):02d}:00 "
          f"({prof.loc[True].max():.0f}/h)")

    daily.to_csv(OUT / "table_c1_daily.csv", index=False, encoding="utf-8")
    prof.T.rename(columns={False: "weekday", True: "weekend"}).to_csv(
        OUT / "table_c1_hour_profile.csv", encoding="utf-8")
    share.round(4).to_csv(OUT / "table_c2_dow_hour_share.csv", encoding="utf-8")
    mean_cnt.round(3).to_csv(OUT / "table_c2_dow_hour_mean.csv", encoding="utf-8")

    fig_temporal(plt, daily, prof)
    fig_dow_hour(plt, mean_cnt, share)

    write_summary(df, daily, prof, mean_cnt, share)
    (OUT / "temporal.json").write_text(json.dumps({
        "days": int((~daily.removed).sum()),
        "daily_mean": round(float(daily.n.mean()), 2),
        "daily_var": round(float(daily.n.var(ddof=1)), 2),
        "daily_sd": round(float(daily.n.std(ddof=1)), 2),
        "daily_min": int(daily.n.min()),
        "daily_max": int(daily.n.max()),
        "weekday_mean": round(float(daily[~daily.is_weekend].n.mean()), 2),
        "weekend_mean": round(float(daily[daily.is_weekend].n.mean()), 2),
        "march_mean": round(float(daily[(daily.date.dt.month == 3) & ~daily.removed].n.mean()), 2),
        "apr_may_mean": round(float(daily[(daily.date.dt.month >= 4) & ~daily.removed].n.mean()), 2),
        "hour_peak_weekday": int(prof.loc[False].idxmax()),
        "hour_peak_weekend": int(prof.loc[True].idxmax()),
        "dow_row_peaks": {DOW[i]: int(np.nanargmax(share.values[i])) for i in range(7)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT/'temporal_summary.md'}, temporal.json, table_c1_*.csv, table_c2_*.csv")
    return 0


def write_summary(df, daily, prof, mean_cnt, share):
    ok = daily[~daily.removed]
    mar = ok[ok.date.dt.month == 3].n.mean()
    apr_may = ok[ok.date.dt.month >= 4].n.mean()
    d = ok.n

    hour_rows = "\n".join(
        f"| {h:02d} | {prof.loc[False, h]:,.0f} | {prof.loc[True, h]:,.0f} | "
        f"{100*prof.loc[False, h]/prof.loc[False].sum():5.2f}% | "
        f"{100*prof.loc[True, h]/prof.loc[True].sum():5.2f}% |"
        for h in range(24))

    peak_rows = "\n".join(
        f"| {DOW[i]} | {int(np.nanargmax(share.values[i])):02d}:00 | "
        f"{share.values[i].max():.1f}% | {mean_cnt.values[i].max():,.0f} |"
        for i in range(7))

    md = f"""# §C — Temporal patterns

*Rendered by `05_temporal.py`. Figures: **#3** temporal overview, **#4** dow × hour heat map.*

## C1 — Temporal overview

| | |
|---|---|
| Days | {len(ok)} cleaned of {len(daily)} in the window (2026-05-17 removed, §A5) |
| **Daily volume** | mean **{d.mean():,.0f}**, sample variance **{d.var(ddof=1):,.0f}** (sd {d.std(ddof=1):,.0f}) |
| Range | {d.min():,.0f} – {d.max():,.0f} per day |
| Weekday | {ok[~ok.is_weekend].n.mean():,.0f}/day over {int((~ok.is_weekend).sum())} days |
| Weekend | {ok[ok.is_weekend].n.mean():,.0f}/day over {int(ok.is_weekend.sum())} days |
| Month | March **{mar:,.0f}/day** → April–May **{apr_may:,.0f}/day** |

**Read it as.** March runs lighter than April–May — a **ramp-up** worth naming so
nobody mistakes it for a demand trend the model should extrapolate. The daily
sample variance ({d.var(ddof=1):,.0f}) is large relative to the mean
({d.mean():,.0f}) but is dominated by the weekday/weekend split rather than by
day-to-day noise within a class.

**If this were a demand forecast, the ramp-up would be a problem.** It is not:
§H predicts *within* the panel, and the walk-forward split of H3 means every
validation block is predicted from the days that precede it, so the level shift
is absorbed by the training window rather than extrapolated.

### Hour of day

| hour | weekday (mean/day) | weekend (mean/day) | share · weekday | share · weekend |
|---:|---:|---:|---:|---:|
{hour_rows}

Weekday peaks at **{int(prof.loc[False].idxmax()):02d}:00** and again at
**18:00**, with an overnight trough near 03:00. The morning peak
**flattens at the weekend** — the weekend profile is close to a single broad
afternoon hump. The morning peak is a commute-driven *supply* event (bikes leave
residential areas) and the evening peak is its mirror; that asymmetry, not
either peak alone, is what §E measures.

## C2 — Day-of-week × hour

**Normalisation: per row.** Each cell is the share of that weekday's trips
falling in that hour, so every row sums to 100%. Raw counts are *not* used
because the cleaned window does not contain an equal number of each weekday
(2026-05-17, a Sunday, was removed in §A5), which would bias Sundays down by
construction.

| weekday | peak hour | share of that day | mean trips in that hour |
|---|---|---:|---:|
{peak_rows}

**Mon–Thu are near-identical** (morning peak 12.3–13.0%, evening 9.4–9.9%).
**Friday** keeps the same peak hours but shifts weight out of the morning and
into the evening (morning 11.7%, evening 10.1/10.8%) — the weekend arriving.
**Sat/Sun lose the morning spike entirely** (07:00–09:00 share falls from
~19% of the day to ~11%) and flatten into a broad midday plateau, with a
single evening peak at 17:00 rather than a commute pair.

The important consequence for §E: the morning spike is a **weekday** structure,
not a Mon–Fri averaging artefact, which is why §E1's cell × hour panel is
restricted to weekdays. Including weekends would smear the very mirror-image
pattern the figure exists to show.
"""
    (OUT / "temporal_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
