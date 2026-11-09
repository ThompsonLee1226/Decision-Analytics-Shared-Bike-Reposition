#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate each GP's 06:30 inventory and capacity, OD-only.

Follows HANDOFF_0630_gp_inventory.md:
  m_i   lower bound from the morning flow  (inventory must never go negative)
  w_i   spatial prior = net arrivals over [previous workday 22:00, target 06:00)
  I0    projection of N * w_i onto [m_i, inf) with sum_i I0_i = N, N = sum_i m_i
  q_i   smallest capacity that never binds on any observed workday morning

OD counts cannot identify the inventory level on their own, so by default
I0 is the minimum-total / minimum-deficit solution (N = sum m_i); pass --n
if a fleet count is known.  q_i is the smallest capacity consistent with the
observed mornings, also a lower bound.  Only GP-assigned cells and the
[06:30, 10:30) morning window are used.
"""
import argparse
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
START_MIN = 6 * 60 + 30
SLOTS = range(8)


def workday(d):
    d = d.dt.normalize()
    return ((d.dt.dayofweek < 5) & ~d.isin(HOLIDAYS)) | d.isin(MAKEUP)


def slot(t):
    return (t.dt.hour * 60 + t.dt.minute - START_MIN) // 30


def panel(df, time, gp, days, gps):
    """Trips -> (date, gp) x slot counts inside the morning window."""
    x = df.assign(date=df[time].dt.normalize(), slot=slot(df[time]))
    x = x[x.date.isin(days) & x[gp].isin(gps) & x.slot.isin(SLOTS)]
    idx = pd.MultiIndex.from_product([days, gps], names=["date", "gp"])
    return (x.groupby(["date", gp, "slot"]).size().unstack("slot", fill_value=0)
              .reindex(idx, fill_value=0).reindex(columns=SLOTS, fill_value=0))


def window(df, time, gp, gps, days, hours):
    """Trips per (calendar date, gp) whose hour-of-day lies in `hours`."""
    x = df[df[time].dt.hour.isin(hours)]
    x = x.assign(date=x[time].dt.normalize(), gp=x[gp])
    x = x[x.date.isin(days) & x.gp.isin(gps)]
    return (x.groupby(["date", "gp"]).size().unstack("gp", fill_value=0)
              .reindex(index=days, columns=gps, fill_value=0))


def project(w, m, n):
    """min sum (x - n*w)^2 / (2*n*w)  s.t.  x >= m,  sum x = n.

    Separable quadratic + one sum constraint => exact water-filling: with
    a_i = n*w_i, the optimal x_i is a_i*r for free GPs and m_i for the
    binding ones, where r makes the total equal n.
    """
    w = np.maximum(w / w.sum(), 1e-9)
    a = n * w / w.sum()
    order = np.argsort(-(m / a))                 # most binding first
    ms, as_ = m[order], a[order]
    cum_m, cum_a = np.cumsum(ms), np.cumsum(as_)
    for k in range(len(m) + 1):                  # k GPs pinned at m_i
        need = n - (cum_m[k - 1] if k else 0)
        free = n - (cum_a[k - 1] if k else 0)
        r = need / free if free > 1e-12 else np.inf
        if (k == 0 or ms[k - 1] >= r * (1 - 1e-9)) and \
           (k == len(m) or ms[k] / as_[k] <= r * (1 + 1e-9)):
            x = np.empty_like(m)
            x[order[:k]] = ms[:k]
            x[order[k:]] = as_[k:] * r
            return x
    return m.copy()


def main() -> None:
    own = pd.read_csv(DATA / "hw6_grid_ownership.csv")
    gh2gp = dict(zip(own.geohash, own.gp))
    gps = sorted(set(own.gp) - {"UNASSIGNED"})

    tr = pd.read_csv(ORDERS, usecols=[
        "start_time", "end_time", "start_geohash", "end_geohash"])
    for c in ("start_time", "end_time"):
        tr[c] = pd.to_datetime(tr[c], format="%Y/%m/%d %H:%M")
    tr["sgp"] = tr.start_geohash.map(gh2gp)
    tr["egp"] = tr.end_geohash.map(gh2gp)

    all_wd = sorted(set(tr.start_time.dt.normalize()[workday(tr.start_time)]))
    # 2026-05-22 is a half day; drop it as a target.
    days = [d for d in all_wd if d != pd.Timestamp("2026-05-22")]

    D = panel(tr, "start_time", "sgp", days, gps)
    A = panel(tr, "end_time", "egp", days, gps)

    # m_i: mean over workdays of max_s(cumD - cumA_{s-1}), clipped at 0.
    m_daily = (D.cumsum(axis=1)
               - A.cumsum(axis=1).shift(axis=1, fill_value=0)).clip(lower=0)
    m = m_daily.max(axis=1).groupby(level="gp").mean().reindex(gps).to_numpy()

    # w_i: net arrivals in [p(d) 22:00, d 06:00), p(d) = previous workday.
    # The late-evening half comes from p(d), the early-morning half from d
    # itself, so Monday pairs Friday's late evening with Monday's morning.
    all_wd_set = set(all_wd)
    prev = {}
    for d in days:
        p = d - pd.Timedelta(days=1)
        while p not in all_wd_set and p >= all_wd[0]:
            p -= pd.Timedelta(days=1)
        if p in all_wd_set:
            prev[d] = p
    tgt = list(prev)                       # target days that have a p(d)
    ev_days = sorted(set(prev.values()))
    A_late = window(tr, "end_time", "egp", gps, ev_days, range(22, 24))
    D_late = window(tr, "start_time", "sgp", gps, ev_days, range(22, 24))
    A_early = window(tr, "end_time", "egp", gps, days, range(0, 6))
    D_early = window(tr, "start_time", "sgp", gps, days, range(0, 6))
    A_ev = (A_late.loc[[prev[d] for d in tgt]].to_numpy()
            + A_early.loc[tgt].to_numpy())
    D_ev = (D_late.loc[[prev[d] for d in tgt]].to_numpy()
            + D_early.loc[tgt].to_numpy())
    raw = np.maximum(A_ev - D_ev, 0)
    empty = raw.sum(axis=1) == 0
    raw[empty] = A_ev[empty]               # fallback: arrivals only
    w = raw.sum(axis=0)

    ap = argparse.ArgumentParser(description="OD-only 06:30 GP inventory")
    ap.add_argument("--n", type=float,
                    help="total bikes parked at the 90 GPs at 06:30 "
                         "(default: sum m_i, the minimum feasible total)")
    n = ap.parse_args().n or m.sum()
    if n < m.sum() - 1e-9:
        raise SystemExit(f"--n={n:.1f} < sum m_i={m.sum():.1f}: infeasible "
                         "without shortages or rebalancing")
    i0 = project(w, m, n)

    # q_i: smallest capacity that never binds on any observed workday
    # morning, i.e. ceil of the peak of the no-rebalancing trajectory.
    A3 = A.to_numpy().reshape(len(days), len(gps), 8)
    D3 = D.to_numpy().reshape(len(days), len(gps), 8)
    traj = i0[None, :, None] + np.cumsum(A3 - D3, axis=2)
    q = np.ceil(np.maximum(i0, traj.max(axis=(0, 2))))
    I_before = np.concatenate([i0[None, :, None].repeat(len(days), axis=0),
                               traj[:, :, :-1]], axis=2)
    short_days = ((I_before - D3) < -1e-9).any(axis=2).sum(axis=0)

    out = pd.DataFrame({
        "gp": gps, "m_i": m, "w_i": w / w.sum(), "I0_0630": i0,
        "q_i": q.astype(int), "short_days": short_days,
    })
    out_path = HERE / "out" / "gp_0630_inventory.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig",
               float_format="%.6g")

    print(f"workdays used: {len(days)} ({days[0].date()} .. {days[-1].date()}), "
          f"w_i from {len(tgt)} prior windows")
    print(f"N = {n:,.1f}   sum m_i (N_min) = {m.sum():,.1f}   "
          f"sum I0 = {i0.sum():,.1f}   sum q_i = {q.sum():,.0f}")
    print(f"I0: min {i0.min():.1f} / median {np.median(i0):.1f} / max {i0.max():.1f}")
    print(f"q_i: min {q.min():.0f} / median {np.median(q):.0f} / max {q.max():.0f}")
    print(f"binding lower bounds (I0 == m_i): {(i0 - m < 1e-6).sum()}/{len(gps)}")
    print(f"GPs with any negative-stock day: {(short_days > 0).sum()}/{len(gps)} "
          f"(max {short_days.max()} days)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
