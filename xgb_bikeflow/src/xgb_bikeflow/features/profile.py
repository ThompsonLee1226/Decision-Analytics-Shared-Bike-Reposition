#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/profile.py
================================
Same-hour profile block — how busy this unit usually is at this clock hour.

The one data-derived family
---------------------------
Everything else is derivable from the calendar or from the unit's own geometry.
This block reads history, so it is the one that needs an argument for why it is
not a time-series feature and does not leak.

The argument: a unit's behaviour at 08:00 is a *property of the unit*, not a
point in a series.  It is stable across days, and an operator planning 08:00
tomorrow knows it.  It is fitted on the fold's training days only and applied
unchanged to validation days, so a validation row never contributes to its own
feature.  What is banned is *lags of the target* — "what happened at this unit in
the previous hour / same hour yesterday" — and none of these six columns are
that.

Why six columns and not two
---------------------------
`mean` is the level; `share` is the shape (mean divided by the unit's daily
mean); `std` is a difficulty measure, not a level predictor.  Keeping all three
matters on the full panel, where units differ by two orders of magnitude in
volume: without `share` a tree has to rediscover "this unit is busy in the
morning" from an absolute count it cannot compare across units.

The zero-fill, and why it is a wart
-----------------------------------
After the merge, any (unit, hour) missing from the training rows is filled with
0.0.  On the shipped panel every unit and hour is present in every fold's
training set, so the fill never fires.  For genuine forward prediction it does:
a unit that did not exist during training gets six zeros, which reads to the
model as "never busy" rather than "no information".  `fallback: global` uses the
all-unit mean for that hour instead, which is the honest prior; it is off by
default because switching it on changes the zero rows and therefore the scores,
and the published numbers were produced with the zeros.  Same reasoning as
`window_origin` in the calendar block: keep the historical behaviour as the
default, offer the defensible one, and say which is which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Context, FeatureBlock

COLUMNS = ("arr_slot_mean", "dep_slot_mean", "arr_slot_std", "dep_slot_std",
           "arr_slot_share", "dep_slot_share")


class ProfileBlock(FeatureBlock):
    """Per (unit, hour) statistics over training rows only."""

    name = "profile"
    columns = COLUMNS
    # `std(ddof=0)` on a single observation is 0.0, which is what the original
    # used; keeping ddof is not optional, it moves the numbers.
    fitted = True

    def __init__(self, fallback: str = "zero", ddof: int = 0, **options):
        super().__init__(**options)
        if fallback not in ("zero", "global"):
            raise ValueError(
                f"profile.fallback must be 'zero' or 'global', got {fallback!r}")
        self.fallback = fallback
        self.ddof = int(ddof)
        self.table: pd.DataFrame | None = None
        self.global_mean: dict[int, dict[str, float]] = {}

    # -- fit ---------------------------------------------------------------
    def fit(self, train: pd.DataFrame, ctx: Context) -> None:
        g = train.groupby(["block_id", "slot"])
        prof = pd.DataFrame({
            "arr_slot_mean": g["arrivals"].mean(),
            "dep_slot_mean": g["departures"].mean(),
            "arr_slot_std": g["arrivals"].std(ddof=self.ddof),
            "dep_slot_std": g["departures"].std(ddof=self.ddof),
        })
        # the unit's overall daily mean, used to turn the slot level into a shape
        arr_daily = (train.groupby("block_id")["arrivals"].mean()
                     .rename("arr_day_mean"))
        dep_daily = (train.groupby("block_id")["departures"].mean()
                     .rename("dep_day_mean"))
        prof = prof.join(arr_daily, on="block_id").join(dep_daily, on="block_id")
        prof["arr_slot_share"] = (prof["arr_slot_mean"]
                                  / prof["arr_day_mean"].replace(0, np.nan))
        prof["dep_slot_share"] = (prof["dep_slot_mean"]
                                  / prof["dep_day_mean"].replace(0, np.nan))
        prof = prof.fillna(0.0)
        self.table = prof.reset_index()[["block_id", "slot", *COLUMNS]]

        gm = train.groupby("slot")[["arrivals", "departures"]].mean()
        self.global_mean = {
            int(h): {"arr_slot_mean": float(r.arrivals),
                     "dep_slot_mean": float(r.departures)}
            for h, r in gm.iterrows()
        }

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        if self.table is None:
            raise RuntimeError(
                "ProfileBlock.transform called before fit(); a fitted block "
                "cannot be applied without its fitted state")
        keys = pd.DataFrame({"block_id": df["block_id"].to_numpy(),
                             "slot": df["slot"].to_numpy()})
        out = keys.merge(self.table, on=["block_id", "slot"], how="left")

        if self.fallback == "global":
            miss = out["arr_slot_mean"].isna().to_numpy()
            if miss.any():
                hours = out.loc[miss, "slot"].to_numpy()
                for col in ("arr_slot_mean", "dep_slot_mean"):
                    fill = np.array([self.global_mean.get(int(h), {}).get(col, 0.0)
                                     for h in hours], dtype=float)
                    out.loc[miss, col] = fill
                # std and share stay zero: no within-unit information at all
        out[list(COLUMNS)] = out[list(COLUMNS)].fillna(0.0)
        return self._check(out, df)

    # -- persistence -------------------------------------------------------
    def state(self) -> dict:
        if self.table is None:
            return {}
        return {
            "fallback": self.fallback,
            "ddof": self.ddof,
            "global_mean": {str(k): v for k, v in self.global_mean.items()},
            # (unit, hour) -> six numbers, kept column-wise to stay compact
            "units": sorted(self.table["block_id"].unique().tolist()),
            "slots": [int(h) for h in self.table["slot"].tolist()],
            "block_id": self.table["block_id"].astype(str).tolist(),
            "values": {c: [float(v) for v in self.table[c]] for c in COLUMNS},
        }

    def load_state(self, d: dict) -> None:
        if not d:
            self.table = None
            return
        self.fallback = d.get("fallback", self.fallback)
        self.ddof = int(d.get("ddof", self.ddof))
        self.global_mean = {int(k): {kk: float(vv) for kk, vv in v.items()}
                            for k, v in d.get("global_mean", {}).items()}
        self.table = pd.DataFrame({
            "block_id": d["block_id"],
            "slot": np.asarray(d["slots"], dtype="int64"),
            **{c: np.asarray(d["values"][c], dtype="float64") for c in COLUMNS},
        })
