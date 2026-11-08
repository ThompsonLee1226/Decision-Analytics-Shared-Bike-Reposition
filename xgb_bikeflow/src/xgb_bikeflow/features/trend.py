#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/trend.py
==============================
Trend block — how much busier the window is at the target date.

What it is
----------
An OLS slope of ``log(daily total arrivals)`` on day-of-window, fitted
separately for weekdays and weekend days, evaluated at the row's date, and
centred so an average training day scores 1.0.

Why it is not a time-series feature
-----------------------------------
It is a *calendar* trend: fitted on training days only, evaluated at the target
date, and computable in advance for any future date.  It carries no lag of the
target and never looks at a neighbouring hour.  At prediction time it is the
same number a planner writes on a whiteboard: "volume is running 20% above
March".

Why it is needed
----------------
With lags banned, nothing else tells the model that the validation block sits
later in the window than its training data.  Measured on this panel, weekday
volume rises 1,480 → 1,782 (+20.4%) from the first 20 days to the last 20, and a
model without a trend term under-predicts by exactly that much in the last fold
(calibration 0.83 → 0.76 → 0.64).

Fitted per weekday class because the ramp is not the same on Saturdays — and
classified with the *calendar* day type, not the raw weekday.  A make-up workday
belongs to the weekday series; if it were classified as a Saturday it would
inflate the Saturday series with a weekday-sized day and flatten the slope.

Note the asymmetry, which the leak guard makes explicit: `fit` reads `arrivals`,
`transform` does not touch it.  A saved model therefore needs only the fitted
slope and intercepts, plus the row's date — never the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Context, FeatureBlock

CLIP_LO, CLIP_HI = 0.3, 3.0
MIN_DAYS_PER_CLASS = 4


def _pick(d: dict, key: bool) -> float:
    """`d[key]`, treating a stored None as "no slope fitted" rather than a value.

    A day type absent from the training window has no slope.  `fit` now fills it
    by borrowing the other class, but a bundle saved before that fix stores null,
    and `dict.get(key, 0.0)` returns None -- not the default -- when the key is
    present with a null value.  That is how a missing slope became
    `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`.
    """
    v = d.get(key)
    return 0.0 if v is None else float(v)


class TrendBlock(FeatureBlock):
    """`trend_scale`: expected level on this date relative to the training mean."""

    name = "trend"
    columns = ("trend_scale",)
    fitted = True

    def __init__(self, target: str = "arrivals", **options):
        super().__init__(**options)
        self.target = target
        self.w0: pd.Timestamp | None = None
        self.slope: dict[bool, float] = {}
        self.intercept: dict[bool, float] = {}
        self.block_share: dict[str, float] = {}
        self.borrowed: list[bool] = []
        self._ref: float = 0.0

    # -- fit ---------------------------------------------------------------
    def fit(self, train: pd.DataFrame, ctx: Context) -> None:
        tr = train
        # Use the calendar day type, not the raw weekday: a make-up workday
        # belongs to the weekday class.  `dow` is present because the calendar
        # block runs before this one; fall back to the calendar rule if a
        # configuration disables the calendar block.
        if "dow" in tr.columns:
            dow_ = tr["dow"]
        else:
            from .calendar import CalendarBlock
            dow_ = CalendarBlock().effective_dow(tr["date"])

        daily = (pd.DataFrame({"date": tr["date"].to_numpy(),
                               "n": tr[self.target].to_numpy(float),
                               "dow": np.asarray(dow_, dtype="int64")})
                 .groupby("date", as_index=False)
                 .agg(n=("n", "sum"), dow=("dow", "first")))
        w0 = pd.to_datetime(tr["date"]).min()
        daily["t"] = (pd.to_datetime(daily["date"]) - w0).dt.days

        self.w0 = w0
        self.slope, self.intercept = {}, {}
        for cls, sub in daily.groupby(daily["dow"] >= 5):
            key = bool(cls)
            if len(sub) < MIN_DAYS_PER_CLASS:
                self.slope[key] = 0.0
                self.intercept[key] = float(np.log(max(sub.n.mean(), 1.0)))
                continue
            y = np.log(np.maximum(sub.n.to_numpy(float), 1.0))
            x = sub.t.to_numpy(float)
            xc = x - x.mean()
            slope = float((xc * (y - y.mean())).sum() / max((xc ** 2).sum(), 1e-9))
            self.slope[key] = slope
            self.intercept[key] = float(y.mean() - slope * x.mean())

        share = train.groupby("block_id")[self.target].sum()
        total = share.sum()
        self.block_share = (share / total if total > 0
                            else share * 0.0).to_dict()

        # A day type with no rows in the training window never enters the loop
        # above, so it would have no slope at all.  That is not hypothetical: the
        # default config trains on workdays only, so every weekend is a day type
        # the booster never saw fitted.  Mirror the class that does exist rather
        # than leaving a hole -- a borrowed slope is a defensible guess, a missing
        # one is a crash, and a silent 0.0 would pin those rows at the intercept.
        self.borrowed = sorted({k for k in (True, False) if k not in self.slope})
        for k in self.borrowed:
            other = self.slope.get(not k)
            self.slope[k] = 0.0 if other is None else other
            other_i = self.intercept.get(not k)
            self.intercept[k] = 0.0 if other_i is None else other_i

        # centre on the mean intercept so the feature is 1.0 on an average day
        self._ref = float(np.mean(list(self.intercept.values())))

    def notes(self) -> list[str]:
        if not self.borrowed:
            return []
        names = ", ".join("weekend" if b else "weekday" for b in self.borrowed)
        return [f"trend: no {names} row in the training window, so its slope and "
                f"intercept were borrowed from the other day type.  Predictions for "
                f"those days carry that assumption."]

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        if self.w0 is None:
            raise RuntimeError(
                "TrendBlock.transform called before fit(); a fitted block "
                "cannot be applied without its fitted state")
        t = (pd.to_datetime(df["date"]) - self.w0).dt.days.to_numpy(float)
        # `is_weekend` is a calendar column; the pipeline emits it before this
        # block.  If the calendar block is off, fall back to the raw weekday.
        if "is_weekend" in df.columns:
            is_we = df["is_weekend"].to_numpy() == 1
        else:
            is_we = pd.to_datetime(df["date"]).dt.dayofweek.to_numpy() >= 5

        pred_log = np.where(
            is_we,
            _pick(self.intercept, True) + _pick(self.slope, True) * t,
            _pick(self.intercept, False) + _pick(self.slope, False) * t,
        )
        out = pd.DataFrame(index=df.index)
        out["trend_scale"] = np.clip(np.exp(pred_log - self._ref),
                                     CLIP_LO, CLIP_HI)
        return self._check(out, df)

    # -- persistence -------------------------------------------------------
    def state(self) -> dict:
        # Bool dict keys do not survive JSON (True -> "true"), and a silent
        # str/bool mismatch would make `.get(True)` return the default.  Store
        # the two classes under explicit names instead.
        return {
            "target": self.target,
            "w0": str(self.w0.date()) if self.w0 is not None else None,
            "slope": {"weekend": self.slope.get(True),
                      "weekday": self.slope.get(False)},
            "intercept": {"weekend": self.intercept.get(True),
                          "weekday": self.intercept.get(False)},
            "ref": self._ref,
            "block_share": self.block_share,
            "clip": [CLIP_LO, CLIP_HI],
            "borrowed": [bool(b) for b in self.borrowed],
        }

    def load_state(self, d: dict) -> None:
        self.target = d.get("target", self.target)
        self.w0 = pd.Timestamp(d["w0"]) if d.get("w0") else None

        def _num(x):
            # `fit` now fills both day types (borrowing from the other class when
            # a type never occurs in the training window), so a fresh bundle has
            # no nulls here.  Older bundles do -- they stored None for the absent
            # class -- and `float(None)` raises, which would make them unloadable.
            # Reading a hole back as a hole keeps them loadable.
            return None if x is None else float(x)

        self.slope = {True: _num(d["slope"]["weekend"]),
                      False: _num(d["slope"]["weekday"])}
        self.intercept = {True: _num(d["intercept"]["weekend"]),
                          False: _num(d["intercept"]["weekday"])}
        self._ref = float(d["ref"])
        self.block_share = dict(d.get("block_share", {}))
        self.borrowed = [bool(b) for b in d.get("borrowed", [])]
