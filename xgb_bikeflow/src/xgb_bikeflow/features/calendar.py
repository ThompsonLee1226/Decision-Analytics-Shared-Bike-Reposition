#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/calendar.py
=================================
Calendar block — everything derivable from the date alone.

Column order is the historical one and is load-bearing: xgboost walks feature
columns in matrix order, so permuting them changes tie-breaking and therefore
changes the published numbers in the fourth decimal.  Keeping the order fixed is
what lets the golden tests assert exact equality instead of a tolerance.

Two mappings that are *not* the weekday of the date
---------------------------------------------------
`dow`, `is_weekend` and `is_weekday` encode the **day type the calendar
implies**, not the weekday the date falls on.  Both mappings are decisions that
came out of measurement, and both are pure calendar statements made before any
split, so neither leaks:

1. **Make-up workdays (调休)** are Saturdays that run on weekday rhythms.  On
   this panel 2026-05-09 carried 99% of the preceding Friday's volume (3.99 vs
   4.04 arrivals), not a Saturday's (2.01).  The flag `is_makeup_workday` is 1 on
   exactly one date, and that date falls in fold 3's validation set, so the flag
   is **zero in every training set** — a tree cannot split on what it never sees,
   and it fell back to the Saturday pattern: predicted 2.06 against an actual
   3.99, deviance 3.68 (the worst single day in the panel).  Setting `dow` to the
   make-up day's weekday fixes it: −2.6% overall deviance.

2. **Holidays** run on weekend rhythms *regardless of which weekday they land
   on*.  `is_holiday` alone does not convey that, and fold 1's training window
   (03-02..03-22) contains no holiday at all while validating 04-04/05/06.  The
   two weekend holidays survived by accident (the weekend pattern is already
   low); Monday 04-06 was predicted at weekday volume, 2.99 against an actual
   1.47.  Mapping holidays onto Saturday rhythms fixes it: −3.7% overall.

Note what did *not* work: adding an explicit `is_holiday_weekday` interaction
flag scored −0.02%.  The problem was the wrong rhythm, not a missing
interaction, which is why the fix is a remapping rather than a new column.

The window-origin quirk
-----------------------
`week_of_window` / `day_of_window` are offsets from the earliest date **in the
frame being transformed** (`window_origin: frame`, the default).  The original
code did this because it called `add_calendar` once per split, so validation
rows were numbered from the first validation day, not from the first training
day.  That is arguably wrong — the two splits then count from different
origins — but it is what every published number was produced with, so it stays
the default.  `window_origin: train` is the semantically consistent variant; it
changes these two columns' values and therefore the scores, so it is opt-in and
is not covered by the golden anchors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Context, FeatureBlock

#: 2026 statutory holidays inside the observation window (State Council notice).
DEFAULT_HOLIDAYS = (
    "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
)
#: Saturday make-up workdays (调休).
DEFAULT_MAKEUP = ("2026-05-09",)
#: `dow` a make-up workday is encoded as.  Friday is the closest weekday
#: neighbour and the measured behaviour is indistinguishable; a sweep over
#: Thursday/Friday differs by 0.01%.
DEFAULT_MAKEUP_DOW = 4
#: `dow` a statutory holiday is encoded as.  5 = Saturday = "weekend rhythms".
DEFAULT_HOLIDAY_DOW = 5


class CalendarBlock(FeatureBlock):
    """The slot index plus the day-type flags, the window index and holiday
    adjacencies."""

    name = "calendar"
    columns = (
        "slot",
        "dow", "is_weekend", "is_weekday", "is_holiday",
        "is_day_before_holiday", "is_day_after_holiday", "is_makeup_workday",
        "month", "day_of_month", "week_of_window", "day_of_window",
    )
    fitted = False
    requires = ()

    def __init__(self, holidays=DEFAULT_HOLIDAYS, makeup=DEFAULT_MAKEUP,
                 makeup_dow: int = DEFAULT_MAKEUP_DOW,
                 holiday_dow: int = DEFAULT_HOLIDAY_DOW,
                 window_origin: str = "frame", **options):
        super().__init__(**options)
        if window_origin not in ("frame", "train"):
            raise ValueError(
                f"calendar.window_origin must be 'frame' or 'train', "
                f"got {window_origin!r}")
        self.holidays = pd.to_datetime(list(holidays))
        self.makeup = pd.to_datetime(list(makeup))
        self.makeup_dow = int(makeup_dow)
        self.holiday_dow = int(holiday_dow)
        self.window_origin = window_origin
        #: origin pinned by `fit` when window_origin == "train"
        self._w0: pd.Timestamp | None = None
        #: origin frozen by `pin_origin` when a model is saved
        self._pinned = False

    def pin_origin(self, date) -> None:
        """Freeze the window origin so a saved model is frame-independent.

        With `window_origin: frame` the origin is the earliest date *in the frame
        being transformed*, so the same block-hour gets a different
        `day_of_window` depending on what else is in the request: predicting a
        whole month and predicting one day of it disagree.  Fine for a fixed
        train/valid split, fatal for a saved model, so `Bundle.save` pins the
        origin and `state` records it.
        """
        self._w0 = pd.to_datetime(date)
        self._pinned = True

    # -- day type ----------------------------------------------------------
    def effective_dow(self, date) -> pd.Series:
        """Weekday implied by the calendar: make-up workdays take `makeup_dow`."""
        d = pd.to_datetime(pd.Series(date).reset_index(drop=True))
        dow = d.dt.dayofweek.astype("int8")
        return dow.where(~d.isin(self.makeup), self.makeup_dow)

    # -- lifecycle ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, ctx: Context) -> None:
        if self.window_origin == "train":
            self._w0 = pd.to_datetime(train["date"]).min()

    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        d = pd.to_datetime(df["date"])
        out = pd.DataFrame(index=df.index)

        out["slot"] = df["slot"].astype("int64")
        out["is_holiday"] = d.isin(self.holidays).astype("int8")
        out["is_makeup_workday"] = d.isin(self.makeup).astype("int8")

        eff = self.effective_dow(df["date"])
        is_hol = (out["is_holiday"].to_numpy() > 0)
        is_mk = (out["is_makeup_workday"].to_numpy() > 0)
        out["dow"] = np.where(is_hol, self.holiday_dow, eff.to_numpy())
        # A make-up workday is a working day; a holiday is not.
        out["is_weekend"] = np.where(
            is_hol, 1,
            np.where(is_mk, 0, (d.dt.dayofweek >= 5))).astype("int8")
        out["is_weekday"] = (1 - out["is_weekend"]).astype("int8")
        out["month"] = d.dt.month.astype("int8")
        out["day_of_month"] = d.dt.day.astype("int8")

        hol = set(self.holidays)
        dates = pd.DatetimeIndex(sorted(d.unique()))
        pre = {x for x in dates if (x + pd.Timedelta(days=1)) in hol}
        post = {x for x in dates if (x - pd.Timedelta(days=1)) in hol}
        out["is_day_before_holiday"] = d.isin(pre).astype("int8")
        out["is_day_after_holiday"] = d.isin(post).astype("int8")

        w0 = self._w0 if (self._pinned or self.window_origin == "train") else dates.min()
        out["week_of_window"] = ((d - w0).dt.days // 7).astype("int8")
        out["day_of_window"] = ((d - w0).dt.days).astype("int16")

        return self._check(out, df)

    # -- persistence -------------------------------------------------------
    def state(self) -> dict:
        return {
            "holidays": [str(x.date()) for x in self.holidays],
            "makeup": [str(x.date()) for x in self.makeup],
            "makeup_dow": self.makeup_dow,
            "holiday_dow": self.holiday_dow,
            "window_origin": self.window_origin,
            "w0": str(self._w0.date()) if self._w0 is not None else None,
            "pinned": bool(self._pinned),
        }

    def load_state(self, d: dict) -> None:
        if not d:
            return
        self.holidays = pd.to_datetime(list(d["holidays"]))
        self.makeup = pd.to_datetime(list(d["makeup"]))
        self.makeup_dow = int(d["makeup_dow"])
        self.holiday_dow = int(d["holiday_dow"])
        self.window_origin = d["window_origin"]
        self._w0 = pd.Timestamp(d["w0"]) if d.get("w0") else None
        self._pinned = bool(d.get("pinned", False))
