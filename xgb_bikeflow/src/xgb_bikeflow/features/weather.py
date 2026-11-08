#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/weather.py
================================
Weather block — the one information source that is neither calendar nor target.

Why weather is allowed under a no-lags rule
-------------------------------------------
The ban is on lags of the *target*: nothing that looks at what happened at this
unit on an earlier day or hour.  Weather is not a lag — it is an exogenous driver
known at prediction time (it is in the forecast).  An operator dispatching for
08:00 tomorrow reads the rain forecast; that is exactly why weather belongs in
the model.

What it bought, measured
------------------------
Without any model, rain suppresses demand hard:

    precipitation      arrivals   departures
    0 (dry)               3.06        2.69
    2+ mm                 1.54        0.90       (-50% / -67%)

and the residual check showed the model had the sign backwards: on hours with
>2 mm it predicted 4.17 against an actual 1.63 — it *raised* the prediction when
it should have halved it.  That was the worst cell in the whole error table
(deviance 4.005 against 1.33 when dry).  Adding this block: arrivals 1.3672 →
1.3057, departures 1.5129 → 1.4552.

Why this encoding and not a rain bucket
---------------------------------------
Six encodings were compared on the same folds with the same parameters.  The
continuous amount plus two flags won; collapsing to "did it rain" cost 0.89%,
a four-bucket one-hot cost 0.67%, and dropping the flags entirely cost 0.15%.
The reason the flags are nearly free is that a tree can already split the
continuous column at 2.0 mm — the flag is duplicate information, kept because it
is what makes the fitted 2 mm threshold visible rather than implicit.  The
reason a bucket loses is the usual one: hand-binning throws away resolution a
tree would have chosen for itself.

Why nine columns and not twenty
-------------------------------
Only 164 of 2,064 hours are wet.  Every extra rain interaction is a chance to
over-fit a rare condition, so the set is monotone transforms plus flags, not a
spline basis.  The encoding sweep is what justifies stopping there.

Missing hours stay missing
--------------------------
No imputation.  xgboost routes NaN down a default branch, which is a better
answer than teaching the model that gaps in the weather feed mean sunshine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Context, FeatureBlock

COLUMNS = ("precip_mm", "log_precip", "is_raining", "is_heavy_rain",
           "temp_c", "is_cold", "is_hot", "wind_ms", "cloud_pct")

#: Thresholds, here rather than buried in an expression.
HEAVY_MM = 2.0
COLD_C = 2.0
HOT_C = 26.0

#: Column renames from the Open-Meteo export to the names this block emits.
RENAMES = {
    "time": "ts",
    "precipitation": "precip_mm",
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_ms",
    "cloud_cover": "cloud_pct",
}
KEEP = ("ts", "precip_mm", "temp_c", "wind_ms", "cloud_pct")


def load_weather(path: str | Path) -> pd.DataFrame:
    """Read an hourly weather CSV and normalise it to (ts, precip, temp, ...).

    `ts` is the real timestamp.  Joining on it (rather than on date+hour) is what
    makes this block survive a panel whose date column is a string, a period, or
    a timezone-aware timestamp.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"weather file not found: {p.resolve()}\n"
            f"Point features.weather.path at it in the config, or disable the "
            f"weather block.")
    w = pd.read_csv(p)
    w = w.rename(columns={k: v for k, v in RENAMES.items() if k in w.columns})
    missing = [c for c in KEEP if c not in w.columns]
    if missing:
        raise ValueError(
            f"weather file {p.name} is missing {missing}. Expected the "
            f"Open-Meteo hourly export with at least "
            f"{sorted(RENAMES)} present.")
    w["ts"] = pd.to_datetime(w["ts"])
    return w[list(KEEP)].drop_duplicates("ts").sort_values("ts")


def _flag(mask: pd.Series, source: pd.Series) -> pd.Series:
    """Turn a boolean comparison into 1.0/0.0, or NaN where `source` is missing."""
    return mask.astype(float).where(source.notna(), np.nan)


class WeatherBlock(FeatureBlock):
    """Hourly precipitation, temperature, wind and cloud."""

    name = "weather"
    columns = COLUMNS
    fitted = False
    requires = ()

    def __init__(self, path: str | Path | None = None,
                 table: pd.DataFrame | None = None,
                 heavy_mm: float = HEAVY_MM, cold_c: float = COLD_C,
                 hot_c: float = HOT_C, **options):
        super().__init__(**options)
        self.path = str(path) if path is not None else ""
        self.heavy_mm = float(heavy_mm)
        self.cold_c = float(cold_c)
        self.hot_c = float(hot_c)
        # Loaded lazily: `Bundle.load` constructs every block before handing it
        # the saved state, and the state *is* the table, so touching the file at
        # construction time would make a bundle refuse to load on a machine that
        # no longer has the CSV.
        self._table = table

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            if not self.path:
                raise FileNotFoundError(
                    "the weather block has no table loaded and no path set; "
                    "give it `path:` in the config or restore it from a bundle")
            self._table = load_weather(self.path)
        return self._table

    @table.setter
    def table(self, value: pd.DataFrame) -> None:
        self._table = value

    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        # The slot index is not an hour offset.  On a 07:30-10:30 half-hour grid
        # slot 0 is 07:30, so adding it as hours would put every row an hour
        # early and join it to the wrong weather.  Ask the horizon instead.
        minutes = ctx.minute_of_slot(df["slot"].to_numpy())
        ts = pd.to_datetime(df["date"]) + pd.to_timedelta(minutes, unit="m")
        # The weather table is hourly.  A sub-hourly horizon therefore has no
        # entry of its own, and the value for the containing hour is used rather
        # than interpolated: interpolation would invent a rainfall rate between
        # two hourly totals, and the columns here are already rates and flags
        # (is_raining, is_heavy_rain) that do not average into anything
        # meaningful.  Flooring keeps every row's weather equal to a number that
        # was actually observed.
        if ctx.slots is not None and ctx.slots.width_minutes < 60:
            ts = ts.dt.floor("h")
        out = pd.DataFrame({"ts": ts})
        out = out.merge(self.table, on="ts", how="left")
        out["log_precip"] = np.log1p(out["precip_mm"])
        # The flags must stay NaN where the reading is missing.  Comparing NaN
        # directly gives False, so `(precip > 0)` turns "no weather row for this
        # date" into a confident "it is not raining" -- while precip_mm beside it
        # stays NaN and takes xgboost's default branch.  Two columns then tell the
        # model opposite stories about the same missing hour.  `where` on the
        # notna mask keeps the pair consistent: unknown precipitation means an
        # unknown rain flag, and the tree decides how to treat it.
        out["is_raining"] = _flag(out["precip_mm"] > 0, out["precip_mm"])
        out["is_heavy_rain"] = _flag(out["precip_mm"] >= self.heavy_mm,
                                     out["precip_mm"])
        out["is_cold"] = _flag(out["temp_c"] <= self.cold_c, out["temp_c"])
        out["is_hot"] = _flag(out["temp_c"] >= self.hot_c, out["temp_c"])
        return self._check(out, df)

    def state(self) -> dict:
        # The fitted state is the *table*, not a summary: a saved model that
        # cannot see the weather hours for the dates it will score is not
        # reproducible.  Stored as column arrays to keep the JSON compact.
        t = self.table
        return {
            "path": self.path,
            "thresholds": {"heavy_mm": self.heavy_mm, "cold_c": self.cold_c,
                           "hot_c": self.hot_c},
            "ts": [str(x) for x in pd.to_datetime(t["ts"])],
            "values": {c: [None if pd.isna(v) else float(v) for v in t[c]]
                       for c in KEEP if c != "ts"},
        }

    def load_state(self, d: dict) -> None:
        if not d:
            return
        self.path = d.get("path", self.path)
        th = d.get("thresholds", {})
        self.heavy_mm = float(th.get("heavy_mm", self.heavy_mm))
        self.cold_c = float(th.get("cold_c", self.cold_c))
        self.hot_c = float(th.get("hot_c", self.hot_c))
        self.table = pd.DataFrame({
            "ts": pd.to_datetime(d["ts"]),
            **{c: np.asarray(d["values"][c], dtype="float64") for c in KEEP
               if c != "ts"},
        })
