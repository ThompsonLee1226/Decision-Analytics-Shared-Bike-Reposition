#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/__init__.py
=================================
The feature registry and the assembler.

Six registered families, five of which any shipped config enables:

    calendar      day type, holiday adjacencies, window index   12 cols
    geometry      x/y metres, radius, bearing                    5 cols
    trend         how busy this date is relative to training     1 col    fitted
    profile       (unit, slot) level, shape and variability      6 cols   fitted
    weather       rain, temperature, wind, cloud                 9 cols
    panel_context a static per-unit column, when the panel has one 0-1 col

`panel_context` is the odd one out and ships disabled: it exists for a genuine
per-unit constant (dock count, capacity) and emits nothing when the panel has no
such column.  See `features/context.py` for the leakage it must never carry.

`FeatureSet` assembles whichever of them the config enables, in a fixed order,
and hands back a plain frame of float columns.  The order is
calendar → geometry → trend → profile → weather → panel_context, which is the
order the original `build_features` used; see `features/calendar.py` for why the
order is load-bearing rather than cosmetic.

Ordering inside `profile` and `weather` is the historical column order, not
alphabetical.  Do not sort it.
"""

from __future__ import annotations

import pandas as pd

from .base import KEY_COLS, Context, FeatureBlock
from .calendar import CalendarBlock
from .context import PanelContextBlock
from .geometry import GeometryBlock
from .profile import ProfileBlock
from .trend import TrendBlock
from .weather import WeatherBlock

#: Registry: YAML block name -> class.  Order here is the emission order.
REGISTRY: dict[str, type[FeatureBlock]] = {
    "calendar": CalendarBlock,
    "geometry": GeometryBlock,
    "trend": TrendBlock,
    "profile": ProfileBlock,
    "weather": WeatherBlock,
    "panel_context": PanelContextBlock,
}

#: Blocks whose emitted columns depend on statistics from the training rows.
FITTED = tuple(name for name, cls in REGISTRY.items() if cls.fitted)


def _join(work: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """Fold a block's columns into the working frame.

    Three rules, each of which fixes a way the naive version failed silently:

    1. **Emitted columns win.**  A panel ships raw `dow` / `is_weekend` computed
       from the weekday, but the calendar block deliberately *corrects* those
       for make-up workdays and holidays.  Letting the input win meant
       `trend_scale` read the raw flag: on 2026-04-06 (a Monday holiday) the
       block saw `is_weekend=0`, took the weekday slope, and returned 3.0 where
       the legacy pipeline returned 1.30 — 480 rows wrong, no error raised.
       The block's value is the considered one, so it replaces the input.

    2. **No duplicate names.**  `calendar` republishes `slot` so its own output
       is readable on its own; two columns called `slot` make a later
       `groupby("slot")` raise `ValueError: Grouper for 'slot' not
       1-dimensional`.  Covered by rule 1.

    3. **Positional alignment.**  Blocks return `reset_index(drop=True)`, but
       `work` arrives from a panel slice carrying the panel's index (fold 1's
       validation rows are 10080..20159).  `concat(axis=1)` aligns on index, so
       it would *union* the ranges and hand back 20,160 rows for 10,080 inputs.
       Row i of `add` was computed from row i of `work`, so align by position.
    """
    out = work.drop(columns=[c for c in add.columns if c in work.columns])
    return pd.concat([out.reset_index(drop=True),
                      add.reset_index(drop=True)], axis=1)


class FeatureSet:
    """An ordered list of enabled blocks; fit once, transform many times."""

    def __init__(self, blocks: list[FeatureBlock], targets: tuple[str, ...],
                 slots=None):
        self.blocks = list(blocks)
        self.targets = tuple(targets)
        # The horizon the slot column indexes into.  A block that turns a slot
        # index into a clock time cannot do so without it, and the failure is
        # silent (07:30 read as 07:00), so the set carries it rather than each
        # block being trusted to be told.
        self.slots = slots
        self._fitted = False

    # -- assembly ----------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """Every column the enabled blocks emit, in emission order."""
        cols: list[str] = []
        for b in self.blocks:
            for c in b.columns:
                if c not in cols:
                    cols.append(c)
        return cols

    @property
    def block_columns(self) -> dict[str, list[str]]:
        """Which block owns which columns — used for reporting and ablation."""
        return {b.name: list(b.columns) for b in self.blocks}

    def describe(self) -> list[dict]:
        return [{"block": b.name, "fitted": b.fitted,
                 "n_columns": len(b.columns), "columns": list(b.columns)}
                for b in self.blocks]

    def notes(self) -> list[str]:
        """Everything the fitted blocks want the user to know, in block order."""
        out: list[str] = []
        for b in self.blocks:
            try:
                out.extend(b.notes())
            except Exception:          # a note must never break a run
                continue
        return out

    # -- lifecycle ---------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FeatureSet":
        """Fit every fitted block on the training rows.

        The frame handed to `fit` keeps the target columns: fitting is allowed to
        read the target, it is emitting features from it that is not.
        """
        ctx = Context(targets=self.targets, slots=self.slots)
        # Blocks are fitted in order so a block may rely on a column emitted by
        # an earlier one (trend needs the calendar's `dow` / `is_weekend`).
        #
        # Both count columns are always carried into `fit`, not just the targets
        # being modelled: the profile block emits arrival *and* departure shape
        # columns whoever is the target, and a run that models only arrivals
        # would otherwise have no `departures` to take a mean of.
        fit_cols = [c for c in (*KEY_COLS, "arrivals", "departures",
                                *self._available_inputs(train))
                    if c in train.columns]
        work = train[fit_cols].copy()
        for b in self.blocks:
            b.fit(work, ctx)
            if b.fitted or b.name in ("calendar", "geometry"):
                work = _join(work, b.transform(work, ctx))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Emit the feature frame for `df` (same rows, same order, float64)."""
        if not self._fitted and any(b.fitted for b in self.blocks):
            raise RuntimeError(
                "FeatureSet.transform called before fit(); call .fit(train) first")
        ctx = Context(targets=self.targets, slots=self.slots)
        # The leak guard: the frame a block sees has no target columns, so a
        # transform cannot read `arrivals` even by accident.
        clean = df.drop(columns=[c for c in self.targets if c in df.columns],
                        errors="ignore")
        # Calendar and geometry publish columns that later blocks consume, so
        # they are transformed first and merged into the working frame.
        work = clean.copy()
        pieces: list[pd.DataFrame] = []
        for b in self.blocks:
            add = b.transform(work, ctx)
            if isinstance(b, PanelContextBlock) and not b.available(work):
                continue
            pieces.append(add)
            work = _join(work, add)
        out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
        out = out.loc[:, ~out.columns.duplicated()]
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        return out.reset_index(drop=True)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # -- persistence -------------------------------------------------------
    def state(self) -> dict:
        return {"blocks": [{"name": b.name, "state": b.state()}
                           for b in self.blocks]}

    def load_state(self, d: dict) -> "FeatureSet":
        saved = {e["name"]: e["state"] for e in d.get("blocks", [])}
        for b in self.blocks:
            if b.name in saved:
                b.load_state(saved[b.name])
        self._fitted = True
        return self

    # -- helpers -----------------------------------------------------------
    def _available_inputs(self, df: pd.DataFrame) -> list[str]:
        need: list[str] = []
        for b in self.blocks:
            for c in b.requires:
                if c not in need:
                    need.append(c)
        return [c for c in need if c in df.columns]


def build(cfg: dict, targets: tuple[str, ...] = ("arrivals", "departures"),
          slots=None) -> FeatureSet:
    """Instantiate the blocks a config enables.

    `cfg` is the `features:` section of a run config:

        features:
          blocks:
            - calendar
            - geometry
            - trend
            - profile
            - weather
          options:
            weather: {path: data/weather_hourly.csv}
            profile: {fallback: zero}

    A block may also be given as ``- name: weather`` with its options inline;
    both spellings are accepted because YAML users reach for both.

    `slots` is the horizon the `slot` column indexes into (a
    `slots.SlotSpec`).  It is passed to the blocks rather than looked up,
    because a block that maps a slot index to a clock time has no other way to
    know where slot 0 starts.
    """
    spec = cfg or {}
    entries = spec.get("blocks")
    if entries is None:
        entries = ["calendar", "geometry", "trend", "profile", "weather"]
    options = dict(spec.get("options") or {})

    blocks: list[FeatureBlock] = []
    for entry in entries:
        if isinstance(entry, str):
            name, inline = entry, {}
        elif isinstance(entry, dict):
            inline = dict(entry)
            name = inline.pop("name", None)
            if name is None:
                raise ValueError(f"feature block entry {entry!r} has no 'name'")
        else:
            raise ValueError(f"feature block entry {entry!r} is neither a name "
                             f"nor a mapping with a 'name'")
        if name not in REGISTRY:
            raise ValueError(
                f"unknown feature block {name!r}; known blocks: "
                f"{sorted(REGISTRY)}")
        opts = {**options.get(name, {}), **inline}
        blocks.append(REGISTRY[name](**opts))
    return FeatureSet(blocks, targets, slots=slots)


__all__ = [
    "REGISTRY", "FITTED", "FeatureSet", "build", "Context", "FeatureBlock",
    "KEY_COLS", "CalendarBlock", "GeometryBlock", "TrendBlock", "ProfileBlock",
    "WeatherBlock", "PanelContextBlock",
]
