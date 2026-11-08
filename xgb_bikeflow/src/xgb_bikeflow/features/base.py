#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/base.py
=============================
The feature-block protocol.

A feature block is a small object that owns one *family* of columns and knows
three things: what it needs as input, how to fit itself on training rows, and how
to emit its columns for any row set.  Blocks are the unit of the "which features
do we use" decision, so adding or dropping a family is a line in a YAML file
rather than an edit to a 400-line function.

Why blocks and not a list of column names
-----------------------------------------
Two of the five families are *fitted*: their values depend on statistics taken
from the training rows (`trend`, `profile`).  A bare column list cannot express
that, so the same training frame would have to be threaded through by hand — which
is how an earlier `build_features(panel, train)` ended up taking the training frame
as its second argument and being called twice, once per split, with the caller
responsible for passing the *right* one.  In the pipeline the fitted blocks are
fitted once and then applied to both splits, so the caller cannot get that wrong.

The leak guard
--------------
`transform` receives a view of the frame **with the target columns removed**.  A
fitted block therefore cannot read `arrivals` while emitting a feature, even by
accident.  That is stronger than a convention in a docstring: the column is not
there.  Blocks that legitimately need the target during `fit` still have it,
because `fit` gets the unmodified training frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Columns the pipeline always provides to a block's `fit`.
KEY_COLS = ("block_id", "date", "slot")


@dataclass
class Context:
    """Everything a block may need beyond the frame itself."""

    #: target column names; removed from the frame handed to `transform`
    targets: tuple[str, ...] = ("arrivals", "departures")
    #: extra options from the YAML block, already merged with defaults
    options: dict[str, Any] = field(default_factory=dict)
    #: the horizon the slot column indexes into.  Blocks that turn a slot index
    #: into a clock time (weather) need it; without it a 30-minute slot would be
    #: read as an hour offset and every weather join would land on 07:00 instead
    #: of 07:30.
    slots: Any = None

    def opt(self, key: str, default=None):
        return self.options.get(key, default)

    def minute_of_slot(self, slot) -> int:
        """Clock minute (past midnight) of a slot index."""
        if self.slots is None:
            raise ValueError(
                "this block needs the slot grid to map a slot index to a clock "
                "time, but Context.slots is unset. The pipeline always sets it; "
                "constructing a Context by hand must pass slots=SlotSpec(...).")
        s = pd.Series(np.asarray(slot))
        # Vectorised through the spec's own arithmetic so it cannot disagree with
        # how the panel was built.
        return (self.slots.start_minute + s.to_numpy() * self.slots.width_minutes)


class FeatureBlock:
    """Base class.  Subclasses set `name` / `columns` and implement the rest.

    Contract
    --------
    fit(train, ctx)
        Optional.  Learn whatever the block needs from training rows only.
        Must not modify `train`.
    transform(df, ctx) -> DataFrame
        Return a frame with **the same rows in the same order** and exactly
        `self.columns` as its columns.  The frame handed in has no target
        columns; `block_id`, `date` and `slot` are always present.
    state() / load_state(d)
        JSON-serialisable snapshot of whatever `fit` learned.  Required for any
        block whose output depends on fitted values — otherwise a saved model
        cannot be applied to new rows without refitting on data that may have
        changed underneath it.
    notes() -> list[str]
        Optional.  Things about the *fitted* state the user should hear: a day
        type that had to borrow its trend, a fallback that fired.  Empty by
        default.  These are not errors — the model runs — but a run that quietly
        guesses is worse than one that says where it guessed.
    """

    name: str = "base"
    columns: tuple[str, ...] = ()
    #: True when `transform` output depends on `fit`
    fitted: bool = False
    #: input columns required from the panel (besides the key columns)
    requires: tuple[str, ...] = ()

    def __init__(self, **options):
        self.options = options

    # -- lifecycle ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, ctx: Context) -> None:  # noqa: D401
        """Learn from training rows.  No-op for non-fitted blocks."""
        return None

    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:  # noqa: D401
        raise NotImplementedError

    # -- persistence -------------------------------------------------------
    def state(self) -> dict:
        """Fitted parameters, JSON-ready.  Empty for non-fitted blocks."""
        return {}

    def load_state(self, d: dict) -> None:
        """Restore what `state()` produced."""
        if d:
            raise ValueError(f"{type(self).__name__} has no fitted state to load")

    def notes(self) -> list[str]:
        """Worth-saying facts about the fitted state.  Empty by default."""
        return []

    # -- helpers -----------------------------------------------------------
    def _check(self, out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        if len(out) != len(df):
            raise ValueError(
                f"block {self.name!r} returned {len(out)} rows for {len(df)} "
                f"input rows")
        missing = [c for c in self.columns if c not in out.columns]
        if missing:
            raise ValueError(f"block {self.name!r} did not emit {missing}")
        return out[list(self.columns)].reset_index(drop=True)


def as_float(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Cast the emitted columns to float64, the dtype xgboost sees."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("float64")
    return out
