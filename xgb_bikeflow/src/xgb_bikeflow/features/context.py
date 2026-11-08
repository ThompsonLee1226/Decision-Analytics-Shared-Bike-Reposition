#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/context.py
================================
Panel context block — a stable, per-unit attribute carried by the panel itself.

The canonical example is the number of docks or the physical capacity of a
station: a fact about the place that does not change with the day's demand, which
the model can use to tell a small station from a large one.

Why it is a block and not a passthrough
---------------------------------------
The column is optional: a panel of hand-picked catchments has nothing to carry,
and the block must then emit nothing rather than a column of NaN that xgboost
would happily split on.

But "optional" has to be decided from the column's *presence*, not from its
variance in whatever slice the block happens to be handed.  An earlier version
required `nunique > 1`, which looked like a sensible guard against a useless
constant column and was in fact a bug: a single fold's training slice can be
constant while the panel is not.  The block would then emit the column for the
training slice and drop it for the validation slice, and the pipeline's column
intersection would quietly remove the feature from the model — a different
feature set per fold, with nothing in the output to say so.  Presence is a
property of the data contract; variance is a property of the sample.

What must NOT go in here
------------------------
Anything computed from the trips being predicted.  An earlier pipeline carried a
`n_cells_touched` column defined as "how many distinct cells inside this unit saw
a trip today", counted from the same rows that produce the target.  At unit shape
(1, 1) it is exactly `arrivals + departures > 0` — a perfect predictor of the
zero/non-zero structure of the label, available only because the label had already
been observed.  It scored well and was pure leakage: a planner does not know how
many cells will be touched before the morning happens.

`build_panel` therefore no longer emits it, and no shipped config turns this block
on.  `tests/test_golden.py::test_no_feature_is_derived_from_the_label` fails if any
two-valued panel column lines up with the label's own support, which is the general
form of the mistake.

A unit-level static attribute (dock count, capacity) is a different thing and is
what this block is for: a property of the infrastructure, not of the demand.  The
block stays in the registry because that case is real and a caller with a capacity
column should not have to write a block to use it.
"""

from __future__ import annotations

import pandas as pd

from .base import Context, FeatureBlock


class PanelContextBlock(FeatureBlock):
    """Emit the configured column when the panel carries a usable version of it."""

    name = "panel_context"
    columns = ("n_cells_touched",)
    fitted = False
    #: may legitimately emit zero columns
    optional = True

    def __init__(self, column: str = "n_cells_touched", **options):
        super().__init__(**options)
        self.column = column
        self.columns = (column,)
        #: None until we have seen the frame the model was actually built from.
        #: Once known, `requires` reports the answer, because a block that emits
        #: a column it read from the input is an input requirement, not a
        #: derivation -- and `schema.json` has to say so or a caller has no way
        #: to know the column is needed.
        self._emitted: bool | None = None

    @property
    def requires(self) -> tuple[str, ...]:
        return (self.column,) if self._emitted else ()

    def available(self, df: pd.DataFrame) -> bool:
        """Present, numeric, and not entirely missing or negative."""
        if self.column not in df.columns:
            return False
        v = pd.to_numeric(df[self.column], errors="coerce")
        return bool(v.notna().any() and (v >= 0).any())

    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        present = self.available(df)
        # Record the decision once: it is a property of the panel, and a
        # validation slice with a constant column must not flip it.
        if self._emitted is None or present:
            self._emitted = present
        if present:
            out[self.column] = pd.to_numeric(df[self.column],
                                             errors="coerce").astype("float64")
        else:
            # Return an empty frame with the declared columns so the caller's
            # concatenation stays shape-consistent; the pipeline drops it.
            out[self.column] = float("nan")
        return out[[self.column]]

    def state(self) -> dict:
        return {"column": self.column, "emitted": bool(self._emitted)}

    def load_state(self, data: dict) -> None:
        self.column = data.get("column", self.column)
        self.columns = (self.column,)
        self._emitted = bool(data.get("emitted", False))
