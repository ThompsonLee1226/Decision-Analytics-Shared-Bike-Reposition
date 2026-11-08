#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/data.py
====================
Reading and validating the panel — the *model input contract*.

A panel is a long table with one row per (unit, date, slot) and two counts:

    unit          block_id / tile_id / anything the unit spec produces
    date          YYYY-MM-DD
    slot          0..n-1, the index of a slot in the configured horizon
    arrivals      count of trips ending in the unit-slot
    departures    count of trips starting in the unit-slot

plus optional per-unit constants (`lat`, `lon`) that feed the geometry features,
and any further constant column a caller wants to expose through the
`panel_context` block (a dock count, a capacity).  A panel that violates the
contract is rejected with a message naming the offending column, rather than
failing three hundred lines later inside a groupby.

`slot`, not `hour`
------------------
The index is a **slot**, because the planning horizon is 07:30-10:30 and its
boundaries fall inside clock hours.  A panel that predates that change carries
an `hour` column instead; it is accepted as an alias and renamed, but only when
the configured slot grid is the trivial hourly one (`SlotSpec.is_hourly`).
Otherwise an hourly column cannot describe the horizon and is rejected rather
than silently reinterpreted — that check lives in `sources`, where the slot spec
is known.

Required vs optional
--------------------
`block_id, date, slot, arrivals, departures` are required.  `lat, lon` are
required only when the geometry feature block is enabled — a panel that already
carries x/y need not carry lat/lon.

The name column is configurable (`unit_column`) because the same schema is used
for geohash cells, aligned tiles, and hand-picked catchments.  Renaming is
cheaper than supporting three table layouts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("block_id", "date", "slot", "arrivals", "departures")
#: Legacy name for the slot column, accepted only on a genuinely hourly grid.
SLOT_ALIASES = ("hour",)
GEOMETRY_INPUT = ("lat", "lon")


class PanelError(ValueError):
    """The panel does not satisfy the input contract."""


def read_panel(path: str | Path, unit_column: str = "block_id",
               require_geometry: bool = True,
               sort_rows: bool = False,
               n_slots: int | None = None) -> pd.DataFrame:
    """Load a panel from CSV/CSV.GZ/Parquet and check the contract.

    The file suffix picks the reader; anything ending in ``.gz`` is treated as
    gzipped CSV, anything ending in ``.parquet`` uses pyarrow.  See
    `validate_panel` for what `sort_rows` changes and why it is off by default.
    `n_slots` bounds the slot index; pass the configured slot count so a panel
    from a different horizon is rejected here instead of mis-grouping later.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"panel not found: {p.resolve()}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return validate_panel(df, unit_column=unit_column,
                          require_geometry=require_geometry,
                          sort_rows=sort_rows, n_slots=n_slots)


def validate_panel(df: pd.DataFrame, unit_column: str = "block_id",
                   require_geometry: bool = True,
                   sort_rows: bool = False,
                   n_slots: int | None = None) -> pd.DataFrame:
    """Check columns, dtypes and the (unit, date, slot) key; return a clean copy.

    `sort_rows`
    -----------
    Off by default, and that default is load-bearing rather than lazy.

    xgboost's `subsample` draws its subset **by row position**, not by any stable
    key, so two files with identical contents in a different order do not train
    the same model.  Measured on this panel with the tuned parameters, the same
    rows in the delivered file order score 1.2167 and sorted by
    (date, slot, unit) score 1.2209 — a 0.35% gap that comes from the 60.5%
    subset being a different 60.5%, and nothing else.  That is the size of the
    effect the published numbers are attached to.

    Preserving the input order is therefore what makes a config reproduce a
    published score, and reordering is a real, if small, modelling choice.  Pass
    `sort_rows: true` (or `data.sort_rows` in a config) when a stable output
    matters more than matching a past number: sorted rows are reproducible from
    the file contents alone, file order is reproducible from the file.
    """
    out = df.copy()

    if unit_column != "block_id":
        if unit_column in out.columns:
            out = out.drop(columns=["block_id"], errors="ignore")
            out = out.rename(columns={unit_column: "block_id"})
        elif "block_id" not in out.columns:
            raise PanelError(
                f"unit_column={unit_column!r} not in panel columns "
                f"{sorted(out.columns)}")

    if "slot" not in out.columns:
        for alias in SLOT_ALIASES:
            if alias in out.columns:
                out = out.rename(columns={alias: "slot"})
                break
    else:
        # Both present: they must agree, or the panel is internally inconsistent
        # and which one is "the" index is a coin flip.
        for alias in SLOT_ALIASES:
            if alias in out.columns:
                same = (pd.to_numeric(out["slot"], errors="coerce")
                        == pd.to_numeric(out[alias], errors="coerce")).all()
                if not same:
                    raise PanelError(
                        f"panel carries both 'slot' and {alias!r} and they "
                        f"disagree; keep one")
                out = out.drop(columns=[alias])

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise PanelError(
            f"panel is missing required column(s) {missing}. "
            f"Present: {sorted(out.columns)}")

    if require_geometry:
        miss_geo = [c for c in GEOMETRY_INPUT if c not in out.columns]
        if miss_geo:
            raise PanelError(
                f"geometry feature block is enabled but the panel has no "
                f"{miss_geo}. Either supply them or turn the block off in the "
                f"config (features.blocks: '- geometry').")

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for c in ("slot", "arrivals", "departures"):
        out[c] = pd.to_numeric(out[c], errors="raise").astype("int64")

    lo, hi = 0, (n_slots - 1 if n_slots else 23)
    if not out["slot"].between(lo, hi).all():
        bad = sorted(out.loc[~out["slot"].between(lo, hi), "slot"].unique())[:5]
        raise PanelError(
            f"slot must be {lo}..{hi}; found {bad}. "
            + (f"The configured horizon has {n_slots} slot(s), so a larger index "
               f"means the panel was built for a different horizon."
               if n_slots else
               "Pass the configured slot count so this is checked against the "
               "horizon rather than a fixed 0..23."))

    dup = out.duplicated(["block_id", "date", "slot"])
    if dup.any():
        n = int(dup.sum())
        ex = out.loc[dup, ["block_id", "date", "slot"]].head(3).to_dict("records")
        raise PanelError(f"{n} duplicate (unit, date, slot) key(s), e.g. {ex}")

    if (out[["arrivals", "departures"]] < 0).to_numpy().any():
        raise PanelError("negative counts in arrivals/departures")

    if sort_rows:
        return out.sort_values(["date", "slot", "block_id"]).reset_index(drop=True)
    return out.reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    """One-line summary of a panel, printed at the start of every run."""
    return {
        "rows": int(len(df)),
        "units": int(df.block_id.nunique()),
        "days": int(df.date.nunique()),
        "slots": int(df.slot.nunique()),
        "date_min": str(df.date.min()),
        "date_max": str(df.date.max()),
        "arrivals_sum": int(df.arrivals.sum()),
        "departures_sum": int(df.departures.sum()),
        "zero_share_arrivals": float((df.arrivals == 0).mean()),
        "zero_share_departures": float((df.departures == 0).mean()),
    }

