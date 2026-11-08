#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/rollup.py
======================
From a cell-level forecast to the things the network is actually about: a
gathering point's expected departures, and an arc's expected load.

Why this is not a groupby
-------------------------
The two rollups look alike and are not:

    gathering points   the member cells are disjoint, so a GP's total is the
                       sum of its cells' predictions and the totals may
                       themselves be added up across GPs.  A share of the city
                       is meaningful.

    arcs               an arc's service cells overlap heavily — 5,060 distinct
                       cells carry 28,686 arc memberships, a mean multiplicity
                       of 5.7, because a cell belongs to every arc whose
                       patches reach it.  An arc's catchment total is a real
                       number ("how many bikes appear within reach of this
                       route") but adding two arcs' totals double-counts the
                       cells they share.

`Rollup.partitioned` says which one you are holding, and `check_total` exists so
a caller that wants to add them up has to say so deliberately.  This is the same
trap the grid module refuses on the spatial axis, one level up.

What "a rate for an arc" means
------------------------------
An arc is a directed road route between two GPs.  The demand attached to it is
the demand in the patches its vehicles can service — not the demand *on* the
road, which is a flow, not a count of bikes becoming available.  So the number
reported per arc is a catchment total, and `per_cell` is the same total divided
by the number of distinct cells, which is the only version that can be compared
between a 2-cell arc and a 252-cell one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("arrivals", "departures")


class RollupError(ValueError):
    """The network file or the predictions cannot be combined."""


# ---------------------------------------------------------------------------
# the network file
# ---------------------------------------------------------------------------

@dataclass
class Network:
    """A GP/arc structure read from `network_v1.json` (or any file of that shape)."""

    version: str
    gps: dict[str, dict]
    arcs: dict[int, dict]
    rows: list[dict]
    raw: dict = field(default_factory=dict, repr=False)

    # -- convenience -------------------------------------------------------
    @property
    def gp_cells(self) -> set[str]:
        out: set[str] = set()
        for g in self.gps.values():
            out.update(g["cells"])
        return out

    @property
    def arc_cells(self) -> set[str]:
        out: set[str] = set()
        for a in self.arcs.values():
            out.update(a["service_cells"])
        return out

    @property
    def cells(self) -> set[str]:
        return {r["geohash"] for r in self.rows}

    def describe(self) -> dict:
        gp_only = self.gp_cells - self.arc_cells
        arc_only = self.arc_cells - self.gp_cells
        return {
            "version": self.version,
            "n_gps": len(self.gps),
            "n_arcs": len(self.arcs),
            "n_rows": len(self.rows),
            "n_cells": len(self.cells),
            "n_gp_cells": len(self.gp_cells),
            "n_arc_cells": len(self.arc_cells),
            "n_gp_only_cells": len(gp_only),
            "n_arc_only_cells": len(arc_only),
            "arc_cell_multiplicity": round(
                sum(len(a["service_cells"]) for a in self.arcs.values())
                / max(len(self.arc_cells), 1), 2),
        }


def load_network(path: str | Path) -> Network:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"network file not found: {p}")
    blob = json.loads(p.read_text(encoding="utf-8"))
    for key in ("gps", "arcs", "rows"):
        if key not in blob:
            raise RollupError(
                f"{p.name} has no {key!r} list; keys were {sorted(blob)}. "
                f"This file is not a network definition.")
    gps = {g["gp_id"]: g for g in blob["gps"]}
    arcs = {int(a["arc_id"]): a for a in blob["arcs"]}
    return Network(version=str(blob.get("version", "?")), gps=gps, arcs=arcs,
                   rows=list(blob["rows"]), raw=blob)


# ---------------------------------------------------------------------------
# rollups
# ---------------------------------------------------------------------------

def _cell_index(preds: pd.DataFrame) -> pd.DataFrame:
    """Validate a cell-level prediction frame and normalise its columns."""
    need = {"block_id", "date", "slot", "pred"}
    missing = need - set(preds.columns)
    if missing:
        raise RollupError(
            f"cell predictions are missing {sorted(missing)}; columns are "
            f"{list(preds.columns)}")
    out = preds[["block_id", "date", "slot", "pred"]].copy()
    out["pred"] = out["pred"].astype(float)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def gp_rollup(net: Network, preds: pd.DataFrame) -> pd.DataFrame:
    """Per (gp, date, slot): the summed forecast over the GP's cells.

    Also returns `n_cells` (member cells present in the predictions) and
    `missing_cells`, so a GP that only half-arrived in the panel is visible rather
    than quietly reported as a smaller dock.
    """
    cell = _cell_index(preds)
    rows = []
    for gid, g in net.gps.items():
        cells = set(g["cells"])
        sub = cell[cell.block_id.isin(cells)]
        if sub.empty:
            continue
        have = set(sub.block_id.unique())
        agg = (sub.groupby(["date", "slot"], as_index=False)["pred"].sum()
               .rename(columns={"pred": "pred_total"}))
        agg.insert(0, "gp_id", gid)
        agg["n_cells"] = len(have)
        agg["missing_cells"] = len(cells - have)
        agg["pred_per_cell"] = agg["pred_total"] / max(len(have), 1)
        rows.append(agg)
    if not rows:
        raise RollupError(
            "no gathering point has a single cell in these predictions. The "
            "network and the panel are at different grid levels, or describe "
            "different areas.")
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["gp_id", "date", "slot"], ignore_index=True)


def arc_rollup(net: Network, preds: pd.DataFrame) -> pd.DataFrame:
    """Per (arc, date, slot): the forecast summed over the arc's catchment.

    This is a catchment total, **not** additive across arcs: see the module
    docstring and `Rollup.partitioned`.
    """
    cell = _cell_index(preds)
    rows = []
    for aid, a in net.arcs.items():
        cells = set(a["service_cells"])
        sub = cell[cell.block_id.isin(cells)]
        if sub.empty:
            continue
        have = set(sub.block_id.unique())
        agg = (sub.groupby(["date", "slot"], as_index=False)["pred"].sum()
               .rename(columns={"pred": "pred_total"}))
        agg.insert(0, "arc_id", aid)
        agg.insert(1, "from_gp", a["from_gp"])
        agg.insert(2, "to_gp", a["to_gp"])
        agg["n_cells"] = len(have)
        agg["pred_per_cell"] = agg["pred_total"] / max(len(have), 1)
        rows.append(agg)
    if not rows:
        raise RollupError("no arc has a single service cell in these predictions")
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["arc_id", "date", "slot"], ignore_index=True)


def row_rollup(net: Network, preds: pd.DataFrame,
               kinds: tuple[str, ...] = ("gp", "arc")) -> pd.DataFrame:
    """One tidy frame over both kinds, keyed by the network file's `row_id`.

    This is the output the brief asks for: a forecast on every gathering point and
    every arc, in one table.  `row_id` is the network file's own id (`<cell>` for a
    GP row, `arc<id>:<cell>` for an arc row), so a row here can be traced back to a
    member cell without re-deriving anything.
    """
    cell = _cell_index(preds)
    by_cell = cell.groupby(["block_id", "date", "slot"])["pred"].sum()

    parts = []
    if "gp" in kinds:
        g = gp_rollup(net, cell)
        parts.append(g.rename(columns={"gp_id": "id"}).assign(kind="gp",
                                                             entity="gp"))
    if "arc" in kinds:
        a = arc_rollup(net, cell)
        parts.append(a.rename(columns={"arc_id": "id"}).assign(kind="arc",
                                                              entity="arc"))
    out = pd.concat(parts, ignore_index=True)
    del by_cell
    return out.sort_values(["kind", "id", "date", "slot"], ignore_index=True)


# ---------------------------------------------------------------------------
# scoring the rollups
# ---------------------------------------------------------------------------

def with_actuals(net: Network, panel: pd.DataFrame, roll: pd.DataFrame,
                 target: str) -> pd.DataFrame:
    """Attach the observed total for each rolled-up row.

    The observed total is built the same way as the forecast — sum the member
    cells' counts — so the comparison is between two numbers computed the same
    way.  Doing it any other way (observing the GP directly, say) would compare a
    sum of predictions with something that is not that sum.

    Dispatching is **per row**, on that row's `kind`.  An earlier version read
    `roll["kind"].iloc[0]` and picked one membership map for the whole frame,
    which silently attached arc totals to the GP rows whenever the frame happened
    to start with an arc (it does: `row_rollup` sorts by kind, and "arc" < "gp").
    """
    if target not in panel.columns:
        raise RollupError(f"{target!r} is not a column of the panel")
    if "kind" not in roll.columns:
        raise RollupError(
            "with_actuals needs a 'kind' column to know which membership map "
            "applies to each row; build the frame with row_rollup")
    obs = panel[["block_id", "date", "slot", target]].copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")

    # One index per kind, built once: the panel is large and this is called for
    # every entity.
    maps = {"gp": {gid: set(g["cells"]) for gid, g in net.gps.items()},
            "arc": {aid: set(a["service_cells"]) for aid, a in net.arcs.items()}}
    totals: dict[str, dict] = {}
    for kind, members in maps.items():
        if kind not in set(roll["kind"]):
            continue
        per_id = {}
        for rid, cells in members.items():
            sub = obs[obs.block_id.isin(cells)]
            if sub.empty:
                continue
            per_id[rid] = sub.groupby(["date", "slot"])[target].sum()
        totals[kind] = per_id

    out = roll.copy()
    out["actual_total"] = [
        float(totals.get(r.kind, {}).get(r.id,
              pd.Series(dtype=float)).get((r.date, r.slot), np.nan))
        for r in out.itertuples()]
    return out
