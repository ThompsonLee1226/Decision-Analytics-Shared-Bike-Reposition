#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/rows.py
====================
The row-selection axis: which units are trained on, which are scored, and how
much the rows you do not care about are allowed to pull on the model.

Three settings that were previously three scripts
-------------------------------------------------
An earlier study trained on 20 named units and scored on the same 20; another
trained on all 2,000 and scored on those 20; a third trained on everything but
multiplied the loss of off-target rows by a constant (0.2, 0.5, 0).  Those are
all the same pipeline with different numbers, so here they are numbers:

    train_units        all | top_n | list | file
    score_units        None (= same as train) | same choices
    off_target_weight  1.0 = plain training, 0.0 = hard mask, 0.2 = soft

`off_target_weight` is the one that needs explaining
----------------------------------------------------
xgboost accepts a per-row weight, so weighted training is exact rather than
approximated by subsampling.  The Poisson objective with sample weight w is
``w * (mu - y*log(mu))`` up to a constant, i.e. the deviance scaled — which is
why weight 0 really does remove a row and weight 0.2 really does down-weight it
by 80%, with no re-derivation needed.

Two traps this module closes
----------------------------
1. **The mask must be applied at scoring time too.**  A model trained on 2,000
   units must not be allowed to look good on unit #1,500, so `score_units`
   defaults to `train_units` only when they are the same thing; when they differ
   the scoring mask is explicit (`eval_blocks` in the original).
2. **Top-N must be ranked by something, and the ranking must be fixed across
   folds.**  Ranking inside each fold would leak: fold 3 would pick its top 20
   using data from folds 1-2.  A `top_n` selection is therefore computed **once,
   on the whole panel**, from the panel's own totals — which is a choice about
   what the study area *is*, not a prediction.  `units_file` exists so the
   selection can be pinned instead of recomputed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGETS_DEFAULT = ("arrivals", "departures")


class RowSpecError(ValueError):
    """The row selection is inconsistent or would silently leak."""


def _read_units_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rows.units_file not found: {p.resolve()}")
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        col = "block_id" if "block_id" in df.columns else df.columns[0]
        return [str(x) for x in df[col].dropna().unique()]
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def select_units(panel: pd.DataFrame, rows: dict,
                 targets: tuple[str, ...] = TARGETS_DEFAULT) -> tuple[list[str], dict]:
    """Resolve `rows.train_units` against the panel; return (ids, how)."""
    mode = rows.get("train_units", "all")
    available = sorted(panel.block_id.unique())
    info: dict = {"mode": mode if isinstance(mode, str) else "list"}

    if isinstance(mode, (list, tuple, set)):
        want = [str(x) for x in mode]
        missing = sorted(set(want) - set(available))
        if missing:
            raise RowSpecError(
                f"rows.train_units lists {len(missing)} unit(s) that are not in "
                f"the panel, e.g. {missing[:3]}. The panel has "
                f"{len(available)} units.")
        chosen = [u for u in available if u in set(want)]
        info["n_requested"] = len(want)
    elif mode == "all":
        chosen = available
    elif mode == "top_n":
        n = int(rows.get("top_n") or 20)
        tot = panel.groupby("block_id")[list(targets)].sum().sum(axis=1)
        chosen = sorted(tot.sort_values(ascending=False).head(n).index.tolist())
        info["criterion"] = ("gross " + "+".join(targets) +
                             " over all rows of the panel; computed once, not "
                             "per fold")
        info["n_requested"] = n
    elif mode == "file":
        chosen = [u for u in _read_units_file(rows["units_file"])
                  if u in set(available)]
        info["n_requested"] = len(_read_units_file(rows["units_file"]))
    else:
        raise RowSpecError(
            f"rows.train_units = {mode!r}; expected 'all', 'top_n', 'file', or "
            f"the explicit list of unit ids")

    if not chosen:
        raise RowSpecError("row selection produced an empty unit set")
    info["n_units"] = len(chosen)
    if info.get("n_requested") and info["n_requested"] != len(chosen):
        info["warning"] = (f"requested {info['n_requested']} unit(s), kept "
                           f"{len(chosen)} present in the panel")
    return chosen, info


def resolve_score_units(train_units: list[str], rows: dict,
                        panel: pd.DataFrame,
                        targets: tuple[str, ...] = TARGETS_DEFAULT
                        ) -> tuple[list[str], dict]:
    """Resolve `rows.score_units`; defaults to the trained set."""
    raw = rows.get("score_units")
    if raw in (None, "same"):
        return list(train_units), {"mode": "same_as_train"}
    sub = dict(rows)
    sub["train_units"] = raw
    ids, info = select_units(panel, sub, targets)
    info["mode"] = f"explicit:{info['mode']}"
    return ids, info


def row_weights(df: pd.DataFrame, score_units: set[str],
                off_target_weight: float) -> np.ndarray:
    """Per-row training weight: 1 on score units, `off_target_weight` elsewhere."""
    w = np.ones(len(df), dtype=float)
    if off_target_weight >= 1.0:
        return w
    off = ~df["block_id"].isin(score_units).to_numpy()
    w[off] = float(off_target_weight)
    return w


def score_mask(df: pd.DataFrame, score_units: set[str],
               slots: object = "all") -> np.ndarray:
    """Boolean mask of the rows that count toward the reported score."""
    m = df["block_id"].isin(score_units).to_numpy()
    if slots not in (None, "all"):
        hs = set(int(h) for h in slots)
        m &= df["slot"].isin(hs).to_numpy()
    return m


def slot_filter(df: pd.DataFrame, slots: object) -> pd.DataFrame:
    """Restrict a frame to `slots` (a list) or leave it alone ('all').

    The slots here are indices into the configured horizon, not clock hours.  On
    the 07:30-10:30 grid slot 0 is 07:30, so filtering to "slot 0" selects the
    first half hour of the horizon and not the 00:00 hour.
    """
    if slots in (None, "all"):
        return df
    hs = [int(h) for h in slots]
    out = df[df["slot"].isin(hs)]
    if out.empty:
        raise RowSpecError(f"rows.train_slots={slots} selected no rows")
    return out
