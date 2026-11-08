#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/split.py
=====================
Walk-forward folds, and the calibration rule that goes with them.

Why not a random split
----------------------
A random row split would put 08:00 of week 8 into training while predicting
08:00 of week 6.  Even with lags banned, the same-hour profile block is fitted on
training days, so a random split would fit it on future days — the model would be
scored on days whose profile it had already seen.  Walk-forward is the only split
under which "predict tomorrow's 08:00" means what it says.

The fold geometry
-----------------
Contiguous day blocks; block 1 trains only, blocks 2-4 each validate in turn,
always forward — so a fold's score is also a read on whether more history helps.
(It does not, uniformly: the last fold, with the most history, is often the worst,
because it validates the May demand ramp.)

Two ways to place the block edges, and why the second is the default
--------------------------------------------------------------------
`walk_forward` splits the *distinct days it sees* into equal-count blocks.  That
is fine on a panel with every calendar day in it, and wrong the moment a filter
removes some: `days: workday` drops every Saturday and Sunday, so 57 working days
are spread over 12 calendar weeks, and 57/4 = 14.25 days per block lands the edges
on a Thursday and a Friday.  The consequence is not leakage — nothing here uses a
lag — but the boundary no longer means what a walk-forward boundary is supposed to
mean: a validation block that starts on Friday shares its Monday-Thursday with the
training set, and the block's weekday mix differs from the training set's.

`week_forward` places the edges on Mondays instead, counting in whole calendar
weeks.  A fold then trains on everything before a given week and validates on
whole weeks, which is the claim the split is asked to support when someone says
"predict next week".  It is the default.  `walk_forward` is kept because the
equal-count geometry is what the published numbers were produced under, and a
comparison between the two is worth being able to run.

Calibration, and why fold 1 is never calibrated
-----------------------------------------------
A single multiplicative factor `sum(obs) / sum(pred)` applied **forward**: fold k
is multiplied by the factor fitted on folds < k.  Fold 1 is therefore
uncalibrated by construction.  Fitting the factor on a fold's own predictions
would be a self-fulfilling correction that cannot be reproduced in production,
where tomorrow's actuals do not exist yet.

The factor is a *level* correction, not a distributional one: it moves the whole
fold up or down by a constant.  Anything fancier — per slot, per unit — was
measured and did not pay: the per-slot variant was worth −0.13% on arrivals and was
actively worse on departures (+0.97%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCHEMES = ("week_forward", "walk_forward")


class SplitError(ValueError):
    """The split is impossible on this panel, or would leak."""


@dataclass
class Fold:
    """One train/validate pair.  `train` and `valid` are row subsets."""

    number: int
    train: pd.DataFrame
    valid: pd.DataFrame
    start: str
    end: str

    @property
    def train_days(self) -> int:
        return int(self.train.date.nunique())

    @property
    def valid_days(self) -> int:
        return int(self.valid.date.nunique())

    def describe(self) -> dict:
        return {"fold": self.number, "valid": f"{self.start}..{self.end}",
                "train_days": self.train_days, "valid_days": self.valid_days,
                "train_rows": int(len(self.train)),
                "valid_rows": int(len(self.valid))}


def _fold_from_days(panel: pd.DataFrame, tr_days: set, va_days: set,
                    number: int) -> Fold:
    tr = panel[panel.date.isin(tr_days)]
    va = panel[panel.date.isin(va_days)]
    if tr.empty or va.empty:
        raise SplitError(
            f"fold {number} is empty (train {len(tr)}, valid {len(va)})")
    # The one invariant that makes this walk-forward: no training day may fall
    # after the first validation day.
    if not (tr.date.max() < va.date.min()):
        raise SplitError(
            f"fold {number} violates walk-forward: train ends {tr.date.max()}, "
            f"validation starts {va.date.min()}")
    return Fold(number, tr, va, str(min(va_days))[:10], str(max(va_days))[:10])


def walk_forward(panel: pd.DataFrame, n_blocks: int = 4) -> list[Fold]:
    """Contiguous day blocks of equal *count*; the first trains only.

    Equal count is not equal weeks: on a panel that drops weekends, 57 days over
    four blocks puts the edges mid-week.  Use `week_forward` unless you are
    reproducing numbers that were produced this way.
    """
    days = sorted(panel.date.unique())
    if len(days) < n_blocks + 1:
        raise SplitError(
            f"walk_forward needs at least {n_blocks + 1} distinct days, panel has "
            f"{len(days)}")
    edges = np.linspace(0, len(days), n_blocks + 1).astype(int)
    folds: list[Fold] = []
    for i in range(1, n_blocks):
        a, b = int(edges[i]), int(edges[i + 1])
        folds.append(_fold_from_days(panel, set(days[:a]), set(days[a:b]), i))
    return folds


def _monday_edges(days: list, n_blocks: int) -> list[int]:
    """Index cuts placed on Mondays, as near to equal week counts as possible.

    Returns `n_blocks + 1` indices into `days`, first 0 and last `len(days)`.
    Anchors are the indices of Mondays (plus index 0 when the panel does not
    start on one), and each interior cut takes the anchor nearest its equal-count
    target.  Two cuts never land on the same anchor, and anchors are taken in
    order, so the blocks stay non-empty and forward.
    """
    d = pd.to_datetime(pd.Series(days))
    # An interior cut must leave at least one day on each side of it, so the very
    # first and very last day are never candidates.
    anchors = [0] if d.iloc[0].dayofweek == 0 else []
    anchors += [i for i in range(1, len(days) - 1) if d.iloc[i].dayofweek == 0]
    if not anchors:
        raise SplitError(
            "week_forward found no Monday to cut on; the panel spans less than a "
            "week or starts late in one. Use scheme: walk_forward instead")
    target = len(days) / n_blocks
    cuts = [0]
    prev = 0
    for k in range(1, n_blocks):
        want = prev + target
        cand = [a for a in anchors if a > prev]
        if not cand:
            raise SplitError(
                f"week_forward ran out of Mondays after cut {k} of {n_blocks - 1}; "
                f"there are {len(anchors)} usable Monday(s) for {n_blocks - 1} "
                f"interior cut(s). Use fewer blocks or scheme: walk_forward")
        pick = min(cand, key=lambda a: (abs(a - want), a))
        cuts.append(int(pick))
        prev = int(pick)
    cuts.append(len(days))
    return cuts


def week_forward(panel: pd.DataFrame, n_blocks: int = 4) -> list[Fold]:
    """Contiguous blocks cut on Mondays, so a fold validates whole weeks."""
    days = sorted(panel.date.unique())
    if len(days) < n_blocks + 1:
        raise SplitError(
            f"week_forward needs at least {n_blocks + 1} distinct days, panel has "
            f"{len(days)}")
    cuts = _monday_edges(days, n_blocks)
    folds: list[Fold] = []
    for i in range(1, len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        folds.append(_fold_from_days(panel, set(days[:a]), set(days[a:b]), i))
    return folds


def make(cfg: dict, panel: pd.DataFrame) -> list[Fold]:
    scheme = str(cfg.get("scheme", "week_forward"))
    if scheme not in SCHEMES:
        raise SplitError(
            f"unknown fold scheme {scheme!r}; known: {', '.join(SCHEMES)}")
    n_blocks = int(cfg.get("n_blocks", 4))
    return {"week_forward": week_forward,
            "walk_forward": walk_forward}[scheme](panel, n_blocks)


class ForwardCalibrator:
    """Multiplicative level correction, fitted on earlier folds only.

    `compound=False` (the default) refits the factor from the raw prediction each
    fold, which is the correct thing: the factor should answer "how far off is
    this model", not "how far off is this model times last fold's correction".

    `compound=True` feeds the corrected prediction back in, which is what the
    original scripts did.  It is kept because the published numbers were produced
    with it, and the difference is visible in the third decimal.
    """

    def __init__(self, compound: bool = False):
        self.factor: float | None = None
        self.history: list[float] = []
        self.compound = bool(compound)

    def apply(self, mu: np.ndarray) -> np.ndarray:
        return mu if self.factor is None else mu * self.factor

    def update(self, y: np.ndarray, mu: np.ndarray) -> None:
        """Fold in this fold's out-of-sample ratio for the *next* fold.

        `mu` must be the **uncalibrated** prediction.  Feeding in an already
        corrected `mu` compounds the factor instead of replacing it: fold 2
        would then set factor = obs/(pred*f1), which is 1/f1 too small, and the
        third fold inherits the error.  The legacy pipeline had exactly this
        shape — it passed the calibrated `mu_masked` back in — so reproducing
        its published number means reproducing that.  `compound=True` does it,
        and the difference is measurable: 1.1930 vs 1.1924 on this panel.
        """
        s = float(np.sum(mu))
        if s <= 0:
            return
        self.factor = float(np.sum(y) / s)
        self.history.append(self.factor)

    @property
    def is_identity(self) -> bool:
        return self.factor is None
