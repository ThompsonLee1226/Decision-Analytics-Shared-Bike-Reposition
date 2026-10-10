#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_common.py
===============
Shared harness for the `model_<name>/` folders at the repository root.

Why this exists
---------------
Each `model_*/` folder holds one alternative section-H demand-rate model.  Every
one of them needs the *same* four things, and none of them is model-specific:

    the panel              EDA/out/trips_clean.csv.gz -> (cell, hour, day), zero-filled
    the walk-forward split section H3, four blocks of ~19 days
    the metrics            MAE / RMSE / Poisson deviance / calibration
    a comparison basis     EDA/out/table_h5_metrics.csv -- the existing pipeline's
                           own numbers, produced on *these same folds*

Copying that into every folder is the copy-paste drift the project style guide
forbids.  The walk-forward split especially must not differ between folders, or
the numbers stop being comparable.  So it lives here once and each
`model_*/<name>.py` is thin: parameters, fit, predict.

Nothing in `EDA/` is modified
------------------------------
`09_demand_model.py` is loaded **read-only** through `importlib` -- its name
starts with a digit, so `import 09_demand_model` is a syntax error, not a style
choice -- and its `build_panel`, `add_cell_prior` and `metrics` are called
directly.  The panel and the metrics therefore cannot drift from the pipeline
that produced the committed numbers.

The one copied seam
-------------------
`09_demand_model.py` builds its walk-forward split **inline inside `main()`**
rather than in a function, so there is nothing to call.  `walk_forward_blocks()`
below copies it -- deliberately the same expression, using the imported
`N_BLOCKS`, so the block *count* cannot drift -- and `warn_if_split_drifted()`
checks the original still contains that expression.

That check is a **source-text check, not a proof of agreement.**  It catches
"someone rewrote the split in the original"; it cannot tell whether two different
rewrites happen to agree.  If it fires, re-read `09_demand_model.main()` before
trusting the comparison.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EDA = ROOT / "EDA"
sys.path.insert(0, str(EDA))            # so `09_demand_model` and its imports resolve

base = importlib.import_module("09_demand_model")
import viz                              # noqa: E402  (needs EDA on sys.path first)

# Re-exported so a model folder imports one module, not two.
TRIPS = base.TRIPS
SUMM = base.SUMM
OUT = base.OUT                          # EDA/out -- read-only, the pipeline's own
TOP_CELLS = base.TOP_CELLS
N_BLOCKS = base.N_BLOCKS
FEATS = base.FEATS_NUM
CAT = "cell"
metrics = base.metrics
poisson_deviance = base.poisson_deviance
NICE = base.NICE


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def model_out(script_file) -> Path:
    """The `out/` folder next to a `model_*/` script, created if missing.

    Models write here rather than into `EDA/out/` so that this area and the
    existing pipeline stay independent -- neither overwrites the other's
    figures or tables.
    """
    d = Path(script_file).resolve().parent / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Panel (H1, H2) -- built by the existing pipeline, unchanged
# ---------------------------------------------------------------------------

def load_panel(top_cells: int | None = None) -> pd.DataFrame:
    """The (cell, hour, day) panel with lags, from the cleaned data.

    Exactly `09_demand_model.build_panel` over `EDA/out/trips_clean.csv.gz`: the
    top `top_cells` L6 hotspots by volume, zero-filled cell-hours, lags computed,
    rows before the longest lag dropped rather than back-filled.
    """
    n = TOP_CELLS if top_cells is None else top_cells
    summ = pd.read_csv(SUMM)
    top = list(summ.nlargest(n, "n_touching").cell)
    df = pd.read_csv(TRIPS, usecols=["start_cell6", "date", "hour"])
    return base.build_panel(df, top)


# ---------------------------------------------------------------------------
# Walk-forward split (H3) -- the one copied seam
# ---------------------------------------------------------------------------

# The exact expression in `09_demand_model.main()`, kept as a literal so
# `warn_if_split_drifted` can look for it.
_SPLIT_EXPR = "edges = np.linspace(0, len(days), N_BLOCKS + 1).astype(int)"


def walk_forward_blocks(days) -> list[tuple[str, str, int, int]]:
    """(start, end, day_index_lo, day_index_hi) per block.

    Copied from the body of `09_demand_model.main()`, which does not expose it.
    Same expression, same imported `N_BLOCKS`.
    """
    edges = np.linspace(0, len(days), N_BLOCKS + 1).astype(int)
    return [(days[edges[i]], days[edges[i + 1] - 1], edges[i], edges[i + 1])
            for i in range(N_BLOCKS)]


def warn_if_split_drifted() -> None:
    """Warn if the original no longer contains the split this module mirrors."""
    src = (EDA / "09_demand_model.py").read_text(encoding="utf-8")
    if _SPLIT_EXPR not in src:
        print("WARNING: 09_demand_model.py no longer contains\n"
              f"    {_SPLIT_EXPR}\n"
              "  This module mirrors that split. Re-check that both still use the "
              "same folds before reading any comparison against "
              "table_h5_metrics.csv as like-for-like.", file=sys.stderr)


@dataclass(frozen=True)
class Fold:
    """One walk-forward validation fold, ready to fit on and predict into."""

    number: int          # 1-based, matching the original's fold numbering
    train: pd.DataFrame  # training days only, `cell_prior` from those days
    valid: pd.DataFrame  # the following block, `cell_prior` from training
    start: str
    end: str


def blocks(days) -> list[tuple[str, str, int, int]]:
    """The four blocks, warning first if the original's split has changed."""
    warn_if_split_drifted()
    return walk_forward_blocks(days)


def folds(panel: pd.DataFrame):
    """Yield the validation folds -- block 1 is train-only, blocks 2-4 validate.

    `cell_prior` is recomputed from each fold's **training days only**.  A cell
    mean taken over the whole period would carry the future into the model and
    the split would stop being walk-forward; this is the same rule
    `09_demand_model.main()` applies.
    """
    days = sorted(panel.date.unique())
    for i, (d0, d1, a, b) in enumerate(walk_forward_blocks(days), start=1):
        if i == 1:
            continue
        tr = panel[panel.day_index < a]
        va = panel[(panel.day_index >= a) & (panel.day_index < b)]
        yield Fold(i - 1, base.add_cell_prior(tr, tr), base.add_cell_prior(tr, va),
                   d0, d1)


# ---------------------------------------------------------------------------
# Design matrix for the tree models
# ---------------------------------------------------------------------------

def tree_design(df: pd.DataFrame, categories=None) -> pd.DataFrame:
    """Numeric features plus `cell` as a categorical with a **pinned** category set.

    Pinning matters.  pandas derives a `category` dtype from the values present,
    so a cell absent from one side of a split would shift every code and be read
    as a *different* cell.  Fitted models pass back the category set they were
    trained on; an unseen cell then becomes NaN, which both xgboost and
    scikit-learn treat as missing rather than as some other cell.
    """
    X = df[FEATS].copy()
    X[CAT] = pd.Categorical(df[CAT], categories=categories)
    return X


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def reference_metrics() -> pd.DataFrame | None:
    """The existing pipeline's pooled numbers, or None if not built yet."""
    p = OUT / "table_h5_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def report(name: str, pooled: dict, ref: pd.DataFrame | None) -> None:
    """Print one model's pooled metrics, then the existing pipeline's, same folds."""
    print(f"\n{name}")
    print(f"  MAE {pooled['mae']:.2f}   RMSE {pooled['rmse']:.2f}   "
          f"Poisson dev {pooled['poisson_deviance']:.3f}   "
          f"calibration {pooled['calibration']:.3f}")
    if ref is None or ref.empty:
        print("\n  (no EDA/out/table_h5_metrics.csv to compare against)")
        return
    print("\n  for reference, the existing pipeline's own numbers on these same folds:")
    for _, r in ref.sort_values("poisson_deviance").iterrows():
        print(f"    {NICE.get(r.model, r.model):<34} MAE {r.mae:5.2f}   "
              f"RMSE {r.rmse:6.2f}   Poisson dev {r.poisson_deviance:.3f}   "
              f"calibration {r.calibration:.3f}")
    best = ref.loc[ref.poisson_deviance.idxmin()]
    delta = 100 * (pooled["poisson_deviance"] / best.poisson_deviance - 1)
    print(f"\n  {name} vs the best of those ({best.model}): "
          f"{delta:+.1f}% Poisson deviance")
