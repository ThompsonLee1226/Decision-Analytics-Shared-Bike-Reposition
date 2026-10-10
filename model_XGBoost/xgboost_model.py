#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_XGBoost/xgboost_model.py
==============================
Section-H demand-rate prediction with **XGBoost, Poisson objective**
(`count:poisson`) -- assignment section 2.

Thin on purpose.  The panel, the walk-forward split, the metrics and the
comparison against the existing pipeline all come from `model_common.py`, shared
by every `model_<name>/` folder.  What is left here is only what is XGBoost's:
its hyperparameters, its fit, its predict.

Target (H1)
-----------
Hourly **departure count** per (L6 cell, hour, day), for the top `TOP_CELLS`
hotspots, zero-filled.  Same population and same folds as the models already in
`EDA/`, so the numbers are directly comparable.

Split (H3)
----------
Walk-forward, four blocks of ~19 days, block 1 train-only and blocks 2-4 the
evaluation folds -- from `model_common`, i.e. the same split the existing
pipeline uses, not a second one.

Hyperparameters -- fixed, not searched
--------------------------------------
Mirrored from `09_demand_model.fit_gbdt`: the same `n_estimators=400` and
`learning_rate=0.06`, and a tree depth chosen as the closest equivalent of
scikit-learn's `max_leaf_nodes=31` (2**5 = 32 leaves).  That is deliberate, so
that any difference between this model and the scikit-learn one is a *library*
difference rather than a tuning difference -- and so that "how were the
parameters selected" has a real answer rather than a search that would have to be
trusted.

No early stopping: the only labelled data a fold can spare is its own validation
set, and stopping on that would leak the answer into the model.

Outliers (H6)
-------------
Inherited, not re-decided: `trips_clean.csv.gz` already has every QC tier
dropped (98.57% retained).  See the section-A3 taxonomy.  No second re-run.

Usage:
    python model_XGBoost/xgboost_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))           # so `model_common` resolves
import model_common as C                # noqa: E402

NAME = "XGBoost (Poisson)"
OUT = C.model_out(__file__)

PARAMS = dict(
    objective="count:poisson",
    tree_method="hist",
    n_estimators=400,
    learning_rate=0.06,
    max_depth=6,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=-1,
    enable_categorical=True,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def fit(train: pd.DataFrame):
    """Fit on one fold's training days. Returns (estimator, pinned categories)."""
    import xgboost as xgb

    cats = pd.Index(sorted(train[C.CAT].unique()))
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(C.tree_design(train, cats), train.n.values)
    return model, cats


def predict(fitted, df: pd.DataFrame) -> np.ndarray:
    model, cats = fitted
    return np.asarray(model.predict(C.tree_design(df, cats)), dtype=float)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_pred_actual(plt, preds: pd.DataFrame, path: Path) -> None:
    """Predicted vs actual on log-log axes, one panel per validation fold.

    Log-log because the volume range spans three decades; on linear axes the
    nine-tenths of cell-hours at low counts collapse into one indistinguishable
    blob, which is exactly where the systematic over-prediction lives.
    """
    from matplotlib.colors import LogNorm

    numbers = sorted(preds.fold.unique())
    fig, axes = plt.subplots(1, len(numbers), figsize=(5.0 * len(numbers) + 0.4, 5.1),
                             squeeze=False)
    axes = axes[0]
    lim = (0.7, preds.actual.max() * 1.6)
    handle = None
    for ax, f in zip(axes, numbers):
        d = preds[preds.fold == f]
        ax.plot(lim, lim, color=C.viz.INK, lw=1.3, ls="--", zorder=3)
        handle = ax.hexbin(d.actual + 1, d.pred + 1, gridsize=44, xscale="log",
                           yscale="log", cmap="Blues", mincnt=1, linewidths=0,
                           norm=LogNorm(vmin=1), zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("actual departures in the hour")
        ax.set_title(f"fold {f}  ({d.date.min()} .. {d.date.max()})", fontsize=10.5)
        m = C.metrics(d.actual, d.pred)
        ax.annotate(f"MAE {m['mae']:.2f}\nRMSE {m['rmse']:.2f}\n"
                    f"Poisson dev {m['poisson_deviance']:.2f}\n"
                    f"calibration {m['calibration']:.3f}",
                    xy=(.03, .97), xycoords="axes fraction", fontsize=9, va="top",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd",
                              alpha=.92))
    cb = fig.colorbar(handle, ax=axes, fraction=.018, pad=.010)
    cb.set_label("cell-hours in the bin (log)", fontsize=9)
    axes[0].set_ylabel("predicted departures in the hour")
    fig.suptitle(f"{NAME} — predicted vs actual hourly departures, walk-forward "
                 f"validation\nlog-log · dashed line is perfect prediction · "
                 f"points above it are over-predictions",
                 fontsize=12, weight="semibold", y=1.03)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"    wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    panel = C.load_panel()
    print(f"panel : {len(panel):,} cell-hours, {panel.cell.nunique()} cells x "
          f"{panel.date.nunique()} days x 24 h, {int(panel.n.sum()):,} departures")

    for i, b in enumerate(C.blocks(sorted(panel.date.unique())), start=1):
        print(f"  block {i}: {b[0]} .. {b[1]}  ({b[3] - b[2]} days)  "
              f"{'train only' if i == 1 else 'validate'}")

    preds, rows = [], []
    for f in C.folds(panel):
        t0 = time.time()
        fitted = fit(f.train)
        mu = predict(fitted, f.valid)
        print(f"  fold {f.number}: train {len(f.train):,} -> validate "
              f"{len(f.valid):,}  ({f.start}..{f.end})  fitted in {time.time()-t0:.1f}s")
        preds.append(pd.DataFrame({
            "fold": f.number, "cell": f.valid.cell.values,
            "date": f.valid.date.values, "hour": f.valid.hour.values,
            "actual": f.valid.n.values, "pred": mu}))
        rows.append({"fold": f.number, **C.metrics(f.valid.n.values, mu)})

    preds = pd.concat(preds, ignore_index=True)
    per_fold = pd.DataFrame(rows)
    pooled = C.metrics(preds.actual, preds.pred)
    ref = C.reference_metrics()

    C.report(NAME, pooled, ref)

    per_fold.to_csv(OUT / "metrics_by_fold.csv", index=False, encoding="utf-8")
    preds.to_csv(OUT / "predictions.csv.gz", index=False, encoding="utf-8",
                 compression="gzip")

    plt = C.viz.setup()
    fig_pred_actual(plt, preds, OUT / "pred_vs_actual.png")

    (OUT / "report.json").write_text(json.dumps({
        "model": "xgboost",
        "name": NAME,
        "objective": PARAMS["objective"],
        "params": PARAMS,
        "features": C.FEATS + [f"{C.CAT} (categorical)"],
        "top_cells": C.TOP_CELLS,
        "panel_rows": int(len(panel)),
        "per_fold_metrics": per_fold.to_dict("records"),
        "pooled_metrics": pooled,
        "reference_models": None if ref is None else ref.to_dict("records"),
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.name}/metrics_by_fold.csv, report.json, "
          f"predictions.csv.gz, pred_vs_actual.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
