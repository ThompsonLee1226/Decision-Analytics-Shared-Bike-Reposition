#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_XGBoost/train_final.py
============================
Refit the XGBoost Poisson model on **every** panel day and save it as a
deployable artifact.

Read this before quoting the artifact's performance
---------------------------------------------------
`xgboost_model.py` exists to hold days back: it fits three models, each on a
prefix of the panel, and scores each on the block that follows.  The numbers in
`out/report.json` belong to *those* three models.

This script fits **one** model on all of the days, which is what you would
actually run tomorrow.  The consequence is that **this artifact has no honest
validation score**: no held-out day is left to score it on.

    quote performance  <- out/report.json   (three fold models)
    ship / deploy      <- models/*.ubj      (this model, un-scored)

Those two never get interchanged.  The script therefore prints no score of any
kind, not even in-sample: a fitted-on-everything model's in-sample error is
near-zero and would be quoted as if it meant something.

Inference needs the pinned categories
-------------------------------------
Native formats store the trees, not the pandas category set.  `cell` is a
categorical feature, so a reloaded model asked to score a frame whose `cell`
column pandas re-encodes -- a cell missing, a cell ordering changed -- will read
one category as another and be quietly wrong.  The `cell_categories` list in the
sidecar must be passed back through `model_common.tree_design(df, cats)`.

The per-cell priors used at fit time are in the sidecar too, so that a replay
reproduces bit-for-bit rather than approximately.

Usage:
    python model_XGBoost/train_final.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))           # so `model_common` resolves
import model_common as C                # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"


def fold_script():
    """`xgboost_model.py` as a module -- the fold script's own PARAMS and NAME,
    imported rather than retyped, so the two models cannot drift apart."""
    spec = importlib.util.spec_from_file_location("xgboost_model", HERE / "xgboost_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    XG = fold_script()
    base = C.base

    panel = C.load_panel()
    n_days, n_cells = panel.date.nunique(), panel.cell.nunique()
    print(f"panel : {len(panel):,} cell-hours  {n_cells} cells x {n_days} days x 24 h")

    if len(panel) != n_cells * n_days * 24:
        raise SystemExit("panel is not a complete (cell, day, hour) grid -- stop and look")
    if n_cells != C.TOP_CELLS:
        raise SystemExit(f"expected {C.TOP_CELLS} hotspot cells, panel has {n_cells}")

    # A full refit has to be the *same experiment* as the reported folds, with
    # only the holdout given back. If the panel moved, the comparison stops being
    # like-for-like and the artifact below would silently be a different model.
    reported = HERE / "out" / "report.json"
    if reported.exists():
        was = json.loads(reported.read_text(encoding="utf-8"))["panel_rows"]
        if was != len(panel):
            raise SystemExit(f"panel changed: report.json says {was:,} rows, "
                             f"this run has {len(panel):,}")

    # cell_prior over *all* days. The fold models had to take theirs from
    # training days only, or the split would stop being walk-forward. With no
    # holdout there is no such constraint, and a serving model does have the
    # whole history -- so this prior is deliberately richer than any fold's, and
    # differs from all three.
    full = base.add_cell_prior(panel, panel)
    cats = pd.Index(sorted(full[C.CAT].unique()))

    t0 = time.time()
    model = xgb.XGBRegressor(**XG.PARAMS)
    model.fit(C.tree_design(full, cats), full.n.values)
    print(f"fit   : {len(full):,} rows, {len(cats)} categories, "
          f"in {time.time() - t0:.1f}s")

    MODELS.mkdir(parents=True, exist_ok=True)
    ubj = MODELS / "xgboost_poisson_full.ubj"
    meta = MODELS / "xgboost_poisson_full.meta.json"
    model.get_booster().save_model(ubj)

    # Round-trip before anything is left on disk claiming to work. A file that
    # loads but predicts differently is worse than no file at all.
    reloaded = xgb.XGBRegressor()
    reloaded.load_model(ubj)
    before = np.asarray(model.predict(C.tree_design(full, cats)), dtype=float)
    after = np.asarray(reloaded.predict(C.tree_design(full, cats)), dtype=float)
    delta = float(np.max(np.abs(before - after)))
    if delta != 0.0:
        ubj.unlink()
        raise SystemExit(f"reloaded model disagrees with the fitted one: "
                         f"max|delta|={delta:.3e}  (removed the artifact)")
    print(f"reload: max|delta| = {delta:.3e} over {len(before):,} predictions")

    prior = full.groupby("cell").cell_prior.first()
    meta.write_text(json.dumps({
        "name": XG.NAME,
        "artifact": ubj.name,
        "objective": XG.PARAMS["objective"],
        "params": XG.PARAMS,
        "features": C.FEATS + [f"{C.CAT} (categorical)"],
        "feature_note": "cell_prior is the *serving* prior; the per-cell values this "
                        "model was fitted with are listed below to make a replay exact",
        "cell_categories": [str(c) for c in cats],
        "cell_categories_note": "pinned; pass these to model_common.tree_design(df, cats) "
                                "or the model will read one cell as another",
        "cell_prior_used": {str(k): float(v) for k, v in prior.items()},
        "top_cells": C.TOP_CELLS,
        "trained_on": "every panel day",
        "train_rows": int(len(full)),
        "train_days": int(n_days),
        "train_cells": int(n_cells),
        "train_date_min": str(panel.date.min()),
        "train_date_max": str(panel.date.max()),
        "validation_score": None,
        "validation_note": "Fitted on all days, so this model has NO held-out "
                           "evaluation -- do not quote a score for it. The numbers in "
                           "out/report.json belong to the three walk-forward fold "
                           "models, which are different models fitted on fewer days.",
        "xgboost_version": xgb.__version__,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nwrote models/{ubj.name}  ({ubj.stat().st_size / 1024:.0f} KB)")
    print(f"wrote models/{meta.name}")
    print("\nno score printed on purpose -- this model has no held-out evaluation;\n"
          "the fold models' numbers are in out/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
