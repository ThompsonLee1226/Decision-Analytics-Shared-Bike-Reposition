#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow -- a configurable XGBoost pipeline for bike-sharing arrival and
departure rates.

The things we had not settled on are *configuration*, not code:

    source      raw orders or a prepared panel      (sources.SourceSpec)
    slots       what the planning horizon is        (slots.SlotSpec)
    unit        what a row's spatial id is          (grid.GridSpec)
    rows        which units and slots count         (rows.py)
    features    which feature families emit         (features.REGISTRY)
    loss        how gathering points are weighted   (groups.GroupSpec)

Everything else -- the panel schema, walk-forward folds, Poisson deviance,
forward calibration, and a saved bundle that can be loaded on a machine that no
longer has the training data -- is fixed.

    from xgb_bikeflow import spec, sources, grid, model
    sp = spec.resolve(spec.load_config("configs/horizon_am.yaml"))
    panel, info = sources.build_panel(sp.source, grid.resolve(sp.unit), sp.slots)
    res = model.run(sp, panel, "arrivals")
    res.bundle.save("somewhere/arrivals")

The shipped configs are `horizon_am` (the real one: 07:30-10:30 over the whole
study area with the gathering-point loss), `tile3x3_l8` (the same orders grouped
into 3x3 blocks), `smoke_gp` (40 units, seconds rather than minutes) and
`compat/cell_l8_published` (the legacy split and calibration settings).

See `python -m xgb_bikeflow.cli --help`.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import (  # noqa: F401,E402
    data, geo, grid, groups, metrics, model, rows, slots, sources, spec, split,
)
from .features import REGISTRY, FeatureSet, build  # noqa: F401,E402
from .groups import GroupSpec, gamma_loss  # noqa: F401,E402
from .metrics import metrics as score, poisson_deviance  # noqa: F401,E402
from .model import Bundle, ModelError  # noqa: F401,E402
from .slots import SlotSpec  # noqa: F401,E402

__all__ = [
    "__version__",
    "data", "geo", "grid", "groups", "metrics", "model", "rows", "slots",
    "sources", "spec", "split",
    "score", "poisson_deviance", "gamma_loss",
    "Bundle", "ModelError", "FeatureSet", "GroupSpec", "SlotSpec",
    "REGISTRY", "build",
]
