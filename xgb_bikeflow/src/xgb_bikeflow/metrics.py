#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/metrics.py
=======================
Scoring rules for count predictions.

Poisson deviance is the primary metric: it is the proper scoring rule for
counts, so it cannot be gamed by pushing predictions toward the mean, which is
exactly what MAE rewards on a panel that is 45% zeros.

Why this is a module and not a function copied into every script
---------------------------------------------------------------
`poisson_deviance` had been copy-pasted into eight files across the two earlier
folders.  Eight copies means eight chances for one of them to drift, and the
numbers they produce are compared against each other in the docs.  There is
now one implementation; the golden tests pin its output to the published
values so a drift becomes a failing test rather than a quietly wrong table.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
#: Floor applied to a prediction before taking its log.  Separate from EPS
#: because they guard different denominators: EPS keeps a `sum(obs)`-normalised
#: ratio finite when the observations are all zero, while this keeps `log(mu)`
#: finite.  Merging them would silently change one of the two.
MIN_MU = 1e-9


def poisson_deviance(y, mu) -> float:
    """Mean Poisson deviance, ``2 * mean(y*log(y/mu) - (y - mu))``.

    The ``y > 0`` term is computed only where it is defined.  ``np.where``
    evaluates both branches, so the one-liner form computes ``0 * log(0) = nan``
    on every zero row and floods the log with RuntimeWarnings.  Masked
    assignment is algebraically identical and silent.
    """
    return float(np.mean(poisson_deviance_rows(y, mu)))


def poisson_deviance_rows(y, mu) -> np.ndarray:
    """Per-row deviance.

    Exposed because the gamma loss needs the mean *within* two groups, and a
    mean-of-means is not the mean.  Having this as the primitive and
    `poisson_deviance` as its mean is what keeps the grouped score and the plain
    score from being two independent implementations that can drift.
    """
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), MIN_MU, None)
    term = np.zeros_like(y)
    nz = y > 0
    term[nz] = y[nz] * np.log(y[nz] / mu[nz])
    return 2.0 * (term - (y - mu))


def metrics(y, mu) -> dict:
    """Deviance plus the four numbers that say whether it is operationally usable.

    ``calibration`` is ``sum(pred) / sum(obs)``: 0.90 means the model moves 10%
    less volume than actually moved, which no deviance value makes obvious.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(mu - y))),
        "rmse": float(np.sqrt(np.mean((mu - y) ** 2))),
        "poisson_deviance": poisson_deviance(y, mu),
        "pred_sum": float(mu.sum()),
        "obs_sum": float(y.sum()),
        "calibration": float(mu.sum() / max(y.sum(), EPS)),
        "bias": float(np.mean(mu - y)),
    }
