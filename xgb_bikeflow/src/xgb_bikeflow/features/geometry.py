#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/features/geometry.py
=================================
Geometry block — where the unit is, as five numbers.

Constant per unit, so it never carries time information.  It earns its place on
the *full* panel (2,000 units), where it is the only thing distinguishing one
unit from another for units that have too little history for the profile to
mean anything.  On the 20-unit panel it is close to redundant — measured cost of
dropping it there is −0.2%, i.e. within noise either way — which is why it is a
block you can switch off rather than a hard-coded column group.

Metres are computed against a **fixed** centroid taken from the study-area
boundary, not from the data's own centre.  That matters for a block: if the
centroid moved when the panel changed, every `x_m` would change with it, and a
saved model would silently apply radii learned on a different origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Context, FeatureBlock

#: Study-area centroid (lat, lon).  From the supplied boundary polygon's extent,
#: so it does not depend on which units happen to be in the panel.
DEFAULT_CENTROID = (39.9760, 116.3168)
EARTH_R = 6_371_000.0


class GeometryBlock(FeatureBlock):
    """Local equirectangular x/y, radius from the centroid, and bearing."""

    name = "geometry"
    columns = ("x_m", "y_m", "dist_to_centroid_m", "bearing_sin", "bearing_cos")
    fitted = False
    requires = ("lat", "lon")

    def __init__(self, centroid=DEFAULT_CENTROID, **options):
        super().__init__(**options)
        self.centroid = (float(centroid[0]), float(centroid[1]))

    def _metres(self, lat, lon):
        """Metres east/north of the centroid. Good to <0.1% over 7 km."""
        clat, clon = self.centroid
        x = np.radians(lon - clon) * EARTH_R * np.cos(np.radians(clat))
        y = np.radians(lat - clat) * EARTH_R
        return x, y

    def transform(self, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        x, y = self._metres(df["lat"].to_numpy(float), df["lon"].to_numpy(float))
        out["x_m"] = x
        out["y_m"] = y
        r = np.hypot(x, y)
        out["dist_to_centroid_m"] = r
        with np.errstate(invalid="ignore", divide="ignore"):
            out["bearing_sin"] = np.where(r > 0, y / np.maximum(r, 1e-9), 0.0)
            out["bearing_cos"] = np.where(r > 0, x / np.maximum(r, 1e-9), 0.0)
        return self._check(out, df)

    def state(self) -> dict:
        return {"centroid": list(self.centroid)}

    def load_state(self, d: dict) -> None:
        if d:
            self.centroid = (float(d["centroid"][0]), float(d["centroid"][1]))
