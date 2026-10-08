#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
basemap.py
==========
Amap (高德) raster tiles as a background for the matplotlib figures.

`02_region_map.py` draws its map twice: once as a Folium HTML for *checking* the
boundary, once as a static PNG for the deck.  The static PNG mosaics Amap tiles
with PIL.  The remaining map figures (05, 06, 07, 09 -- departure/arrival, net
flow, anchor-hour net flow, model residuals) need the same background under a
matplotlib scatter, so the mosaicking lives here and `02_region_map.py`'s copy
is left alone rather than half-refactored.

Coordinates
-----------
The axes are **Web-Mercator pixels**, not degrees.  Amap serves the same
Mercator pyramid as everyone else, so a pixel is a linear unit in the projection
and the tile mosaic can be dropped in with `imshow` at its exact extent.  A
lon/lat axis would be wrong: Mercator latitude is not linear in degrees, so the
imagery would shear against the data.  Ticks are therefore placed by projecting
round lon/lat values into pixel space, which keeps the labels readable and the
imagery exact.

Tiles are cached under `EDA/.tilecache/`, outside `out/` so the deliverable
folder holds only results.  The cache makes reruns offline and free.

Datum: these tiles are **GCJ-02** and so is every coordinate in this project
(see `region.py`).  Nothing is converted.
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".tilecache"

TILES = ("http://webrd0{s}.is.autonavi.com/appmaptile"
         "?size=1&scale=1&style=7&x={x}&y={y}&z={z}")
SUBDOMAINS = "1234"
ATTRIBUTION = "© 高德地图"

TILE = 256


def lon_to_x(lon, zoom: float):
    return (np.asarray(lon, dtype=float) + 180.0) / 360.0 * (2 ** zoom) * TILE


def lat_to_y(lat, zoom: float):
    s = np.sin(np.radians(np.asarray(lat, dtype=float)))
    s = np.clip(s, -1 + 1e-12, 1 - 1e-12)
    return (0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)) * (2 ** zoom) * TILE


def x_to_lon(x, zoom: float):
    return np.asarray(x, dtype=float) / (2 ** zoom * TILE) * 360.0 - 180.0


def y_to_lat(y, zoom: float):
    n = math.pi - 2 * math.pi * np.asarray(y, dtype=float) / (2 ** zoom * TILE)
    return np.degrees(np.arctan(np.sinh(n)))


class AmapMosaic:
    """A mosaic of Amap tiles covering a lon/lat window, drawn in pixel space."""

    def __init__(self, zoom: int, lat0: float, lat1: float, lon0: float, lon1: float):
        self.zoom = zoom
        x0 = int(lon_to_x(lon0, zoom) // TILE)
        x1 = int(lon_to_x(lon1, zoom) // TILE)
        y0 = int(lat_to_y(lat1, zoom) // TILE)      # north edge = smaller y
        y1 = int(lat_to_y(lat0, zoom) // TILE)
        self.x0, self.y0 = x0, y0
        self.nx, self.ny = x1 - x0 + 1, y1 - y0 + 1
        self.ox, self.oy = x0 * TILE, y0 * TILE
        self.got = 0
        self.img = self._mosaic()

    # -- fetching ---------------------------------------------------------
    def _mosaic(self):
        from PIL import Image
        img = Image.new("RGB", (self.nx * TILE, self.ny * TILE), "#f2f2f2")
        for tx in range(self.x0, self.x0 + self.nx):
            for ty in range(self.y0, self.y0 + self.ny):
                t = self._tile(tx, ty)
                if t is not None:
                    img.paste(t, ((tx - self.x0) * TILE, (ty - self.y0) * TILE))
                    self.got += 1
        return img

    def _tile(self, tx: int, ty: int):
        from PIL import Image
        CACHE.mkdir(exist_ok=True)
        f = CACHE / f"{self.zoom}_{tx}_{ty}.png"
        if f.exists():
            try:
                return Image.open(f).convert("RGB")
            except Exception:
                f.unlink(missing_ok=True)
        for sub in SUBDOMAINS:
            url = TILES.format(s=sub, x=tx, y=ty, z=self.zoom)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=25).read()
                if len(data) > 200:
                    f.write_bytes(data)
                    return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                continue
        return None

    # -- coordinates ------------------------------------------------------
    def xy(self, lat, lon):
        """(lat, lon) -> mosaic pixel coordinates."""
        return (np.asarray(lon_to_x(lon, self.zoom), dtype=float) - self.ox,
                np.asarray(lat_to_y(lat, self.zoom), dtype=float) - self.oy)

    def ll(self, x, y):
        """Mosaic pixel coordinates -> (lat, lon)."""
        return (y_to_lat(np.asarray(y, dtype=float) + self.oy, self.zoom),
                x_to_lon(np.asarray(x, dtype=float) + self.ox, self.zoom))

    @property
    def extent(self):
        """imshow extent (left, right, bottom, top) in pixel space."""
        return (0, self.nx * TILE, self.ny * TILE, 0)

    def __repr__(self):
        return (f"<AmapMosaic z{self.zoom} {self.nx}x{self.ny} tiles, "
                f"{self.got} fetched, {self.nx*TILE}x{self.ny*TILE} px>")

    # -- drawing ----------------------------------------------------------
    def draw(self, ax, alpha: float = 1.0, attribution: bool = True,
             ticks: bool = True, n_ticks: int = 6):
        ax.imshow(self.img, extent=self.extent, interpolation="bilinear",
                  alpha=alpha, zorder=0)
        ax.set_xlim(0, self.nx * TILE)
        ax.set_ylim(self.ny * TILE, 0)
        ax.set_aspect("equal")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

        if ticks:
            self._ticks(ax, n_ticks)
        if attribution:
            ax.annotate(ATTRIBUTION, xy=(1, 0), xycoords="axes fraction",
                        xytext=(-4, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7.5, color="#444444",
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                                  alpha=.72), zorder=20)

    def _ticks(self, ax, n: int):
        """Label round lon/lat values by projecting them into pixel space."""
        (la_lo, lo_lo), (la_hi, lo_hi) = self.ll(0, self.ny * TILE), self.ll(self.nx * TILE, 0)
        step = _nice_step((lo_hi - lo_lo) / n)
        los = np.arange(math.ceil(lo_lo / step) * step, lo_hi + 1e-9, step)
        lats = np.arange(math.ceil(la_lo / step) * step, la_hi + 1e-9, step)
        xt = [float(self.xy(la_lo, v)[0]) for v in los]
        yt = [float(self.xy(v, lo_lo)[1]) for v in lats]
        ax.set_xticks([v for v in xt if 0 <= v <= self.nx * TILE])
        ax.set_xticklabels([f"{v:.3f}" for v, t in zip(los, xt) if 0 <= t <= self.nx * TILE],
                           fontsize=8)
        ax.set_yticks([v for v in yt if 0 <= v <= self.ny * TILE])
        ax.set_yticklabels([f"{v:.3f}" for v, t in zip(lats, yt) if 0 <= t <= self.ny * TILE],
                           fontsize=8, rotation=90, va="center")
        ax.tick_params(length=2)


def _nice_step(raw: float) -> float:
    """Round a desired tick spacing up to the nearest 1/2/5 x 10^k."""
    if raw <= 0:
        return 0.001
    k = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * k:
            return m * k
    return 10 * k


def tile_url_for_check(x: int, y: int, zoom: int) -> str:
    """The URL a tile came from -- handy when a slide needs the source."""
    return TILES.format(s=SUBDOMAINS[0], x=x, y=y, z=zoom)
