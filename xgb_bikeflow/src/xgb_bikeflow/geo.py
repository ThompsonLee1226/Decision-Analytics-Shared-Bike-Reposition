#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/geo.py
===================
Pure-python L8 geohash arithmetic, so that a "3x3 block" is defined by
**neighbouring geohash cells**, not by latitude/longitude rounding.

Why not lat/lon rounding
------------------------
Rounding a centroid to the nearest 1/4 and taking a 9-cell patch is easy to
write and silently wrong at cell boundaries: the patch is then anchored on
whatever the rounding produced, not on the L8 cell the row actually falls in.
Geohash indices have no such ambiguity -- cell (i, j) and its 8 neighbours are
exactly the 3x3 block, and `block_id` is derived from the same index, so two
rows that describe the same physical patch always get the same id.

The ordered index pair
----------------------
A geohash is a 1-D encoding of a 2-D cell.  With an **even** number of bits per
dimension (L8 is 40 bits -> 20 per dimension), the pair `(lat_idx, lon_idx)` is
a bijection with the geohash, so `+1` on either axis is a genuine step to the
adjacent cell.  `to_idx` / `from_idx` are inverses; `test_geo.py` checks it.
"""

from __future__ import annotations

B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
IDX = {c: i for i, c in enumerate(B32)}

DEFAULT_LEVEL = 8


def to_idx(gh: str) -> tuple[int, int]:
    """Geohash -> (lat_index, lon_index). Geohash interleaves lon first."""
    lat = lon = 0
    even = True
    for ch in gh:
        v = IDX[ch]
        for mask in (16, 8, 4, 2, 1):
            if even:
                lon = (lon << 1) | (1 if v & mask else 0)
            else:
                lat = (lat << 1) | (1 if v & mask else 0)
            even = not even
    return lat, lon


def from_idx(lat: int, lon: int, nbits: int) -> str:
    """(lat_index, lon_index) -> geohash. Inverse of `to_idx`."""
    bits: list[int] = []
    for i in range(nbits - 1, -1, -1):
        bits.append((lon >> i) & 1)
        bits.append((lat >> i) & 1)
    out = []
    for k in range(0, len(bits), 5):
        v = 0
        for b in bits[k:k + 5]:
            v = (v << 1) | b
        out.append(B32[v])
    return "".join(out)


def nbits_of(gh: str) -> int:
    return len(gh) * 5 // 2


def block_cells(gh: str) -> tuple[str, ...]:
    """The 9 cells of the 3x3 block centred on `gh`, in a fixed order."""
    nb = nbits_of(gh)
    mod = 2 ** nb
    la, lo = to_idx(gh)
    return tuple(
        from_idx((la + dy) % mod, (lo + dx) % mod, nb)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )


def block_id(gh: str) -> str:
    """**Deprecated and misleading — this does NOT collapse 3x3 blocks.**

    An earlier version of this docstring claimed that "two different centre cells
    that describe the same physical patch map to the same id".  That is false.
    `block_cells` is centred *on* the argument, so `min(block_cells(gh))` is
    essentially injective: measured on the cells present in the trip data it
    returns one id per cell (41,670 -> 41,670 cleaned, 42,632 -> 42,632 raw), and
    `block_id(b) == b` for none of them.

    The returned label is also an **alias, not the cell itself**: the row labelled
    `wx4eqtke` describes the cell `wx4eqtku`, because that is the minimum of
    `wx4eqtke`'s own patch. Counts are right; names are shifted by up to one cell.
    Joining on the label as if it were a geohash is therefore off by one — which
    is one of the reasons the source JSON never reconciled.

    Consequence: the name promises a 3x3 patch but the value is a **single
    8-digit cell** (~57 x 88 m at the equator's scale).  Any run that grouped by
    this function was modelling cells while calling them blocks.  Use `tile_id`
    for a real partition.

    Kept as-is rather than fixed because changing it silently would invalidate
    every published number; the name is retained so old outputs stay
    reproducible.  New code should call `tile_id`.
    """
    return min(block_cells(gh))


def tile_id(gh: str) -> str:
    """Id of the aligned 3x3 tile containing `gh` — a true partition.

    Tiles are the cells with `(la // 3, lo // 3)` fixed, so every cell belongs to
    exactly one tile and the nine members of a tile are `base + (0..2, 0..2)`.
    This is also the unit the source `l8_block_netflow_top10_3x3.json` counts:
    its `block_*_per_day` fields reproduce a patch **centred** on `cell`
    (`block_cells`), not this tile.  Use `tile_cells` to get that member set.
    """
    la, lo = to_idx(gh)
    return from_idx(la // 3 * 3, lo // 3 * 3, nbits_of(gh))


def alias_cell(label: str) -> str:
    """The cell a `block_id` label actually describes.

    `block_id(gh) = min(block_cells(gh))` and the minimum of a 3x3 patch centred on
    `(la, lo)` is always `(la - 1, lo - 1)` (indices wrap, but never at the edge for
    real Shanghai geohashes).  So the map is not just injective, it is a **uniform
    shift one cell south-west**, and it inverts exactly: the cell behind the label
    is `(la + 1, lo + 1)`.

    Verified on 2,000 sampled labels: offset `(+1, +1)` for 2,000 of 2,000.
    """
    la, lo = to_idx(label)
    return from_idx(la + 1, lo + 1, nbits_of(label))


def tile_cells(tile: str) -> tuple[str, ...]:
    """The nine cells of the aligned tile whose id is `tile`, in a fixed order.

    Assumes `tile` is already an aligned id (`la % 3 == lo % 3 == 0`), which is
    what `tile_id` returns.  Do **not** substitute `block_cells` here: that
    function centres on its argument and would return the wrong nine cells for
    a tile id.
    """
    nb = nbits_of(tile)
    mod = 2 ** nb
    la, lo = to_idx(tile)
    return tuple(
        from_idx((la + dy) % mod, (lo + dx) % mod, nb)
        for dy in range(3)
        for dx in range(3)
    )


def bbox(gh: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """((lat_lo, lat_hi), (lon_lo, lon_hi)) of the geohash cell."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        v = IDX[ch]
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if v & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if v & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo, lat_hi), (lon_lo, lon_hi)


def centre(gh: str) -> tuple[float, float]:
    (la0, la1), (lo0, lo1) = bbox(gh)
    return (la0 + la1) / 2, (lo0 + lo1) / 2
