#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/grid.py
====================
The spatial-unit axis: how a geohash cell is grouped into the thing a row is
about, and whether a given panel agrees with that grouping.

Why this replaces a "unit kind" enum
------------------------------------
The first version of this module offered three strings — `cell`, `aligned_tile`,
`centred_patch` — because those were the three panels that existed.  That is a
list of the answers we happened to try, not a description of the choice.  The
choice actually has three free parameters:

    level     geohash level of the *cell* the trips are counted on   (8)
    shape     how cells group into a unit                            (1x1, 3x3,
              or an arbitrary offset table)
    aligned   whether the grouping is a partition or a centred patch

Written that way, `cell` is `shape=1x1`, `aligned_tile` is `shape=3x3,
aligned=True`, and `centred_patch` is `shape=3x3, aligned=False` — the old enum
becomes three points in a space instead of the space itself.  A 2x2 tile, a 5x5
patch, or an L7 cell are then a config line rather than a new module.

Partition or not is the one property that must never be guessed
---------------------------------------------------------------
`centred_patch` overlaps: on this study area 39 patches produce 12 overlapping
pairs, so summing "total flow" over them double-counts while every individual
number still looks right.  `GridSpec.partitioned` exists so downstream code can
refuse to add up a non-partition rather than quietly reporting a total that is
40% too high.

Rebinning is not done here, and that is deliberate
--------------------------------------------------
A panel holds only the cells that were selected, so grouping them into a 3x3
tile covers part of the tile's area and the count comes out low with nothing to
signal it.  Changing the grid therefore means rebuilding the panel from trips.
This module's job is to make the *declared* grid checkable against the panel it
is pointed at, so a config that says 3x3 cannot silently be run on a 1x1 panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import geo

#: Cell levels the geohash arithmetic supports.  L8 is ~19x29 m at this latitude
#: and is the level the source data is counted on; the others are here so a
#: coarser cell can be tested without touching the trip table.
LEVELS = (6, 7, 8)

#: Groupings that are a partition of the plane (no unit shares a cell).
PARTITION_SHAPES = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5))


class GridError(ValueError):
    """The panel's ids do not match the declared grid."""


@dataclass(frozen=True)
class GridSpec:
    """How cells group into units.

    `shape` is `(rows, cols)` in geohash-index steps.  `aligned=True` means the
    unit is the tile whose south-west corner is at a multiple of `shape`, which
    makes the units disjoint; `aligned=False` centres the window on the cell, so
    neighbouring units overlap.
    """

    level: int = 8
    shape: tuple[int, int] = (1, 1)
    aligned: bool = True
    #: Only used when `shape` is not a square; kept explicit rather than inferred
    #: so a config cannot mean two things.
    note: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise GridError(
                f"grid.level = {self.level}; supported levels are {LEVELS}. "
                f"Level 8 is what the trip table is counted on.")
        r, c = self.shape
        if r < 1 or c < 1:
            raise GridError(f"grid.shape must be positive, got {self.shape}")
        if self.aligned and (r, c) not in PARTITION_SHAPES:
            raise GridError(
                f"grid.shape = {(r, c)} is not a known partition shape "
                f"{PARTITION_SHAPES}. An aligned grouping must divide the index "
                f"lattice evenly or it leaves a ragged edge; use aligned: false "
                f"for a centred window of that size.")

    # -- properties --------------------------------------------------------
    @property
    def partitioned(self) -> bool:
        """True when units tile the plane without overlap."""
        return bool(self.aligned)

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def label(self) -> str:
        r, c = self.shape
        base = "cell" if (r, c) == (1, 1) else f"{r}x{c}"
        if (r, c) != (1, 1) and not self.aligned:
            base += "-centred"
        return f"L{self.level}/{base}"

    def describe(self) -> dict:
        return {
            "label": self.label,
            "level": self.level,
            "shape": list(self.shape),
            "aligned": self.aligned,
            "partition": self.partitioned,
            "n_cells_per_unit": self.n_cells,
            "note": ("disjoint: totals may be summed" if self.partitioned else
                     "overlapping windows: summing totals double-counts"),
        }

    # -- the actual arithmetic --------------------------------------------
    def unit_of(self, cell: str) -> str:
        """The unit id a cell belongs to."""
        if self.shape == (1, 1) and self.aligned:
            return _at_level(cell, self.level)
        return _group_id(cell, self.level, self.shape, self.aligned)

    def cells_of(self, unit: str) -> tuple[str, ...]:
        """The member cells of a unit id, in a fixed order.

        Only exact for a unit built by `unit_of`; a centred window's id records
        its centre, so this recovers the same nine cells.
        """
        r, c = self.shape
        if (r, c) == (1, 1):
            return (_at_level(unit, self.level),)
        la, lo = geo.to_idx(unit)
        mod = 2 ** geo.nbits_of(unit)
        if self.aligned:
            offs = [(dy, dx) for dy in range(r) for dx in range(c)]
        else:
            ry, rx = (r - 1) // 2, (c - 1) // 2
            offs = [(dy, dx) for dy in range(-ry, r - ry) for dx in range(-rx, c - rx)]
        return tuple(geo.from_idx((la + dy) % mod, (lo + dx) % mod,
                                  geo.nbits_of(unit)) for dy, dx in offs)


def _at_level(gh: str, level: int) -> str:
    """Truncate (or zero-pad) a geohash to `level` characters."""
    gh = str(gh)
    if len(gh) >= level:
        return gh[:level]
    raise GridError(
        f"cell {gh!r} is shorter than grid.level = {level}; it cannot be "
        f"promoted to a coarser cell without the parent's bits")


def _group_id(cell: str, level: int, shape: tuple[int, int],
              aligned: bool) -> str:
    """Unit id for `cell` under this grouping.

    For an aligned grouping the id is the south-west corner of the block, so it
    is stable for every member cell.  For a centred window the id is the centre
    cell, and two centres one step apart produce overlapping windows — which is
    exactly why `partitioned` is False there.
    """
    cell = _at_level(cell, level)
    la, lo = geo.to_idx(cell)
    r, c = shape
    if aligned:
        la = la // r * r
        lo = lo // c * c
    return geo.from_idx(la, lo, geo.nbits_of(cell))


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve(cfg: dict) -> GridSpec:
    """Build a GridSpec from the `unit:` section of a run config.

    Accepts both the current spelling and the legacy `kind:` enum, because the
    shipped configs and the report's reproduction recipes use the old one and a
    reader should not have to translate by hand.
    """
    kind = cfg.get("kind")
    if kind in ("cell", "aligned_tile", "centred_patch"):
        level = int(cfg.get("level", 8))
        if kind == "cell":
            return GridSpec(level=level)
        size = int(cfg.get("size", 3))
        return GridSpec(level=level, shape=(size, size),
                        aligned=(kind == "aligned_tile"),
                        note=f"legacy kind={kind}")
    shape = cfg.get("shape", (1, 1))
    if isinstance(shape, int):
        shape = (shape, shape)
    if isinstance(shape, str):
        a, _, b = shape.lower().partition("x")
        shape = (int(a), int(b or a))
    return GridSpec(level=int(cfg.get("level", 8)),
                    shape=tuple(shape),
                    aligned=bool(cfg.get("aligned", True)))


# ---------------------------------------------------------------------------
# Checking a panel against a declared grid
# ---------------------------------------------------------------------------

def _share_aligned(ids: list[str], shape: tuple[int, int], level: int) -> float:
    """Share of ids that sit on a block boundary — the fingerprint of a grid."""
    r, c = shape
    if (r, c) == (1, 1):
        return 0.0
    hits = 0
    for u in ids:
        try:
            la, lo = geo.to_idx(_at_level(u, max(len(u), level)))
        except (KeyError, IndexError):
            continue
        if la % r == 0 and lo % c == 0:
            hits += 1
    return hits / max(len(ids), 1)


def verify(ids, spec: GridSpec, *, sample: int | None = 2000) -> dict:
    """Check that `ids` look like `spec`; return a diagnostic dict.

    A cell panel drawn from a fine lattice *will* contain boundary cells by
    chance — one in nine for a 3x3 grid — so the test is on the share, not on a
    single id.  For a 3x3 grouping the expected share is ~11% for a cell panel
    and 100% for a tile panel, which leaves plenty of room for a 90% threshold.

    A 1x1 grid cannot be checked this way at all (every id is its own boundary),
    so the check is skipped and reported as `checked: False` rather than passing
    silently as if it had been verified.
    """
    uniq = list(dict.fromkeys(str(x) for x in pd.Series(list(ids)).unique()))
    if sample is not None:
        uniq = uniq[:sample]
    if not uniq:
        raise GridError("the panel has no units")

    lengths = {len(u) for u in uniq}
    if len(lengths) > 1:
        raise GridError(
            f"unit ids have mixed lengths {sorted(lengths)}; a panel must be "
            f"counted at one geohash level")
    level = lengths.pop()

    diag = {"n_units": len(uniq), "id_level": level,
            "declared": spec.label, "examples": uniq[:3]}

    if level != spec.level:
        raise GridError(
            f"grid.level = {spec.level} but the panel's ids are {level} "
            f"characters long (examples {uniq[:3]}). These are different cells; "
            f"set unit.level: {level}, or rebuild the panel at level "
            f"{spec.level}.")

    if spec.shape == (1, 1):
        diag.update(share_aligned=None, checked=False,
                    note="a 1x1 grid is not distinguishable from the ids alone")
        return diag

    try:
        share = _share_aligned(uniq, spec.shape, spec.level)
    except GridError:
        raise
    except Exception as exc:                                  # noqa: BLE001
        raise GridError(
            f"unit ids do not parse as geohashes ({exc}); the first few are "
            f"{uniq[:3]}") from exc

    diag.update(share_aligned=round(share, 4), checked=True,
                expected_share=round(1 / spec.n_cells, 3))
    r, c = spec.shape
    if spec.aligned and share < 0.9:
        raise GridError(
            f"config declares an aligned {r}x{c} grid but only {share:.0%} of "
            f"the {len(uniq)} ids are block origins (examples {uniq[:3]}).\n"
            f"  This is a cell panel. It cannot be re-grouped here: the panel "
            f"holds only the selected cells, so a block built from them would "
            f"cover part of its area and come out low without saying so.\n"
            f"  Either set unit.shape: [1, 1], or rebuild the panel at the "
            f"{r}x{c} grid and point data.panel at it.")
    if not spec.aligned and share > 0.9:
        raise GridError(
            f"config declares a centred {r}x{c} window but {share:.0%} of the "
            f"ids are block origins, which is the signature of an aligned tile "
            f"panel (examples {uniq[:3]}).\n"
            f"  Set unit.aligned: true. Note the two are not interchangeable: a "
            f"centred window overlaps its neighbours and its totals must not be "
            f"summed.")
    return diag


def is_partition(ids) -> bool | None:
    """Do these unit ids cover disjoint cells?  None when it cannot be decided.

    Used by anything that wants to add up per-unit counts: a partition's totals
    are the total, an overlap's are not.
    """
    uniq = list(dict.fromkeys(str(x) for x in pd.Series(list(ids)).unique()))
    if not uniq:
        return None
    seen: set[str] = set()
    for u in uniq[:400]:
        try:
            cells = geo.tile_cells(u) if _looks_aligned(u) else geo.block_cells(u)
        except Exception:                                     # noqa: BLE001
            return None
        if seen & set(cells):
            return False
        seen.update(cells)
    return True


def _looks_aligned(gh: str) -> bool:
    try:
        la, lo = geo.to_idx(gh)
    except (KeyError, IndexError):
        return False
    return la % 3 == 0 and lo % 3 == 0
