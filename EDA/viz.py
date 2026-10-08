#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz.py
======
Shared plotting setup for the EDA figures.

Every figure script imports this, so the sixteen figures in the deck read as one
deck: same fonts, same grid, same palette, same DPI.  It also fixes the one
convention that matters when four people assemble slides -- **figure numbers**.
`save()` takes the number from the plan's inventory (§I1) and writes
``out/figNN_<slug>.png``, so a filename maps onto a slide without a lookup table.

The palette
-----------
Small, and used consistently across every figure so the reader learns it once:

    blue  = bikes arriving / accumulating      red   = bikes leaving / depleting
    grey  = neutral, context                   amber = a second series that is
                                                      neither inflow nor outflow

Diverging maps are always centred at zero with **symmetric** limits.  A net-flow
map that runs -500..+1200 hides which side dominates, which is the whole point
of the figure.

Text on figures is English only.  Chinese place names (北四环, 中关村大街 ...)
belong in the tables and the prose, where the font is known good; a CJK font is
registered anyway so a stray label degrades gracefully instead of rendering as
hollow boxes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# --- palette ----------------------------------------------------------------
INK = "#1f1f1f"
GREY = "#8c8c8c"
GRID = "#d9d9d9"
BLUE = "#0958d9"          # arriving / accumulating
RED = "#d4380d"           # leaving / depleting
AMBER = "#d48806"         # a third series
GREEN = "#389e0d"

DIVERGING = "RdBu_r"      # low (negative) = blue = accumulation, high = red = depletion


def setup():
    """Configure matplotlib for slide output and return `plt`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#8a8a8a",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
    })
    return plt


def save(fig, n: int, slug: str, dpi: int = 150) -> Path:
    """Write `fig` as out/fig<NN>_<slug>.png using the plan's figure number."""
    path = OUT / f"fig{n:02d}_{slug}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"    wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


def sym_lim(v, q: float = 98.0, floor: float = 1.0):
    """Symmetric colour limits for a diverging map, robust to a few outliers.

    Returns ``(-vmax, +vmax)`` where vmax is the `q`-th percentile of |v|.  Using
    the extreme value instead lets one hotspot flatten everything else to white.
    """
    a = np.abs(np.asarray(v, dtype=float))
    a = a[np.isfinite(a)]
    vmax = float(np.percentile(a, q)) if a.size else floor
    return (-max(vmax, floor), max(vmax, floor))


def wrap(s: str, width: int = 46) -> str:
    """Naive word wrap for figure titles (matplotlib has no built-in)."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)
