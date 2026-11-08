#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write the two flat cell lists the v1.0 config points at.

`network_v1.json` carries the structure (which GP owns which cell, which arc
serves which cells).  The source and loss sections of a config want a flat list
of cell ids, so that is what this writes:

  data/network_v1_cells.json      every cell in the network  (the panel universe)
  data/network_v1_gp_cells.json   only the gathering-point cells (the loss split)

Keeping them as separate files rather than teaching the config loader to dig
into `network_v1.json` means the same two lists can be handed to any other tool
that wants "the cells we model" without knowing this project's schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Running a script straight from a clone: the package lives in src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"


def main() -> None:
    net = json.loads((DATA / "network_v1.json").read_text(encoding="utf-8"))

    universe: list[str] = []
    seen: set[str] = set()
    for r in net["rows"]:
        gh = r["geohash"]
        if gh not in seen:
            seen.add(gh)
            universe.append(gh)
    gp_cells: list[str] = []
    gseen: set[str] = set()
    for g in net["gps"]:
        for gh in g["cells"]:
            if gh not in gseen:
                gseen.add(gh)
                gp_cells.append(gh)

    for name, ids in (("network_v1_cells.json", universe),
                      ("network_v1_gp_cells.json", gp_cells)):
        p = DATA / name
        p.write_text(json.dumps(ids, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {p.name:28s} {len(ids):6,d} cells  {p.stat().st_size/1024:6.1f} KB")

    missing = [c for c in gp_cells if c not in seen]
    if missing:
        raise SystemExit(f"{len(missing)} GP cells are not in the universe, "
                         f"e.g. {missing[:3]}")
    print(f"gp cells inside the universe: {len(gp_cells):,} of {len(universe):,} "
          f"({len(gp_cells)/len(universe):.1%})")


if __name__ == "__main__":
    main()
