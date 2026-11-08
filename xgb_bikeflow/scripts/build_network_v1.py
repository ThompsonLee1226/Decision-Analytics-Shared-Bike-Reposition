#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the v1.0 network definition: HW6 GPs + repro arcs, both on geohash8.

Writes one self-contained JSON that the model config points at, so nothing
downstream has to know about the repro package's cell_id lattice:

  data/network_v1.json
    version, horizon, sources
    gps:      [{gp_id, name, class, cells, lat, lon, n_cells, ...}]
    arcs:     [{arc_id, from_gp, to_gp, length_m, service_cells, ...}]
    rows:     [{row_id, kind, gp_id|arc_id, geohash|cells}]   (what gets predicted)

Mapping rule for repro GPs -> HW6 GPs: majority vote over shared cells.
A repro GP whose cells are majority-outside HW6 is kept as its own GP (the
network needs it as an arc endpoint), under id "R<gp_id>".

All three inputs ship in `data/`, as CSV copies of the three files this script
actually opens: two sheets of the HW6 workbook and two members of the repro
package.  Nothing outside the repository is read, so this runs from a clone:

  data/repro_network.json      <- GP_arc_patch_repro_.../expected_current/network.json
  data/repro_cell_mapping.csv  <- .../expected_current/cell_mapping.csv
  data/repro_cells.csv         <- .../frozen/2026-10-27/.../HW5_GP_Handoff/data/cells.csv
  data/hw6_gp_master.csv       <- HW6_A_结果总览.xlsx :: GP主表
  data/hw6_grid_ownership.csv  <- HW6_A_结果总览.xlsx :: 网格归属
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Running a script straight from a clone: the package lives in src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

DATA = ROOT / "data"
OUT = DATA / "network_v1.json"


def main() -> None:
    cells = pd.read_csv(DATA / "repro_cells.csv")
    cm = pd.read_csv(DATA / "repro_cell_mapping.csv")
    net = json.loads((DATA / "repro_network.json").read_text(encoding="utf-8"))
    cells = cells.merge(cm[["cell_id", "type", "gp_id", "patch_id"]], on="cell_id")

    hwgp = pd.read_csv(DATA / "hw6_gp_master.csv")
    hwcell = pd.read_csv(DATA / "hw6_grid_ownership.csv")
    hwcell = hwcell[hwcell.gp != "UNASSIGNED"]
    gh2hw = dict(zip(hwcell.geohash, hwcell.gp))

    # -- HW6 GPs, keyed "H<n>" ---------------------------------------------
    gps: dict[str, dict] = {}
    for _, r in hwgp.iterrows():
        member = sorted(hwcell[hwcell.gp == r.gp].geohash)
        gps[r.gp] = {
            "gp_id": r.gp, "source": "HW6", "class": r.cls,
            "lat": float(r.lat), "lon": float(r.lon),
            "cells": member, "n_cells": len(member),
            "dep_per_day": float(r.dep_day), "arr_per_day": float(r.arr_day),
            "n_source_groups": int(r.n_src),
        }

    # -- map repro GPs onto HW6 GPs by majority vote -----------------------
    repro_cells = (cells[cells.type == "GP"].groupby("gp_id").geohash.apply(list))
    vote, purity = {}, {}
    for gid, ghs in repro_cells.items():
        c = Counter(gh2hw[g] for g in ghs if g in gh2hw)
        if not c:
            continue
        tot = sum(c.values())
        best, n = c.most_common(1)[0]
        vote[int(gid)] = best
        purity[int(gid)] = n / tot

    # repro GPs that no HW6 GP explains stay their own endpoint
    standalone = {int(g): "R%d" % int(g) for g in repro_cells.index
                  if int(g) not in vote}
    for gid, rid in standalone.items():
        ghs = sorted(repro_cells[gid])
        sub = cells[cells.gp_id == gid]
        gps[rid] = {
            "gp_id": rid, "source": "repro-only", "class": "需求补充",
            "lat": float(sub.latitude_wgs84.mean()),
            "lon": float(sub.longitude_wgs84.mean()),
            "cells": ghs, "n_cells": len(ghs),
            "dep_per_day": None, "arr_per_day": None, "n_source_groups": None,
        }
    canonical = {**vote, **standalone}

    # -- arcs --------------------------------------------------------------
    patch_gh: dict[int, set[str]] = {}
    for p, g in zip(cells[cells.type == "patch"].patch_id,
                    cells[cells.type == "patch"].geohash):
        patch_gh.setdefault(int(p), set()).add(g)

    arcs, dropped = [], Counter()
    for a in net["arcs"]:
        f = canonical.get(int(a["from_gp"]))
        t = canonical.get(int(a["to_gp"]))
        if f is None or t is None:
            dropped["unmapped endpoint"] += 1
            continue
        if f == t:
            dropped["collapsed to a self-loop"] += 1
            continue
        sc: set[str] = set()
        for p in a["patch_ids"]:
            sc |= patch_gh.get(int(p), set())
        arcs.append({
            "arc_id": int(a["arc_id"]),
            "from_gp": f, "to_gp": t,
            "length_m": round(float(a["length_m"]), 3),
            "n_patches": len(a["patch_ids"]),
            "service_cells": sorted(sc),
            "repro_from_gp": int(a["from_gp"]),
            "repro_to_gp": int(a["to_gp"]),
        })

    # -- the rows the model predicts ---------------------------------------
    rows_out = []
    for gid, g in gps.items():
        for gh in g["cells"]:
            rows_out.append({"row_id": gh, "kind": "gp", "gp_id": gid,
                             "geohash": gh})
    for a in arcs:
        for gh in a["service_cells"]:
            rows_out.append({"row_id": f"arc{a['arc_id']}:{gh}", "kind": "arc",
                             "arc_id": a["arc_id"], "geohash": gh})

    by_cell: dict[str, set[str]] = {}
    for r in rows_out:
        by_cell.setdefault(r["geohash"], set()).add(r["row_id"])

    blob = {
        "version": "1.0",
        "built_from": {
            "gps": "HW6_A_结果总览.xlsx :: GP主表 (90 GPs)",
            "network": "GP_arc_patch_repro_20261106/expected_current/network.json "
                       "(67 GPs / 401 arcs)",
            "cells": "HW5_GP_Handoff/data/cells.csv (12,149 geohash8 cells)",
        },
        "horizon": "06:30-10:30, 30-minute slots",
        "counts": {
            "n_gps": len(gps), "n_gps_hw6": sum(1 for g in gps.values()
                                                if g["source"] == "HW6"),
            "n_gps_repro_only": len(standalone),
            "n_arcs": len(arcs),
            "n_arc_rows": sum(1 for r in rows_out if r["kind"] == "arc"),
            "n_gp_rows": sum(1 for r in rows_out if r["kind"] == "gp"),
            "n_rows": len(rows_out),
            "n_cells": len(by_cell),
            "n_cells_multi_row": sum(1 for v in by_cell.values() if len(v) > 1),
            "arcs_dropped": dict(dropped),
        },
        "mapping": {
            "rule": "repro GP -> HW6 GP by majority vote over shared cells; a "
                    "repro GP with no shared cell stays its own endpoint as R<id>",
            "n_voted": len(vote),
            "n_standalone": len(standalone),
            "n_matched": len(set(vote.values())),
            "purity": {str(k): round(v, 4) for k, v in sorted(purity.items())},
        },
        "gps": list(gps.values()),
        "arcs": arcs,
        "rows": rows_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
    for k, v in blob["counts"].items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
