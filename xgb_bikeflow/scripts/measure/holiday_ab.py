#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Isolate what the eight statutory holidays are worth, with the fold boundaries held
fixed.

The first comparison let `week_forward` place edges on each arm's own panel, so
adding eight days also moved the edges: arm B trained on a week less and was
scored on a different window.  That measures "a different panel", not "holidays".

Here both arms use the SAME fold edges, taken from the panel that contains the
holidays.  Arm B keeps the holiday rows; arm A drops them.  Training windows,
validation windows and scoring rows are then identical row-for-row apart from the
holiday days themselves.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Running a script straight from a clone: the package lives in src/.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb

from xgb_bikeflow import grid as G
from xgb_bikeflow import groups as GR
from xgb_bikeflow import metrics as MT
from xgb_bikeflow import model as M
from xgb_bikeflow import rows as RW
from xgb_bikeflow import slots as SL
from xgb_bikeflow import sources as SRC
from xgb_bikeflow import spec as SP2
from xgb_bikeflow import split as SP

OUT = ROOT / "docs" / "eda_comparison"
HOLS = ["2026-04-04", "2026-04-05", "2026-04-06",
        "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"]


def build(days):
    slot_spec = SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=30)
    gspec = G.GridSpec(level=8, shape=(1, 1), aligned=True)
    src = SRC.SourceSpec(kind="orders", path=ROOT / "data" / "orders_2026.csv.gz",
                         days=days, cells="active_in_window", min_events=2,
                         holidays=tuple(HOLS), makeup=("2026-05-09",))
    p, _ = SRC.build_panel(src, gspec, slot_spec, verbose=False)
    p["date"] = pd.to_datetime(p["date"])
    return p


def strip_holidays(df: pd.DataFrame) -> pd.DataFrame:
    hol = pd.to_datetime(HOLS)
    return df.loc[~df.date.isin(hol)].reset_index(drop=True)


def main() -> int:
    t0 = time.time()
    spec = SP2.resolve(SP2.load_config(ROOT / "configs" / "horizon_am.yaml"))

    wd = build("workday")
    hol = pd.to_datetime(HOLS)
    with_hol = build(sorted({str(pd.Timestamp(d).date()) for d in
                             set(wd.date.unique()) | set(hol)}))
    print(f"workday only   {len(wd):,} rows / {wd.date.nunique()} days")
    print(f"with holidays  {len(with_hol):,} rows / {with_hol.date.nunique()} days")

    folds_b = SP.make(spec.fold, with_hol)
    print("\nfolds (shared edges):")
    for f in folds_b:
        print(f"  {f.number}  valid {f.start}..{f.end}"
              f"  train {f.train_days}d  valid {f.valid_days}d")

    train_units, _ = RW.select_units(with_hol, spec.rows, spec.targets)
    score_units, _ = RW.resolve_score_units(train_units, spec.rows, with_hol,
                                           spec.targets)
    score_set = set(score_units)
    loss = GR.resolve_membership(
        GR.resolve(spec.loss), Path(spec.config_path).parent,
        available=with_hol.block_id)

    out = {"note": "fold edges identical across arms; only holiday rows differ",
           "folds": [f"{f.start}..{f.end}" for f in folds_b], "targets": {}}

    for target in ("arrivals", "departures"):
        res = {}
        for arm in ("A_no_holiday", "B_with_holiday"):
            devs, shares = [], []
            for fold in folds_b:
                tr = fold.train if arm == "B_with_holiday" else strip_holidays(fold.train)
                va = fold.valid
                tr = tr[tr.block_id.isin(set(train_units))]
                f = SP.Fold(fold.number, tr, va, fold.start, fold.end)
                Xtr, Xva, _, cols, trf = M._prepare_fold(
                    f, spec.features, spec.rows, spec.targets, spec.slots)
                w = RW.row_weights(trf, score_set,
                                   float(spec.rows.get("off_target_weight", 1.0)))
                gw = GR.fit_weights(loss.group_of(trf["block_id"].to_numpy()), loss)
                if gw is not None:
                    w = w * gw
                m = xgb.XGBRegressor(**spec.params_for(target))
                m.fit(Xtr[cols], trf[target].to_numpy(float),
                      **({} if np.allclose(w, 1.0) else {"sample_weight": w}))
                mask = RW.score_mask(f.valid, score_set,
                                     spec.rows.get("score_slots", "all"))
                y = va[target].to_numpy(float)[mask]
                mu = np.clip(np.asarray(m.predict(Xva[cols][mask]), float), 0, None)
                # The holiday rows inside this validation window, if any.
                is_h = va.date.isin(hol).to_numpy()[mask]
                devs.append(float(MT.poisson_deviance(y, mu)))
                shares.append((len(y), int(is_h.sum()),
                               float(MT.poisson_deviance(y[~is_h], mu[~is_h]))
                               if (~is_h).any() else float("nan")))
            res[arm] = {
                "per_fold_dev": [round(d, 6) for d in devs],
                "pooled_like": round(float(np.mean(devs)), 6),
                "n_rows": int(sum(s[0] for s in shares)),
                "n_holiday_rows": int(sum(s[1] for s in shares)),
                "workday_only_dev": round(float(np.nanmean([s[2] for s in shares])), 6),
            }
            print(f"\n{target} {arm}: folds {res[arm]['per_fold_dev']}"
                  f"  mean {res[arm]['pooled_like']:.4f}"
                  f"  hol rows {res[arm]['n_holiday_rows']:,}"
                  f"  ({time.time() - t0:.0f}s)")

        a, b = res["A_no_holiday"], res["B_with_holiday"]
        delta = (b["pooled_like"] / a["pooled_like"] - 1) * 100
        wdelta = (b["workday_only_dev"] / a["workday_only_dev"] - 1) * 100
        res["delta_pct"] = round(delta, 3)
        res["workday_only_delta_pct"] = round(wdelta, 3)
        print(f"  all validated rows : {delta:+.2f}%")
        print(f"  workday rows only  : {wdelta:+.2f}%   <- the rows that matter")
        out["targets"][target] = res

    (OUT / "holiday_ab_same_folds.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote : {OUT / 'holiday_ab_same_folds.json'}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
