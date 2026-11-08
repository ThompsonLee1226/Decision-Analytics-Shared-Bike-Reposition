#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Should the eight statutory holidays stay in the panel, labelled, instead of being
dropped by `source.days: workday`?

Two arms, one fold each, the shipped parameters, no tuning:

    A  workday only                 57 days   <- what ships now
    B  workday + the 8 holidays     65 days   holidays keep is_holiday=1, dow=5

B is built by handing `source.days` an explicit list of dates, which the source
layer already supports, so this measures the modelling question rather than a
hypothetical feature.

The comparison is made on rows both arms can score: the working days in the last
fold's validation window.  B trains on eight extra days; if that helps the days we
actually plan for, the deviance on those shared rows falls.  B's own holiday rows
are reported separately, and they are in-sample for the folds whose training
window contains them -- that is stated, not hidden.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

# Running a script straight from a clone: the package lives in src/.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb

from xgb_bikeflow import features as F
from xgb_bikeflow import grid as G
from xgb_bikeflow import groups as GR
from xgb_bikeflow import metrics as MT
from xgb_bikeflow import model as M
from xgb_bikeflow import rows as RW
from xgb_bikeflow import slots as SL
from xgb_bikeflow import sources as SRC
from xgb_bikeflow import spec as SP2
from xgb_bikeflow import split as SP
import xgb_bikeflow as X

OUT = ROOT / "docs" / "eda_comparison"
HOLS = ["2026-04-04", "2026-04-05", "2026-04-06",
        "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"]


def build(days, slot_spec, gspec):
    src = SRC.SourceSpec(kind="orders", path=ROOT / "data" / "orders_2026.csv.gz",
                         days=days, cells="active_in_window", min_events=2,
                         holidays=tuple(HOLS), makeup=("2026-05-09",))
    return SRC.build_panel(src, gspec, slot_spec, verbose=False)


def fit_score(panel, spec, target, fold_no):
    """Train on one fold, score its own validation rows. Returns (pred df, params)."""
    train_units, _ = RW.select_units(panel, spec.rows, spec.targets)
    score_units, _ = RW.resolve_score_units(train_units, spec.rows, panel,
                                            spec.targets)
    score_set = set(score_units)
    loss = GR.resolve_membership(
        GR.resolve(spec.loss), Path(spec.config_path).parent,
        available=panel.block_id)
    folds = SP.make(spec.fold, panel)
    fold = next(f for f in folds if f.number == fold_no)
    f = SP.Fold(fold.number,
                fold.train[fold.train.block_id.isin(set(train_units))],
                fold.valid, fold.start, fold.end)
    prep = M._prepare_fold(f, spec.features, spec.rows, spec.targets, spec.slots)
    Xtr, Xva, _, cols, tr = prep
    w = RW.row_weights(tr, score_set, float(spec.rows.get("off_target_weight", 1.0)))
    gw = GR.fit_weights(loss.group_of(tr["block_id"].to_numpy()), loss)
    if gw is not None:
        w = w * gw
    params = spec.params_for(target)
    m = xgb.XGBRegressor(**params)
    kw = {} if np.allclose(w, 1.0) else {"sample_weight": w}
    m.fit(Xtr[cols], tr[target].to_numpy(float), **kw)

    mask = RW.score_mask(f.valid, score_set, spec.rows.get("score_slots", "all"))
    v = f.valid[mask].copy()
    v["pred"] = np.clip(np.asarray(m.predict(Xva[cols][mask]), float), 0, None)
    return v, params


def main() -> int:
    t0 = time.time()
    slot_spec = SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=30)
    gspec = G.GridSpec(level=8, shape=(1, 1), aligned=True)
    spec = SP2.resolve(SP2.load_config(ROOT / "configs" / "horizon_am.yaml"))

    pan_a, _ = build("workday", slot_spec, gspec)
    hol_days = pd.to_datetime(HOLS)
    wd_days = sorted(set(pd.to_datetime(pan_a.date.unique())) | set(hol_days))
    pan_b, _ = build([str(pd.Timestamp(d).date()) for d in wd_days],
                     slot_spec, gspec)
    print(f"arm A  {len(pan_a):,} rows  {pan_a.date.nunique()} days")
    print(f"arm B  {len(pan_b):,} rows  {pan_b.date.nunique()} days")

    folds_a = SP.make(spec.fold, pan_a)
    folds_b = SP.make(spec.fold, pan_b)
    print("\nfolds A:", [(f.number, f.start, f.end) for f in folds_a])
    print("folds B:", [(f.number, f.start, f.end) for f in folds_b])

    out = {"days_A": int(pan_a.date.nunique()),
           "days_B": int(pan_b.date.nunique()),
           "fold_windows": {"A": [(f.start, f.end) for f in folds_a],
                            "B": [(f.start, f.end) for f in folds_b]},
           "targets": {}}

    for target in ("arrivals", "departures"):
        res = {"A": {}, "B": {}}
        preds = {}
        for arm, panel in (("A", pan_a), ("B", pan_b)):
            v, params = fit_score(panel, spec, target, 3)
            preds[arm] = v
            y = v[target].to_numpy(float)
            res[arm] = {"n": int(len(v)),
                        "dev": float(MT.poisson_deviance(y, v.pred.to_numpy())),
                        "mae": float(np.abs(y - v.pred.to_numpy()).mean()),
                        "pred_sum": float(v.pred.sum()), "obs_sum": float(y.sum())}
            print(f"\n{target} arm {arm}: n {len(v):,} dev {res[arm]['dev']:.4f} "
                  f"mae {res[arm]['mae']:.4f} ({time.time() - t0:.0f}s)")

        # Shared rows: working days in A's validation window.
        va_days = sorted(set(pd.to_datetime(preds["A"].date.unique())))
        shared = pd.to_datetime(preds["B"].date).isin(va_days).to_numpy()
        hol_in_va = pd.to_datetime(preds["B"].date).isin(hol_days).to_numpy()
        for arm in ("A", "B"):
            v = preds[arm]
            m = (pd.to_datetime(v.date).isin(va_days)
                 & ~pd.to_datetime(v.date).isin(hol_days)).to_numpy()
            res[arm]["shared_n"] = int(m.sum())
            res[arm]["shared_dev"] = float(
                MT.poisson_deviance(v[target].to_numpy(float)[m],
                                    v.pred.to_numpy()[m]))
        b_hol = preds["B"][hol_in_va]
        res["B"]["holiday_n"] = int(hol_in_va.sum())
        res["B"]["holiday_dev"] = float(MT.poisson_deviance(
            b_hol[target].to_numpy(float), b_hol.pred.to_numpy())) if len(b_hol) else None

        print(f"  shared workday rows: n {res['A']['shared_n']:,}  "
              f"A {res['A']['shared_dev']:.4f}  B {res['B']['shared_dev']:.4f}  "
              f"delta {(res['B']['shared_dev'] / res['A']['shared_dev'] - 1) * 100:+.2f}%")
        if res["B"]["holiday_n"]:
            print(f"  B on holiday rows  : n {res['B']['holiday_n']:,}  "
                  f"dev {res['B']['holiday_dev']:.4f}")
        out["targets"][target] = res

    (OUT / "holiday_ab.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote : {OUT / 'holiday_ab.json'}   ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
