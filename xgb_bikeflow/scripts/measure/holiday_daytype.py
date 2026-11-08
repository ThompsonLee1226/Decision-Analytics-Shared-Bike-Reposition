#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What does the shipped (workday-trained) model say about a holiday, and what does
it say once it has seen one?

Fold 3 only, both arms, split the score by day type.  This is the number that
decides whether keeping the holidays is a nicety or a fix: the shipped model has
`is_holiday == 0` on every training row, so it cannot have learned a holiday
level.  A growing trend scale is all it has, and holidays come out high.
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


def main() -> int:
    t0 = time.time()
    spec = SP2.resolve(SP2.load_config(ROOT / "configs" / "horizon_am.yaml"))
    hol = pd.to_datetime(HOLS)

    wd = build("workday")
    with_hol = build(sorted({str(pd.Timestamp(d).date()) for d in
                             set(wd.date.unique()) | set(hol)}))
    folds = SP.make(spec.fold, with_hol)
    fold = folds[-1]
    print(f"fold {fold.number}  valid {fold.start}..{fold.end}"
          f"  train {fold.train_days}d")

    train_units, _ = RW.select_units(with_hol, spec.rows, spec.targets)
    score_units, _ = RW.resolve_score_units(train_units, spec.rows, with_hol,
                                           spec.targets)
    score_set = set(score_units)
    loss = GR.resolve_membership(
        GR.resolve(spec.loss), Path(spec.config_path).parent,
        available=with_hol.block_id)

    out = {"fold": fold.number, "window": [fold.start, fold.end], "targets": {}}
    for target in ("arrivals", "departures"):
        res = {}
        for arm, drop in (("A_no_holiday", True), ("B_with_holiday", False)):
            tr = fold.train
            if drop:
                tr = tr.loc[~tr.date.isin(hol)]
            tr = tr[tr.block_id.isin(set(train_units))]
            f = SP.Fold(fold.number, tr, fold.valid, fold.start, fold.end)
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
            va = f.valid[mask]
            y = va[target].to_numpy(float)
            mu = np.clip(np.asarray(m.predict(Xva[cols][mask]), float), 0, None)
            is_h = va.date.isin(hol).to_numpy()
            res[arm] = {
                "n_workday": int((~is_h).sum()), "n_holiday": int(is_h.sum()),
                "workday_dev": float(MT.poisson_deviance(y[~is_h], mu[~is_h])),
                "holiday_dev": float(MT.poisson_deviance(y[is_h], mu[is_h])),
                "holiday_actual": float(y[is_h].mean()),
                "holiday_pred": float(mu[is_h].mean()),
                "holiday_calibration": float(mu[is_h].sum() / y[is_h].sum()),
            }
            r = res[arm]
            print(f"\n{target} {arm}  (train {f.train_days}d,"
                  f" {time.time() - t0:.0f}s)")
            print(f"  workday rows n {r['n_workday']:,}  dev {r['workday_dev']:.4f}")
            print(f"  holiday rows n {r['n_holiday']:,}  dev {r['holiday_dev']:.4f}"
                  f"  actual {r['holiday_actual']:.4f}  pred {r['holiday_pred']:.4f}"
                  f"  calib {r['holiday_calibration']:.3f}x")
        a, b = res["A_no_holiday"], res["B_with_holiday"]
        res["holiday_dev_delta_pct"] = round(
            (b["holiday_dev"] / a["holiday_dev"] - 1) * 100, 2)
        res["workday_dev_delta_pct"] = round(
            (b["workday_dev"] / a["workday_dev"] - 1) * 100, 2)
        print(f"  -> holiday dev {res['holiday_dev_delta_pct']:+.2f}%"
              f"   workday dev {res['workday_dev_delta_pct']:+.2f}%")
        out["targets"][target] = res

    (OUT / "holiday_daytype.json").write_text(json.dumps(out, indent=2),
                                              encoding="utf-8")
    print(f"\nwrote : {OUT / 'holiday_daytype.json'}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
