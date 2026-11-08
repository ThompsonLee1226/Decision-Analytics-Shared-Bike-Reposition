#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
How much does `gamma` actually control, and what does a workday-trained model do
on the eight days it has never seen?

`gamma` never enters the booster -- it weights the two group *means* after the
fact (groups.py explains why no `sample_weight` can do that).  So one fitted model
can be scored at every gamma, which is what a hyper-parameter search would have
had to do by retraining.  Sweeping it here answers "is gamma optimised or hard
coded" with a measurement.

The holiday rows are genuinely out of sample for a model trained on working days,
so no leakage question arises: those days are absent from every fold's training
set.
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

OUT = ROOT / "docs" / "eda_comparison"
HOLS = ["2026-04-04", "2026-04-05", "2026-04-06",
        "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"]


def panel_for(days):
    slot_spec = SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=30)
    gspec = G.GridSpec(level=8, shape=(1, 1), aligned=True)
    src = SRC.SourceSpec(kind="orders", path=ROOT / "data" / "orders_2026.csv.gz",
                         days=days, cells="active_in_window", min_events=2,
                         holidays=tuple(HOLS), makeup=("2026-05-09",))
    return SRC.build_panel(src, gspec, slot_spec, verbose=False)


def main() -> int:
    t0 = time.time()
    spec = SP2.resolve(SP2.load_config(ROOT / "configs" / "horizon_am.yaml"))

    # Training panel: working days, i.e. what ships.
    pan, _ = panel_for("workday")
    pan["date"] = pd.to_datetime(pan["date"])
    # Scoring panel: every calendar day.  The holiday rows exist only here.
    allpan, _ = panel_for("all")
    allpan["date"] = pd.to_datetime(allpan["date"])

    hol_days = pd.to_datetime(HOLS)
    is_hol_row = allpan.date.isin(hol_days).to_numpy()
    print(f"train panel {len(pan):,} rows / {pan.date.nunique()} days")
    print(f"score panel {len(allpan):,} rows / {allpan.date.nunique()} days"
          f"  holiday rows {int(is_hol_row.sum()):,}")

    train_units, _ = RW.select_units(pan, spec.rows, spec.targets)
    score_units, _ = RW.resolve_score_units(train_units, spec.rows, pan, spec.targets)
    score_set = set(score_units)
    loss0 = GR.resolve_membership(
        GR.resolve(spec.loss), Path(spec.config_path).parent, available=pan.block_id)

    out = {"targets": {}, "gamma_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                           0.6, 0.7, 0.8, 0.9, 1.0]}
    for target in ("arrivals", "departures"):
        fold = SP.make(spec.fold, pan)[-1]
        f = SP.Fold(fold.number,
                    fold.train[fold.train.block_id.isin(set(train_units))],
                    fold.valid, fold.start, fold.end)
        Xtr, Xva, fset, cols, tr = M._prepare_fold(
            f, spec.features, spec.rows, spec.targets, spec.slots)
        w = RW.row_weights(tr, score_set, float(spec.rows.get("off_target_weight", 1.0)))
        gw = GR.fit_weights(loss0.group_of(tr["block_id"].to_numpy()), loss0)
        if gw is not None:
            w = w * gw
        m = xgb.XGBRegressor(**spec.params_for(target))
        m.fit(Xtr[cols], tr[target].to_numpy(float),
              **({} if np.allclose(w, 1.0) else {"sample_weight": w}))
        print(f"\n{target}: fold {fold.number} {fold.start}..{fold.end} fitted"
              f"  ({time.time() - t0:.0f}s)")

        # Score the whole calendar by asking the feature set for those days.
        ctx = fset
        Xall = fset.transform(allpan) if hasattr(fset, "transform") else None
        if Xall is None:
            raise SystemExit("FeatureSet has no transform(); cannot score unseen days")
        mu = np.clip(np.asarray(m.predict(Xall[cols]), float), 0, None)
        y_hol = allpan[target].to_numpy(float)[is_hol_row]
        mu_hol = mu[is_hol_row]
        dev_hol = MT.poisson_deviance(y_hol, mu_hol)
        print(f"  holiday rows  actual {y_hol.mean():.3f}  pred {mu_hol.mean():.3f}"
              f"  calib {mu_hol.sum() / y_hol.sum():.4f}  dev {dev_hol:.4f}")

        per_day = []
        for d in sorted(hol_days):
            sel = (allpan.date == d).to_numpy()
            per_day.append({"date": str(d.date()),
                            "actual": float(allpan[target].to_numpy(float)[sel].mean()),
                            "pred": float(mu[sel].mean()),
                            "dev": float(MT.poisson_deviance(
                                allpan[target].to_numpy(float)[sel], mu[sel]))})
            print(f"    {d.date()}  actual {per_day[-1]['actual']:.3f}"
                  f"  pred {per_day[-1]['pred']:.3f}  dev {per_day[-1]['dev']:.4f}")

        # gamma sweep on the fold's own validation rows, where the group split is
        # the one the run reports.
        mask = RW.score_mask(f.valid, score_set, spec.rows.get("score_slots", "all"))
        v = f.valid[mask]
        yv = v[target].to_numpy(float)
        muv = np.clip(np.asarray(m.predict(Xva[cols][mask]), float), 0, None)
        is_g = loss0.group_of(v["block_id"].to_numpy())
        sweep = {}
        for gam in out["gamma_values"]:
            gspec = GR.GroupSpec(gathering=loss0.gathering, mode="group_mean",
                                 gamma=gam, members=loss0.members)
            gl = GR.gamma_loss(yv, muv, is_g, gspec)
            sweep[f"{gam:.1f}"] = {"loss": gl["loss"],
                                   "gathering": gl.get("loss_gathering"),
                                   "other": gl.get("loss_other")}
        best = min(sweep, key=lambda k: sweep[k]["loss"])
        print(f"  gamma: gathering rows {int(is_g.sum()):,} / {len(is_g):,}")
        for gam, row in sweep.items():
            print(f"    {gam}  {row['loss']:.5f}   g {row['gathering']:.4f}"
                  f"  o {row['other']:.4f}")
        print(f"  best gamma {best} ({sweep[best]['loss']:.5f});"
              f" 0.5 gives {sweep['0.5']['loss']:.5f}"
              f" ({(sweep['0.5']['loss'] / sweep[best]['loss'] - 1) * 100:+.2f}%)")

        out["targets"][target] = {
            "fold": fold.number, "fold_window": [fold.start, fold.end],
            "holiday_rows": int(is_hol_row.sum()),
            "holiday_actual_mean": float(y_hol.mean()),
            "holiday_pred_mean": float(mu_hol.mean()),
            "holiday_calibration": float(mu_hol.sum() / y_hol.sum()),
            "holiday_dev": float(dev_hol),
            "holiday_by_day": per_day,
            "gamma_sweep": sweep, "gamma_best": best,
        }

    (OUT / "holiday_and_gamma.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote : {OUT / 'holiday_and_gamma.json'}   ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
