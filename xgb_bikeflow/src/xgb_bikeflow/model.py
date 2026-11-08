#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/model.py
=====================
Fit, score, and save.

What a saved model is here
--------------------------
Not just the booster.  `booster.save_model()` writes the trees and the
`feature_names` it was trained with, and nothing else.  But this pipeline's
input to the booster is not the panel — it is the panel run through five feature
blocks, two of which carry **fitted parameters that live outside the booster**:

===========================  ==========================================
`trend_scale`                slope + intercept per day class, window origin
`arr_slot_*` / `dep_slot_*`  the (unit, slot) statistic table
===========================  ==========================================

The other three are stateless and need no state: `calendar` derives everything
from the date, `geometry` from the unit id, and `weather` from a CSV keyed by
timestamp (the path and the values are still stored, so a bundle does not depend
on that file staying put).

Load only the booster tomorrow and `trend_scale` has to be recomputed by refitting
the trend block on whatever training data is at hand.  If that data changed, the
feature is outside the distribution the trees were grown on and **nothing
raises** — the predictions just drift.  That is the failure this module exists to
prevent: the bundle stores the feature state, and `load` refuses a bundle whose
fingerprint does not match the data it is being applied to.

The five files
--------------
    model.json          booster, xgboost's own format (self-describing, portable)
    features.json       the exact column list and order
    feature_state.json  every fitted block's learned parameters
    schema.json         the input columns a caller must supply
    meta.json           target, unit, folds, hyperparameters, calibration,
                        fingerprint, row counts, timings

`features.json` is not redundant with the booster's own `feature_names`: it also
pins the order, and it survives a re-serialisation by a tool that drops names.

The fingerprint
---------------
A hash over the feature state plus the column list.  It answers the question the
booster cannot: "were these features built by the model that was trained?"  Not a
data hash — a hash of the *fitted* inputs, so it is stable across machines and
across a re-read of the same panel.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import __version__, grid as grid_mod, groups as groups_mod
from . import rows as rows_mod, slots as slots_mod, split as split_mod
from .features import FeatureSet, build as build_features
from .metrics import metrics, poisson_deviance

MODEL_FILE = "model.json"
FEATURES_FILE = "features.json"
STATE_FILE = "feature_state.json"
SCHEMA_FILE = "schema.json"
META_FILE = "meta.json"
BUNDLE_FILES = (MODEL_FILE, FEATURES_FILE, STATE_FILE, SCHEMA_FILE, META_FILE)

#: Columns a caller must supply to score new rows; the rest are derived.
INPUT_COLUMNS = ("block_id", "date", "slot")


class ModelError(RuntimeError):
    """The bundle is incomplete, or does not match the data it is applied to."""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _canonical(obj) -> str:
    """Stable JSON for hashing: sorted keys, no whitespace, floats rounded."""
    def norm(x):
        if isinstance(x, float):
            # 10 significant digits: enough to catch a real change in a fitted
            # slope, immune to the last-bit noise of a different BLAS.
            return float(f"{x:.10g}")
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"))


def fingerprint(feature_state: dict, columns: list[str]) -> str:
    payload = _canonical({"state": feature_state, "columns": list(columns)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class Bundle:
    """A trained model plus everything needed to rebuild its inputs."""

    target: str
    features: FeatureSet
    booster: xgb.XGBRegressor
    columns: list[str]
    meta: dict = field(default_factory=dict)

    # -- saving ------------------------------------------------------------
    def save(self, directory: str | Path, origin: str | None = None) -> Path:
        """Write the five files.  Returns the directory.

        `origin` pins the calendar window origin.  Without it the `frame` mode
        numbers `day_of_window` from whatever rows a caller happens to pass, so
        the same block-hour would score differently in a month-long request and
        a one-day request.  A saved model must not have that property, so the
        caller is required to say what "day 0" is -- normally the first training
        day, which is what the anchor runs effectively used.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / MODEL_FILE))

        for b in self.features.blocks:
            pin = getattr(b, "pin_origin", None)
            if pin is not None and origin is not None:
                pin(origin)

        state = self.features.state()
        cols = list(self.columns)
        (d / FEATURES_FILE).write_text(
            json.dumps({"columns": cols,
                        "n_features": len(cols),
                        "blocks": self.features.describe()},
                       indent=1, ensure_ascii=False), encoding="utf-8")
        (d / STATE_FILE).write_text(
            json.dumps(state, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8")
        required = set(INPUT_COLUMNS)
        for b in self.features.blocks:
            required.update(b.requires)
        (d / SCHEMA_FILE).write_text(
            json.dumps({
                "required_input_columns": sorted(required),
                "derived_columns": [c for c in cols
                                    if c not in required],
                "key_columns": list(INPUT_COLUMNS),
                "target": self.target,
                "note": ("required_input_columns must be present on any frame "
                         "passed to Bundle.predict(); the derived columns are "
                         "rebuilt from them by the stored feature state"),
            }, indent=1, ensure_ascii=False), encoding="utf-8")

        meta = dict(self.meta)
        meta.update({
            "target": self.target,
            "n_features": len(cols),
            "feature_columns": cols,
            "fingerprint": fingerprint(state, cols),
            "package_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        (d / META_FILE).write_text(
            json.dumps(meta, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8")
        # Keep the in-memory copy equal to what was written.  Otherwise
        # `Bundle.save(d); Bundle.load(d)` gives two objects that disagree
        # about their own fingerprint, and a caller comparing them sees a
        # spurious difference.
        self.meta = meta
        return d

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path, verify: bool = True) -> "Bundle":
        """Read the bundle back.  Missing files are an error, not a fallback."""
        d = Path(directory)
        absent = [f for f in BUNDLE_FILES if not (d / f).exists()]
        if absent:
            raise ModelError(
                f"{d} is not a model bundle: missing {absent}. A bundle needs "
                f"all of {list(BUNDLE_FILES)} — a booster on its own cannot "
                f"rebuild its features.")

        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        feats_json = json.loads((d / FEATURES_FILE).read_text(encoding="utf-8"))
        state = json.loads((d / STATE_FILE).read_text(encoding="utf-8"))
        cols = list(feats_json["columns"])

        booster = xgb.XGBRegressor()
        booster.load_model(str(d / MODEL_FILE))

        names = booster.get_booster().feature_names
        if names is not None and list(names) != cols:
            raise ModelError(
                f"{d.name}: the booster was trained on a different column order "
                f"than features.json records.\n  booster: {list(names)[:6]}...\n"
                f"  bundle : {cols[:6]}...")

        # A bundle written before the horizon was part of meta cannot map a slot
        # index back to a clock time.  Fail here, naming the cause, rather than
        # letting a weather block raise an unrelated "Context.slots is unset"
        # three call frames deep on the first predict.
        names = {e.get("name") for e in state.get("blocks", [])}
        if "weather" in names and not meta.get("slots"):
            raise ModelError(
                f"{d.name}: this bundle predates the recorded horizon (meta.json "
                f"has no 'slots'), so its weather features cannot resolve a slot "
                f"index to a clock time. Retrain to write a complete bundle -- "
                f"the fix is in the writer, not in this file.")

        fset = _rebuild_feature_set(state, meta)
        if verify:
            want = meta.get("fingerprint")
            got = fingerprint(state, cols)
            if want and want != got:
                raise ModelError(
                    f"{d.name}: feature state does not match the fingerprint in "
                    f"meta.json ({want} != {got}). The bundle was edited or "
                    f"assembled from two different runs.")
            meta["fingerprint_ok"] = True

        return cls(target=meta.get("target", meta.get("targets", ["?"])[0]
                                   if isinstance(meta.get("targets"), list)
                                   else "arrivals"),
                   features=fset, booster=booster, columns=cols, meta=meta)

    # -- scoring -----------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Feature-engineer `df` with the stored state, then predict.

        `df` needs only the columns in `schema.json` (plus `lat`/`lon` when the
        geometry block is on).  No target column is read, and no refitting is
        done: the whole point of the stored state.
        """
        missing = [c for c in INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ModelError(f"predict needs {missing} on the input frame")
        for b in self.features.blocks:
            for c in b.requires:
                if c not in df.columns:
                    raise ModelError(
                        f"predict needs {c!r}: the {b.name!r} feature block is "
                        f"part of this model and it reads that column")
        clean = df.copy()
        clean["date"] = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m-%d")
        X = self.features.transform(clean)
        absent = [c for c in self.columns if c not in X.columns]
        if absent:
            raise ModelError(
                f"the stored feature state did not produce {absent}; the bundle "
                f"and the feature blocks disagree")
        mu = self.booster.predict(X[self.columns].astype(float))
        mu = np.clip(np.asarray(mu, float), 0.0, None)
        factor = self.meta.get("calibration_factor")
        if factor:
            mu = mu * float(factor)
        return mu


def _rebuild_feature_set(state: dict, meta: dict) -> FeatureSet:
    """Re-instantiate the blocks from a saved state.

    The blocks are rebuilt from their saved states rather than from the run
    config, so a bundle stays loadable after the config is edited or deleted.
    The class comes from the registry by name, and the constructor is given the
    block's own saved options where it needs them (weather needs a path, but
    `load_state` replaces the table anyway, so a placeholder is fine).
    """
    from .features import REGISTRY

    saved = {e["name"]: e["state"] for e in state.get("blocks", [])}
    targets = tuple(meta.get("targets") or (meta.get("target", "arrivals"),))
    # The slot grid is part of the model, not of the caller's frame: a weather
    # feature needs to turn "slot 3" back into a clock time, and that mapping came
    # from the training horizon.  Rebuilding the set without it made every loaded
    # bundle raise "Context.slots is unset" at predict time -- the blocks were
    # fine, the context they were handed was not.
    slots_meta = meta.get("slots")
    slot_spec = slots_mod.resolve(slots_meta) if slots_meta else None
    blocks = []
    for name, s in saved.items():
        cls = REGISTRY.get(name)
        if cls is None:
            raise ModelError(
                f"the bundle uses feature block {name!r}, which this version of "
                f"the package does not know. Known: {sorted(REGISTRY)}")
        try:
            blk = cls()
        except TypeError as exc:
            raise ModelError(
                f"feature block {name!r} cannot be rebuilt from a bundle: "
                f"{exc}. It must be constructible with no arguments."
            ) from exc
        blk.load_state(s)
        blocks.append(blk)
    fset = FeatureSet(blocks, targets, slots=slot_spec)
    # `load_state` on the set (rather than just on each block) is what flips the
    # fitted flag; without it `transform` refuses to run even though every block
    # carries its fitted state.
    fset.load_state(state)
    return fset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    number: int
    window: str
    train_days: int
    valid_days: int
    metrics_raw: dict
    metrics_cal: dict
    calib_factor: float
    #: the gamma-weighted score, or {} when the loss is off.  Kept beside the
    #: plain metrics rather than replacing them: the gamma loss is a *different*
    #: question ("does it do well where the docks are"), and a run that reported
    #: only the blend could not show whether the blend improved because the
    #: gathering points improved or because the rest did.
    grouped: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict, repr=False)


@dataclass
class RunResult:
    target: str
    pooled_raw: dict
    pooled_cal: dict
    per_fold: list[dict]
    preds: pd.DataFrame
    bundle: Bundle
    info: dict


def _prepare_fold(fold: split_mod.Fold, features_cfg: dict, rows_cfg: dict,
                  targets: tuple[str, ...],
                  slots: "slots_mod.SlotSpec | None" = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSet, list[str],
                             pd.DataFrame]:
    """Fit the feature blocks on the fold's training rows, apply to both sides.

    Returns the filtered training frame alongside the matrices, because
    `rows.train_slots` narrows the training side and the caller needs the target
    and the weights for *those* rows.  Taking them from `fold.train` instead
    silently misaligns the vectors the moment `train_slots` stops being `all`.
    """
    slots_train = rows_cfg.get("train_slots", "all")
    tr = rows_mod.slot_filter(fold.train, slots_train)
    va = fold.valid

    fset = build_features(features_cfg, targets=targets, slots=slots)
    Xtr = fset.fit(tr).transform(tr)
    Xva = fset.transform(va)
    cols = [c for c in fset.columns if c in Xtr.columns and c in Xva.columns]
    if len(Xtr) != len(tr):
        raise ModelError(
            f"the feature blocks returned {len(Xtr)} training rows for a "
            f"{len(tr)}-row fold; the target and the weights would not line up")
    return Xtr, Xva, fset, cols, tr


def run(spec, panel: pd.DataFrame, target: str,
        tune: bool = True) -> RunResult:
    """Train and score one target over the walk-forward folds."""
    t_start = time.time()
    targets = spec.targets
    if target not in panel.columns:
        raise ModelError(f"target {target!r} is not a column of the panel")

    gspec = grid_mod.resolve(spec.unit)
    grid_info = grid_mod.verify(panel.block_id, gspec)

    # Which units are gathering points.  Resolved against the panel here rather
    # than in the spec, because the spec is built before the panel exists and a
    # gathering-point file at the wrong grid level looks like a valid one.
    loss_spec = groups_mod.resolve_membership(
        groups_mod.resolve(spec.loss), Path(spec.config_path).parent,
        available=panel.block_id)

    train_units, train_info = rows_mod.select_units(panel, spec.rows, targets)
    score_units, score_info = rows_mod.resolve_score_units(
        train_units, spec.rows, panel, targets)
    score_set = set(score_units)

    # Folds are cut on the *whole* panel and the training side is narrowed inside
    # the loop by `_prepare_fold`.  Cutting a narrowed panel here instead would
    # move the fold boundaries whenever the unit selection changed, and the
    # validation windows would no longer line up between configs.
    # Validation rows come from the *panel*, so a unit outside the training set
    # is still present and can be scored (that is the full-model study).
    folds = split_mod.make(spec.fold, panel)

    off_w = float(spec.rows.get("off_target_weight", 1.0))
    params = spec.params_for(target)

    fold_results: list[FoldResult] = []
    preds_all: list[pd.DataFrame] = []
    calib = split_mod.ForwardCalibrator(
        compound=bool(spec.calibration_cfg.get("compound", False)))
    last_fset: FeatureSet | None = None
    last_booster: xgb.XGBRegressor | None = None
    cols_final: list[str] = []
    said: set[str] = set()

    for k, fold in enumerate(folds, 1):
        # The training side is restricted to the selected units; the validation
        # side is the full fold so the scoring mask can be applied afterwards.
        f = split_mod.Fold(fold.number,
                           fold.train[fold.train.block_id.isin(set(train_units))],
                           fold.valid, fold.start, fold.end)
        Xtr, Xva, fset, cols, tr = _prepare_fold(f, spec.features, spec.rows,
                                                 targets, spec.slots)
        ytr = tr[target].to_numpy(float)
        w = rows_mod.row_weights(tr, score_set, off_w)
        gw = groups_mod.fit_weights(
            loss_spec.group_of(tr["block_id"].to_numpy()), loss_spec)
        if gw is not None:
            # `fit_weight` and `off_target_weight` multiply: one says how much a
            # row matters by *where* it is, the other by *whether it is scored*.
            w = w * gw

        booster = xgb.XGBRegressor(**params)
        fit_kwargs = {} if np.allclose(w, 1.0) else {"sample_weight": w}
        booster.fit(Xtr[cols], ytr, **fit_kwargs)

        mu = np.clip(np.asarray(booster.predict(Xva[cols]), float), 0.0, None)

        mask = rows_mod.score_mask(f.valid, score_set,
                                   spec.rows.get("score_slots", "all"))
        y_m = f.valid[target].to_numpy(float)[mask]

        raw = metrics(y_m, mu[mask])
        cal_factor = calib.factor
        mu_cal = calib.apply(mu)
        cal = metrics(y_m, mu_cal[mask])
        gmask = loss_spec.group_of(f.valid["block_id"].to_numpy())[mask]
        grouped = {"raw": groups_mod.gamma_loss(y_m, mu[mask], gmask, loss_spec),
                   "calibrated": groups_mod.gamma_loss(y_m, mu_cal[mask], gmask,
                                                       loss_spec)}

        fold_results.append(FoldResult(
            number=f.number, window=f"{f.start}..{f.end}",
            train_days=f.train_days, valid_days=f.valid_days,
            metrics_raw=raw, metrics_cal=cal, calib_factor=float(cal_factor or 1.0),
            grouped=grouped))

        preds_all.append(pd.DataFrame({
            "fold": f.number, "block_id": f.valid.block_id.to_numpy()[mask],
            "date": f.valid.date.to_numpy()[mask],
            "slot": f.valid.slot.to_numpy()[mask],
            "actual": y_m, "pred_raw": mu[mask], "pred": mu_cal[mask],
            "is_gathering": gmask,
        }))

        if spec.calibrate:
            # `compound` decides whether the next factor is fitted against the
            # raw prediction (correct) or the corrected one (what the original
            # scripts did, and what the published numbers reflect).
            calib.update(y_m, mu_cal[mask] if calib.compound else mu[mask])

        last_fset, last_booster, cols_final = fset, booster, cols
        # A borrowed slope or a fired fallback is not an error, but a user who is
        # not told about it will read the prediction as if it rested on data.  Say
        # once, on the fold where it first applies.
        if spec.verbose:
            for note in fset.notes():
                if note not in said:
                    said.add(note)
                    print(f"  note    : {note}")
        if spec.verbose:
            extra = ""
            if loss_spec.enabled:
                g_cal = grouped["calibrated"]
                if g_cal.get("mode") == "group_mean_empty":
                    # One group is empty, so there is no blend.  Printing "g / o"
                    # here would show a number against nan and read as a broken
                    # loss rather than a degenerate grouping.  Which group is
                    # empty matters: "no gathering points configured" is a
                    # deliberate setting, "every unit is one" is a tiny test panel.
                    if not g_cal.get("n_gathering"):
                        extra = (f"  dev {g_cal['loss']:.4f} "
                                 f"(no gathering points)")
                    else:
                        extra = f"  dev {g_cal['loss']:.4f} (single group)"
                else:
                    extra = (f"  gamma dev {g_cal['loss']:.4f}"
                             f" (g {g_cal['loss_gathering']:.4f}"
                             f" / o {g_cal['loss_other']:.4f})")
            print(f"  fold {f.number}  {f.start}..{f.end}  "
                  f"train {f.train_days:>2}d  dev {raw['poisson_deviance']:.4f} "
                  f"-> cal {cal['poisson_deviance']:.4f}  "
                  f"calib {fold_results[-1].calib_factor:.4f}{extra}")

    preds = pd.concat(preds_all, ignore_index=True)
    pooled_raw = metrics(preds.actual, preds.pred_raw)
    pooled_cal = metrics(preds.actual, preds.pred)
    pooled_grouped = {
        "raw": groups_mod.gamma_loss(
            preds.actual, preds.pred_raw, preds.is_gathering.to_numpy(), loss_spec),
        "calibrated": groups_mod.gamma_loss(
            preds.actual, preds.pred, preds.is_gathering.to_numpy(), loss_spec),
    }

    assert last_fset is not None and last_booster is not None
    # The saved bundle uses the LAST fold's feature state and booster, which is
    # the only pair that could be deployed: it is trained on the most history.
    # Its out-of-sample score is the fold it validated, not `pooled` — those are
    # different numbers and the bundle says which is which.
    last = fold_results[-1]
    bundle = Bundle(
        target=target, features=last_fset, booster=last_booster,
        columns=cols_final,
        meta={
            "targets": list(targets),
            "params": params,
            "grid": gspec.describe(),
            **{f"unit_{k}": v for k, v in grid_info.items()},
            "train_units": train_info,
            "score_units": score_info,
            "n_train_units": len(train_units),
            "n_score_units": len(score_units),
            "off_target_weight": off_w,
            "loss": loss_spec.describe(),
            "gamma_loss": pooled_grouped,
            # The horizon is part of the model: a saved bundle must be able to
            # turn a stored slot index back into a clock time without the caller
            # supplying the config.  Without this key, load() had no slot grid to
            # rebuild the Context with, and every prediction on a 30-minute
            # horizon raised "Context.slots is unset".
            "slots": spec.slots.describe(),
            "fold_scheme": spec.fold,
            "folds": [fr.__dict__ | {"frames": None} for fr in
                      (FoldResult(fr.number, fr.window, fr.train_days,
                                  fr.valid_days, fr.metrics_raw, fr.metrics_cal,
                                  fr.calib_factor, grouped=fr.grouped)
                       for fr in fold_results)],
            "calibration": {
                "enabled": spec.calibrate,
                "rule": "single multiplicative factor = sum(obs)/sum(pred), "
                        "fitted on earlier folds only; fold 1 uncalibrated",
                "factors": [fr.calib_factor for fr in fold_results],
                "applied_factor": float(last.calib_factor),
            },
            "calibration_factor": float(last.calib_factor),
            "deployment_fold": {"fold": last.number, "window": last.window,
                                "deviance_raw": last.metrics_raw["poisson_deviance"],
                                "deviance_cal": last.metrics_cal["poisson_deviance"],
                                "note": "the saved booster is the last fold's; its "
                                        "in-sample-trained window is everything "
                                        "before that fold"},
            "pooled": {"raw": pooled_raw, "calibrated": pooled_cal},
            "pooled_grouped": pooled_grouped,
            "n_rows": int(len(preds)),
            "config_path": spec.config_path,
            "elapsed_s": round(time.time() - t_start, 1),
        })

    info = {
        "grid": gspec.describe(), "unit_check": grid_info,
        "train_units": train_info, "score_units": score_info,
        "n_train_units": len(train_units), "n_score_units": len(score_units),
        "off_target_weight": off_w,
        "loss": loss_spec.describe(),
        "gamma_loss": pooled_grouped,
        "slots": spec.slots.describe(),
        "folds": [fr.__dict__ | {"frames": None} for fr in fold_results],
        "elapsed_s": round(time.time() - t_start, 1),
    }
    return RunResult(target=target, pooled_raw=pooled_raw, pooled_cal=pooled_cal,
                     per_fold=[fr.__dict__ | {"frames": None}
                               for fr in fold_results],
                     preds=preds, bundle=bundle, info=info)


def tune_params(spec, panel: pd.DataFrame, target: str,
                n_trials: int | None = None) -> dict:
    """Optuna search on the folded deviance, masked to the scoring rows.

    The objective is the *masked* deviance, not the training loss: a model
    trained on 2,000 units must not be tuned to be accurate on unit #1,500, so
    the search scores exactly the rows that will be reported.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials = int(n_trials or (spec.tune or {}).get("trials", 20))
    base = spec.params_for(target)
    targets = spec.targets

    train_units, _ = rows_mod.select_units(panel, spec.rows, targets)
    score_units, _ = rows_mod.resolve_score_units(train_units, spec.rows,
                                                  panel, targets)
    score_set = set(score_units)
    loss_spec = groups_mod.resolve_membership(
        groups_mod.resolve(spec.loss), Path(spec.config_path).parent,
        available=panel.block_id)
    folds = split_mod.make(spec.fold, panel)
    off_w = float(spec.rows.get("off_target_weight", 1.0))
    prepared = []
    for fold in folds:
        f = split_mod.Fold(fold.number,
                           fold.train[fold.train.block_id.isin(set(train_units))],
                           fold.valid, fold.start, fold.end)
        Xtr, Xva, _, cols, tr = _prepare_fold(f, spec.features, spec.rows, targets,
                                              spec.slots)
        # The weights are fitted here, once, exactly as `run` will fit them.  A
        # search that ignored them would optimise a differently-weighted model
        # than the one it hands its answer back to -- visible only as a best
        # value that the subsequent run cannot reproduce.
        w = rows_mod.row_weights(tr, score_set, off_w)
        gw = groups_mod.fit_weights(
            loss_spec.group_of(tr["block_id"].to_numpy()), loss_spec)
        if gw is not None:
            w = w * gw
        prepared.append((f, Xtr, Xva, cols, w, tr))

    def objective(trial) -> float:
        params = dict(base)
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0,
                                                    log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        })
        devs = []
        for f, Xtr, Xva, cols, w, tr in prepared:
            m = xgb.XGBRegressor(**params)
            fit_kwargs = {} if np.allclose(w, 1.0) else {"sample_weight": w}
            m.fit(Xtr[cols], tr[target].to_numpy(float), **fit_kwargs)
            mu = np.clip(np.asarray(m.predict(Xva[cols]), float), 0.0, None)
            mask = rows_mod.score_mask(f.valid, score_set,
                                       spec.rows.get("score_slots", "all"))
            y_m = f.valid[target].to_numpy(float)[mask]
            if loss_spec.enabled:
                # The search optimises what the run will report.  Scoring a
                # gamma-weighted run by plain deviance would tune it to be good
                # on average over 16k cells, which is not what it is for.
                gmask = loss_spec.group_of(f.valid["block_id"].to_numpy())[mask]
                devs.append(groups_mod.gamma_loss(y_m, mu[mask], gmask,
                                                  loss_spec)["loss"])
            else:
                devs.append(poisson_deviance(y_m, mu[mask]))
        return float(np.mean(devs))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=spec.seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(base)
    best.update(study.best_params)
    return {"params": best, "best_value": float(study.best_value),
            "n_trials": trials,
            "trials": [{"number": t.number, "value": t.value,
                        "params": t.params} for t in study.trials]}
