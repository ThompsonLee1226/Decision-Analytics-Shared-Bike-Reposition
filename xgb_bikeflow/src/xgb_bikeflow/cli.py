#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/cli.py
===================
    python -m xgb_bikeflow.cli build     --config CONFIG
    python -m xgb_bikeflow.cli run       --config CONFIG [--target arrivals]
    python -m xgb_bikeflow.cli check     --config CONFIG
    python -m xgb_bikeflow.cli evaluate  --model DIR --panel PANEL
    python -m xgb_bikeflow.cli inspect   --model DIR

`build`    turn the configured source into the panel and write it to disk
`run`      train, score, and write the run directory (predictions + bundles)
`check`    validate the config, the source, and the derived panel, no training
`evaluate` score a saved bundle on the deployment fold's out-of-sample window
`inspect`  print what a saved bundle contains and what input it needs

Design note: there is no `predict` command, because there is no held-out data
yet. `Bundle.predict(df)` exists and is used by `run` to score, so the path that
would be used for prediction is exercised on every run rather than only when a
prediction command is typed.  `evaluate` is the one command that loads a bundle
from disk, and it exists precisely so that path is checked too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import (__version__, data, grid, groups as groups_mod,
               metrics as metrics_mod, model,
               rollup as rollup_mod, rows, slots as slots_mod,
               sources, spec as spec_mod)
from .features import build as build_features


def _load(spec_i) -> tuple[pd.DataFrame, dict]:
    """Materialise the panel from whatever `source` says it is.

    Returns the panel and the provenance dict.  Kept as a single entry point so
    that `run` and `check` cannot disagree about what they are looking at — an
    earlier version had `check` read the panel file directly while `run` built
    it, which meant `check` could pass on a stale panel.
    """
    panel, info = sources.build_panel(spec_i.source, grid.resolve(spec_i.unit),
                                      spec_i.slots, verbose=False,
                                      sort_rows=spec_i.sort_rows)
    data.validate_panel(panel, n_slots=spec_i.slots.n_slots,
                        unit_column="block_id")
    return panel, info


def _banner(spec_i, panel: pd.DataFrame, info: dict | None = None) -> dict:
    d = data.describe(panel)
    gspec = grid.resolve(spec_i.unit)
    d["grid"] = gspec.describe()
    d["grid_check"] = grid.verify(panel.block_id, gspec)
    print(f"config   : {Path(spec_i.config_path).name}")
    if info:
        sources.print_source(info)
    print(f"rows     : {d['rows']:,}  units {d['units']}  "
          f"days {d['days']}  slots {d['slots']}  "
          f"{d['date_min']}..{d['date_max']}")
    print(f"horizon  : {spec_i.slots.label}  "
          f"({spec_i.slots.n_slots} x {spec_i.slots.width_minutes} min, "
          f"labels {', '.join(spec_i.slots.labels)})")
    share = d["grid_check"].get("share_aligned")
    tag = ("partition" if gspec.partitioned else "OVERLAPPING")
    tail = (f"{share:.0%} of ids on a block boundary" if share is not None
            else "not checkable from ids alone")
    print(f"grid     : {gspec.label}  ({tag}), {tail}")
    print(f"zeros    : arrivals {d['zero_share_arrivals']:.1%}  "
          f"departures {d['zero_share_departures']:.1%}")
    return d


def _gamma_note(gl: dict) -> str:
    """One readable line for a gamma-loss result, or "" when there is nothing to say.

    The empty-group case gets its own wording: when every unit in the panel is a
    gathering point there is no "other" half, so the blend equals the gathering
    mean and reporting a gamma would imply a balance that never happened.  This
    is a normal configuration (run the pipeline on the gathering points alone),
    not an error, so it says so rather than printing nan twice.
    """
    if not gl or gl.get("loss") is None or gl.get("mode") == "off":
        return ""
    if gl.get("mode") == "group_mean_empty":
        if not gl.get("n_gathering"):
            # `gathering: none`, or a point list that matched nothing here.  Say
            # which, because the first is a deliberate setting and the second is
            # usually a mistake that `check` should have caught earlier.
            return (f"  deviance {gl['loss']:.4f}  (no gathering points configured, "
                    f"so gamma has nothing to balance; {gl.get('n_other', 0):,} rows)")
        return (f"  deviance {gl['loss']:.4f}  (every unit is a gathering point, so "
                f"gamma balances nothing; {gl.get('n_gathering', 0):,} rows)")
    return (f"  gamma {gl.get('gamma')}  weighted dev {gl['loss']:.4f}  "
            f"(gathering {gl.get('loss_gathering', float('nan')):.4f} on "
            f"{gl.get('n_gathering', 0):,} rows / other "
            f"{gl.get('loss_other', float('nan')):.4f} on "
            f"{gl.get('n_other', 0):,} rows)")


def _write_run(spec_i, target: str, res: model.RunResult) -> Path:
    """Create the run directory for one target and write everything into it."""
    out = Path(spec_i.outdir) / target
    out.mkdir(parents=True, exist_ok=True)

    if spec_i.save.get("predictions", True):
        res.preds.to_csv(out / "predictions.csv.gz", index=False,
                         compression="gzip")

    if spec_i.save.get("models", True):
        mdir = out / str(spec_i.save.get("dir", "models"))
        res.bundle.save(mdir)

    summary = {
        "config": spec_i.config_path,
        "config_name": spec_i.name,
        "target": target,
        "package_version": __version__,
        "panel": str(spec_i.panel_path),
        "slots": spec_i.slots.describe(),
        "pooled_raw": res.pooled_raw,
        "pooled_calibrated": res.pooled_cal,
        "gamma_loss": res.info.get("gamma_loss"),
        "loss": res.info.get("loss"),
        "per_fold": res.per_fold,
        **{k: v for k, v in res.info.items() if k != "folds"},
        "folds": res.info["folds"],
        "params": res.bundle.meta["params"],
        "n_features": len(res.bundle.columns),
        "features": res.bundle.columns,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=float),
        encoding="utf-8")
    return out


def cmd_build(args) -> int:
    """Materialise the panel and write it, without training anything.

    Worth having as its own command because building the panel from 666k orders
    takes minutes while a fold takes seconds: if the source rule is wrong you
    want to find that out before, not inside, a training run.
    """
    cfg = spec_mod.load_config(args.config)
    sp = spec_mod.resolve(cfg)
    del cfg
    panel, info = _load(sp)
    _banner(sp, panel, info)
    out = Path(args.output) if args.output else sp.panel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"wrote    : {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_run(args) -> int:
    cfg = spec_mod.load_config(args.config)
    sp = spec_mod.resolve(cfg)
    del cfg
    panel, info = _load(sp)
    _banner(sp, panel, info)

    # Building the panel is the expensive, error-prone step, and `evaluate` needs
    # a labelled panel to score a saved bundle.  Always write the one we actually
    # trained on: skipping the write when a file already exists at that path would
    # leave `evaluate` scoring against rows built by an earlier config while the
    # run says nothing about the difference.
    sp.panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(sp.panel_path, index=False, compression="gzip")
    mb = sp.panel_path.stat().st_size / 1e6
    print(f"panel    : wrote {sp.panel_path}  ({mb:.1f} MB)")

    targets = ([args.target] if args.target else list(sp.targets))
    for t in targets:
        if t not in panel.columns:
            raise SystemExit(f"target {t!r} is not a column of the panel")

    print()
    overall = {}
    for target in targets:
        print(f"=== {target} " + "=" * 60)
        params = sp.params_for(target)
        if sp.tune.get("enabled"):
            n = args.trials or sp.tune.get("trials", 20)
            print(f"  tuning on {n} trials ...")
            tuned = model.tune_params(sp, panel, target, n)
            params = tuned["params"]
            print(f"  best masked deviance {tuned['best_value']:.4f}")
            sp.params_by_target.setdefault(target, {}).update(
                {k: v for k, v in params.items()
                 if k not in ("objective", "tree_method", "n_jobs",
                              "verbosity", "random_state")})
        sp.params_by_target.setdefault(target, {})
        # run() reads params through spec.params_for(), so push the tuned set
        sp.params_by_target[target] = {**params}
        res = model.run(sp, panel, target)
        print(f"  pooled raw  dev {res.pooled_raw['poisson_deviance']:.4f}  "
              f"mae {res.pooled_raw['mae']:.4f}  "
              f"calib {res.pooled_raw['calibration']:.4f}")
        print(f"  pooled cal  dev {res.pooled_cal['poisson_deviance']:.4f}  "
              f"mae {res.pooled_cal['mae']:.4f}  "
              f"calib {res.pooled_cal['calibration']:.4f}")
        gl = (res.info.get("gamma_loss") or {}).get("calibrated") or {}
        note = _gamma_note(gl)
        if note:
            print(note)
        out = _write_run(sp, target, res)
        print(f"  wrote {out}")
        overall[target] = {"raw": res.pooled_raw["poisson_deviance"],
                           "calibrated": res.pooled_cal["poisson_deviance"],
                           "gamma": gl.get("loss")}

    print("\n" + "=" * 74)
    print(f"{'target':<12}{'raw dev':>12}{'calibrated':>14}{'gamma loss':>14}")
    for t, v in overall.items():
        g = v.get("gamma")
        gtxt = f"{g:>14.4f}" if isinstance(g, float) else f"{'-':>14}"
        print(f"{t:<12}{v['raw']:>12.4f}{v['calibrated']:>14.4f}{gtxt}")
    return 0


def cmd_rollup(args) -> int:
    """Roll a run's cell-level predictions up to the network's gathering points
    and arcs.

    Reads what `run` already wrote rather than re-predicting: the predictions on
    disk are the calibrated ones that were scored, so the rolled-up totals and the
    cell-level deviance describe the same forecast.  Re-predicting here would give
    a second, subtly different set of numbers with no way to tell which one the
    report used.
    """
    from . import data as data_mod
    from . import rollup as roll_mod

    cfg = spec_mod.load_config(args.config)
    sp = spec_mod.resolve(cfg)
    del cfg
    if not sp.rollup_enabled:
        raise SystemExit(
            "this config has no `rollup:` section with a `network:` path, so "
            "there is nothing to roll up to.  Add one, e.g.\n"
            "  rollup:\n"
            "    network: ../data/network_v1.json")

    net = roll_mod.load_network(sp.rollup["network"])
    kinds = tuple(sp.rollup.get("kinds") or ("gp", "arc"))
    print(f"network  : {Path(sp.rollup['network']).name}  "
          f"version {net.version}")
    d = net.describe()
    print(f"           {d['n_gps']} gps ({d['n_gp_cells']:,} cells)  "
          f"{d['n_arcs']} arcs ({d['n_arc_cells']:,} cells, mean membership "
          f"{d['arc_cell_multiplicity']})  {d['n_cells']:,} distinct cells")

    # The observed totals are read from the panel `run` wrote, so the rollup and
    # the cell-level score are built from the same rows.
    panel = None
    if sp.rollup.get("score", True) and sp.panel_path.exists():
        panel = data_mod.read_panel(sp.panel_path, unit_column="block_id",
                                   require_geometry=False)
        print(f"panel    : {sp.panel_path.name}  {len(panel):,} rows")

    targets = ([args.target] if args.target else
               [t for t in sp.targets
                if (sp.outdir / t / "predictions.csv.gz").exists()])
    if not targets:
        raise SystemExit(
            f"no predictions under {sp.outdir}; run the config first "
            f"(`cli run --config ...`)")

    outdir = sp.outdir / str(sp.rollup.get("outdir", "entities"))
    outdir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        src = sp.outdir / target / "predictions.csv.gz"
        preds = pd.read_csv(src)
        print(f"\n=== {target} " + "=" * 58)
        print(f"  cells    {preds.block_id.nunique():,}  "
              f"rows {len(preds):,}  folds {sorted(preds.fold.unique())}")

        roll = roll_mod.row_rollup(net, preds, kinds=kinds)
        if panel is not None and target in panel.columns:
            if "actual_total" in roll.columns:
                roll = roll.drop(columns=["actual_total"])
            roll = roll_mod.with_actuals(net, panel, roll, target)

        path = outdir / f"{target}.csv.gz"
        roll.to_csv(path, index=False, compression="gzip")
        # One line per kind, because a GP total and an arc total are not the same
        # quantity: the GP cells partition the area, the arc catchments overlap it
        # (mean membership 5.7), so their sums are not comparable and neither is a
        # pooled error over both.
        for kind in kinds:
            sub = roll[roll.kind == kind]
            if sub.empty:
                print(f"  {kind:<8} no rows")
                continue
            has = sub["actual_total"].notna() if "actual_total" in sub else None
            if has is None or not has.any():
                print(f"  {kind:<8} {sub.id.nunique():>4} entities  "
                      f"{len(sub):,} rows  "
                      f"total {sub.pred_total.sum():,.1f}")
                continue
            mm = metrics_mod.metrics(sub.loc[has, "actual_total"].to_numpy(float),
                                     sub.loc[has, "pred_total"].to_numpy(float))
            print(f"  {kind:<8} {sub.id.nunique():>4} entities  "
                  f"{len(sub):,} rows  "
                  f"dev {mm['poisson_deviance']:.4f}  "
                  f"mae {mm['mae']:.3f}  "
                  f"calib {mm['calibration']:.4f}  "
                  f"sum pred {sub.loc[has, 'pred_total'].sum():,.0f} / "
                  f"obs {sub.loc[has, 'actual_total'].sum():,.0f}")
        print(f"  wrote    {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    return 0


def cmd_evaluate(args) -> int:
    """Score a **saved** bundle on a panel.

    This is deliberately not the same thing as `run`.  `run` trains and scores
    in one pass; `evaluate` loads a bundle written earlier and scores it, which
    is the only way to check that what was saved still behaves like what was
    trained.  It needs a panel that carries the target columns.
    """
    from . import data as data_mod

    bundle = model.Bundle.load(args.model)
    panel = data_mod.read_panel(args.panel, unit_column=args.unit_column,
                                require_geometry=True)
    if bundle.target not in panel.columns:
        raise data_mod.PanelError(
            f"the panel has no {bundle.target!r} column, so there is nothing to "
            f"score against")

    # Score the deployment fold's own out-of-sample window, not the whole panel.
    # Every row before that window is training data the booster has already seen,
    # so including it reports a number no honest evaluation would produce (0.95
    # against a true 1.30 on this panel).
    dep = bundle.meta.get("deployment_fold") or {}
    window = dep.get("window")
    if window and ".." in str(window):
        lo, hi = [w.strip() for w in str(window).split("..")]
        dates = pd.to_datetime(panel["date"])
        keep = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        if not keep.any():
            raise data_mod.PanelError(
                f"the panel has no rows in the bundle's deployment window "
                f"{window}; it cannot be scored out of sample")
        frame = panel[keep].reset_index(drop=True)
    else:
        frame = panel
        print("WARNING   : the bundle does not record its deployment window; "
              "scoring every row, which includes training data")

    mu = bundle.predict(frame)
    y = frame[bundle.target].to_numpy()
    m = metrics_mod.metrics(y, mu)
    print(f"model     : {args.model}")
    print(f"target    : {bundle.target}")
    print(f"fingerprint {bundle.meta.get('fingerprint')}")
    print(f"window    : {window}  (out of sample)")
    print(f"rows      : {m['n']:,}")
    print()
    print(f"  deviance    {m['poisson_deviance']:.6f}")
    print(f"  mae         {m['mae']:.6f}")
    print(f"  rmse        {m['rmse']:.6f}")
    print(f"  calibration {m['calibration']:.4f}")
    print(f"  bias        {m['bias']:+.4f}")
    return 0


def cmd_check(args) -> int:
    cfg = spec_mod.load_config(args.config)
    sp = spec_mod.resolve(cfg)
    del cfg
    panel, src_info = _load(sp)
    info = _banner(sp, panel, src_info)

    tu, tinfo = rows.select_units(panel, sp.rows, sp.targets)
    su, sinfo = rows.resolve_score_units(tu, sp.rows, panel, sp.targets)
    print(f"train    : {len(tu)} units  ({tinfo.get('mode')})")
    print(f"score    : {len(su)} units  ({sinfo.get('mode')})")
    if tinfo.get("warning") or sinfo.get("warning"):
        print(f"WARNING  : {tinfo.get('warning') or sinfo.get('warning')}")

    # The gathering-point rule is the one part of the loss that can silently do
    # nothing (an id list at the wrong grid level intersects the panel in zero
    # units), so `check` reports it before a single tree is fitted.
    try:
        loss_spec = groups_mod.resolve_membership(
            groups_mod.resolve(sp.loss), Path(sp.config_path).parent,
            available=panel.block_id)
    except groups_mod.GroupError as exc:
        print(f"loss     : MISCONFIGURED - {exc}")
        return 2
    print(f"loss     : {loss_spec.describe()}")
    if loss_spec.enabled:
        members = panel.block_id.isin(loss_spec.members)
        print(f"           gathering rows {int(members.sum()):,} of "
              f"{len(panel):,}  ({members.mean():.2%})")

    from . import split as split_mod
    folds = split_mod.make(sp.fold, panel)
    for f in folds:
        d = f.describe()
        print(f"fold {d['fold']}  valid {d['valid']}  "
              f"train {d['train_days']}d/{d['train_rows']:,} rows  "
              f"valid {d['valid_days']}d/{d['valid_rows']:,} rows")

    sub = panel[panel.block_id.isin(set(tu))]
    fset = build_features(sp.features, targets=sp.targets, slots=sp.slots)
    X = fset.fit(sub).transform(sub)
    print(f"features : {len(fset.columns)} columns from "
          f"{len(fset.blocks)} blocks")
    for b in fset.describe():
        print(f"  {b['block']:<14}{b['n_columns']:>3} cols"
              f"{'  (fitted)' if b['fitted'] else ''}")
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const:
        print(f"  constant: {const}  (no split value; harmless but useless)")
    for note in fset.notes():
        print(f"  note    : {note}")
    print("check    : OK")
    del info
    return 0


def cmd_inspect(args) -> int:
    b = model.Bundle.load(args.model)
    m = b.meta
    print(f"model     : {Path(args.model).resolve()}")
    print(f"target    : {b.target}")
    print(f"features  : {len(b.columns)}")
    print(f"fingerprint {m.get('fingerprint')}  "
          f"(verified {m.get('fingerprint_ok', False)})")
    print(f"trained   : {m.get('saved_at')}  package {m.get('package_version')}  "
          f"xgboost {m.get('xgboost')}")
    g = m.get("grid") or {}
    if g:
        print(f"grid      : {g.get('label')}  "
              f"({g.get('n_cells_per_unit')} cells/unit, "
              f"{'partition' if g.get('partition') else 'OVERLAPPING'})")
    else:
        print("grid      : unspecified")
    # `run` writes the grid diagnostics flattened with a `unit_` prefix, not as a
    # nested `grid_check` dict; reading the nested key silently printed nothing.
    if m.get("unit_checked") is False:
        print(f"            {m.get('unit_note', 'the ids cannot be checked')}")
    elif m.get("unit_checked"):
        print(f"            ids agree with the declared grid "
              f"({m.get('unit_share_aligned', 0):.0%} on a block boundary)")
    print(f"rows      : {m.get('n_rows'):,} scored"
          if m.get("n_rows") else "rows      : ?")
    dep = m.get("deployment_fold") or {}
    if dep:
        print(f"deploys   : fold {dep.get('fold')} ({dep.get('window')}), "
              f"dev {dep.get('deviance_cal'):.4f} calibrated")
    cal = m.get("calibration") or {}
    print(f"calibrate : factor {cal.get('applied_factor')}  "
          f"{'on' if cal.get('enabled') else 'off'}")
    sl = m.get("slots") or {}
    if sl:
        print(f"horizon   : {sl.get('label')}  "
              f"({len(sl.get('slot_labels') or [])} slots, "
              f"{sl.get('width_minutes')} min each)")
    loss = m.get("loss") or {}
    if loss:
        print(f"loss      : {loss.get('mode')}  gamma {loss.get('gamma')}  "
              f"gathering {loss.get('gathering')}  "
              f"({loss.get('n_gathering')} points)")
        gl = (m.get("gamma_loss") or {}).get("calibrated") or {}
        note = _gamma_note(gl)
        if note:
            print(f"           {note.strip()}")
    for b in b.features.describe():
        print(f"  {b['block']:<14}{b['n_columns']:>3} cols"
              f"{'  (fitted)' if b['fitted'] else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="xgb_bikeflow",
        description="Configurable XGBoost pipeline for bike arrival/departure "
                    "rates at a chosen spatial unit.")
    ap.add_argument("--version", action="version",
                    version=f"xgb_bikeflow {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, help_ in (
        ("build", cmd_build, "build the panel from the configured source"),
        ("run", cmd_run, "train, score, and write the run directory"),
        ("check", cmd_check, "validate the config and the derived panel"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True,
                       help="path to a run config (.yaml)")
        if name == "build":
            p.add_argument("--output", default=None,
                           help="where to write the panel (default: run.outdir/panel.csv.gz)")
        else:
            p.add_argument("--target", default=None,
                           help="one target only: arrivals or departures")
        if name == "run":
            p.add_argument("--trials", type=int, default=None,
                           help="override model.tune.trials")
        p.set_defaults(func=fn)

    p = sub.add_parser("evaluate",
                       help="score a saved bundle against a labelled panel")
    p.add_argument("--model", required=True, help="path to a model directory")
    p.add_argument("--panel", required=True, help="labelled panel to score on")
    p.add_argument("--unit-column", default="block_id")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("rollup",
                       help="roll a run's cell predictions up to GPs and arcs")
    p.add_argument("--config", required=True,
                   help="the config the run was produced with")
    p.add_argument("--target", default=None,
                   help="one target only: arrivals or departures")
    p.set_defaults(func=cmd_rollup)

    p = sub.add_parser("inspect", help="print what a saved bundle contains")
    p.add_argument("--model", required=True, help="path to a model directory")
    p.set_defaults(func=cmd_inspect)
    return ap


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args(argv)
    t0 = time.time()
    try:
        rc = args.func(args)
    except (spec_mod.ConfigError, data.PanelError, grid.GridError,
            rows.RowSpecError, model.ModelError, sources.SourceError,
            slots_mod.SlotError, groups_mod.GroupError,
            rollup_mod.RollupError) as exc:
        cls = type(exc).__name__
        print(f"\n{cls}: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"\nFileNotFoundError: {exc}", file=sys.stderr)
        return 2
    print(f"\ndone in {time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
