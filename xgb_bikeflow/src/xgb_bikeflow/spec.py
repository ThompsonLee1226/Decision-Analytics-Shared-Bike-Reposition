#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_bikeflow/spec.py
====================
Configuration: the whole experiment as one YAML file.

The pipeline has five independent choices, and earlier work encoded the corners
of that space as separate scripts:

    source    where rows come from           (raw orders / a prepared panel)
    slots     what the horizon is            (07:30-10:30 in 30-min slots / hourly)
    unit      what a row's spatial unit is   (cell / aligned tile / centred patch)
    rows      which rows are trained/scored  (all units / chosen / masked)
    features  which feature families are on  (calendar ... weather)
    loss      how predictions are scored     (gamma-weighted gathering points)

Writing a new combination meant copying a script and editing constants -- which
is how the same `poisson_deviance` ended up in eight files.

Here each axis is a value in a YAML file.  Nothing is hard-coded, and every
config is a record of what was run.

Why YAML and not JSON
---------------------
Comments.  Half of these settings need a reason next to them ("n_jobs is pinned
because xgboost is not bit-reproducible across thread counts"), and JSON cannot
hold one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import groups as groups_mod
from . import sources as sources_mod
from . import slots as slots_mod
from . import split as split_mod

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "run": {
        "name": "run",
        "outdir": "runs",
        "seed": 0,
        "verbose": True,
    },
    # Where the rows come from.  `orders` counts the raw trip table into the slot
    # grid; `panel` uses a table that is already one row per (unit, date, slot).
    "source": {
        "kind": "orders",
        "path": None,
        "date_column": "date",
        "departures_cell": "start_geohash",
        "arrivals_cell": "end_geohash",
        "departures_time": "start_time",
        "arrivals_time": "end_time",
        "time_format": None,
        #: 'workday' drops weekends and statutory holidays, keeping make-up days;
        #: this is what the morning peak is planned over.  'labelled' keeps every
        #: day and leaves the distinction to calendar.is_holiday, which is what a
        #: holiday-level estimate needs.  See sources.DAY_RULES.
        "days": "workday",
        "holidays": [],
        "makeup": [],
        #: which cells the panel is built over: a study-area definition, not a
        #: feature.  'active_in_window' keeps cells with >= min_events in the
        #: horizon; 'all' keeps the whole lattice.
        "cells": "active_in_window",
        "min_events": 1,
        #: keep the file's row order.  xgboost's `subsample` draws by row
        #: position, so re-sorting the same rows trains a different model; see
        #: `data.validate_panel` for the measurement.
        "sort_rows": False,
        "describe": True,
    },
    # The horizon.  A SlotSpec: start and width are in minutes past midnight, so
    # 07:30-10:30 in half hours is start: 450, width_minutes: 30.
    "slots": {
        "start": 450,
        "span_minutes": 180,
        "width_minutes": 30,
    },
    # The grid the panel is counted on.  See grid.py: a unit is a level, a
    # shape and an alignment, so there is no enumeration to pick from.
    "unit": {
        "level": 8,             # geohash characters of one cell: 6 | 7 | 8
        "shape": [1, 1],        # cells per unit; [1, 1] is a single cell
        "aligned": True,        # False = a window centred on the cell (overlaps)
    },
    "rows": {
        #: train on all units in the panel, or only these
        "train_units": "all",     # all | top_n | explicit list | from_file
        "top_n": 20,          # a default rank, not a study design
        "units_file": None,
        #: score on these units (None -> same as train_units)
        "score_units": None,
        #: slots per day trained/scored on: all | list of slot indices
        "train_slots": "all",
        "score_slots": "all",
        #: per-row weight applied to rows NOT in score_units (0 = the mask)
        "off_target_weight": 1.0,
    },
    # The loss.  gamma mixes the two groups' *means*, not their rows: the shipped
    # panel has 801 gathering points against 8,793 other cells, so a per-row weight
    # of gamma would let gamma=0.5 put 8.3% of the weight on gathering points.
    "loss": {
        "gathering": "none",     # none | all | list of ids | {file: path}
        "mode": "group_mean",    # group_mean | row_weighted | off
        "gamma": 0.5,
        # Also weight the *fit* by group membership.  Off because it was swept
        # over 0/0.25/0.5/0.75 and moved the gamma loss by at most 0.53% with no
        # consistent sign; see the table in README.  Kept because the question is
        # legitimate and a future panel may answer differently.
        "fit_weight": 0.0,
    },

    "features": {"blocks": None, "options": {}},
    "fold": {
        "scheme": "walk_forward",
        "n_blocks": 4,
    },
    "model": {
        "params": {
            "objective": "count:poisson",
            "tree_method": "hist",
            "n_estimators": 800,
            "learning_rate": 0.0280,
            "max_depth": 7,
            "min_child_weight": 15.1486,
            "subsample": 0.605,
            "colsample_bytree": 0.726,
            "reg_lambda": 1.0,
            "reg_alpha": 0.001,
            "gamma": 1e-8,
            "random_state": 0,
            "n_jobs": 1,
            "verbosity": 0,
        },
        #: per-target overrides merged over `params`.  The optimal capacity for
        #: arrivals is not the optimal capacity for departures — on this data
        #: arrivals wants 800 trees at depth 7, departures 600 at depth 7 with a
        #: different regularisation — so sharing one parameter set would
        #: silently assume they agree.
        "params_by_target": {},
        "tune": {"enabled": False, "trials": 20},
        "calibrate": True,
        #: how the forward calibrator is refreshed.  `compound: false` refits
        #: from the raw prediction each fold (correct); `true` feeds the already
        #: corrected prediction back in, which is what the original scripts did
        #: and what the published numbers were produced with.
        "calibration": {"compound": False},
    },
    "save": {
        "models": True,
        "dir": "models",
        "predictions": True,
    },
    # Turning a cell-level forecast into the entities a network is described in.
    # Off unless a config asks for it: most studies are about cells, and a rollup
    # needs a network file that most studies do not have.
    "rollup": {
        "network": None,          # path to a network_v1.json-shaped file
        "kinds": ["gp", "arc"],
        "outdir": "entities",
        "score": True,
    },
}

VALID_TRAIN_UNITS = ("all", "top_n", "list", "file")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """The YAML is missing a key, has a bad value, or contradicts itself."""


# ---------------------------------------------------------------------------
# Merge / validate
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive merge with `over` winning.  Lists are replaced, not extended."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{p.name} must be a YAML mapping at the top level, got "
            f"{type(data).__name__}")
    return data


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Read a YAML config, merge it over the defaults, and validate."""
    raw = load_yaml(path)
    cfg = _deep_merge(DEFAULTS, raw)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["_config_path"] = str(Path(path).resolve())
    cfg["_config_name"] = Path(path).stem
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """Fail loudly and specifically.  Every message names the YAML key."""
    src = cfg["source"]
    if src.get("kind") not in ("orders", "panel"):
        raise ConfigError(
            f"source.kind = {src.get('kind')!r}; expected 'orders' (count the raw "
            f"trip table into the slot grid) or 'panel' (a table that is already "
            f"one row per unit/date/slot)")
    if not src.get("path"):
        raise ConfigError(
            "source.path is empty. An orders source is the raw trip table; a "
            "panel source is a prepared (unit, date, slot) table.")
    days = src.get("days")
    if not (isinstance(days, (list, tuple)) or days in sources_mod.DAY_RULES):
        raise ConfigError(
            f"source.days = {days!r}; expected one of "
            f"{', '.join(sources_mod.DAY_RULES)}, or an explicit list of dates. "
            f"'labelled' keeps every day and lets calendar.is_holiday say which "
            f"are holidays -- use it when a holiday level is the thing being "
            f"estimated rather than removed.")
    cells = src.get("cells")
    ok_cells = (isinstance(cells, (list, tuple))
                or cells in ("all", "active_in_window")
                or (isinstance(cells, dict) and (cells.get("file") or cells.get("json"))))
    if not ok_cells:
        raise ConfigError(
            f"source.cells = {cells!r}; expected 'all', 'active_in_window', a list "
            f"of unit ids, or {{file: path.csv|path.json}}")

    # The horizon.  Validated by constructing it, so an impossible span fails here
    # with the slot module's own message rather than at the first panel build.
    try:
        slots_mod.resolve(cfg["slots"])
    except Exception as exc:
        raise ConfigError(f"slots: {exc}") from exc
    span = float(cfg["slots"].get("span_minutes", 180))
    width = float(cfg["slots"].get("width_minutes", 30))
    if width > 120:
        raise ConfigError(
            f"slots.width_minutes = {width}; a slot wider than two hours is a "
            f"daily aggregate, and the profile block would have one value per day")

    unit = cfg["unit"]
    if unit.get("kind") is not None:
        raise ConfigError(
            f"unit.kind = {unit.get('kind')!r} is the old spelling. A unit is now "
            f"three values:\n"
            f"  unit: {{level: 8, shape: [3, 3], aligned: true}}\n"
            f"  shape [1, 1] = single cells; aligned true = a partition, false = a "
            f"window centred on the cell (overlapping).")

    rows = cfg["rows"]
    tu = rows.get("train_units")
    if isinstance(tu, str) and tu not in VALID_TRAIN_UNITS:
        raise ConfigError(
            f"rows.train_units = {tu!r}; expected 'all', 'top_n', 'list' or "
            f"'file', or give the explicit list of unit ids")
    if tu == "top_n" and not rows.get("top_n"):
        raise ConfigError("rows.train_units is 'top_n' but rows.top_n is empty")
    if tu == "file" and not rows.get("units_file"):
        raise ConfigError("rows.train_units is 'file' but rows.units_file is None")
    w = float(rows.get("off_target_weight", 1.0))
    if not 0.0 <= w <= 1.0:
        raise ConfigError(
            f"rows.off_target_weight = {w}; must be in [0, 1].  0 reproduces the "
            f"masked loss (rows off the target units contribute nothing), "
            f"1 reproduces plain training on every row.")

    loss = cfg["loss"]
    try:
        groups_mod.resolve(loss)
    except Exception as exc:
        raise ConfigError(f"loss: {exc}") from exc
    g = loss.get("gathering")
    if g not in (None, "none", "all") and not isinstance(g, (list, tuple, dict)):
        raise ConfigError(
            f"loss.gathering = {g!r}; expected 'none', 'all', an explicit list of "
            f"unit ids, or {{file: gp.json}}")
    if isinstance(g, dict) and not (g.get("file") or g.get("json")):
        raise ConfigError(
            "loss.gathering is a mapping but has no 'file' or 'json' key")

    fold = cfg["fold"]
    if fold.get("scheme") not in split_mod.SCHEMES:
        raise ConfigError(
            f"fold.scheme = {fold.get('scheme')!r}; expected one of "
            f"{', '.join(split_mod.SCHEMES)}. A random split would fit the "
            f"same-slot profile on future days, so neither option is random.")
    if int(fold.get("n_blocks", 4)) < 3:
        raise ConfigError(
            f"fold.n_blocks = {fold.get('n_blocks')}; need >= 3 (the first block "
            f"is train-only, the rest validate)")

    params = cfg["model"]["params"]
    if params.get("objective") != "count:poisson":
        raise ConfigError(
            f"model.params.objective = {params.get('objective')!r}; this "
            f"pipeline scores Poisson deviance, so the training objective must "
            f"be count:poisson")
    if "n_jobs" not in params:
        raise ConfigError(
            "model.params.n_jobs is required. xgboost is not bit-reproducible "
            "across thread counts (measured 1.324414 vs 1.328081 on the same "
            "fold), so the thread count is part of the experiment, not an "
            "environment detail. Pin it, default 1.")

    if cfg["save"].get("models") and not cfg["save"].get("dir"):
        raise ConfigError("save.models is true but save.dir is empty")

    by_target = cfg["model"].get("params_by_target") or {}
    if not isinstance(by_target, dict):
        raise ConfigError(
            "model.params_by_target must be a mapping of target -> params, e.g.\n"
            "  params_by_target:\n"
            "    departures: {n_estimators: 600}")
    for tgt, over in by_target.items():
        if not isinstance(over, dict):
            raise ConfigError(
                f"model.params_by_target.{tgt} must be a mapping, got "
                f"{type(over).__name__}")
        if "objective" in over and over["objective"] != params.get("objective"):
            raise ConfigError(
                f"model.params_by_target.{tgt}.objective = {over['objective']!r} "
                f"differs from model.params.objective = "
                f"{params.get('objective')!r}. The scoring rule is Poisson "
                f"deviance, so every target must train with it.")

    roll = cfg.get("rollup") or {}
    if roll.get("network") is not None:
        if not str(roll.get("network", "")).strip():
            raise ConfigError(
                "rollup.network is empty. Either name a network file or remove "
                "the `rollup:` section; a rollup with no network has nothing to "
                "roll up to.")
        kinds = roll.get("kinds") or []
        if not kinds:
            raise ConfigError("rollup.kinds is empty; expected a subset of "
                              "['gp', 'arc']")
        bad = [k for k in kinds if k not in ("gp", "arc")]
        if bad:
            raise ConfigError(
                f"rollup.kinds has {bad}; the only kinds a network file defines "
                f"are 'gp' and 'arc'")


def params_for(cfg: dict, target: str) -> dict:
    """Base params with this target's overrides merged in."""
    params = dict(cfg["model"]["params"])
    params.update((cfg["model"].get("params_by_target") or {}).get(target) or {})
    return params


# ---------------------------------------------------------------------------
# Resolved specs
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """Everything a run needs, resolved from the YAML."""

    name: str
    config_path: str
    #: where rows come from, with every path already absolute
    source: "sources_mod.SourceSpec"
    #: the horizon
    slots: "slots_mod.SlotSpec"
    #: grouping options, unresolved membership (needs the panel's unit ids)
    loss: dict
    #: data.read_panel options, kept for `source.kind: panel`
    sort_rows: bool
    unit: dict
    rows: dict
    features: dict
    fold: dict
    params: dict
    params_by_target: dict
    tune: dict
    calibrate: bool
    calibration_cfg: dict
    save: dict
    outdir: Path
    seed: int
    verbose: bool
    rollup: dict = field(default_factory=dict)
    config: dict = field(repr=False, default_factory=dict)

    @property
    def rollup_enabled(self) -> bool:
        return bool(self.rollup.get("network"))

    @property
    def targets(self) -> tuple[str, ...]:
        t = self.rows.get("targets") or ("arrivals", "departures")
        return tuple(t) if not isinstance(t, str) else (t,)

    def params_for(self, target: str) -> dict:
        """Base hyperparameters with this target's overrides merged in."""
        params = dict(self.params)
        params.update(self.params_by_target.get(target) or {})
        params.setdefault("random_state", self.seed)
        return params

    # -- convenience ------------------------------------------------------
    @property
    def panel_path(self) -> Path:
        """Where a built panel is cached between the build and the run.

        A panel build reads the whole order table and is the slowest step, so it
        is written once next to the run directory and reused by subsequent
        targets.  Keeping the path on the spec means `build` and `run` cannot
        disagree about where it went.
        """
        return self.outdir / "panel.csv.gz"


def resolve(cfg: dict) -> RunSpec:
    """Turn the merged dict into a typed spec with paths resolved."""
    base = Path(cfg["_config_path"]).parent

    # Paths in a config are relative to the config file, so a config is movable
    # together with the data it points at.
    def _p(value, default):
        if value in (None, ""):
            return (base / default).resolve()
        p = Path(value)
        return (p if p.is_absolute() else (base / p)).resolve()

    # Block options may themselves point at a file (`weather.path`).  Those are
    # resolved here rather than inside the block, because a block has no idea
    # which config file constructed it and would end up resolving against the
    # process working directory — which is how a config that works from one
    # directory silently fails from another.
    features = dict(cfg["features"])
    options = dict(features.get("options") or {})
    for block, opts in list(options.items()):
        if isinstance(opts, dict) and opts.get("path"):
            options[block] = {**opts, "path": str(_p(opts["path"], "."))}
    features["options"] = options

    rows_cfg = dict(cfg["rows"])
    if rows_cfg.get("units_file"):
        rows_cfg["units_file"] = str(_p(rows_cfg["units_file"], "."))

    # `cells: {file: ...}` in the source is a study-area definition, so its path
    # is resolved the same way as everything else here.
    source_cfg = dict(cfg["source"])
    if isinstance(source_cfg.get("cells"), dict) and source_cfg["cells"].get("file"):
        source_cfg["cells"] = {**source_cfg["cells"],
                               "file": str(_p(source_cfg["cells"]["file"], "."))}

    rollup_cfg = dict(cfg.get("rollup") or {})
    if rollup_cfg.get("network"):
        rollup_cfg["network"] = str(_p(rollup_cfg["network"], "."))

    run = cfg["run"]
    spec = RunSpec(
        name=str(run["name"]),
        config_path=cfg["_config_path"],
        source=sources_mod.resolve(source_cfg, base),
        slots=slots_mod.resolve(cfg["slots"]),
        loss=dict(cfg["loss"]),
        sort_rows=bool((cfg.get("source") or {}).get("sort_rows", False)),
        unit=dict(cfg["unit"]),
        rows=rows_cfg,
        features=features,
        fold=dict(cfg["fold"]),
        params=dict(cfg["model"]["params"]),
        params_by_target=dict(cfg["model"].get("params_by_target") or {}),
        tune=dict(cfg["model"].get("tune") or {}),
        calibrate=bool(cfg["model"].get("calibrate", True)),
        calibration_cfg=dict(cfg["model"].get("calibration") or {}),
        save=dict(cfg["save"]),
        outdir=_p(run.get("outdir"), "runs") / str(run["name"]),
        seed=int(run.get("seed", 0)),
        verbose=bool(run.get("verbose", True)),
        rollup=rollup_cfg,
        config=cfg,
    )
    return spec
