"""Regression anchors for the v2 pipeline.

What changed and why these tests look different from v1
-------------------------------------------------------
v1 read a pre-built ``(unit, date, hour)`` panel.  v2 reads the **raw order
table** and counts it into the configured horizon, so the panel itself is now
derived state: same orders, different horizon, different rows.  That makes the
old anchors meaningless (they were numbers on a panel that no longer exists as
an input) and it makes three new things worth pinning instead:

1. **The counting.** A known shard of orders must produce exact departure and
   arrival counts inside a horizon, and the two sides must select independently
   (a trip whose departure is inside 07:30-10:30 but whose arrival is outside
   contributes exactly one departure and zero arrivals).  This is the step where
   an off-by-one in the half-open interval would quietly shift every number.
2. **The gamma loss.** ``gamma*mean(G) + (1-gamma)*mean(not G)`` is the metric the
   brief asks for, and it is easy to get subtly wrong in three ways that all look
   plausible: weighting rows instead of means, weighting the wrong group, or
   silently falling back to the plain mean when a group is empty.  Each is pinned
   separately.
3. **The end-to-end numbers.** One small config is trained and its pooled
   deviance recorded, so a change to a feature block, a default or a rounding
   shows up as a failing test rather than as a slightly different slide.

Why the end-to-end anchor is allowed to be a specific number
------------------------------------------------------------
It runs at ``n_jobs: 1`` on a fixed order shard, so it is reproducible.  xgboost's
histogram build is *not* bit-reproducible across thread counts (v1 measured a
0.018 deviance spread between 1 and 32 threads), which is why every shipped
config pins the thread count and this suite asserts a value instead of a range.
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import xgb_bikeflow.data as D
import xgb_bikeflow.grid as G
import xgb_bikeflow.groups as GR
import xgb_bikeflow.model as M
import xgb_bikeflow.rollup as R
import xgb_bikeflow.slots as SL
import xgb_bikeflow.sources as SR
import xgb_bikeflow.spec as S
import xgb_bikeflow.split as SP

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent              # repository root: configs/, data/, src/
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"

#: Calendar facts for the observation window (State Council notice).
HOLIDAYS = ("2026-04-04", "2026-04-05", "2026-04-06",
            "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05")
MAKEUP = ("2026-05-09",)


@pytest.fixture(scope="module")
def orders():
    """The shipped order projection, read once."""
    return pd.read_csv(DATA / "orders_2026.csv.gz",
                       parse_dates=["date"])


# ---------------------------------------------------------------------------
# 1. the counting
# ---------------------------------------------------------------------------

def test_horizon_is_half_open():
    """A trip exactly at 10:30 belongs to the next horizon, not this one."""
    spec = SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=30)
    assert spec.n_slots == 6
    assert tuple(spec.labels) == ("07:30", "08:00", "08:30", "09:00",
                                  "09:30", "10:00")
    assert spec.label == "07:30-10:30/30min"
    # 07:29:59 -> out; 07:30:00 -> slot 0; 10:29:59 -> slot 5; 10:30:00 -> out.
    got = spec.index_of(np.array([449, 450, 479, 480, 629, 630]))
    assert got.tolist() == [-1, 0, 0, 1, 5, -1]


def test_slot_span_must_tile_the_window():
    """A horizon that does not divide evenly is refused, not rounded."""
    with pytest.raises(SL.SlotError):
        SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=40)
    with pytest.raises(SL.SlotError):
        SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=200)


def test_departures_and_arrivals_select_independently(orders):
    """One trip can be a departure inside the horizon and an arrival outside it.

    Counted on the shipped file and compared against a hand filter that makes no
    assumption about the two sides being linked.  If the implementation ever
    reused one mask for both, the arrival count would come out too high.
    """
    spec = SL.SlotSpec(start_minute=450, span_minutes=180, width_minutes=30)
    dep_min = SR._to_minutes(orders["start_time"], None).to_numpy()
    arr_min = SR._to_minutes(orders["end_time"], None).to_numpy()
    dep_in = spec.index_of(dep_min) >= 0
    arr_in = spec.index_of(arr_min) >= 0

    src = SR.SourceSpec(kind="orders", path=DATA / "orders_2026.csv.gz",
                        days="workday", holidays=HOLIDAYS, makeup=MAKEUP,
                        cells="all")
    _, info = SR.build_panel(src, G.GridSpec(level=8),
                             spec, verbose=False)
    workday = SR.workday_mask(orders["date"], HOLIDAYS, MAKEUP)
    assert info["departures_events"] == int((dep_in & workday).sum())
    assert info["arrivals_events"] == int((arr_in & workday).sum())
    # And the two counts really are different: the link is not being reused.
    assert info["departures_events"] != info["arrivals_events"]
    # Some trips straddle the horizon edge in exactly one direction.
    straddle = int((dep_in & ~arr_in).sum() + (~dep_in & arr_in).sum())
    assert straddle > 0


def test_workday_rule_drops_weekends_and_holidays_but_keeps_makeup(orders):
    days = pd.Series(sorted(pd.unique(orders["date"].dt.strftime("%Y-%m-%d"))))
    keep = SR.workday_mask(pd.to_datetime(days), HOLIDAYS, MAKEUP)
    kept = set(days[keep])
    assert "2026-05-09" in kept, "the make-up Saturday is a working day"
    for gone in ("2026-03-07", "2026-03-08", "2026-04-06", "2026-05-01"):
        assert gone not in kept, gone


def test_the_raw_export_drives_the_same_pipeline():
    """Both shipped order files must build a panel, and the same shape of one.

    `dataset_train_20260921_lsz.csv.gz` is the vendor export untouched;
    `orders_2026.csv.gz` is the same trips after a cleaning pass that drops 1,246
    rows.  Renaming the date column is the only change either needs.  The keys
    have to line up -- an extra or missing (cell, day, slot) row would mean the
    two files disagree about the calendar or the universe, which is a different
    problem from a count being higher.
    """
    spec = SL.SlotSpec(450, 180, 60)
    grid = G.GridSpec(level=8)
    common = dict(days="labelled", holidays=HOLIDAYS, makeup=MAKEUP,
                  cells={"file": DATA / "gathering_points.json"})

    cleaned, a = SR.build_panel(
        SR.SourceSpec(kind="orders", path=DATA / "orders_2026.csv.gz",
                      date_column="date", **common), grid, spec, verbose=False)
    raw, b = SR.build_panel(
        SR.SourceSpec(kind="orders",
                      path=DATA / "dataset_train_20260921_lsz.csv.gz",
                      date_column="action_dt", **common), grid, spec, verbose=False)

    assert b["rows_read"] > a["rows_read"], "the raw export is the bigger file"
    assert raw.shape == cleaned.shape
    assert list(raw.columns) == list(cleaned.columns)

    key = ["block_id", "date", "slot"]
    ra = cleaned.sort_values(key).reset_index(drop=True)
    rb = raw.sort_values(key).reset_index(drop=True)
    assert ra[key].equals(rb[key]), "the two files disagree about which rows exist"

    # Cleaning only removes trips, so every count in the raw panel is >= the
    # cleaned one, and the totals differ by exactly the dropped trips.
    for col in ("arrivals", "departures"):
        assert (rb[col] >= ra[col]).all(), col
    dropped = (rb["arrivals"].sum() - ra["arrivals"].sum())
    assert dropped > 0


def test_panel_is_dense_over_the_horizon(orders):
    """Every (unit, day, slot) inside the universe appears, zeros included.

    The model is trained on a zero-inflated target, so a panel that silently
    dropped zero rows would look cleaner and train on a different distribution.
    """
    spec = SL.SlotSpec(450, 180, 60)
    src = SR.SourceSpec(kind="orders", path=DATA / "orders_2026.csv.gz",
                        days="workday", holidays=HOLIDAYS, makeup=MAKEUP,
                        cells={"file": DATA / "gathering_points.json"})
    panel, info = SR.build_panel(src, G.GridSpec(level=8), spec, verbose=False)
    assert len(panel) == info["cells_universe"] * info["days_kept"] * spec.n_slots
    assert not panel.duplicated(["block_id", "date", "slot"]).any()
    assert (panel[["arrivals", "departures"]] >= 0).all().all()
    # The universe is the configured cells, not whatever the orders happened to
    # touch: all 835 gathering points are present even though only 801 of them
    # see morning traffic.
    assert info["cells_universe"] == 835
    assert int((panel.groupby("block_id")[["arrivals", "departures"]].sum().sum(axis=1) == 0).sum()) > 0


# ---------------------------------------------------------------------------
# 2. the gamma loss
# ---------------------------------------------------------------------------

def test_gamma_weights_the_two_means_not_the_rows():
    """gamma=0.5 puts half the weight on each group *mean*.

    With 8 gathering rows against 2 others, a row-weighted reading would give the
    big group 80% of the weight.  The two readings coincide only when the groups
    are the same size, so this uses unequal groups on purpose.
    """
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0])
    is_g = np.array([True] * 8 + [False] * 2)
    spec = GR.GroupSpec(members=frozenset({"a"}), mode="group_mean", gamma=0.5)
    got = GR.gamma_loss(y, mu, is_g, spec)
    lg, lo = got["loss_gathering"], got["loss_other"]
    assert got["mode"] == "group_mean"
    assert lg < lo
    assert got["loss"] == pytest.approx(0.5 * lg + 0.5 * lo, abs=1e-9)
    # The row-weighted reading would be dominated by the gathering rows, so the
    # two must differ -- if this ever stops being true the test is vacuous.
    rw = GR.gamma_loss(y, mu, is_g, GR.GroupSpec(
        members=frozenset({"a"}), mode="row_weighted", gamma=0.5))
    assert rw["loss"] != pytest.approx(got["loss"], abs=1e-6)


def test_gamma_endpoints_are_the_pure_groups():
    y = np.array([1.0, 4.0, 0.0, 3.0])
    mu = np.array([1.5, 2.0, 0.5, 2.5])
    is_g = np.array([True, True, False, False])
    full = GR.gamma_loss(y, mu, is_g, GR.GroupSpec(
        members=frozenset({"a"}), gamma=1.0))
    none = GR.gamma_loss(y, mu, is_g, GR.GroupSpec(
        members=frozenset({"a"}), gamma=0.0))
    assert full["loss"] == pytest.approx(full["loss_gathering"], abs=1e-12)
    assert none["loss"] == pytest.approx(none["loss_other"], abs=1e-12)


def test_empty_group_is_reported_not_hidden():
    """All units gathering => no blend, and the result says so.

    Silently returning the overall mean would look like a working gamma score.
    """
    y = np.zeros(5)
    mu = np.ones(5)
    got = GR.gamma_loss(y, mu, np.ones(5, bool), GR.GroupSpec(
        members=frozenset({"a"}), gamma=0.5))
    assert got["mode"] == "group_mean_empty"
    assert got["n_other"] == 0
    assert np.isnan(got["loss_other"])
    assert got["loss"] == pytest.approx(got["loss_gathering"], abs=1e-12)


def test_gamma_out_of_range_is_refused():
    for bad in (-0.1, 1.1):
        with pytest.raises(GR.GroupError):
            GR.resolve({"gamma": bad})
    with pytest.raises(GR.GroupError):
        GR.resolve({"mode": "weighted_rows"})


def test_gathering_membership_must_intersect_the_panel():
    """A rule that matches nothing is an error, not an empty group.

    This is the failure mode that matters in practice: a gathering-point file at
    a different grid level parses fine and then applies gamma to nothing.
    """
    spec = GR.resolve({"gathering": ["notacell1", "notacell2"]})
    with pytest.raises(GR.GroupError, match="none of the 2 gathering-point ids"):
        GR.resolve_membership(spec, ROOT, available=pd.Series(["wx4eqtke"]))


# ---------------------------------------------------------------------------
# 3. end to end
# ---------------------------------------------------------------------------

def _tiny_spec(tmp: pathlib.Path):
    """A small but real config: 40 units, hourly slots, few trees."""
    spec = SL.SlotSpec(450, 180, 60)
    src = SR.SourceSpec(kind="orders", path=DATA / "orders_2026.csv.gz",
                        days="workday", holidays=HOLIDAYS, makeup=MAKEUP,
                        cells={"file": DATA / "gathering_points.json"})
    panel, _ = SR.build_panel(src, G.GridSpec(level=8), spec, verbose=False)
    keep = sorted(panel.block_id.unique())[:40]
    panel = panel[panel.block_id.isin(keep)].reset_index(drop=True)

    cfg = S.load_config(str(CONFIGS / "smoke_gp.yaml"))
    cfg["model"]["params"].update({"n_estimators": 40, "max_depth": 4})
    cfg["loss"]["gathering"] = list(keep[:10])
    cfg["loss"]["gamma"] = 0.5
    cfg["_config_path"] = str((CONFIGS / "smoke_gp.yaml").resolve())
    S.validate(cfg)
    return S.resolve(cfg), panel


def test_end_to_end_deviance(scratch):
    """A trained run lands on its recorded pooled deviance.

    Small enough to be quick, large enough that the feature blocks and the fold
    logic are actually exercised.  Update this constant only in the same commit as
    the change that moves it, and say why in the message.
    """
    spec, panel = _tiny_spec(scratch)
    res = M.run(spec, panel, "arrivals", tune=False)
    assert res.pooled_raw["n"] == len(res.preds)
    assert res.pooled_raw["poisson_deviance"] == pytest.approx(ANCHOR_ARR, abs=5e-4)

    # The gamma loss is reported alongside, and on this config both groups exist.
    g = res.info["gamma_loss"]["raw"]
    assert g["mode"] == "group_mean"
    assert g["n_gathering"] > 0 and g["n_other"] > 0


#: pooled raw deviance of the tiny config, arrivals.
#:
#: This config is 40 units and 40 trees, so it is underfit on purpose -- it is a
#: test of the pipeline (panel -> folds -> features -> loss -> calibration), not a
#: model quality benchmark.
#:
#: Two things moved this number and both are deliberate:
#:
#:   * `smoke_gp.yaml` switched from `walk_forward` to `week_forward`, so the folds
#:     are whole weeks and validate 14 / 18 / 10 days rather than 14 / 14 / 15.
#:     The row count went 5,160 -> 5,040.
#:   * Fold 3's calibration factor dropped from 2.25 to 1.58, because a 40-tree
#:     booster predicting three whole May weeks is less wrong than one predicting
#:     two and a half.
#:
#: Renamed from ANCHOR_ARR's old value 0.696294 so the diff shows why it moved.
ANCHOR_ARR = 0.641987


def test_narrowing_train_slots_keeps_the_target_aligned(scratch):
    """`rows.train_slots` must narrow X, y and the weights together.

    Regression test.  `_prepare_fold` returns the feature matrix built from the
    *slot-filtered* training frame, but `run` and `tune_params` were taking the
    target and the weights from the unfiltered fold.  With the shipped configs
    (`train_slots: all`) the two happen to be the same length, so nothing failed --
    the moment anyone filters, xgboost gets mismatched vectors, or worse, silently
    pairs each row with another row's label.
    """
    spec, panel = _tiny_spec(scratch)
    narrowed = dataclasses.replace(spec, rows={**spec.rows, "train_slots": [0]})
    res = M.run(narrowed, panel, "arrivals", tune=False)
    # Every fold trained on slot 0 only, and every row still got a prediction.
    assert res.pooled_raw["n"] == len(res.preds)
    assert np.isfinite(res.pooled_raw["poisson_deviance"])

    # A different slot subset must produce a different model, or the setting is
    # being ignored rather than applied.
    other = dataclasses.replace(spec, rows={**spec.rows, "train_slots": [1]})
    res_other = M.run(other, panel, "arrivals", tune=False)
    assert res.pooled_raw["poisson_deviance"] != pytest.approx(
        res_other.pooled_raw["poisson_deviance"], abs=1e-9)


def test_a_borrowed_trend_says_so():
    """A day type absent from training must be reported, not silently guessed.

    The shipped config trains on workdays only, so the trend block has no weekend
    slope to fit.  Borrowing the weekday slope is a defensible guess and much
    better than a crash, but a run that guesses silently reads as if it rested on
    data.  `notes()` is how it says so, and `check` prints it.
    """
    import xgb_bikeflow.features.trend as TR

    spec = SL.SlotSpec(450, 180, 60)
    # Ten weekdays and no weekend: exactly the shipped config's `days: workday`.
    days = pd.bdate_range("2026-05-04", periods=10)
    train = pd.DataFrame({
        "block_id": ["a"] * 10,
        "date": days.strftime("%Y-%m-%d"),
        "slot": [0] * 10,
        "arrivals": np.arange(10, dtype=float),
        "departures": np.arange(10, dtype=float),
    })
    block = TR.TrendBlock(target="arrivals", window_origin="train")
    ctx = TR.Context(targets=("arrivals", "departures"), slots=spec)
    # The calendar block normally supplies `dow` / `is_weekend`; feed them here.
    train["dow"] = days.dayofweek
    train["is_weekend"] = (days.dayofweek >= 5).astype("int8")
    block.fit(train, ctx)

    assert set(block.borrowed) == {True}, block.borrowed
    notes = block.notes()
    assert notes, "a borrowed slope must produce a note"
    assert "weekend" in notes[0]
    # And the block still predicts rather than raising on the missing class.
    assert block.slope.get(True) is not None


def test_a_bundle_predicts_without_the_training_frame(scratch):
    """Saving, loading, and predicting must not need the panel or the labels."""
    spec, panel = _tiny_spec(scratch)
    res = M.run(spec, panel, "arrivals", tune=False)

    directory = scratch / "arrivals"
    res.bundle.save(directory)
    loaded = M.Bundle.load(directory)

    assert loaded.columns == res.bundle.columns
    assert loaded.meta["fingerprint"] == res.bundle.meta["fingerprint"]

    keys = res.preds[["block_id", "date", "slot"]].copy()
    frame = keys.merge(
        panel[["block_id", "date", "slot", "lat", "lon"]],
        on=["block_id", "date", "slot"], how="left")
    a = res.bundle.predict(frame)
    b = loaded.predict(frame)
    assert (a == b).all()

    # And the guard: a frame missing what the blocks read is refused by name.
    with pytest.raises(M.ModelError, match="predict needs"):
        loaded.predict(keys)


def test_saved_bundle_reproduces_its_deployment_fold(scratch):
    """disk round trip + `evaluate`'s window == the number training recorded."""
    spec, panel = _tiny_spec(scratch)
    res = M.run(spec, panel, "arrivals", tune=False)

    directory = scratch / "arrivals"
    res.bundle.save(directory)
    loaded = M.Bundle.load(directory)

    dep = loaded.meta["deployment_fold"]
    lo, hi = [w.strip() for w in dep["window"].split("..")]
    dates = pd.to_datetime(panel["date"])
    frame = panel[(dates >= pd.Timestamp(lo))
                  & (dates <= pd.Timestamp(hi))].reset_index(drop=True)
    pred = loaded.predict(frame)
    m = M.metrics(frame["arrivals"].to_numpy(), pred)
    assert m["poisson_deviance"] == pytest.approx(dep["deviance_cal"], abs=1e-6)


# ---------------------------------------------------------------------------
# grid and config plumbing
# ---------------------------------------------------------------------------

def test_grid_shape_is_read_from_config():
    """`unit.shape` must reach the GridSpec.

    Regression test for a real bug: DEFAULTS used to carry `kind: cell`, which
    grid.resolve() checks first, so every config was silently forced to single
    cells whatever `shape:` said.
    """
    for name, shape, label in [("horizon_am", (1, 1), "L8/cell"),
                               ("tile3x3_l8", (3, 3), "L8/3x3")]:
        cfg = S.load_config(str(CONFIGS / f"{name}.yaml"))
        spec = S.resolve(cfg)
        assert tuple(spec.unit["shape"]) == shape, name
        assert G.resolve(spec.unit).label == label, name


def test_grid_rejects_ragged_alignment():
    with pytest.raises(G.GridError):
        G.GridSpec(level=8, shape=(2, 3), aligned=True)


def test_centred_patch_is_not_a_partition():
    spec = G.GridSpec(level=8, shape=(3, 3), aligned=False)
    assert spec.partitioned is False
    assert len(spec.cells_of("wx4eqtke")) == 9
    assert "wx4eqtke" in spec.cells_of("wx4eqtke")


def test_hourly_source_cannot_make_a_half_hour_horizon():
    """A pre-built panel has integer hours, so it cannot express 07:30-10:30."""
    src = SR.SourceSpec(kind="panel", path=DATA / "panel_cell_l8.csv.gz")
    with pytest.raises(SR.SourceError, match="slots.*is not hour-aligned|panel carries"):
        SR.build_panel(src, G.GridSpec(level=8), SL.SlotSpec(450, 180, 30),
                       verbose=False)


def test_no_feature_is_derived_from_the_label(orders):
    """The panel must not carry a column that is a function of arrivals/departures.

    Regression test for a real leak.  v1 shipped `n_cells_touched`, defined as the
    number of distinct cells inside a unit that saw a trip on that day.  It is a
    plausible-sounding static attribute and it is in fact a restatement of the
    label: at unit shape (1, 1) it equals `(arrivals + departures) > 0` exactly, so
    the model was handed the zero/non-zero structure of what it was predicting.
    `build_panel` no longer emits it.

    The check is deliberately crude -- any panel column perfectly predicted by the
    target's own support would fail it -- because a leak this shape will not look
    like a leak when it is added.
    """
    spec = SL.SlotSpec(450, 180, 60)
    src = SR.SourceSpec(kind="orders", path=DATA / "orders_2026.csv.gz",
                        days="workday", holidays=HOLIDAYS, makeup=MAKEUP,
                        cells={"file": DATA / "gathering_points.json"})
    panel, _ = SR.build_panel(src, G.GridSpec(level=8), spec, verbose=False)
    assert "n_cells_touched" not in panel.columns

    nz = (panel["arrivals"] + panel["departures"]) > 0
    for col in panel.columns:
        if col in ("arrivals", "departures", "block_id", "date", "slot",
                   "lat", "lon"):
            continue
        series = panel[col]
        if series.nunique(dropna=False) > 2:
            continue
        # A two-valued column that lines up with "did anything happen" is the
        # label wearing a different name.
        if (series.astype(bool) == nz).all():
            raise AssertionError(
                f"column {col!r} is an exact restatement of "
                f"(arrivals + departures) > 0 -- that is the label, not a feature")


def test_cli_check_runs():
    """`check` is the cheap pre-flight; it must work on the shipped config.

    The child's output is decoded as UTF-8 here, so it has to be *told* to write
    UTF-8.  Redirected stdout otherwise picks up the console code page -- cp936
    on a Chinese Windows install -- and the path printed on the `source :` line
    contains this repository's directory name.  That decoded as a
    `UnicodeDecodeError` inside `subprocess`, which `text=True` swallowed by
    handing back `stdout = None`, and the test then failed on a `TypeError`
    several lines away from the actual cause.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "xgb_bikeflow.cli", "check",
         "--config", str(CONFIGS / "horizon_am.yaml")],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        env={**__import__("os").environ,
             "PYTHONPATH": str(SRC),
             "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout is not None, "the child's output could not be decoded"
    assert "check    : OK" in proc.stdout
    assert "07:30-10:30" in proc.stdout


def test_missing_weather_hour_keeps_its_flags_unknown():
    """A date the weather table does not cover must not read as "dry".

    Regression test.  `is_raining` was `(precip > 0)` computed after a left join, so
    an hour with no weather row became a confident 0.0 while `precip_mm` beside it
    stayed NaN.  The two columns then said opposite things about the same missing
    hour, and xgboost took its default branch for one but not the other.  Unknown
    input has to stay unknown in every column derived from it.
    """
    from xgb_bikeflow.features.base import Context
    from xgb_bikeflow.features.weather import WeatherBlock

    table = pd.DataFrame({
        "ts": pd.to_datetime(["2026-05-01 08:00"]),
        "precip_mm": [0.0], "temp_c": [20.0], "wind_ms": [1.0],
        "cloud_pct": [0.0],
    })
    block = WeatherBlock(table=table)
    # slot 0 of 07:30-10:30/30min is 07:30, which floors to the absent 07:00 hour.
    df = pd.DataFrame({"block_id": ["a"], "date": ["2026-05-01"], "slot": [0]})
    out = block.transform(df, Context(slots=SL.SlotSpec(450, 180, 30)))
    assert out.iloc[0].isna().all(), out.to_dict("records")

    # And the hour that *is* covered keeps real values, not NaN.
    df2 = pd.DataFrame({"block_id": ["a"], "date": ["2026-05-01"], "slot": [1]})
    out2 = block.transform(df2, Context(slots=SL.SlotSpec(450, 180, 30)))
    assert out2.loc[0, "is_raining"] == 0.0
    assert out2.loc[0, "temp_c"] == 20.0


def test_tuning_weights_the_fit_the_same_way_run_does():
    """`tune` must optimise the model `run` will actually fit.

    Regression test for a real inconsistency: `tune_params` fitted every trial
    without `sample_weight`, while `run` applied `off_target_weight` and
    `loss.fit_weight`.  With either setting off its default the search optimised a
    differently-weighted model than the one it handed its answer back to, and the
    only symptom was a best value the following run could not reproduce.
    """
    import inspect

    src = inspect.getsource(M.tune_params)
    assert "row_weights" in src, "tune_params does not apply rows.row_weights"
    assert "fit_weights" in src, "tune_params does not apply groups.fit_weights"
    assert "sample_weight" in src, "tune_params fits without the computed weights"


def test_week_forward_cuts_on_mondays():
    """A week-aligned split must actually start every block on a Monday.

    The panel here is five working weeks, which is the case that exposed the bug:
    equal-count blocks over a weekend-free panel cut mid-week, so a validation block
    shares most of its weekdays with the training set and the fold stops meaning
    "predict next week".
    """
    dates = pd.bdate_range("2026-05-04", periods=25)          # 5 working weeks
    panel = pd.DataFrame({
        "block_id": np.repeat("wx4eqtke", len(dates)),
        "date": np.repeat(dates.strftime("%Y-%m-%d"), 1),
        "slot": 0,
        "arrivals": 1.0, "departures": 1.0,
    })
    folds = SP.week_forward(panel, 4)
    assert len(folds) == 3
    for f in folds:
        assert pd.Timestamp(f.start).dayofweek == 0, (
            f"fold {f.number} validates from {f.start}, a "
            f"{pd.Timestamp(f.start).day_name()}")
        assert pd.Timestamp(f.end).dayofweek == 4, (
            f"fold {f.number} ends on {f.end}, a "
            f"{pd.Timestamp(f.end).day_name()}; a whole-week block ends on Friday")
        assert f.valid_days % 5 == 0, (
            f"fold {f.number} validates {f.valid_days} working days, not whole weeks")


def test_week_forward_refuses_when_mondays_run_out():
    """Three weeks cannot hold four blocks: say so rather than cutting mid-week.

    This is the honest failure.  Silently falling back to equal-count blocks would
    produce a number labelled "week_forward" that came from a different geometry,
    which is worse than refusing.
    """
    dates = pd.bdate_range("2026-05-04", periods=15)          # only 3 Mondays
    panel = pd.DataFrame({
        "block_id": np.repeat("wx4eqtke", len(dates)),
        "date": np.repeat(dates.strftime("%Y-%m-%d"), 1),
        "slot": 0,
        "arrivals": 1.0, "departures": 1.0,
    })
    with pytest.raises(SP.SplitError, match="ran out of Mondays"):
        SP.week_forward(panel, 4)
    # Three blocks fit, and they are still whole weeks.
    assert len(SP.week_forward(panel, 3)) == 2


def test_equal_count_blocks_do_cut_mid_week():
    """The reason `week_forward` exists, kept as a demonstration.

    If this ever stops being true -- because someone "fixes" walk_forward to
    snap to Mondays -- then the two schemes have collapsed into one and
    `week_forward`'s docstring is telling a story that no longer happens.
    """
    dates = pd.bdate_range("2026-05-04", periods=15)
    panel = pd.DataFrame({
        "block_id": np.repeat("wx4eqtke", len(dates)),
        "date": np.repeat(dates.strftime("%Y-%m-%d"), 1),
        "slot": 0,
        "arrivals": 1.0, "departures": 1.0,
    })
    starts = [pd.Timestamp(f.start).dayofweek for f in SP.walk_forward(panel, 4)]
    assert any(d != 0 for d in starts), (
        "equal-count blocks came out week-aligned; week_forward is now redundant")


def test_week_forward_needs_a_monday_to_cut_on():
    """Four days inside one week cannot be split into week blocks -- say so."""
    dates = pd.bdate_range("2026-05-05", periods=3)            # Tue-Thu, no Monday
    panel = pd.DataFrame({
        "block_id": np.repeat("wx4eqtke", len(dates)),
        "date": np.repeat(dates.strftime("%Y-%m-%d"), 1),
        "slot": 0,
        "arrivals": 1.0, "departures": 1.0,
    })
    with pytest.raises(SP.SplitError, match="Monday"):
        SP.week_forward(panel, 2)


def test_unknown_fold_scheme_names_the_options():
    with pytest.raises(SP.SplitError, match="week_forward, walk_forward"):
        SP.make({"scheme": "random"}, pd.DataFrame())


def test_labelled_keeps_holidays_but_still_drops_weekends():
    """`labelled` is workday plus the holidays, not everything.

    The distinction is the whole point: a holiday is a quieter working day, and a
    Saturday in the morning peak is a different regime.  A rule that kept both
    would make this look like it worked while answering a different question.
    """
    import pandas as pd

    from xgb_bikeflow import sources as SRC

    hol = ["2026-04-04", "2026-04-05", "2026-04-06"]
    mk = ["2026-05-09"]
    days = pd.Series(pd.to_datetime([
        "2026-04-03",   # Friday, ordinary
        "2026-04-04",   # Saturday, holiday   -> in
        "2026-04-05",   # Sunday, holiday     -> in
        "2026-04-06",   # Monday, holiday     -> in
        "2026-04-07",   # Tuesday, ordinary
        "2026-04-11",   # Saturday, ordinary  -> out
        "2026-05-09",   # Saturday, make-up   -> in
    ]))
    got = SRC.day_mask(days, "labelled", hol, mk)
    assert list(got) == [True, True, True, True, True, False, True]

    # workday drops the three holidays and keeps the make-up Saturday.
    assert list(SRC.day_mask(days, "workday", hol, mk)) == \
        [True, False, False, False, True, False, True]
    # weekend is its complement, so the two rules partition the calendar.
    wd = SRC.day_mask(days, "workday", hol, mk)
    we = SRC.day_mask(days, "weekend", hol, mk)
    assert (wd == ~we).all()



# ---------------------------------------------------------------------------
# rollup: cell forecasts to gathering points and arcs
# ---------------------------------------------------------------------------

def _toy_network(cells_a, cells_b, shared):
    """A two-GP, two-arc network small enough to reason about by hand.

    Arc 1 serves A's cells plus `shared`; arc 2 serves B's cells plus `shared`.
    So the arc catchments overlap in exactly one cell while the GPs partition,
    which is the distinction the rollup module exists to keep straight.
    """
    return R.Network(
        version="test", rows=[{"geohash": c} for c in
                             set(cells_a) | set(cells_b) | set(shared)],
        gps={"A": {"gp_id": "A", "cells": list(cells_a)},
             "B": {"gp_id": "B", "cells": list(cells_b)}},
        arcs={1: {"arc_id": 1, "from_gp": "A", "to_gp": "B",
                  "service_cells": list(cells_a) + list(shared)},
              2: {"arc_id": 2, "from_gp": "B", "to_gp": "A",
                  "service_cells": list(cells_b) + list(shared)}},
    )


def _toy_preds(cells, n_days=2, n_slots=2):
    rows = []
    for c in cells:
        for d in ("2026-05-04", "2026-05-05")[:n_days]:
            for s in range(n_slots):
                rows.append({"block_id": c, "date": d, "slot": s,
                             "pred": 1.0, "fold": 3})
    return pd.DataFrame(rows)


def test_gp_totals_add_up_and_arc_totals_do_not():
    """The one invariant a caller can get wrong without noticing.

    GP cells are disjoint — their totals sum to the city total.  Arc catchments
    overlap (5.7-fold on the shipped network) — their totals do not, and adding
    them double-counts the shared cells.  Both facts are asserted here so that a
    future refactor that makes the two rollups share a code path has to think
    about which additivity it is preserving.
    """
    net = _toy_network(["c1", "c2"], ["c3", "c4"], ["s1"])
    preds = _toy_preds(["c1", "c2", "c3", "c4", "s1"])

    gp = R.gp_rollup(net, preds)
    arc = R.arc_rollup(net, preds)

    gp_total = gp.groupby(["date", "slot"])["pred_total"].sum().sum()
    arc_total = arc.groupby(["date", "slot"])["pred_total"].sum().sum()
    cell_total = preds.groupby(["date", "slot"])["pred"].sum().sum()

    # GPs partition: four GP cells at 1.0 each, per (date, slot).
    assert gp_total == pytest.approx(16)                   # 4 cells x 4 bins
    # Arcs each pick up the shared cell, so they double count it.
    assert arc_total == pytest.approx(cell_total + 4)      # s1 counted twice


def test_arc_reports_cell_count_so_totals_stay_comparable():
    """A 2-cell arc and a 154-cell arc cannot be compared on totals alone."""
    net = _toy_network(["c1", "c2"], ["c3", "c4"], ["s1"])
    preds = _toy_preds(["c1", "c2", "c3", "c4", "s1"])
    preds.loc[preds.block_id == "c1", "pred"] = 5.0        # one busy cell

    arc = R.arc_rollup(net, preds)
    a1 = arc[arc.arc_id == 1]
    assert set(a1.n_cells) == {3}                          # c1, c2, s1
    assert a1.pred_total.iloc[0] == pytest.approx(5.0 + 1.0 + 1.0)
    assert a1.pred_per_cell.iloc[0] == pytest.approx(7.0 / 3)


def test_rollup_refuses_a_grid_level_mismatch():
    """Predictions keyed on something else must fail loudly, not return zeros."""
    net = _toy_network(["c1", "c2"], ["c3", "c4"], ["s1"])
    preds = _toy_preds(["c1", "c2", "c3", "c4", "s1"])
    preds["block_id"] = preds["block_id"].str.upper()      # right shape, wrong ids

    with pytest.raises(R.RollupError, match="different grid levels"):
        R.gp_rollup(net, preds)


def test_rollup_keeps_a_partly_observed_gp_visible():
    """A GP with only some cells in the panel reports fewer cells, not a small dock."""
    net = _toy_network(["c1", "c2", "c3"], ["c4"], ["s1"])
    preds = _toy_preds(["c1", "c2", "s1"])                 # c3 missing entirely
    gp = R.gp_rollup(net, preds)
    a = gp[gp.gp_id == "A"]
    assert set(a.n_cells) == {2}
    assert set(a.missing_cells) == {1}


def test_actuals_are_attached_per_kind_not_per_frame():
    """A mixed GP+arc frame must use each row's own membership map.

    Regression test for a real bug: `with_actuals` read `roll["kind"].iloc[0]` and
    picked one map for the whole frame.  `row_rollup` sorts by kind and "arc" <
    "gp", so the frame always started with an arc — every GP row silently got an
    arc total attached, and the GP metrics in the first v1.0 run were nonsense
    (deviance 1.34 for a GP, with `actual_total` missing entirely).
    """
    net = _toy_network(["c1", "c2"], ["c3", "c4"], ["s1"])
    preds = _toy_preds(["c1", "c2", "c3", "c4", "s1"])
    panel = preds.rename(columns={"pred": "arrivals"})[
        ["block_id", "date", "slot", "arrivals"]]

    roll = R.row_rollup(net, preds, kinds=("gp", "arc"))
    assert roll["kind"].iloc[0] == "arc"        # the sort order that exposed it
    out = R.with_actuals(net, panel, roll, "arrivals")

    # Every GP row must have an actual: one GP cell at 1.0 is still an actual.
    gp = out[out.kind == "gp"]
    assert gp["actual_total"].notna().all(), "GP rows got no actual_total"
    # And the GP's actual (2 cells x 1.0) must differ from the arc's (3 cells).
    a = out[(out.kind == "gp") & (out.id == "A")].actual_total
    arc1 = out[(out.kind == "arc") & (out.id == 1)].actual_total
    assert set(a) == {2.0}
    assert set(arc1) == {3.0}


@pytest.fixture
def scratch():
    """A writable directory inside the workspace.

    pytest's own `tmp_path` scans the system temp root, which is read-only under
    some sandbox policies; this keeps the suite working anywhere the package does.
    """
    import shutil

    path = ROOT / ".tests-tmp"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
