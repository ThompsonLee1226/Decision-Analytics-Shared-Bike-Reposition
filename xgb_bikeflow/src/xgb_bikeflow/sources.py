"""Sources: where a panel comes from.

Two kinds, and the difference matters for what the pipeline is allowed to assume:

`sources.kind: panel`
    A table that already has one row per (unit, date, slot) with counts.  The
    pipeline validates it and uses it unchanged.  This is how a pre-aggregated
    extract is fed in, and it is the only source that can carry a *different*
    universe from the one the orders imply.

`sources.kind: orders`
    The raw order table: one row per trip, with a start/end timestamp and a
    start/end geohash.  The pipeline counts them into the slot grid itself.  This
    is the honest input when the horizon is 07:30-10:30, because an hourly panel
    cannot represent a horizon that starts half an hour into an hour.

Which column means what
-----------------------
The two counts are not symmetric and the naming is easy to flip:

    departures  a trip **starts** in the unit   -> `start_geohash`, `start_time`
    arrivals    a trip **ends**   in the unit   -> `end_geohash`,   `end_time`

An earlier version of this project built `arrivals` from `end_geohash` and
`departures` from `start_geohash`; the mapping is kept, and `describe_source`
prints both so a reversed frame is visible before it is trained on.

Zero-filling
------------
A unit-slot with no trip is an observation of **zero**, not a missing value, so
the output is a complete (unit x date x slot) grid.  That is what makes the
Poisson likelihood meaningful and what stops the model from being scored only on
busy places.  Complete means complete over the *universe* — see `cells:` below.

Choosing the universe
---------------------
`cells: active_in_window` keeps every cell that had at least `min_events` events
inside the horizon anywhere in the file.  That is a **study-area definition**, not
a feature: it is applied once, before any split, from the whole file, and it
decides which rows exist.  A cell that is never active in the window cannot be
predicted from a window that never saw it, and including 25,000 all-zero cells
would do nothing but dilute the loss.  The alternative (`cells: all`) is available
for anyone who wants the full lattice, and `cells: <list>` / `cells: {file: ...}`
for a fixed study area handed over from elsewhere (a gathering-point set, say).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo
from . import slots as slots_mod

IDX = {c: i for i, c in enumerate("0123456789bcdefghjkmnpqrstuvwxyz")}


class SourceError(ValueError):
    """The source cannot produce a panel for this configuration."""


# ---------------------------------------------------------------------------
# fast geohash -> integer index, vectorised
# ---------------------------------------------------------------------------

def _char_lut() -> np.ndarray:
    lut = np.full(256, 255, dtype=np.uint8)
    for ch, v in IDX.items():
        lut[ord(ch)] = v
    return lut


_LUT = _char_lut()


def ghash_to_idx(cells: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Vectorised `geo.to_idx` for a whole column.

    Returns (lat_index, lon_index, nbits_per_axis).  The scalar path decodes one
    string at a time and takes ~40 bit operations through Python; this does the
    same arithmetic on arrays, which is what makes a 666k-row order file
    practical.

    `nbits_per_axis` is the *bit* count `geo.from_idx` expects (a character is
    five bits, alternately spent on longitude and latitude, so an 8-character
    geohash is 20 bits per axis).  It is not the character length — confusing the
    two is why `unit_ids` used to reject every 8-character column as "20
    characters long".
    """
    arr = np.asarray(cells)
    if arr.dtype.kind != "U" and arr.dtype.kind != "O":
        arr = arr.astype(str)
    lens = pd.Series(arr).str.len().to_numpy()
    if np.isnan(lens).any() if lens.dtype.kind == "f" else False:
        raise SourceError("geohash column contains missing values")
    uniq = np.unique(lens[~pd.isna(arr)])
    if len(uniq) != 1:
        raise SourceError(
            f"geohash column mixes lengths {sorted(uniq.tolist())}; a panel must "
            f"use one grid level, and a mixed column means two levels were "
            f"concatenated")
    nb = int(uniq[0])

    codes = np.stack([np.frombuffer(s.encode("ascii"), dtype=np.uint8)
                      for s in arr]) if len(arr) < 0 else None
    # np.stack over Python-encoded strings is slow at 600k rows, so go through a
    # fixed-width unicode view instead: every string has the same length, so the
    # buffer is rectangular already.
    flat = np.frombuffer("".join(arr).encode("ascii"), dtype=np.uint8)
    codes = flat.reshape(len(arr), nb)

    vals = _LUT[codes]
    if (vals == 255).any():
        bad = np.unique(codes[vals == 255])[:5]
        raise SourceError(
            f"geohash column contains characters outside the geohash alphabet: "
            f"{[chr(int(b)) for b in bad]}")

    lat = np.zeros(len(arr), dtype=np.int64)
    lon = np.zeros(len(arr), dtype=np.int64)
    even = True
    for i in range(nb):
        v = vals[:, i].astype(np.int64)
        for mask in (16, 8, 4, 2, 1):
            bit = (v & mask) != 0
            if even:
                lon = (lon << 1) | bit
            else:
                lat = (lat << 1) | bit
            even = not even
    return lat, lon, int((nb * 5 + 1) // 2)


def unit_ids(cells: pd.Series, spec) -> pd.Series:
    """Map cells to unit ids for a `grid.GridSpec`.

    Only *aligned* groupings collapse cells one-to-one; a centred patch is one
    unit per cell and is handled by the grid module's own path, because its
    units overlap and a cell belongs to nine of them.
    """
    if not spec.partitioned:
        raise SourceError(
            f"grid {spec.label} is a centred window, so a cell belongs to several "
            f"units and cannot be counted into one. Use an aligned grouping, or "
            f"give cells: <list> of unit ids that are already centres.")
    lat, lon, nbits = ghash_to_idx(cells)
    chars = nbits * 2 // 5
    if chars != spec.level:
        raise SourceError(
            f"grid.level = {spec.level} but the order file's geohashes are "
            f"{chars} characters; they would group onto the wrong lattice. Change "
            f"grid.level, or supply cells already at that level.")
    r, c = spec.shape
    la = (lat // r) * r
    lo = (lon // c) * c

    # Snap to the block's lower-left corner, then encode.  The encoding is done
    # once per *unique* pair rather than per row: a study area has thousands of
    # cells, not one per order, so this stays cheap even on a 666k-row file.
    pairs, inv = np.unique(np.stack([la, lo], axis=1), axis=0, return_inverse=True)
    enc = np.asarray([geo.from_idx(int(a), int(b), nbits) for a, b in pairs],
                     dtype=object)
    return pd.Series(enc[inv], index=cells.index, name="block_id")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    kind: str = "orders"
    path: Path | None = None
    #: raw order columns; defaults match the shared trip extract
    departures_cell: str = "start_geohash"
    arrivals_cell: str = "end_geohash"
    departures_time: str = "start_time"
    arrivals_time: str = "end_time"
    date_column: str = "date"
    time_format: str | None = None       # None -> let pandas infer
    #: all | workday | weekend | list of dates
    days: str = "workday"
    holidays: tuple[str, ...] = ()
    makeup: tuple[str, ...] = ()
    #: all | active_in_window | list | {file: path}
    cells: object = "active_in_window"
    min_events: int = 1
    #: panel-only: the column holding the slot/hour index
    slot_column: str = "hour"
    describe: bool = True
    extra: dict = field(default_factory=dict)


def resolve(cfg: dict, base: Path) -> SourceSpec:
    """Build a SourceSpec from the `source:` mapping, resolving relative paths."""
    cfg = dict(cfg or {})
    path = cfg.get("path")
    if path in (None, ""):
        raise SourceError("source.path is empty; a run needs an input table")
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return SourceSpec(
        kind=str(cfg.get("kind", "orders")).lower(),
        path=p,
        departures_cell=str(cfg.get("departures_cell", "start_geohash")),
        arrivals_cell=str(cfg.get("arrivals_cell", "end_geohash")),
        departures_time=str(cfg.get("departures_time", "start_time")),
        arrivals_time=str(cfg.get("arrivals_time", "end_time")),
        date_column=str(cfg.get("date_column", "date")),
        time_format=cfg.get("time_format"),
        days=str(cfg.get("days", "workday")),
        holidays=tuple(cfg.get("holidays") or ()),
        makeup=tuple(cfg.get("makeup") or ()),
        cells=cfg.get("cells", "active_in_window"),
        min_events=int(cfg.get("min_events", 1)),
        slot_column=str(cfg.get("slot_column", "hour")),
        describe=bool(cfg.get("describe", True)),
        extra=dict(cfg),
    )


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _read(path: Path, usecols=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, usecols=usecols)


def _to_minutes(ts: pd.Series, fmt: str | None) -> pd.Series:
    dt = pd.to_datetime(ts, format=fmt, errors="coerce")
    return dt.dt.hour * 60 + dt.dt.minute


def workday_mask(dates: pd.Series, holidays=(), makeup=()) -> np.ndarray:
    """True on days the scheme treats as working days.

    A make-up Saturday (调休) is a working day; a statutory holiday that falls on
    a weekday is not.  Both are calendar facts independent of the target, so
    applying them before the split leaks nothing.
    """
    d = pd.to_datetime(dates)
    hol = set(pd.to_datetime(list(holidays))) if holidays else set()
    mk = set(pd.to_datetime(list(makeup))) if makeup else set()
    is_hol = d.isin(hol).to_numpy()
    is_mk = d.isin(mk).to_numpy()
    weekday = d.dt.dayofweek.to_numpy() < 5
    return np.where(is_hol, False, np.where(is_mk, True, weekday))


#: `source.days` values.  The first three select a day type; the fourth keeps the
#: working days *and* the statutory holidays, which is what you want when a
#: holiday level is something to estimate rather than something to delete.
DAY_RULES = ("all", "workday", "weekend", "labelled")


def day_mask(dates, rule: str, holidays=(), makeup=()) -> np.ndarray:
    """The rows `rule` keeps.

    `labelled` keeps the working days **and** the statutory holidays, and nothing
    else.  The distinction from `workday` is deliberate and narrow: weekends are
    still out, because the morning-peak operation this models is a working-day
    operation, but the eight holidays stay in so `calendar.is_holiday` can carry
    information instead of being constant zero.

    Dropping those eight days is a modelling choice, not data cleaning -- the days
    are present and they are much quieter than a working day (0.34x in this
    horizon).  A model trained without them has never seen `is_holiday = 1` and
    answers a holiday with its trend extrapolation, which reads high by 3x; see
    docs/holiday_and_gamma.md for the measurement.
    """
    if rule == "all":
        return np.ones(len(dates), dtype=bool)
    if rule == "workday":
        return workday_mask(dates, holidays, makeup)
    if rule == "weekend":
        return ~workday_mask(dates, holidays, makeup)
    if rule == "labelled":
        d = pd.to_datetime(dates)
        hol = set(pd.to_datetime(list(holidays))) if holidays else set()
        return workday_mask(dates, holidays, makeup) | d.isin(hol).to_numpy()
    raise SourceError(f"source.days = {rule!r}; expected one of "
                      f"{', '.join(DAY_RULES)}, or an explicit list of dates")



def build_panel(src: SourceSpec, gspec, slot_spec: slots_mod.SlotSpec,
                verbose: bool = True,
                sort_rows: bool = False) -> tuple[pd.DataFrame, dict]:
    """Count an order table into a complete (unit, date, slot) panel.

    Returns the panel and an info dict describing what was read and discarded, so
    a run can state its own coverage instead of asking the reader to trust it.
    """
    if src.kind == "panel":
        return _panel_from_panel(src, slot_spec, verbose, sort_rows)

    need = [src.date_column, src.departures_cell, src.arrivals_cell,
            src.departures_time, src.arrivals_time]
    df = _read(src.path, usecols=need)
    n_raw = len(df)
    df[src.date_column] = pd.to_datetime(df[src.date_column], errors="coerce")
    bad_date = df[src.date_column].isna()
    if bad_date.any():
        raise SourceError(f"{int(bad_date.sum())} rows have an unparseable "
                          f"{src.date_column}")

    dm = _to_minutes(df[src.departures_time], src.time_format)
    am = _to_minutes(df[src.arrivals_time], src.time_format)

    # Slot membership is decided per event, so one trip can be a departure inside
    # the horizon and an arrival outside it.  That is the correct reading of a
    # horizon and is why the two sides are counted separately.
    dep_slot = slot_spec.index_of(dm.to_numpy())
    arr_slot = slot_spec.index_of(am.to_numpy())

    days = src.days
    if isinstance(days, (list, tuple)):
        keep_day = df[src.date_column].isin(pd.to_datetime(list(days))).to_numpy()
    else:
        keep_day = day_mask(df[src.date_column], days, src.holidays, src.makeup)
    if not keep_day.any():
        raise SourceError(
            f"source.days = {days!r} removed every row. holidays={list(src.holidays)!r} "
            f"makeup={list(src.makeup)!r}")


    dep_cell = unit_ids(df.loc[dep_slot >= 0, src.departures_cell], gspec)
    arr_cell = unit_ids(df.loc[arr_slot >= 0, src.arrivals_cell], gspec)

    # Align the masks back after the two independent selections.
    dep_idx = np.flatnonzero(dep_slot >= 0)
    arr_idx = np.flatnonzero(arr_slot >= 0)
    dep_keep = keep_day[dep_idx]
    arr_keep = keep_day[arr_idx]

    dep = pd.DataFrame({
        "block_id": dep_cell.to_numpy()[dep_keep],
        "date": df[src.date_column].to_numpy()[dep_idx][dep_keep],
        "slot": dep_slot[dep_idx][dep_keep],
    })
    arr = pd.DataFrame({
        "block_id": arr_cell.to_numpy()[arr_keep],
        "date": df[src.date_column].to_numpy()[arr_idx][arr_keep],
        "slot": arr_slot[arr_idx][arr_keep],
    })

    universe = _choose_cells(src, gspec, pd.concat(
        [dep[["block_id"]], arr[["block_id"]]], ignore_index=True), dep, arr)

    days_u = np.sort(pd.unique(df.loc[keep_day, src.date_column]))
    panel = _grid_counts(dep, arr, universe, days_u, slot_spec)
    info = {
        "kind": "orders", "path": str(src.path), "rows_read": int(n_raw),
        "days_kept": int(len(days_u)), "day_rule": days,
        "departures_events": int(dep_keep.sum()), "arrivals_events": int(arr_keep.sum()),
        "cells_universe": int(len(universe)), "cell_rule": _cell_rule(src),
        "slots": slot_spec.describe(), "grid": gspec.describe(),
    }
    if verbose:
        print_source(info)
    return panel, info


def _cell_rule(src: SourceSpec) -> str:
    c = src.cells
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return f"file:{c.get('file')}"
    return f"list({len(c)})"


def _choose_cells(src: SourceSpec, gspec, ends: pd.DataFrame,
                  dep: pd.DataFrame, arr: pd.DataFrame) -> np.ndarray:
    """The set of units the panel is built over."""
    c = src.cells
    if isinstance(c, dict) and c.get("file"):
        f = Path(c["file"])
        if not f.is_absolute():
            f = (Path(src.path).parent / f).resolve()
        if f.suffix == ".json":
            blob = json.loads(f.read_text(encoding="utf-8"))
            ids = _ids_from_json(blob)
        else:
            ids = pd.read_csv(f).iloc[:, 0].astype(str).tolist()
        return np.sort(pd.unique(pd.Series(ids, dtype=object)))
    if isinstance(c, (list, tuple)):
        return np.sort(pd.unique(pd.Series(list(c), dtype=object)))
    if c == "all":
        return np.sort(pd.unique(ends["block_id"].to_numpy()))
    if c == "active_in_window":
        counts = ends["block_id"].value_counts()
        keep = counts[counts >= max(1, src.min_events)].index
        return np.sort(pd.unique(pd.Series(list(keep), dtype=object)))
    raise SourceError(
        f"source.cells = {c!r}; expected 'all', 'active_in_window', a list of "
        f"unit ids, or {{file: path.csv|path.json}}")


def _ids_from_json(blob) -> list[str]:
    """Pull unit ids out of a grouping JSON.

    Accepts the shapes in use: a list of strings, a list of objects with a
    `cells` key, or a mapping of `groups`/`gps` -> objects with `cells`.
    """
    if isinstance(blob, list) and blob and isinstance(blob[0], str):
        return list(blob)
    groups = None
    if isinstance(blob, dict):
        for key in ("groups", "gps", "gathering_points", "cells"):
            if isinstance(blob.get(key), list):
                groups = blob[key]
                break
    elif isinstance(blob, list):
        groups = blob
    if groups is None:
        raise SourceError("could not find a list of groups in the JSON")
    out: list[str] = []
    for g in groups:
        if isinstance(g, str):
            out.append(g)
        elif isinstance(g, dict):
            for key in ("cells", "cell_ids", "geohashes", "members"):
                if isinstance(g.get(key), list):
                    out.extend(str(x) for x in g[key])
                    break
            else:
                raise SourceError(
                    f"group {g!r} has no cells list; keys were {sorted(g)}")
    return out


def _grid_counts(dep: pd.DataFrame, arr: pd.DataFrame, universe: np.ndarray,
                 days: np.ndarray, slot_spec: slots_mod.SlotSpec) -> pd.DataFrame:
    """Complete (unit x day x slot) grid with zero-filled counts."""
    idx = pd.MultiIndex.from_product(
        [universe, pd.to_datetime(days), range(slot_spec.n_slots)],
        names=["block_id", "date", "slot"])

    d = (dep.groupby(["block_id", "date", "slot"]).size()
         .rename("departures").reindex(idx, fill_value=0))
    a = (arr.groupby(["block_id", "date", "slot"]).size()
         .rename("arrivals").reindex(idx, fill_value=0))

    panel = pd.DataFrame({"departures": d.to_numpy(), "arrivals": a.to_numpy()},
                         index=idx).reset_index()
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    panel["arrivals"] = panel["arrivals"].astype("int64")
    panel["departures"] = panel["departures"].astype("int64")

    # Geometry is a constant per unit, and the geometry block needs it.
    centre = {u: geo.centre(u) for u in universe}
    panel["lat"] = panel["block_id"].map(lambda u: centre[u][0])
    panel["lon"] = panel["block_id"].map(lambda u: centre[u][1])
    return panel


def _panel_from_panel(src: SourceSpec, slot_spec, verbose: bool,
                      sort_rows: bool = False):
    from . import data as data_mod
    if not slot_spec.is_hourly:
        # Checked before reading, because the error is about the config, not the
        # file: no amount of re-reading will make an hourly column describe a
        # 07:30-10:30 horizon.
        raise SourceError(
            f"source.kind = 'panel' but slots = {slot_spec.label}. A panel carries "
            f"integer hours, so it cannot represent a horizon that is not "
            f"hour-aligned. Use source.kind = 'orders', or set "
            f"slots.start: 0 and slots.width_minutes: 60.")
    df = data_mod.read_panel(src.path, unit_column="block_id",
                             require_geometry=True, sort_rows=sort_rows,
                             n_slots=slot_spec.n_slots)
    info = {"kind": "panel", "path": str(src.path), **data_mod.describe(df)}
    if verbose:
        print_source(info)
    return df, info


def print_source(info: dict) -> None:
    if info.get("kind") == "orders":
        print(f"source   : {info['path']}")
        print(f"           {info['rows_read']:,} orders -> "
              f"{info['departures_events']:,} departures / "
              f"{info['arrivals_events']:,} arrivals inside "
              f"{info['slots']['label']}")
        print(f"           days {info['days_kept']} ({info['day_rule']})   "
              f"cells {info['cells_universe']:,} ({info['cell_rule']})")
    else:
        print(f"source   : {info['path']}")
        print(f"           {info['rows']:,} rows  units {info['units']:,}  "
              f"days {info['days']}")
