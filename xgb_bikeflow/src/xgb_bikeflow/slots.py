"""Slot grids: the horizon a model predicts over.

A horizon is not the same thing as an hour.  The one this project plans against
is 07:30-10:30, which starts half an hour into an hour, so any code that keys on
`hour` silently answers a different question.  A `SlotSpec` therefore describes
the horizon explicitly as (start, span, width) in minutes past midnight:

    SlotSpec(start_minute=450, span_minutes=180, width_minutes=30)
    -> 6 slots starting 07:30, 08:00, 08:30, 09:00, 09:30, 10:00

The slot column is an integer index 0..n-1, not a clock hour.  Two consequences
worth stating because they are easy to get wrong:

  * a slot index only means something together with the spec that produced it,
    so the spec is saved in the bundle and checked on load;
  * a 60-minute grid starting at midnight gives slot == hour, which is what lets
    the older pre-extracted panels still be used unchanged.

Half-open intervals
-------------------
A slot covers [start, start + width).  At the boundary the event belongs to the
later slot, so 10:30 exactly is outside a 07:30-10:30 horizon.  Getting this
wrong moves a handful of events per boundary and is invisible in aggregate
metrics, so it is applied in one place (`index_of`) and tested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60
SLOT_COLUMN = "slot"
MINUTE_COLUMN = "slot_start_minute"


class SlotError(ValueError):
    """The slot specification is impossible or contradicts the data."""


@dataclass(frozen=True)
class SlotSpec:
    """A horizon cut into equal-width slots.

    `start_minute` and `width_minutes` are in minutes past local midnight.
    The span must be a whole number of widths, because a partial final slot
    would be a different-sized observation that the loss cannot compare.
    """

    start_minute: int = 450      # 07:30
    span_minutes: int = 180      # 07:30 -> 10:30
    width_minutes: int = 30

    def __post_init__(self) -> None:
        if self.width_minutes < 1:
            raise SlotError(f"width_minutes must be >= 1, got {self.width_minutes}")
        if self.span_minutes < self.width_minutes:
            raise SlotError(
                f"span_minutes {self.span_minutes} is shorter than one slot of "
                f"{self.width_minutes} minutes")
        if self.span_minutes % self.width_minutes:
            raise SlotError(
                f"span_minutes {self.span_minutes} is not a whole number of "
                f"{self.width_minutes}-minute slots; a partial final slot would "
                f"be a different-sized observation")
        if not 0 <= self.start_minute < MINUTES_PER_DAY:
            raise SlotError(
                f"start_minute {self.start_minute} is outside the day "
                f"(0..{MINUTES_PER_DAY - 1})")
        if self.end_minute > MINUTES_PER_DAY:
            raise SlotError(
                f"the horizon ends at minute {self.end_minute}, past midnight; "
                f"a horizon that wraps would need two dates per row")

    # -- geometry ---------------------------------------------------------
    @property
    def n_slots(self) -> int:
        return self.span_minutes // self.width_minutes

    @property
    def end_minute(self) -> int:
        return self.start_minute + self.span_minutes

    @property
    def starts(self) -> list[int]:
        """Start minute of each slot."""
        return [self.start_minute + i * self.width_minutes
                for i in range(self.n_slots)]

    @property
    def labels(self) -> list[str]:
        """Clock labels, e.g. ['07:30', '08:00', ...]."""
        return [f"{s // 60:02d}:{s % 60:02d}" for s in self.starts]

    @property
    def hours(self) -> list[int]:
        """Clock hours the horizon touches, in order, without repeats."""
        seen: list[int] = []
        for s in self.starts:
            h = s // 60
            if h not in seen:
                seen.append(h)
        return seen

    @property
    def label(self) -> str:
        return (f"{self.start_minute // 60:02d}:{self.start_minute % 60:02d}"
                f"-{self.end_minute // 60:02d}:{self.end_minute % 60:02d}"
                f"/{self.width_minutes}min")

    def describe(self) -> dict:
        return {"label": self.label, "n_slots": self.n_slots,
                "starts": self.starts, "slot_labels": self.labels,
                "width_minutes": self.width_minutes}

    # -- mapping ----------------------------------------------------------
    def index_of(self, minutes: pd.Series | np.ndarray) -> np.ndarray:
        """Slot index for minutes-past-midnight, or -1 when outside.

        Half-open: `start` is inside slot 0, `end_minute` is outside.
        """
        m = np.asarray(minutes)
        off = m - self.start_minute
        inside = (off >= 0) & (off < self.span_minutes)
        return np.where(inside, off // self.width_minutes, -1).astype(np.int64)

    def from_timestamp(self, ts: pd.Series) -> np.ndarray:
        """Slot index for a datetime series (local clock, as stored)."""
        return self.index_of(ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy())

    def minute_of_slot(self, slot: int) -> int:
        if not 0 <= slot < self.n_slots:
            raise SlotError(f"slot {slot} is outside 0..{self.n_slots - 1}")
        return self.start_minute + slot * self.width_minutes

    # -- compatibility ----------------------------------------------------
    @property
    def is_hourly(self) -> bool:
        """True when slot indices coincide with clock hours (00:00 + 60min)."""
        return self.start_minute == 0 and self.width_minutes == 60

    def matches_panel_hours(self, hours) -> bool:
        """Whether a panel with these integer `hour` values is this grid.

        A pre-extracted panel carries `hour` 0..23.  It can stand in for a slot
        grid only when the grid is the trivial identical one; for 07:30-10:30 it
        cannot, and the panel has to be rebuilt from orders.
        """
        if not self.is_hourly:
            return False
        return set(int(h) for h in pd.unique(pd.Series(list(hours)))) <= set(range(24))


def resolve(cfg: dict | None) -> SlotSpec:
    """Build a SlotSpec from the `slots:` mapping of a config."""
    cfg = dict(cfg or {})
    if "width" in cfg and "width_minutes" not in cfg:
        cfg["width_minutes"] = cfg.pop("width")
    return SlotSpec(
        start_minute=_parse_minute(cfg.get("start", 450), "slots.start"),
        span_minutes=int(cfg.get("span_minutes", 180)),
        width_minutes=int(cfg.get("width_minutes", 30)),
    )


def _parse_minute(value, key: str) -> int:
    """Accept 450, '07:30' or '7:30' for a minute-past-midnight."""
    if isinstance(value, str):
        txt = value.strip()
        if ":" not in txt:
            raise SlotError(f"{key} = {value!r}; expected 'HH:MM' or a minute count")
        hh, mm = txt.split(":", 1)
        try:
            h, m = int(hh), int(mm)
        except ValueError as exc:
            raise SlotError(f"{key} = {value!r} is not 'HH:MM'") from exc
        if not (0 <= h < 24 and 0 <= m < 60):
            raise SlotError(f"{key} = {value!r} is not a valid time of day")
        return h * 60 + m
    return int(value)
