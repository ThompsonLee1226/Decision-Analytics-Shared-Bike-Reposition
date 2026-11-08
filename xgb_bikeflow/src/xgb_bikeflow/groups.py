"""Gathering points and the gamma loss.

The specification calls for a loss that is

    gamma * (loss on gathering points) + (1 - gamma) * (loss on everything else)

Two readings of that exist, and they are not the same function:

**(a) per-row weighted.**  Give every gathering-point row weight `gamma`, every
other row weight `1 - gamma`.  This reads the formula literally, one row at a
time, and it is wrong for this data: the shipped panel has 801 gathering points
against 8,793 other cells, a ratio of 1:11.  At `gamma = 0.5` the gathering
points then carry 8.3% of the total weight, so `gamma` does not mean what the
specification says it means, and sweeping it does almost nothing.

**(b) group-mean weighted.**  Give the two *groups* weights `gamma` and
`1 - gamma`, each group's loss being its own mean per row:

    L = gamma * mean_{i in G} l_i + (1 - gamma) * mean_{i not in G} l_i

Now `gamma = 0.5` genuinely balances a gathering point against a non-gathering
point regardless of how many of each exist.  This is what "gamma times the loss on
gathering points plus (1-gamma) times the loss on all the other cells" means as a
modelling instruction, and it is what `loss.mode: group_mean` implements.

`loss.mode: row_weighted` is kept because it is the literal reading and the
difference is worth being able to measure rather than argue about — the two are
one line of config apart and a sweep shows the consequence directly.

This cannot be done with `sample_weight`
----------------------------------------
xgboost's `sample_weight` scales each row's gradient, and Poisson deviance is
`sum_i w_i * (mu_i - y_i log mu_i)`.  Dividing a group's contribution by its row
count is a *normalisation*, not a weight: no fixed per-row weight makes
`sum_i w_i l_i` equal `mean_{i in G} l_i` unless `w_i` depends on the group's own
size.  So the gamma loss is computed outside the booster, and the booster is
trained on the plain objective.  Two consequences worth stating:

  * `gamma` cannot change which model is fitted, only how it is judged, unless
    the search targets it (it does: `tune` optimises the gamma-weighted score);
  * the saved model is gamma-independent, so one bundle can be scored at any
    gamma.  The gamma it was *selected* at is recorded in `meta.json`.

Weighting the fit
-----------------
`loss.fit_weight` optionally also weights the training rows (default `off`), which
is how the earlier masked-loss work trained.  It is separate from the scoring
gamma precisely because they are different questions: what the model is optimised
for, and what it is judged on.

It defaults to off because sweeping it (0 / 0.25 / 0.5 / 0.75, two targets, the
shipped folds) moved the gamma loss by at most 0.53% with no consistent sign.
`count:poisson` already fits the conditional mean; reweighting changes which rows
push the gradient, not what the model is asked to predict, and the metric is a
blend of two means that no reweighting makes more predictable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import EPS, poisson_deviance, poisson_deviance_rows


class GroupError(ValueError):
    """The gathering-point definition or the loss specification is unusable."""


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

@dataclass
class GroupSpec:
    """Which units are gathering points, and how the loss mixes the two groups."""

    #: all | none | list of unit ids | {file: path} | {json: path, key: ...}
    gathering: object = "none"
    #: group_mean | row_weighted | off
    mode: str = "group_mean"
    gamma: float = 0.5
    #: also weight the *fit* by group membership (the earlier masked-loss regime)
    fit_weight: float = 0.0
    #: resolved membership, filled in by `resolve_membership`
    members: frozenset = field(default_factory=frozenset, repr=False)

    # -- helpers ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def group_of(self, units: pd.Series | np.ndarray) -> np.ndarray:
        """Boolean array: True where the unit is a gathering point."""
        m = self.members
        if not m:
            return np.zeros(len(units), dtype=bool)
        return pd.Series(np.asarray(units)).isin(m).to_numpy()

    def describe(self) -> dict:
        return {"gathering": _rule(self.gathering), "n_gathering": len(self.members),
                "mode": self.mode, "gamma": float(self.gamma),
                "fit_weight": float(self.fit_weight)}


def _rule(g) -> str:
    if isinstance(g, str):
        return g
    if isinstance(g, dict):
        return f"file:{g.get('file') or g.get('json')}"
    return f"list({len(g)})"


def resolve(cfg: dict) -> GroupSpec:
    """Build a GroupSpec from the `loss:` mapping of a config."""
    cfg = dict(cfg or {})
    raw_mode = cfg.get("mode", "group_mean")
    # YAML 1.1 resolves the bare words off/on/yes/no to booleans, so `mode: off`
    # -- the value this module documents -- arrives here as False and would have
    # to be quoted in every config file to work.  Accept the boolean rather than
    # making the documented spelling the one that fails; `True` is genuinely
    # ambiguous (on means what?) and is refused.
    if raw_mode is False:
        mode = "off"
    elif raw_mode is True:
        raise GroupError(
            "loss.mode: true is ambiguous -- YAML reads bare 'on' as a boolean, so "
            "this was probably meant to be 'group_mean'. Write the mode name out.")
    else:
        mode = str(raw_mode).lower()
    if mode not in ("group_mean", "row_weighted", "off"):
        raise GroupError(
            f"loss.mode = {mode!r}; expected 'group_mean' (gamma balances the two "
            f"groups' means), 'row_weighted' (gamma is a per-row weight), or 'off'")
    gamma = float(cfg.get("gamma", 0.5))
    if not 0.0 <= gamma <= 1.0:
        raise GroupError(
            f"loss.gamma = {gamma}; must be in [0, 1]. 0 ignores gathering points, "
            f"1 ignores everything else.")
    fw = float(cfg.get("fit_weight", 0.0))
    if not 0.0 <= fw <= 1.0:
        raise GroupError(f"loss.fit_weight = {fw}; must be in [0, 1]")
    return GroupSpec(gathering=cfg.get("gathering", "none"), mode=mode,
                     gamma=gamma, fit_weight=fw)


def resolve_membership(spec: GroupSpec, base: Path,
                       available: pd.Series | None = None) -> GroupSpec:
    """Fill in `spec.members` from the configured rule, checking it against data."""
    g = spec.gathering
    ids: list[str]
    if g in (None, "none", False):
        ids = []
    elif g == "all":
        if available is None:
            raise GroupError("loss.gathering = 'all' needs the panel's unit ids")
        ids = sorted(pd.unique(available).tolist())
    elif isinstance(g, str):
        raise GroupError(
            f"loss.gathering = {g!r}; expected 'none', 'all', an explicit list of "
            f"unit ids, or a mapping like {{file: gp.json}}")
    elif isinstance(g, (list, tuple)):
        ids = [str(x) for x in g]
    elif isinstance(g, dict):
        p = g.get("file") or g.get("json")
        if not p:
            raise GroupError(
                "loss.gathering is a mapping but has no 'file'/'json' key; pass "
                "{file: groups.json} or {file: cells.csv}")
        f = Path(p)
        if not f.is_absolute():
            f = (base / f).resolve()
        if not f.exists():
            raise FileNotFoundError(f"gathering-point file not found: {f}")
        ids = _ids_from_file(f, g.get("key"))
    else:
        raise GroupError(f"loss.gathering has unusable type {type(g).__name__}")

    ids = [i for i in dict.fromkeys(ids)]
    members = frozenset(ids)
    if available is not None and members:
        have = set(pd.unique(available).tolist())
        overlap = members & have
        if not overlap:
            raise GroupError(
                f"none of the {len(members):,} gathering-point ids appear in the "
                f"panel's {len(have):,} units. Either the gathering-point file is "
                f"at a different grid level than unit.level, or the two describe "
                f"different areas.")
        spec = GroupSpec(gathering=spec.gathering, mode=spec.mode,
                         gamma=spec.gamma, fit_weight=spec.fit_weight,
                         members=frozenset(overlap))
        return spec
    return GroupSpec(gathering=spec.gathering, mode=spec.mode, gamma=spec.gamma,
                     fit_weight=spec.fit_weight, members=members)


def _ids_from_file(path: Path, key: str | None) -> list[str]:
    if path.suffix == ".json":
        blob = json.loads(path.read_text(encoding="utf-8"))
        if key:
            for part in key.split("."):
                blob = blob[part]
        return _ids_from_json(blob)
    df = pd.read_csv(path)
    col = key or df.columns[0]
    if col not in df.columns:
        raise GroupError(
            f"{path.name} has no column {col!r}; columns are {list(df.columns)}")
    return df[col].astype(str).tolist()


def _ids_from_json(blob) -> list[str]:
    """Accept the shapes a grouping file in this project has used."""
    if isinstance(blob, list) and blob and isinstance(blob[0], str):
        return [str(x) for x in blob]
    groups = blob
    if isinstance(blob, dict):
        for k in ("groups", "gps", "gathering_points", "cells", "units"):
            if isinstance(blob.get(k), list):
                groups = blob[k]
                break
        else:
            raise GroupError(
                f"could not find a list of groups in the JSON; keys were "
                f"{sorted(blob)[:8]}")
    out: list[str] = []
    for g in groups:
        if isinstance(g, str):
            out.append(g)
        elif isinstance(g, dict):
            for k in ("cells", "cell_ids", "geohashes", "members", "units"):
                if isinstance(g.get(k), list):
                    out.extend(str(x) for x in g[k])
                    break
            else:
                raise GroupError(
                    f"a group has no cells list; keys were {sorted(g)[:8]}")
        else:
            raise GroupError(f"unexpected group type {type(g).__name__}")
    return out


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------

def gamma_loss(y, mu, is_gathering: np.ndarray, spec: GroupSpec) -> dict:
    """Score a prediction under the configured loss.

    Returns a dict with the mixed `loss` plus both groups' own means, so a run can
    show that gamma did what it claims rather than only reporting the blend (a
    blend can stay flat while both halves move in opposite directions).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    g = np.asarray(is_gathering, dtype=bool)
    if len(g) != len(y):
        raise GroupError(f"group mask has {len(g)} rows, target has {len(y)}")

    rows = poisson_deviance_rows(y, mu)
    n_g, n_o = int(g.sum()), int((~g).sum())

    if spec.mode == "off":
        return {"loss": float(rows.mean()), "mode": "off", "gamma": None,
                "loss_gathering": float("nan"), "loss_other": float("nan"),
                "n_gathering": n_g, "n_other": n_o,
                "share_gathering": float(n_g / max(len(g), 1))}

    if spec.mode == "row_weighted":
        w = np.where(g, spec.gamma, 1.0 - spec.gamma)
        tot = float(w.sum())
        loss = float((w * rows).sum() / tot) if tot > 0 else float("nan")
        return {"loss": loss, "mode": "row_weighted", "gamma": float(spec.gamma),
                "loss_gathering": float(rows[g].mean()) if n_g else float("nan"),
                "loss_other": float(rows[~g].mean()) if n_o else float("nan"),
                "n_gathering": n_g, "n_other": n_o,
                "weight_share_gathering": float(w[g].sum() / tot) if tot else 0.0,
                "share_gathering": float(n_g / max(len(g), 1))}

    # group_mean
    if n_g == 0 or n_o == 0:
        # One group is empty, so gamma has nothing to balance. Fall back to the
        # overall mean and say so instead of returning a number that looks like a
        # successful gamma-weighted score.
        return {"loss": float(rows.mean()), "mode": "group_mean_empty",
                "gamma": float(spec.gamma),
                "loss_gathering": float(rows[g].mean()) if n_g else float("nan"),
                "loss_other": float(rows[~g].mean()) if n_o else float("nan"),
                "n_gathering": n_g, "n_other": n_o,
                "share_gathering": float(n_g / max(len(g), 1))}

    lg, lo = float(rows[g].mean()), float(rows[~g].mean())
    return {"loss": spec.gamma * lg + (1.0 - spec.gamma) * lo,
            "mode": "group_mean", "gamma": float(spec.gamma),
            "loss_gathering": lg, "loss_other": lo,
            "n_gathering": n_g, "n_other": n_o,
            "share_gathering": float(n_g / max(len(g), 1))}


def fit_weights(is_gathering: np.ndarray, spec: GroupSpec) -> np.ndarray | None:
    """Sample weights for the *fit*, or None when weighting is off.

    `fit_weight: 0` trains on every row equally (the default).  Above zero the
    two groups are balanced by count, so the number is interpretable as "this much
    of the gradient mass goes to gathering points" rather than as an arbitrary
    scale factor that depends on how many cells there are.
    """
    if spec.fit_weight <= 0 or not spec.members:
        return None
    g = np.asarray(is_gathering, dtype=bool)
    n_g, n_o = int(g.sum()), int((~g).sum())
    if n_g == 0 or n_o == 0:
        return None
    fw = spec.fit_weight
    w = np.where(g, fw / n_g, (1.0 - fw) / n_o)
    scale = len(g) / w.sum()
    return w * scale
