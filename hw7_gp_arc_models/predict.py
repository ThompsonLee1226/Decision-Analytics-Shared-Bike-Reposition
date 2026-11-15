#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/predict_two_models.py
=============================
Load `{gp,arc}/model.json` from this folder without the training code and
predict from them.

    python scripts/predict_two_models.py --verify          # load both, sanity-check
    python scripts/predict_two_models.py --panel arc --verify

The saved payload holds everything a prediction needs: the booster (XGBoost
JSON), the offset table per (unit, half-hour, channel), the trend index fit,
and the hourly rain flags.  This script rebuilds those objects and then calls
the booster exactly the way training did -- `log mu = log offset + f(x)`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
RUN = ROOT
PANELS = {"gp": ("gp", "departures"), "arc": ("arc", "departures")}


def _booster(raw: dict) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(raw).encode("utf-8")))
    return booster


def load(panel: str, run: Path = RUN) -> dict:
    payload = json.loads((run / panel / "model.json").read_text(encoding="utf-8"))
    model = {"payload": payload, "per_channel": payload.get("per_channel", False)}
    if model["per_channel"]:
        model["channels"] = {}
        for name, channel in payload["channels"].items():
            offset = channel["offset"]
            table = pd.Series(
                offset["value"],
                index=pd.MultiIndex.from_arrays([offset["unit"], offset["slot"]],
                                                names=["unit", "slot"]))
            model["channels"][name] = {"booster": _booster(channel["booster"]),
                                       "offset": table}
    else:
        offset = payload["offset"]
        model["booster"] = _booster(payload["booster"])
        model["offset"] = pd.Series(
            offset["value"],
            index=pd.MultiIndex.from_arrays(
                [offset["unit"], offset["slot"], offset["channel"]],
                names=["unit", "slot", "channel"]))
    return model


def volume_index(state: dict):
    """Rebuild the fitted log-linear index from its four numbers."""
    w0 = pd.Timestamp(state["w0"])

    def transform(dates: pd.Series) -> np.ndarray:
        t = (pd.to_datetime(dates) - w0).dt.days.to_numpy(float)
        out = np.exp(state["intercept"] + state["slope"] * t - state["ref"])
        lo, hi = state["clip"]
        return np.clip(out, lo, hi)

    return transform


def rain_flag(state: dict) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(k): v for k, v in state["hourly"].items()}


def design(model: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    payload = model["payload"]
    features = payload["features"]
    channel = frame["channel"].to_numpy(float) if "channel" in frame else np.where(
        frame["channel_name"].to_numpy() == "departures", 0.0, 1.0)
    slot = frame["slot"].to_numpy(float)
    dates = pd.to_datetime(frame["date"])

    X = pd.DataFrame(index=frame.index)
    X["channel"] = channel
    X["slot"] = slot
    X["dow"] = pd.to_datetime(dates).dt.dayofweek.to_numpy(float)
    # Rain: the hour the half-hour starts in, 0 when the file has no row.
    start = dates + pd.to_timedelta(390 + slot * 30, unit="m")
    hourly = rain_flag(payload["rain"])
    X["is_raining"] = [float(hourly.get(pd.Timestamp(t).floor("h"), 0)) for t in start]
    if "log_distance_km" in features:
        distance = (frame["distance_m"].to_numpy(float) if "distance_m" in frame
                    else frame["unit"].map(lambda u: payload["units"][u]["distance_m"]
                                           ).to_numpy(float))
        X["log_distance_km"] = np.log(np.maximum(distance, 1.0) / 1000.0)
    index = np.full(len(frame), np.nan)
    for value, name in ((0.0, "departures"), (1.0, "arrivals")):
        mask = channel == value
        index[mask] = volume_index(payload["volume_index"][name])(dates[mask])
    X["volume_index"] = index
    X = X[list(features)]

    # The stored offset is the raw training mean; the trend factor lives in
    # payload["trend_caps"] and has to be re-applied here, or the loaded model
    # would quietly predict without the trend it was trained with.
    caps = payload.get("trend_caps") or {}
    factor = np.ones(len(frame))
    for value, name in ((0.0, "departures"), (1.0, "arrivals")):
        mask = channel == value
        cap = caps.get(name)
        if cap is not None and mask.any():
            factor[mask] = np.clip(index[mask], 1.0, float(cap))

    if model["per_channel"]:
        return X, None, factor
    keys = pd.MultiIndex.from_arrays([frame["unit"].to_numpy(), slot, channel])
    offset = np.maximum(model["offset"].reindex(keys).to_numpy(float), 1e-3)
    return X, offset * factor, factor


def predict(model: dict, frame: pd.DataFrame) -> np.ndarray:
    X, offset, factor = design(model, frame)
    # Booster.predict has no base_margin argument; the offset belongs on the
    # DMatrix, which is the same call xgboost's own offset examples make.
    if model["per_channel"]:
        out = np.zeros(len(frame))
        names = (frame["channel_name"] if "channel_name" in frame else
                 np.where(frame["channel"] == 0.0, "departures", "arrivals"))
        for channel, part in model["channels"].items():
            mask = np.asarray(names == channel)
            if not mask.any():
                continue
            keys = pd.MultiIndex.from_arrays(
                [frame.loc[mask, "unit"].to_numpy(), frame.loc[mask, "slot"].to_numpy()])
            offset = np.maximum(part["offset"].reindex(keys).to_numpy(float), 1e-3)
            offset = offset * factor[mask]
            matrix = xgb.DMatrix(X[mask])
            matrix.set_base_margin(np.log(offset))
            out[mask] = np.clip(part["booster"].predict(matrix), 0.0, None)
        return out
    matrix = xgb.DMatrix(X)
    matrix.set_base_margin(np.log(offset))
    return np.clip(model["booster"].predict(matrix), 0.0, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="both", choices=["gp", "arc", "both"])
    ap.add_argument("--run", default=str(RUN), help="run directory holding gp/ and arc/")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    panels = ["gp", "arc"] if args.panel == "both" else [args.panel]
    for panel in panels:
        model = load(panel, run)
        rounds = (sum(c["booster"].num_boosted_rounds() for c in model["channels"].values())
                  if model["per_channel"]
                  else model["booster"].num_boosted_rounds())
        offsets = (sum(len(c["offset"]) for c in model["channels"].values())
                   if model["per_channel"] else len(model["offset"]))
        print(f"{panel}: {rounds} trees, features {model['payload']['features']}, "
              f"offsets {offsets}")
        if not args.verify:
            continue
        preds = pd.read_csv(run / panel / "predictions.csv.gz")
        sample = preds.drop(columns=["pred", "offset"]).drop_duplicates(
            ["unit", "date", "slot", "channel_name"]).head(500)
        mu = predict(model, sample)
        print(f"   loaded and scored {len(sample)} rows: min {mu.min():.4f}, "
              f"max {mu.max():.2f}, mean {mu.mean():.2f}, finite {np.isfinite(mu).all()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
