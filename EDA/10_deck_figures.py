#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_deck_figures.py
==================
The four figures the deck needs that the sixteen of §I1 do not cover.  Each one
exists because the assignment asks for it in so many words and nothing already
built answers it *in a picture*:

    fig17  outlier triage       §2  "report how you handle the outliers"
    fig18  demand quantiles     §1  "report sample variances and sample quantiles"
    fig19  walk-forward split   §2  "report how you divide your data into
                                    training and validation sets"
    fig20  model comparison     §2  "apply (at least) two different
                                    statistical/machine learning methods"

Numbering continues the §I1 inventory (1..16) so the deck's figure order is
unchanged and a filename still maps onto a slide without a lookup table.

**Nothing here recomputes a statistic of record.**  Every number is read from an
artifact an earlier script already wrote and the prose already quotes --
`cleaning_report.json`, `table_f1_variance_quantiles.csv`, `table_c1_daily.csv`,
`table_h3_folds.csv`, `table_h1_panel.csv.gz`, `model_XGBoost/out/report.json`
-- so a figure here cannot silently drift away from the table standing next to
it.  The single exception is fig17(b), which re-reads the raw CSV to get the
*before-cleaning* duration distribution, because that distribution is not
published anywhere else.

Usage:
    python 10_deck_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent

CLEANING = OUT / "cleaning_report.json"
DAILY = OUT / "table_c1_daily.csv"
FOLDS = OUT / "table_h3_folds.csv"
QUANT = OUT / "table_f1_variance_quantiles.csv"
XGB_REPORT = ROOT / "model_XGBoost" / "out" / "report.json"
RAW = ROOT / "Dataset" / "dataset_train_20260921_lsz.csv"

DAY0 = pd.Timestamp("2026-03-02")     # first clean day; day 0 on the fig19 axis


# ---------------------------------------------------------------------------
# fig17 -- outlier handling
# ---------------------------------------------------------------------------
def fig17(plt) -> None:
    """Three panels: what was flagged, what it cost, what it looked like."""
    rep = json.loads(CLEANING.read_text(encoding="utf-8"))
    flags = rep["flags"]
    dropped = set(rep["config"]["clean_drop_flags"])

    live = {k: v["n"] for k, v in flags.items() if v["n"] > 0}
    silent = sorted(k for k, v in flags.items() if v["n"] == 0)
    order = sorted(live, key=live.get)          # smallest at the bottom

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))

    # -- (a) the flag audit ---------------------------------------------------
    ax = axes[0]
    y = np.arange(len(order))
    ax.barh(y, [live[k] for k in order],
            color=[viz.RED if k in dropped else viz.GREY for k in order],
            height=0.62, zorder=2)
    for yi, k in zip(y, order):
        ax.annotate(f"{live[k]:,}", (live[k], yi), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, family="monospace", color=viz.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(order, family="monospace", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(8, 1.2e5)
    ax.set_xlabel("rows carrying the flag   (log scale)")
    ax.set_title("(a) Every QC flag, and what happened to it")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=viz.RED),
                       plt.Rectangle((0, 0), 1, 1, color=viz.GREY)],
              labels=["dropped (tier 1-2)", "kept, annotated only"],
              loc="lower right")
    ax.text(0.02, 0.03, f"{len(silent)} further flags fire on 0 rows:\n"
                        + ", ".join(silent),
            transform=ax.transAxes, fontsize=8, color=viz.GREY,
            family="monospace", va="bottom")

    # -- (b) duration before vs after ----------------------------------------
    ax = axes[1]
    raw = pd.read_csv(RAW, usecols=["start_time", "end_time"])
    before = (pd.to_datetime(raw.end_time)
              - pd.to_datetime(raw.start_time)).dt.total_seconds() / 60.0
    clean = pd.read_csv(OUT / "trips_clean.csv.gz", usecols=["duration_min"])
    after = clean.duration_min.astype(float)

    hi = 200.0
    bins = np.logspace(np.log10(0.5), np.log10(hi), 56)
    for series, colour, label, lw in ((before, viz.RED, "all rows (before)", 2.4),
                                      (after, viz.BLUE, "clean rows (after)", 1.6)):
        h, e = np.histogram(series.clip(0.5, hi), bins=bins, density=True)
        ax.step(e[:-1], h, where="post", color=colour, lw=lw, label=label,
                zorder=3)
        ax.fill_between(e[:-1], h, step="post", color=colour, alpha=0.10,
                        zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("trip duration, minutes   (log scale)")
    ax.set_ylabel("density")
    ax.set_title("(b) Cleaning truncates the tail,\n      it does not reshape the body")
    ax.legend(loc="upper right")
    ax.annotate(f"max {before.max():.0f} -> {after.max():.0f} min",
                xy=(hi, 0), xytext=(0.60, 0.42), textcoords="axes fraction",
                fontsize=9.5, family="monospace", color=viz.INK,
                arrowprops=dict(arrowstyle="->", color=viz.INK, lw=1.0))
    ax.text(0.03, 0.90, f"n {len(before):,} -> {len(after):,}\n"
                        f"median {before.median():.0f} -> {after.median():.0f} min",
            transform=ax.transAxes, fontsize=9, family="monospace",
            va="top", color=viz.GREY)

    # -- (c) the anomaly day --------------------------------------------------
    ax = axes[2]
    d = pd.read_csv(DAILY, parse_dates=["date"])
    ax.plot(d.date, d.n, color=viz.GREY, lw=1.0, marker="o", ms=2.6,
            label="daily departures", zorder=2)
    ax.plot(d.date, d.roll7, color=viz.BLUE, lw=2.0,
            label="7-day centred median", zorder=3)
    bad = d[d.removed.astype(bool)]
    ax.scatter(bad.date, bad.n, s=95, facecolor="none", edgecolor=viz.RED,
               lw=2.2, zorder=4, label="flagged anomaly day")
    for _, r in bad.iterrows():
        ax.annotate(f"{r.date:%b %d}\n{int(r.n):,} trips\n"
                    f"= {r.n / r.roll7:.0%} of baseline",
                    (r.date, r.n), xytext=(14, 26), textcoords="offset points",
                    fontsize=9, family="monospace", color=viz.RED,
                    arrowprops=dict(arrowstyle="->", color=viz.RED, lw=1.0))
    ax.set_ylabel("departures per day")
    ax.set_title("(c) The one anomalous day, kept and labelled")
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Outlier triage: 99.81% retained, every decision visible",
                 fontsize=13.5, fontweight="semibold", y=1.02)
    viz.save(fig, 17, "outlier_triage")


# ---------------------------------------------------------------------------
# fig18 -- sample quantiles
# ---------------------------------------------------------------------------
def fig18(plt) -> None:
    """Mean vs p50 vs p90, per cell, at each cell's own busiest hour."""
    q = pd.read_csv(QUANT).sort_values("total", ascending=True).reset_index(drop=True)
    y = np.arange(len(q))

    fig, ax = plt.subplots(figsize=(11.4, 6.2))

    # A rail from 0 anchors each row, so a low-volume cell reads on the same
    # scale as a hotspot instead of floating in whitespace.
    ax.hlines(y, 0, q.p90, color=viz.GRID, lw=0.9, zorder=1)
    ax.hlines(y, q.p50, q.p90, color=viz.BLUE, lw=3.2, alpha=0.30, zorder=2)
    ax.scatter(q.p50, y, s=34, facecolor="white", edgecolor=viz.BLUE,
               lw=1.8, zorder=5, label="p50  (median day)")
    ax.scatter(q.p90, y, s=34, color=viz.BLUE, zorder=5, label="p90  (1-in-10 day)")
    ax.scatter(q["mean"], y, s=78, marker="D", color=viz.RED,
               edgecolor="white", lw=1.2, zorder=6, label="mean")

    for yi, r in q.iterrows():
        gap = (r.p90 - r["mean"]) / r["mean"]
        ax.annotate(f"{r['mean']:.0f} -> {r.p90:.0f}   +{gap:.0%}",
                    (max(r.p90, r["mean"]), yi), xytext=(14, 0),
                    textcoords="offset points", va="center", fontsize=8.8,
                    family="monospace", color=viz.INK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  @{h:02d}:00" for c, h in zip(q.cell, q.hour)],
                       family="monospace", fontsize=9)
    ax.set_xlabel("departures in that cell-hour")
    ax.set_xlim(0, q.p90.max() * 1.36)
    ax.set_title("Planning on the mean under-prepares: it falls short of the "
                 "1-in-10 busy day\nin every one of the ten hotspots")
    ax.legend(loc="lower right", ncol=1)
    fig.text(0.012, -0.015,
             "Each row is a cell at its own busiest hour, over all 84 clean days. "
             "Red = +increase in a busy day over the mean.",
             fontsize=8.6, color=viz.GREY, family="monospace")
    viz.save(fig, 18, "demand_quantiles")


# ---------------------------------------------------------------------------
# fig19 -- walk-forward split
# ---------------------------------------------------------------------------
def fig19(plt) -> None:
    """The four blocks, and which of them each fold trains and validates on."""
    f = pd.read_csv(FOLDS)
    f["x0"] = (pd.to_datetime(f.start) - DAY0).dt.days
    f["x1"] = (pd.to_datetime(f.end) - DAY0).dt.days + 1     # half-open

    panel = pd.read_csv(OUT / "table_h1_panel.csv.gz", usecols=["date"])
    first_panel = (pd.to_datetime(panel.date).min() - DAY0).days
    last = int(f.x1.max())

    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    rows = {}
    y_full, y_blocks = 5.0, 4.0

    # -- top strip: the whole calendar, warm-up called out --------------------
    ax.barh(y_full, last, left=0, height=0.52, color=viz.GRID, zorder=2)
    ax.barh(y_full, first_panel, left=0, height=0.52, color=viz.GREY, alpha=0.45,
            hatch="///", edgecolor="white", zorder=3)
    ax.text(first_panel / 2, y_full, "warm-up\n7 d", ha="center", va="center",
            fontsize=8.6, family="monospace", color=viz.INK, zorder=4)
    for _, r in f.iterrows():
        ax.text((r.x0 + r.x1) / 2, y_full, f"block {r.block}",
                ha="center", va="center", fontsize=9.4, family="monospace",
                color=viz.INK, zorder=4)
    ax.text(last + 1.2, y_full, "84 clean days  (2026-03-02 .. 05-24)",
            va="center", fontsize=9, family="monospace", color=viz.GREY)

    # -- block roles ----------------------------------------------------------
    ax.barh(y_blocks, last - first_panel, left=first_panel, height=0.52,
            color=viz.GRID, zorder=2)
    ax.text((first_panel + last) / 2, y_blocks,
            "block 1 is used for training only -- never validated on",
            ha="center", va="center", fontsize=9.2, family="monospace",
            color=viz.INK, zorder=4)

    # -- one row per evaluation fold -----------------------------------------
    folds = list(f.itertuples())[1:]          # blocks 2, 3, 4
    for i, v in enumerate(reversed(folds)):
        yv = 2.0 - i
        train_start, train_end = first_panel, v.x0
        n_train = train_end - train_start
        ax.barh(yv, n_train, left=train_start, height=0.52, color=viz.GREY,
                zorder=3)
        ax.barh(yv, v.x1 - v.x0, left=v.x0, height=0.52, color=viz.BLUE, zorder=3)
        ax.text((train_start + train_end) / 2, yv,
                f"train  {n_train:>2} d", ha="center", va="center",
                fontsize=9, family="monospace", color="white", zorder=4)
        ax.text((v.x0 + v.x1) / 2, yv,
                f"validate {v.x1 - v.x0} d", ha="center", va="center",
                fontsize=9, family="monospace", color="white", zorder=4)
        rows[yv] = f"fold {i + 1}"

    ax.set_yticks([y_full, y_blocks] + list(rows))
    ax.set_yticklabels(["calendar", "block role"] + [rows[k] for k in sorted(rows)],
                       fontsize=9.6)
    ax.set_ylim(-0.7, 5.75)
    ax.set_xlim(-0.5, last + 26)

    ax.set_xticks([(pd.Timestamp(s) - DAY0).days for s in
                   ["2026-03-02", "2026-03-16", "2026-03-30", "2026-04-13",
                    "2026-04-27", "2026-05-11", "2026-05-24"]])
    ax.set_xticklabels(["Mar 02", "Mar 16", "Mar 30", "Apr 13", "Apr 27",
                        "May 11", "May 24"])
    ax.set_xlabel("day")
    ax.grid(axis="y", visible=False)
    ax.set_title("Walk-forward validation: every fold is validated strictly after "
                 "its training days")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=viz.GREY),
                       plt.Rectangle((0, 0), 1, 1, color=viz.BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=viz.GREY, alpha=0.45,
                                     hatch="///")],
              labels=["training days", "validation days",
                      "consumed by the lag-168 warm-up"],
              loc="lower right", ncol=1)
    fig.text(0.012, -0.02,
             "The model is refitted per fold and `cell_prior` is recomputed from "
             "that fold's training days only.\nA random row split would let "
             "`lag168` carry the answer across folds.",
             fontsize=8.8, color=viz.GREY, family="monospace")
    viz.save(fig, 19, "walk_forward_split")


# ---------------------------------------------------------------------------
# fig20 -- the models, side by side
# ---------------------------------------------------------------------------
def fig20(plt) -> None:
    """All four models on the four columns the choice actually rests on."""
    rep = json.loads(XGB_REPORT.read_text(encoding="utf-8"))
    rows = [{"model": "XGBoost (Poisson)", **rep["pooled_metrics"]}]
    rows += [{"model": m["model"], **m} for m in rep["reference_models"]]
    t = pd.DataFrame(rows).sort_values("poisson_deviance").reset_index(drop=True)
    t["rmse_over_mae"] = t.rmse / t.mae

    # The suptitle names two columns XGBoost is claimed to lead. A claim in a
    # title is a constant that can decouple from the data underneath it, so
    # check it here rather than trusting the sentence.
    xgb = int(t.index[t.model == "XGBoost (Poisson)"][0])
    if xgb != int(t.poisson_deviance.idxmin()):
        raise SystemExit("fig20 title claims XGBoost leads on deviance -- it does not")
    if xgb != int((t.calibration - 1.0).abs().idxmin()):
        raise SystemExit("fig20 title claims XGBoost leads on calibration -- it does not")

    NICE = {"XGBoost (Poisson)": "XGBoost\n(Poisson)",
            "gbdt_poisson": "Boosted trees\n(Poisson, sklearn)",
            "nb_glm": "Negative\nbinomial GLM",
            "seasonal_naive": "Seasonal naive\n(baseline)"}
    COLOUR = {"XGBoost (Poisson)": viz.BLUE,
              "gbdt_poisson": viz.GREEN,
              "nb_glm": viz.AMBER,
              "seasonal_naive": viz.GREY}

    obs_max = float(pd.read_csv(OUT / "table_h1_panel.csv.gz",
                                usecols=["n"]).n.max())
    xs = np.arange(len(t))
    labels = [NICE[m] for m in t.model]
    colours = [COLOUR[m] for m in t.model]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))

    def bars(ax, col, fmt, title, ylab, refline=None):
        ax.bar(xs, t[col], color=colours, width=0.62, zorder=2)
        for xi, v in zip(xs, t[col]):
            ax.annotate(fmt.format(v), (xi, v), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=9.4, family="monospace")
        if refline is not None:
            ax.axhline(refline[0], color=viz.RED, ls="--", lw=1.3, zorder=3)
            # Left-aligned just above the line: centred, this label runs
            # straight through the tallest bar in panels (c) and (d).
            ax.text(0.015, refline[0], refline[1],
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", fontsize=8.6,
                    family="monospace", color=viz.RED, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.margins(y=0.20)

    ax = axes[0, 0]
    bars(ax, "poisson_deviance", "{:.3f}",
         "(a) Poisson deviance — the ranking column\n      (lower is better)",
         "deviance")

    ax = axes[0, 1]
    bars(ax, "calibration", "{:.3f}",
         "(b) Calibration  Σpred / Σobs\n      (1.000 is perfect)",
         "predicted / observed", refline=(1.0, "1.000 = no systematic bias"))
    ax.set_ylim(min(0.90, t.calibration.min() - 0.02), 1.075)

    ax = axes[1, 0]
    bars(ax, "rmse_over_mae", "{:.2f}",
         "(c) RMSE / MAE — a wide ratio means\n      a few enormous errors",
         "RMSE / MAE")
    ax.axhline(1.0, color=viz.INK, ls=":", lw=1.0, zorder=1)
    ax.annotate("1.00 = every error the same size", xy=(0.5, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, -14),
                textcoords="offset points", fontsize=8.8, family="monospace",
                color=viz.INK, ha="center")

    ax = axes[1, 1]
    bars(ax, "pred_max", "{:.0f}",
         "(d) Largest single prediction\n      vs the largest count ever seen",
         "departures in one cell-hour",
         refline=(obs_max, f"max observed in the panel = {obs_max:.0f}"))

    fig.suptitle("Four models, identical folds — the boosted trees rank first on\n"
                 "deviance and calibration, the two columns the choice rests on",
                 fontsize=13.5, fontweight="semibold", y=1.0)
    fig.text(0.012, 0.005,
             "Pooled over the three walk-forward validation folds in "
             "`table_h3_folds.csv`. XGBoost is the model this project ships; "
             "the other three are scikit-learn / statsmodels references.",
             fontsize=8.6, color=viz.GREY, family="monospace")
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    viz.save(fig, 20, "model_comparison")


# ---------------------------------------------------------------------------
def main() -> None:
    plt = viz.setup()
    for fn in (fig17, fig18, fig19, fig20):
        print(f"  {fn.__name__}")
        fn(plt)


if __name__ == "__main__":
    main()
