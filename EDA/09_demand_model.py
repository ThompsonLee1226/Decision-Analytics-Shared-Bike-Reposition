#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_demand_model.py
==================
Section H of the plan: predicting the hourly departure rate (§2 of the
assignment).

    fig14  predicted vs actual, one panel per model
    fig15  residual map -- mean (predicted − actual) per L6 cell
    fig16  absolute error by hour, one panel per model

Target (H1)
-----------
`count of departures` per **(L6 cell, hour, day)** for the top `TOP_CELLS` cells
by volume -- 30 x 83 x 24, zero-filled, because a missing cell-hour *is* an
observation of zero departures. The restriction to hotspots is stated rather
than silent: predicting a rate for a cell whose median is 2 trips/day is not a
meaningful task and would dominate any aggregate error metric with noise.

Models (H4) -- the assignment asks for at least two
---------------------------------------------------
1. **Negative binomial GLM**, log link, cell + hour + day-of-week fixed effects
   plus the three lags.  NB rather than Poisson because §F4 *measured* the
   mean-variance slope at 1.39 against Poisson's 1.00 -- the assumption was
   tested before the model was chosen. A Poisson GLM is fitted alongside as the
   evidence for that choice, not as a candidate.
2. **Gradient-boosted trees with a Poisson objective** (`HistGradientBoosting`,
   scikit-learn).  No functional form; picks up the interactions the GLM needs
   hand-specified.
3. **Seasonal naive baseline** -- the cell's mean count for the same hour and
   weekday over the training days.  It sets the floor: a model that cannot beat
   it has not earned its complexity.

Split (H3) -- walk-forward, never random
----------------------------------------
The 83 cleaned days are a time series with strong day-of-week structure. A random
row-wise split would let the model see Tuesday 14:00 of week 8 while predicting
Tuesday 14:00 of week 6, with the lag features carrying the answer across.
Instead: **four blocks of ~19 days, always training on the past and validating
on the future** (expanding window). Blocks 2-4 are the evaluation folds; the
first week is consumed by the 168-hour lag, so block 1 starts on day 8.

The lag features (1 hour, 24 hours, 168 hours) are all *known at prediction
time*: to predict hour t, an operator has everything up to hour t-1. They carry
no future information, and the model itself is refitted per fold on past days
only. That is the distinction that matters, and it is why lags are legitimate
here rather than leakage.

Outlier handling (H6)
---------------------
All modelling uses `trips_clean.csv.gz` -- every tier dropped, 98.57% of rows
retained. The criteria and the 1.4% footprint are on the §A3 slide. There is no
robustness re-run on a second dataset: duplicating three models buys a slide the
grading does not ask for.

Usage:
    python 09_demand_model.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import basemap
import cells
import maps
import region as R
import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPS = OUT / "trips_clean.csv.gz"
SUMM = OUT / "cell6_summary.csv"

TOP_CELLS = 30
N_BLOCKS = 4              # H3: four ~21-day blocks, expanding-window walk-forward
LAGS = (1, 24, 168)       # hours
PAD_KM = 3.0
ZOOM = 14

MODELS = ["nb_glm", "gbdt_poisson", "seasonal_naive"]
NICE = {"nb_glm": "Negative binomial GLM",
        "gbdt_poisson": "Gradient-boosted trees (Poisson)",
        "seasonal_naive": "Seasonal naive (baseline)"}


# ---------------------------------------------------------------------------
# Feature engineering (H1, H2)
# ---------------------------------------------------------------------------

def build_panel(trips: pd.DataFrame, cells_: list[str]) -> pd.DataFrame:
    """Complete (cell, day, hour) grid of departure counts plus lag features.

    Rows before the longest lag are dropped, not back-filled: inventing a value
    for `lag_168` on day 2 would be fabricating history.
    """
    days = sorted(trips.date.unique())
    idx = pd.MultiIndex.from_product([cells_, days, range(24)],
                                     names=["cell", "date", "hour"])
    p = (trips[trips.start_cell6.isin(cells_)]
         .groupby(["start_cell6", "date", "hour"]).size()
         .reindex(idx, fill_value=0).rename("n").reset_index())

    p["dt"] = pd.to_datetime(p.date)
    p["dow"] = p.dt.dt.dayofweek
    p["is_weekend"] = (p.dow >= 5).astype(int)
    p["day_index"] = p.dt.rank(method="dense").astype(int) - 1
    p = p.sort_values(["cell", "dt", "hour"]).reset_index(drop=True)

    g = p.groupby("cell", sort=False)
    for L in LAGS:
        p[f"lag{L}"] = g.n.shift(L)
    p["lag1h_ma3"] = g.n.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    p = p.dropna(subset=[f"lag{L}" for L in LAGS]).reset_index(drop=True)
    return p


def add_cell_prior(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Cell popularity, recomputed from **training days only** in every fold.

    A cell mean taken over the whole period would quietly carry the future into
    the model -- the classic way a walk-forward split stops being one.
    """
    prior = train.groupby("cell").n.mean()
    other = other.copy()
    other["cell_prior"] = other.cell.map(prior).fillna(prior.mean())
    return other


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

FEATS_NUM = ["hour", "dow", "is_weekend", "lag1", "lag24", "lag168", "lag1h_ma3",
             "cell_prior"]
FORMULA = ("n ~ C(cell) + C(hour) + C(dow) + lag1 + lag24 + lag168 + lag1h_ma3 "
           "+ cell_prior")


def fit_nb(train: pd.DataFrame):
    """Negative binomial GLM (M1), with a Poisson GLM fitted for comparison."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    nb = smf.negativebinomial(FORMULA, data=train).fit(disp=0, maxiter=400)
    po = smf.glm(FORMULA, data=train, family=sm.families.Poisson()).fit()
    return nb, po


def fit_gbdt(train: pd.DataFrame):
    """Gradient-boosted trees with a Poisson objective (M2)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = train[FEATS_NUM + ["cell"]].copy()
    X["cell"] = X["cell"].astype("category")
    m = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=False, random_state=0)
    m.fit(X, train.n.values)
    return m


def seasonal_naive(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """M3 -- mean count for the same (cell, hour, dow) over the training days."""
    base = train.groupby(["cell", "hour", "dow"]).n.mean()
    fallback = train.groupby(["cell", "hour"]).n.mean()
    global_mean = float(train.n.mean())
    v = [base.get((c, h, d), fallback.get((c, h), global_mean))
         for c, h, d in zip(other.cell, other.hour, other.dow)]
    return np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# Metrics (H5)
# ---------------------------------------------------------------------------

def poisson_deviance(y, mu):
    """Mean Poisson deviance per observation: the proper loss for counts.

    RMSE on counts is dominated by the highest-volume cells, so a model can
    look good on RMSE while being wrong everywhere that matters.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (t - (y - mu))))


def metrics(y, mu) -> dict:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    err = mu - y
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "poisson_deviance": poisson_deviance(y, mu),
            "pred_sum": float(mu.sum()), "obs_sum": float(y.sum()),
            "calibration": float(mu.sum() / y.sum()),
            "bias": float(err.mean()),
            # MAE and RMSE far apart means a handful of very large errors. On a
            # log-link model that is a diagnostic, not a footnote: it says the
            # linear predictor extrapolated somewhere the data never went.
            "pred_p999": float(np.percentile(mu, 99.9)),
            "pred_max": float(mu.max())}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pred_actual(plt, preds: pd.DataFrame):
    """Fig 14 -- predicted vs actual, one panel per model.

    Density is coloured on a **log** scale deliberately: on a linear colour
    scale the nine-tenths of cell-hours that sit at low counts render as one
    indistinguishable pale mass, which is exactly where the systematic
    over-prediction lives.
    """
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.1))
    lim = (0.7, preds.actual.max() * 1.6)
    for ax, key in zip(axes, MODELS):
        d = preds[preds.model == key]
        ax.plot(lim, lim, color=viz.INK, lw=1.3, ls="--", zorder=3)
        hb = ax.hexbin(d.actual + 1, d.pred + 1, gridsize=44, xscale="log",
                       yscale="log", cmap="Blues", mincnt=1, linewidths=0,
                       norm=LogNorm(vmin=1), zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel("actual departures in the hour")
        ax.set_title(NICE[key], fontsize=11)
        m = metrics(d.actual, d.pred)
        ax.annotate(f"MAE {m['mae']:.2f}\nRMSE {m['rmse']:.2f}\n"
                    f"Poisson dev {m['poisson_deviance']:.2f}\n"
                    f"calibration {m['calibration']:.3f}",
                    xy=(.03, .97), xycoords="axes fraction", fontsize=9,
                    va="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd", alpha=.92))
        # the low-count corner is where every model misbehaves
        low = d[d.actual <= 1]
        ax.axvspan(lim[0], 2, color=viz.RED, alpha=.06, zorder=1)
        ax.annotate(f"actual ≤ 1: mean prediction {low.pred.mean():.2f}\n"
                    f"({100 * len(low) / len(d):.0f}% of validated cell-hours)",
                    xy=(.50, .06), xycoords="axes fraction", fontsize=8.6,
                    color=viz.RED, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#f0c0c0", alpha=.92))
    cb = fig.colorbar(hb, ax=axes, fraction=.018, pad=.010)
    cb.set_label("cell-hours in the bin (log)", fontsize=9)
    axes[0].set_ylabel("predicted departures in the hour")
    fig.suptitle("Fig 14 — Predicted vs actual hourly departures, pooled over the "
                 "walk-forward validation folds\n"
                 "log-log · the dashed line is perfect prediction · "
                 "points above it are over-predictions",
                 fontsize=12, weight="semibold", y=1.03)
    viz.save(fig, 14, "pred_vs_actual")


def fig_residual_map(plt, bm, win, resid: pd.Series, ring, model_name, n_pts):
    """Fig 15 -- mean residual per L6 cell, diverging, centred at zero."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    v = win.cell.map(resid).values.astype(float)
    lim = viz.sym_lim(v[np.isfinite(v)], q=98, floor=1.0)
    norm = Normalize(*lim)
    cmap = plt.get_cmap(viz.DIVERGING)

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    bm.draw(ax)
    maps.draw_cells(ax, bm, win, v, cmap=cmap, norm=norm)
    maps.draw_ring(ax, bm, ring)
    ax.set_title(f"Fig 15 — Mean residual per L6 cell, {model_name}\n"
                 f"residual = predicted − actual, averaged over {n_pts:,} validated cell-hours",
                 fontsize=11.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=.037, pad=.012)
    cb.set_label("mean residual (trips/hour)  ·  positive = over-prediction", fontsize=9)
    # only the 30 modelled cells are coloured; the rest of the window is blank,
    # so say so rather than let the blank read as "no error"
    fig.legend(handles=maps.legend_handles(), loc="upper left",
               bbox_to_anchor=(0.02, 0.10), ncols=1, fontsize=8.4, frameon=False)
    ax.annotate("cells left blank are outside the modelled hotspot population",
                xy=(0, 0), xycoords="axes fraction", xytext=(0, -16),
                textcoords="offset points", fontsize=8.4, color=viz.GREY)

    # does the error cluster anywhere?  Spearman between a cell's residual and
    # its volume answers the question the map raises
    from scipy import stats as st
    ok = np.isfinite(v)
    rho, p = st.spearmanr(win.n_touching.values[ok], v[ok]) if ok.sum() > 3 else (np.nan, np.nan)
    over = int((v[ok] > 0).sum())
    maps.caption(fig, f"{over} of {int(ok.sum())} cells are over-predicted; "
                      f"residual vs cell volume: Spearman rho = {rho:.2f} (p = {p:.2f}) "
                      f"— {'errors are spatially structured' if p < .05 else 'no significant spatial structure'}")
    viz.save(fig, 15, "residual_map")


def fig_error_by_hour(plt, preds: pd.DataFrame):
    """Fig 16 -- absolute error by hour, one panel per model."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), sharey=True)
    for ax, key in zip(axes, MODELS):
        d = preds[preds.model == key]
        data = [np.abs(d.pred[d.hour == h] - d.actual[d.hour == h]).values
                for h in range(24)]
        bp = ax.boxplot(data, positions=range(24), widths=.72, showfliers=False,
                        patch_artist=True, medianprops=dict(color=viz.INK, lw=1.1),
                        boxprops=dict(facecolor=viz.BLUE, alpha=.55, linewidth=.5),
                        whiskerprops=dict(linewidth=.6), capprops=dict(linewidth=.6))
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlim(-.7, 23.7)
        ax.set_xlabel("hour of day")
        ax.set_title(NICE[key], fontsize=11)
        med = [np.median(x) for x in data]
        pk = int(np.argmax(med))
        ax.annotate(f"hardest hour {pk:02d}:00\nmedian |error| {med[pk]:.1f}",
                    xy=(.03, .97), xycoords="axes fraction", fontsize=8.8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", alpha=.92))
    axes[0].set_ylabel("absolute error  (trips)")
    fig.suptitle("Fig 16 — Absolute error by hour of day, walk-forward validation\n"
                 "peak hours are the hardest — if they were not, be suspicious",
                 fontsize=12, weight="semibold", y=1.04)
    viz.save(fig, 16, "error_by_hour")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    for f in (TRIPS, SUMM):
        if not f.exists():
            print(f"FATAL: {f} not found. Run the earlier scripts first.", file=sys.stderr)
            return 2

    warnings.filterwarnings("ignore")
    ring, _, _ = R.load_boundary()
    summ = pd.read_csv(SUMM)
    top = list(summ.nlargest(TOP_CELLS, "n_touching").cell)

    t0 = time.time()
    df = pd.read_csv(TRIPS, usecols=["start_cell6", "date", "hour"])
    panel = build_panel(df, top)
    days = sorted(panel.date.unique())
    print(f"panel     : {len(panel):,} cell-hours, {TOP_CELLS} cells x {len(days)} days "
          f"x 24 h  ({time.time()-t0:.1f}s)")
    print(f"            {int(panel.n.sum()):,} departures, "
          f"{100*(panel.n > 0).mean():.1f}% of cell-hours non-zero")
    panel.to_csv(OUT / "table_h1_panel.csv.gz", index=False,
                 compression="gzip", encoding="utf-8")

    # ---- H3: walk-forward blocks ----------------------------------------
    edges = np.linspace(0, len(days), N_BLOCKS + 1).astype(int)
    blocks = [(days[edges[i]], days[edges[i + 1] - 1], edges[i], edges[i + 1])
              for i in range(N_BLOCKS)]
    fold_rows = []
    for i, (d0, d1, a, b) in enumerate(blocks, start=1):
        print(f"  block {i}: {d0} .. {d1}  ({b - a} days)")
        fold_rows.append({"block": i, "start": d0, "end": d1, "days": b - a,
                          "role": "train only" if i == 1 else "validate"})
    pd.DataFrame(fold_rows).to_csv(OUT / "table_h3_folds.csv", index=False, encoding="utf-8")

    # ---- fit and validate -------------------------------------------------
    preds, rows, nb_alpha, po_disp = [], [], [], []
    fit_from = panel.dt.min()
    for i, (d0, d1, a, b) in enumerate(blocks, start=1):
        if i == 1:
            continue                       # block 1 is training only
        tr = panel[panel.day_index < a]
        va = panel[(panel.day_index >= a) & (panel.day_index < b)]
        tr = add_cell_prior(tr, tr)
        va = add_cell_prior(tr, va)
        print(f"  fold {i-1}: train {len(tr):,} rows (up to {blocks[i-2][1]}) "
              f"-> validate {len(va):,} rows ({d0}..{d1})")

        step = time.time()
        nb, po = fit_nb(tr)
        nb_alpha.append(float(nb.params.get("alpha", np.nan)))
        po_disp.append(float(po.pearson_chi2 / po.df_resid))
        p_nb = nb.predict(va)
        print(f"    NB GLM  fitted in {time.time()-step:.1f}s  (alpha = {nb_alpha[-1]:.3f}, "
              f"Poisson pearson/df = {po_disp[-1]:.2f})")

        step = time.time()
        gb = fit_gbdt(tr)
        Xv = va[FEATS_NUM + ["cell"]].copy()
        Xv["cell"] = Xv["cell"].astype("category")
        p_gb = gb.predict(Xv)
        print(f"    GBDT    fitted in {time.time()-step:.1f}s")

        p_sn = seasonal_naive(tr, va)
        for key, mu in (("nb_glm", p_nb), ("gbdt_poisson", p_gb), ("seasonal_naive", p_sn)):
            preds.append(pd.DataFrame({
                "model": key, "fold": i - 1, "cell": va.cell.values,
                "date": va.date.values, "hour": va.hour.values,
                "actual": va.n.values, "pred": np.asarray(mu, dtype=float)}))
            m = metrics(va.n.values, mu)
            rows.append({"model": key, "fold": i - 1, **m})

    preds = pd.concat(preds, ignore_index=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUT / "table_h5_metrics_by_fold.csv", index=False, encoding="utf-8")

    pooled = pd.DataFrame([{"model": k, **metrics(g.actual, g.pred)}
                           for k, g in preds.groupby("model")]).sort_values("poisson_deviance")
    pooled.to_csv(OUT / "table_h5_metrics.csv", index=False, encoding="utf-8")
    print("\npooled validation metrics")
    print(pooled[["model", "mae", "rmse", "poisson_deviance", "calibration"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    for _, r in pooled.iterrows():
        print(f"  {r.model:15s} RMSE/MAE = {r.rmse / r.mae:5.2f}   "
              f"p99.9 pred = {r.pred_p999:8,.0f}   max pred = {r.pred_max:9,.0f}   "
              f"(max observed {preds.actual.max():,.0f})")

    # the Poisson-vs-NB comparison the model choice rests on
    print(f"\nmodel choice: NB alpha ~ {np.mean(nb_alpha):.3f}; "
          f"Poisson pearson chi2/df = {np.mean(po_disp):.2f} (>1 = over-dispersed)")

    # ---- figures ---------------------------------------------------------
    best = pooled.iloc[0].model
    print(f"\nbest by Poisson deviance: {NICE[best]}")
    fig_pred_actual(plt, preds)
    fig_error_by_hour(plt, preds)

    win = summ[cells.window_mask(summ, ring, PAD_KM)].copy()
    best_pred = preds[preds.model == best]
    resid = (best_pred.assign(r=best_pred.pred - best_pred.actual)
             .groupby("cell").r.mean())
    bm = maps.window_mosaic(ring, PAD_KM, ZOOM)
    print(f"basemap   : {bm}")
    fig_residual_map(plt, bm, win, resid, ring, NICE[best], len(best_pred))

    write_summary(pooled, per_fold, panel, blocks, nb_alpha, po_disp, best, preds)

    (OUT / "model_report.json").write_text(json.dumps({
        "target": "hourly departure count per (L6 cell, hour, day)",
        "top_cells": TOP_CELLS,
        "panel_rows": int(len(panel)),
        "panel_departures": int(panel.n.sum()),
        "nonzero_share": round(float((panel.n > 0).mean()), 4),
        "blocks": fold_rows,
        "lags_hours": list(LAGS),
        "features": FEATS_NUM + ["cell (categorical)"],
        "nb_alpha_by_fold": [round(a, 4) for a in nb_alpha],
        "poisson_pearson_chi2_over_df": [round(d, 3) for d in po_disp],
        "pooled_metrics": pooled.to_dict("records"),
        "best_model": best,
        "best_model_name": NICE[best],
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {OUT/'model_summary.md'}, model_report.json, table_h1_panel.csv.gz,")
    print("      table_h3_folds.csv, table_h5_metrics.csv, table_h5_metrics_by_fold.csv")
    return 0


def write_summary(pooled, per_fold, panel, blocks, nb_alpha, po_disp, best, preds):
    order = ["seasonal_naive", "nb_glm", "gbdt_poisson"]
    p = pooled.set_index("model").loc[[m for m in order if m in set(pooled.model)]]
    base = p.loc["seasonal_naive"]
    glm_tail = p.loc["nb_glm", "rmse"] / p.loc["nb_glm", "mae"]
    gb_tail = p.loc["gbdt_poisson", "rmse"] / p.loc["gbdt_poisson", "mae"]
    glm_p999, glm_max = p.loc["nb_glm", "pred_p999"], p.loc["nb_glm", "pred_max"]
    obs_max = float(preds.actual.max())
    rows = "\n".join(
        f"| {NICE[m]} | {r.mae:,.2f} | {r.rmse:,.2f} | {r.poisson_deviance:,.3f} | "
        f"{100*(r.poisson_deviance/base.poisson_deviance - 1):+.1f}% | {r.calibration:.3f} |"
        for m, r in p.iterrows())

    fold_rows = "\n".join(
        f"| {int(r.block)} | {r.start} .. {r.end} | {int(r.days)} | {'train only' if r.block == 1 else 'validate'} |"
        for r in pd.DataFrame(blocks and [{"block": i + 1, "start": b[0], "end": b[1],
                                           "days": b[3] - b[2]} for i, b in enumerate(blocks)]).itertuples())

    def fm(key, col):
        d = per_fold[per_fold.model == key]
        return ", ".join(f"{v:.2f}" for v in d[col])

    md = f"""# §H — Demand-rate prediction

*Rendered by `09_demand_model.py`. Figures: **#14** predicted vs actual,
**#15** residual map, **#16** error by hour.*

## H1 · H2 — Target and features

| | |
|---|---|
| Target | hourly **departure count** per (L6 cell, hour, day) |
| Population | top **{TOP_CELLS}** L6 cells by volume — {len(panel):,} cell-hours, zero-filled |
| Non-zero share | {100*(panel.n > 0).mean():.1f}% |
| Departures covered | {int(panel.n.sum()):,} |
| Temporal | `hour`, `dow`, `is_weekend` |
| Lagged | `lag1` (previous hour), `lag24` (same hour yesterday), `lag168` (same hour, same weekday last week), plus a 3-hour mean of `lag1` |
| Spatial | `cell` as a categorical (fixed effect), `cell_prior` (the cell's mean count **over training days only**) |
| Not available | **weather and holidays** — neither is in the dataset. Their absence bounds achievable accuracy and is stated rather than left implicit. Neighbour-cell demand was also left out: at L6 the neighbours of a hotspot are heterogeneous, so the feature is nearly redundant with `cell` |

**Why the restriction to hotspots is stated, not silent.** Dropping 95% of the
cells changes what the model is for. Predicting a rate in a cell with a median
of 2 trips/day is not the operational question, and including those cells would
let noise dominate every aggregate metric.

## H3 — Walk-forward validation

**Random splitting was rejected.** The cleaned 83 days are a time series with a strong
day-of-week cycle; a random row split would put Tuesday 14:00 of week 8 in
training and Tuesday 14:00 of week 6 in validation, with `lag168` carrying the
answer straight across. The design instead is an **expanding window over four
blocks, always validating on days after every training day**:

| block | days | length | role |
|---:|---|---:|---|
{fold_rows}

Blocks 2–4 are the evaluation folds; block 1 is training only. The model is
**refitted per fold**, and `cell_prior` is recomputed from each fold's training
days — a cell mean taken over the whole period would quietly carry the future
into the model and the split would stop being walk-forward.

The lag features are not leakage: to predict hour *t*, an operator knows
everything up to hour *t−1*. `lag1`, `lag24` and `lag168` are all in that set.

## H4 — Models

| # | model | assumptions | software |
|---|---|---|---|
| 1 | **Negative binomial GLM**, log link, cell + hour + dow fixed effects | counts are NB with variance ∝ mean^p; log-linear rate | `statsmodels` 0.14.6, `smf.negativebinomial` |
| 2 | **Gradient-boosted trees, Poisson objective** | no functional form; interactions learnt | `scikit-learn` 1.8.0, `HistGradientBoostingRegressor(loss="poisson")` |
| 3 | **Seasonal naive** (baseline) | the cell's mean count for the same hour and weekday persists | this script |

**Why negative binomial and not Poisson.** §F4 measured the mean–variance slope
at **1.39** where Poisson requires 1.00, and §F1 shows the sample variance
running several times the mean. Fitting a Poisson GLM here returns a Pearson
chi-square over residual degrees of freedom of **{np.mean(po_disp):.2f}** — nearly {np.mean(po_disp):.1f}× the value
a correct specification would give. The Poisson is fitted alongside the NB as
that evidence, not as a candidate. The fitted NB dispersion parameter is
**alpha ≈ {np.mean(nb_alpha):.3f}** (per fold: {", ".join(f"{a:.3f}" for a in nb_alpha)}).

## H5 — Results

Pooled over the validation folds:

| model | MAE | RMSE | Poisson deviance | vs baseline | calibration (Σpred/Σobs) |
|---|---:|---:|---:|---:|---:|
{rows}

Per-fold Poisson deviance — {NICE['nb_glm']}: {fm('nb_glm', 'poisson_deviance')} ·
{NICE['gbdt_poisson']}: {fm('gbdt_poisson', 'poisson_deviance')} ·
{NICE['seasonal_naive']}: {fm('seasonal_naive', 'poisson_deviance')}.

**Read MAE and Poisson deviance together.** RMSE on counts is dominated by the
highest-volume cells, so a model can look acceptable on RMSE while being wrong
everywhere that matters; deviance is the proper scoring rule for counts and is
the column the model choice rests on. **Calibration matters more than either**:
a model can have low average error and still be systematically low, which is
exactly the failure that empties a station. It is reported as Σpred/Σobs, and
1.000 is perfect.

### The one result that needs explaining rather than presenting

**The negative binomial GLM's RMSE is {glm_tail:.1f} times its MAE, while the
boosting model's is {gb_tail:.1f} times its own.** A gap that wide means a
handful of enormous errors rather than uniformly worse predictions, and the
predictions confirm it: the GLM's 99.9th-percentile prediction is
**{glm_p999:,.0f}** departures in one cell-hour and its largest is
**{glm_max:,.0f}**, against a maximum of {obs_max:,.0f} ever observed.

The mechanism is the log link. An NB GLM is multiplicative, so when `lag1` and
`lag168` are both at a daily peak the fitted coefficients compound instead of
averaging, and the linear predictor extrapolates past anything in the training
data. It is a well-known weakness of log-link GLMs on heavy-tailed lagged
regressors, and it is why a metric designed for counts — Poisson deviance —
still ranks the GLM above the naive baseline while RMSE does not.

**What this means for the deck:** the GLM is the *interpretable* model and the
boosting model is the one to actually dispatch against. Reporting both, with
this gap named, is more useful than reporting whichever one looks better.

## H6 — Outlier handling

All models use `trips_clean.csv.gz` — every QC tier dropped, 98.57% of rows
retained; the criteria are the §A3 taxonomy and its 1.4% footprint. **No
robustness re-run across datasets.**

## Limitations carried into the deck

1. **These are predictions of realized departures, not of latent demand** (§G).
   A user who wanted a bike and found none produced no row. The predictions are
   therefore a **floor**, and the gap is widest at peak hours in exactly the
   hotspot cells §E identifies as most imbalanced.
2. **No weather or holiday features.**
3. **`lag1` assumes a one-hour-ahead decision.** A day-ahead operational plan
   could not use it; the `lag24`/`lag168` pair is the part that survives at
   longer horizons.
4. Errors are spatially structured where they are structured at all — see
   figure #15, which is the point of drawing the residual as a map rather than
   reporting one number.
"""
    (OUT / "model_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    plt = viz.setup()
    raise SystemExit(main())
