#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_l8_netflow_concentration.py
==============================
How concentrated is weekday peak net flow?  This script takes the **top-10
pairwise-disjoint 3 x 3 blocks** written by `11_l8_peak_hotspots.py`
(block-ranked, `out/l8_block_netflow_top10_3x3.json`) and accumulates their
shares of the service-wide net flow.

For each panel (AM/PM x inflow/outflow):

    share_i       = |block net_i| / denominator
    cumulative_N  = sum of share_1..share_N

    denominator   = sum of positive cell net      (inflow)
                  = sum of |negative cell net|    (outflow)

over the in-region L8 cells, per weekday — the same denominators the hotspot
table reports, so the per-block shares are unchanged and only the running
total is new.

Two questions are answered:

    top10 share      do the ten selected blocks carry more than half the
                     service-wide net flow in that direction?
    min N above 50%  if so, how few blocks are needed to pass 50%?

Because the ten blocks are pairwise disjoint, their shares are additive: the
cumulative share is the share of the union of the first N blocks, with no
double counting.

Caveat that governs the reading of "top N": the sequence is the **first N of
the disjoint top-10 selection**, not the N largest blocks citywide.  The
non-overlap rule deliberately skips candidates that overlap an already chosen
block, so rank 2 is not necessarily the second-largest block.  The cumulative
curve is therefore a concentration curve for the reported selection.

Outputs:
    out/table_l8_netflow_concentration.csv
    out/l8_netflow_concentration.json
    out/l8_netflow_concentration_summary.md
    out/fig20_l8_netflow_concentration.png

Usage:
    python 12_l8_netflow_concentration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import l8
import maps
import viz

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SOURCE = OUT / "l8_block_netflow_top10_3x3.json"

DIRECTIONS = ("inflow", "outflow")
THRESHOLD = 50.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_source(path: Path = SOURCE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 11_l8_peak_hotspots.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def panel_table(doc: dict) -> tuple[pd.DataFrame, dict]:
    """Per-rank cumulative shares for the four panels, plus a panel summary.

    Returns the tidy long table (40 rows) and a summary keyed by panel.
    """
    service = doc["service_per_weekday"]
    rows, summary = [], {}

    for panel, blocks in doc["top10"].items():
        peak, direction = panel.split("_")
        denom = float(service[peak]["net_in" if direction == "inflow"
                                   else "net_out"])
        # The file stores signed block net; the outflow ranking uses magnitude.
        signed = np.array([float(b["block_net_per_day"]) for b in blocks])
        magnitude = np.abs(signed)
        share = 100.0 * magnitude / denom
        cumulative = np.cumsum(share)

        # The per-block shares must agree with the file's own column; if they
        # do not, the denominator used here is the wrong one.
        reported = np.array([float(b["share_net_pct"]) for b in blocks])
        if not np.allclose(share, reported, atol=1e-9):
            raise AssertionError(f"{panel}: recomputed shares disagree "
                                 f"with share_net_pct")

        above = np.flatnonzero(cumulative > THRESHOLD)
        n50 = int(above[0] + 1) if above.size else None

        summary[panel] = {
            "peak": peak,
            "direction": direction,
            "denominator_per_weekday": denom,
            "top10_net_per_weekday": float(magnitude.sum()),
            "top10_share_pct": float(cumulative[-1]),
            "top10_exceeds_50": bool(cumulative[-1] > THRESHOLD),
            "min_n_over_50": n50,
            "cumulative_at_min_n_pct": (float(cumulative[n50 - 1])
                                        if n50 else None),
            "cumulative_pct": [float(x) for x in cumulative],
        }

        for i, b in enumerate(blocks, 1):
            rows.append({
                "panel": panel,
                "peak": peak,
                "direction": direction,
                "n": i,
                "rank": int(b["rank"]),
                "cell": b["cell"],
                "denominator_per_weekday": denom,
                "block_net_per_day": float(signed[i - 1]),
                "abs_block_net_per_day": float(magnitude[i - 1]),
                "share_pct": float(share[i - 1]),
                "cumulative_pct": float(cumulative[i - 1]),
                "exceeds_50": bool(cumulative[i - 1] > THRESHOLD),
            })

    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_concentration(plt, table: pd.DataFrame, summary: dict):
    """Fig 20 -- cumulative share against the number of selected blocks.

    Colour encodes direction (the deck's blue = net inflow, red = net
    outflow); line style and marker fill encode the peak (AM solid/filled,
    PM dashed/hollow), so the four series never rely on hue alone.
    """
    fig, ax = plt.subplots(figsize=(11.8, 6.9))
    styles = {
        ("am", "inflow"):  dict(color=viz.BLUE, ls="-", mfc=viz.BLUE),
        ("pm", "inflow"):  dict(color=viz.BLUE, ls=(0, (5, 2.4)), mfc="white"),
        ("am", "outflow"): dict(color=viz.RED, ls="-", mfc=viz.RED),
        ("pm", "outflow"): dict(color=viz.RED, ls=(0, (5, 2.4)), mfc="white"),
    }
    labels = {
        ("am", "inflow"): "AM inflow",
        ("pm", "inflow"): "PM inflow",
        ("am", "outflow"): "AM outflow",
        ("pm", "outflow"): "PM outflow",
    }

    # 50% reference line first, so the data sit on top of it.
    ax.axhline(THRESHOLD, color=viz.GREY, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.70, THRESHOLD + 1.0, "50% threshold", color=viz.GREY,
            fontsize=9, ha="left", va="bottom")

    for (peak, direction), style in styles.items():
        panel = f"{peak}_{direction}"
        t = table.loc[table["panel"] == panel]
        x = t["n"].to_numpy()
        y = t["cumulative_pct"].to_numpy()
        ax.plot(x, y, lw=2.0, zorder=3, label=labels[(peak, direction)],
                marker="o", ms=5.4, mec=style["color"], mew=1.2,
                mfc=style["mfc"], **{k: v for k, v in style.items()
                                     if k not in ("mfc",)})

        # Direct label at the right edge: identity is never colour-alone.
        ax.text(10.28, y[-1], f"{labels[(peak, direction)]} {y[-1]:.1f}%",
                color=viz.INK, fontsize=9.4, va="center", ha="left")

        # Ring the first block that passes 50%; the two crossings are the
        # answer to the question and deserve a direct label.
        s = summary[panel]
        if s["min_n_over_50"]:
            n = s["min_n_over_50"]
            ax.scatter([n], [s["cumulative_at_min_n_pct"]], s=120,
                       facecolor="white", edgecolor=style["color"],
                       linewidth=1.9, zorder=6)
            dy = 3.2 if direction == "inflow" else -3.6
            va = "bottom" if direction == "inflow" else "top"
            ax.text(n + 0.16, s["cumulative_at_min_n_pct"] + dy, f"N={n}",
                    color=style["color"], fontsize=9.6, weight="bold",
                    va=va, ha="left")

    ax.set_xlim(0.55, 13.0)
    ax.set_ylim(0, 70)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Number of selected 3×3 blocks (pairwise disjoint, "
                  "in rank order)")
    ax.set_ylabel("Cumulative share of service net flow (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.02), ncols=2,
              handlelength=2.6, columnspacing=1.4)

    ax.set_title(
        "Fig 20 — Concentration of peak net flow in the top-N disjoint "
        "3×3 blocks\n"
        "cumulative share of service-wide net inflow (blue) / net outflow "
        "(red); only PM inflow (N=4) and AM outflow (N=7) pass 50%",
        fontsize=12.6, loc="left")

    maps.caption(
        fig,
        "Top-N is the first N of the pairwise-disjoint top-10 selection in "
        "11_l8_peak_hotspots.py, not the N largest blocks citywide.  "
        "Denominator = sum of positive cell net (inflow) or |negative cell "
        "net| (outflow) over in-region L8 cells, per weekday.  Disjoint "
        "blocks make the shares additive; the ten blocks still cover only "
        "part of the service area.",
        y=-0.01)
    viz.save(fig, 20, "l8_netflow_concentration")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(doc: dict, table: pd.DataFrame, summary: dict) -> None:
    table.round(4).to_csv(OUT / "table_l8_netflow_concentration.csv",
                          index=False, encoding="utf-8")

    payload = {
        "source": SOURCE.name,
        "weekdays": doc["weekdays"],
        "top_n": doc["top_n"],
        "rank_by": doc["rank_by"],
        "non_overlapping": doc["non_overlapping"],
        "threshold_pct": THRESHOLD,
        "denominator_note": (
            "sum of positive cell net (inflow) or |negative cell net| "
            "(outflow) over in-region L8 cells, per weekday"),
        "panels": summary,
    }
    (OUT / "l8_netflow_concentration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")

    def pct(v):
        return f"{v:.2f}%"

    panel_rows = "\n".join(
        f"| {l8.PEAK_LABELS[s['peak']]} · net {s['direction']} | "
        f"{s['denominator_per_weekday']:,.1f} | "
        f"{s['top10_net_per_weekday']:,.1f} | {pct(s['top10_share_pct'])} | "
        f"{'yes' if s['top10_exceeds_50'] else 'no'} | "
        f"{('N = ' + str(s['min_n_over_50'])) if s['min_n_over_50'] else '—'} | "
        f"{(pct(s['cumulative_at_min_n_pct']) if s['min_n_over_50'] else '—')} |"
        for p in ("am", "pm") for d in DIRECTIONS
        for s in [summary[f"{p}_{d}"]])

    # Wide cumulative table: one row per N, one column per panel.
    wide = table.pivot(index="n", columns="panel", values="cumulative_pct")
    order = ["am_inflow", "am_outflow", "pm_inflow", "pm_outflow"]
    wide = wide[order]
    cum_rows = "\n".join(
        f"| {n} | " + " | ".join(f"{wide.loc[n, c]:.2f}%"
                                 for c in order) + " |"
        for n in wide.index)

    answers = [(summary[p]) for p in order if summary[p]["min_n_over_50"]]
    answer_text = "; ".join(
        f"**{l8.PEAK_LABELS[s['peak']]} · net {s['direction']}**: "
        f"N = {s['min_n_over_50']} "
        f"({pct(s['cumulative_at_min_n_pct'])})"
        for s in answers)
    diffuse = [s for s in (summary[p] for p in order)
               if not s["top10_exceeds_50"]]
    diffuse_text = "; ".join(
        f"**{l8.PEAK_LABELS[s['peak']]} · net {s['direction']}** "
        f"reaches only {pct(s['top10_share_pct'])} by N = 10"
        for s in diffuse)

    md = f"""# L8 peak net-flow concentration in the top-10 blocks (fig 20)

*Rendered by `12_l8_netflow_concentration.py`.*

## Method

Reads `{SOURCE.name}` — the block-ranked, pairwise-disjoint top-10
neighbourhoods from `11_l8_peak_hotspots.py` — and accumulates each block's
share of the service-wide net flow:

    share_i      = |3×3 block net_i| / denominator
    cumulative_N = share_1 + ... + share_N

The denominator is the same one the hotspot table uses, per weekday:

    inflow   sum of positive cell net over in-region L8 cells
    outflow  sum of |negative cell net| over in-region L8 cells

The ten blocks are pairwise disjoint, so the cumulative share is the share of
their union and no cell is counted twice.

**Read the rank as the selection order, not as a citywide top-N.** The
non-overlap rule skips candidates that overlap an already chosen block, so
rank 2 need not be the second-largest block in the study area. The curve is a
concentration curve for the reported selection.

## Panel summary

| panel | denominator (trips/weekday) | top10 block net | top10 share | > 50%? | min N > 50% | cumulative at min N |
|---|---:|---:|---:|---|---:|---:|
{panel_rows}

{answer_text}. {diffuse_text}.

## Cumulative share by N

| N | AM inflow | AM outflow | PM inflow | PM outflow |
|---:|---:|---:|---:|---:|
{cum_rows}

## Files

| file | contents |
|---|---|
| `table_l8_netflow_concentration.csv` | 40 rows — one per panel and N, with the block value, its share, the running total and the 50% flag |
| `l8_netflow_concentration.json` | the same series plus the panel-level summary |
| `fig20_l8_netflow_concentration.png` | cumulative-share curves, 50% threshold, first crossing marked |

## Notes

- Per-block shares are unchanged from `11_l8_peak_hotspots.py`; this script
  only adds the running total.
- The `share of gross` alternative (block net / total departures + arrivals)
  is much smaller — under 15% for every top-10 panel — and no panel passes
  50% on that basis. The net denominators are used here because the question
  is about net inflow / net outflow.
- Disjoint blocks make the shares additive, but they still cover only part of
  the service area, so even N = 10 need not reach 100%.
"""
    (OUT / "l8_netflow_concentration_summary.md").write_text(
        md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    try:
        doc = load_source()
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    table, summary = panel_table(doc)
    plt = viz.setup()

    fig_concentration(plt, table, summary)
    write_report(doc, table, summary)

    for panel in ("am_inflow", "am_outflow", "pm_inflow", "pm_outflow"):
        s = summary[panel]
        print(f"  {panel:>11}: top10 {s['top10_share_pct']:5.2f}%  "
              f"{'> 50' if s['top10_exceeds_50'] else '<= 50'}"
              + (f"  min N = {s['min_n_over_50']} "
                 f"({s['cumulative_at_min_n_pct']:.2f}%)"
                 if s["min_n_over_50"] else ""))
    print("  wrote table_l8_netflow_concentration.csv, "
          "l8_netflow_concentration.json,")
    print("        l8_netflow_concentration_summary.md, "
          "fig20_l8_netflow_concentration.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
