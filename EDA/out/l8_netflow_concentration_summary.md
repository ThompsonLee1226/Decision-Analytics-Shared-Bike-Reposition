# L8 peak net-flow concentration in the top-10 blocks (fig 20)

*Rendered by `12_l8_netflow_concentration.py`.*

## Method

Reads `l8_block_netflow_top10_3x3.json` — the block-ranked, pairwise-disjoint top-10
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
| AM peak 07:00–09:59 · net inflow | 1,183.8 | 322.0 | 27.20% | no | — | — |
| AM peak 07:00–09:59 · net outflow | 1,125.2 | 606.3 | 53.89% | yes | N = 7 | 50.89% |
| PM peak 17:00–19:59 · net inflow | 814.4 | 513.4 | 63.04% | yes | N = 4 | 52.44% |
| PM peak 17:00–19:59 · net outflow | 865.5 | 169.6 | 19.59% | no | — | — |

**AM peak 07:00–09:59 · net outflow**: N = 7 (50.89%); **PM peak 17:00–19:59 · net inflow**: N = 4 (52.44%). **AM peak 07:00–09:59 · net inflow** reaches only 27.20% by N = 10; **PM peak 17:00–19:59 · net outflow** reaches only 19.59% by N = 10.

## Cumulative share by N

| N | AM inflow | AM outflow | PM inflow | PM outflow |
|---:|---:|---:|---:|---:|
| 1 | 4.85% | 29.48% | 33.31% | 3.33% |
| 2 | 8.65% | 37.41% | 45.96% | 6.16% |
| 3 | 11.30% | 42.76% | 49.39% | 8.52% |
| 4 | 13.95% | 46.14% | 52.44% | 10.78% |
| 5 | 16.56% | 48.00% | 54.85% | 12.45% |
| 6 | 18.80% | 49.45% | 57.14% | 13.97% |
| 7 | 21.00% | 50.89% | 58.86% | 15.45% |
| 8 | 23.14% | 52.14% | 60.49% | 16.88% |
| 9 | 25.18% | 53.12% | 61.92% | 18.25% |
| 10 | 27.20% | 53.89% | 63.04% | 19.59% |

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
