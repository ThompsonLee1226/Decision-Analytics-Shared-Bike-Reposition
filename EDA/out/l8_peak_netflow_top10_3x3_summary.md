# L8 peak net-flow neighbourhoods (fig 18)

*Rendered by `11_l8_peak_hotspots.py`.*

## Method

For each three-hour peak (AM 07:00–09:59, PM 17:00–19:59):

1. accumulate departures and arrivals over the 60 weekdays and form the
   per-cell signed net inflow `arrivals − departures`, expressed as
   **trips per weekday**;
2. rank cells by **single-cell** net flow — descending for inflow, ascending for outflow — and expand each selected cell to the **3 × 3 L8 block** centred on it;
3. keep ten blocks, requiring them to be pairwise disjoint.

The centre cell is the "top grid"; the reported value is the **net flow of the nine-cell block**, not of the centre alone. The table's `#` column ranks the selected blocks by that 3×3
net; `single rank` is the cell's position in the single-cell ranking, for
reference. The ten selected neighbourhoods are **pairwise disjoint**, so their shares do not double-count the same cells; they still cover only part of the service area, so the shares need not sum to 100%.

`active cells` counts the in-region L8 cells with any weekday activity in the
block (maximum 9). An inactive cell contributes zero, and blocks on the ring
edge are clipped to their in-region part, so this column can be below 9.

## Service denominators (per weekday)

| peak | sum of positive cell net | sum of abs(negative cell net) | gross flow |
|---|---:|---:|---:|
| AM peak 07:00–09:59 | 1,183.8 | 1,125.2 | 4,308.4 |
| PM peak 17:00–19:59 | 814.4 | 865.5 | 3,869.5 |

`share of net` is the block net divided by the matching column (positive net
for inflow blocks, |negative net| for outflow blocks). `share of gross` is
|block net| divided by the service gross flow; it is a smaller, more
conservative number.

## Leaders

| panel | #1 centre | dominant cell | 3×3 net/weekday | share of net | share of gross |
|---|---|---|---:|---:|---:|
| AM peak 07:00–09:59 · net inflow | `wx4eqtz0` | `wx4eqtz0` | +57.5 | 4.9% | 1.3% |
| AM peak 07:00–09:59 · net outflow | `wx4eqtku` | `wx4eqtku` | -331.7 | 29.5% | 7.7% |
| PM peak 17:00–19:59 · net inflow | `wx4eqtku` | `wx4eqtku` | +270.8 | 33.2% | 7.0% |
| PM peak 17:00–19:59 · net outflow | `wx4eqw7r` | `wx4eqw7r` | -28.8 | 3.3% | 0.7% |

The AM net-outflow leader is the block centred on `wx4eqtku` (dominant cell `wx4eqtku`, -331.7/weekday, 29.5% of service net outflow). The PM net-inflow leader is the block centred on `wx4eqtku` (dominant cell `wx4eqtku`, +270.8/weekday, 33.2% of service net inflow). Both blocks have `wx4eqtku` as their dominant cell, which is the L8-resolution role reversal in one cell.

The 3×3 aggregation strengthens the single-cell result: the surrounding cells belong to the same hotspot, not to its neighbours.

The non-overlap constraint matters here. `wx4eqtku` and `wx4eqtkv`, for
example, are adjacent cells whose 3×3 blocks cover almost the same ground;
keeping both would report the same core twice. The selected top10 therefore
reaches farther down the candidate ranking to find genuinely distinct areas.

## Top 10, panel by panel

### AM peak 07:00–09:59 · net inflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `wx4eqtz0` | `wx4eqtz0` | 5.7 | 63.1 | +57.5 | 4.9% | 1.3% | 9 |
| 2 | 9 | `wx4eqwe2` | `wx4eqwe2` | 10.2 | 55.1 | +44.9 | 3.8% | 1.0% | 9 |
| 3 | 4 | `wx4eqwug` | `wx4eqwug` | 0.9 | 31.8 | +30.9 | 2.6% | 0.7% | 9 |
| 4 | 11 | `wx4eqwy7` | `wx4eqwy7` | 1.1 | 27.2 | +26.1 | 2.2% | 0.6% | 9 |
| 5 | 2 | `wx4ernm9` | `wx4ernm9` | 3.4 | 28.1 | +24.7 | 2.1% | 0.6% | 9 |
| 6 | 8 | `wx4eqwx0` | `wx4eqwx0` | 1.0 | 25.1 | +24.1 | 2.0% | 0.6% | 9 |
| 7 | 3 | `wx4eqtyq` | `wx4eqtyq` | 2.0 | 25.3 | +23.4 | 2.0% | 0.5% | 9 |
| 8 | 5 | `wx4eqwzh` | `wx4eqwzh` | 0.8 | 23.1 | +22.3 | 1.9% | 0.5% | 9 |
| 9 | 6 | `wx4eqym9` | `wx4eqym9` | 25.5 | 42.0 | +16.5 | 1.4% | 0.4% | 9 |
| 10 | 7 | `wx4ermbw` | `wx4ermbw` | 7.6 | 23.4 | +15.7 | 1.3% | 0.4% | 9 |

### AM peak 07:00–09:59 · net outflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `wx4eqtku` | `wx4eqtku` | 385.8 | 54.0 | -331.7 | 29.5% | 7.7% | 9 |
| 2 | 2 | `wx4eqv7m` | `wx4eqv7m` | 110.4 | 21.2 | -89.2 | 7.9% | 2.1% | 9 |
| 3 | 4 | `wx4eqydw` | `wx4eqydw` | 78.1 | 17.9 | -60.3 | 5.4% | 1.4% | 9 |
| 4 | 5 | `wx4eqtkh` | `wx4eqtkh` | 46.4 | 8.4 | -38.0 | 3.4% | 0.9% | 9 |
| 5 | 8 | `wx4erje6` | `wx4erje6` | 26.8 | 5.9 | -20.9 | 1.9% | 0.5% | 9 |
| 6 | 9 | `wx4eqt6s` | `wx4eqt6s` | 19.1 | 5.0 | -14.1 | 1.3% | 0.3% | 9 |
| 7 | 13 | `wx4eqz49` | `wx4eqz49` | 27.1 | 14.8 | -12.2 | 1.1% | 0.3% | 6 |
| 8 | 10 | `wx4erj7m` | `wx4erj7m` | 43.9 | 35.4 | -8.5 | 0.8% | 0.2% | 9 |
| 9 | 18 | `wx4eqeku` | `wx4eqeku` | 12.7 | 5.6 | -7.1 | 0.6% | 0.2% | 6 |
| 10 | 19 | `wx4eqyrf` | `wx4eqyrf` | 17.7 | 12.4 | -5.3 | 0.5% | 0.1% | 9 |

### PM peak 17:00–19:59 · net inflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `wx4eqtku` | `wx4eqtku` | 32.4 | 303.1 | +270.8 | 33.2% | 7.0% | 9 |
| 2 | 2 | `wx4eqt7k` | `wx4eqt7k` | 42.2 | 140.0 | +97.7 | 12.0% | 2.5% | 9 |
| 3 | 4 | `wx4eqydw` | `wx4eqydw` | 15.5 | 43.5 | +27.9 | 3.4% | 0.7% | 9 |
| 4 | 5 | `wx4eqz49` | `wx4eqz49` | 12.6 | 36.1 | +23.5 | 2.9% | 0.6% | 6 |
| 5 | 6 | `wx4erje7` | `wx4erje7` | 9.1 | 27.2 | +18.1 | 2.2% | 0.5% | 9 |
| 6 | 7 | `wx4eqtkh` | `wx4eqtkh` | 17.8 | 34.8 | +17.0 | 2.1% | 0.4% | 9 |
| 7 | 9 | `wx4eqexf` | `wx4eqexf` | 10.1 | 24.1 | +14.0 | 1.7% | 0.4% | 9 |
| 8 | 11 | `wx4erjs2` | `wx4erjs2` | 2.7 | 15.7 | +13.0 | 1.6% | 0.3% | 9 |
| 9 | 12 | `wx4erhwm` | `wx4erhwm` | 1.9 | 10.5 | +8.6 | 1.1% | 0.2% | 9 |
| 10 | 10 | `wx4eqwx0` | `wx4eqwx0` | 27.5 | 35.8 | +8.3 | 1.0% | 0.2% | 9 |

### PM peak 17:00–19:59 · net outflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `wx4eqw7r` | `wx4eqw7r` | 39.8 | 10.9 | -28.8 | 3.3% | 0.7% | 9 |
| 2 | 11 | `wx4eqwm5` | `wx4eqwm5` | 27.5 | 3.1 | -24.4 | 2.8% | 0.6% | 9 |
| 3 | 5 | `wx4eqwj1` | `wx4eqwj1` | 20.5 | 1.0 | -19.6 | 2.3% | 0.5% | 9 |
| 4 | 12 | `wx4eqwsb` | `wx4eqwsb` | 23.4 | 5.5 | -18.0 | 2.1% | 0.5% | 9 |
| 5 | 4 | `wx4ernmd` | `wx4ernmd` | 14.6 | 2.8 | -11.8 | 1.4% | 0.3% | 9 |
| 6 | 8 | `wx4eqtyq` | `wx4eqtyq` | 12.8 | 2.4 | -10.4 | 1.2% | 0.3% | 9 |
| 7 | 17 | `wx4eqtz0` | `wx4eqtz0` | 12.0 | 2.0 | -10.0 | 1.2% | 0.3% | 9 |
| 8 | 9 | `wx4ern9q` | `wx4ern9q` | 10.1 | 3.0 | -7.1 | 0.8% | 0.2% | 9 |
| 9 | 3 | `wx4ermbw` | `wx4ermbw` | 13.6 | 6.6 | -7.0 | 0.8% | 0.2% | 9 |
| 10 | 2 | `wx4ern5k` | `wx4ern5k` | 9.0 | 2.1 | -6.9 | 0.8% | 0.2% | 9 |
