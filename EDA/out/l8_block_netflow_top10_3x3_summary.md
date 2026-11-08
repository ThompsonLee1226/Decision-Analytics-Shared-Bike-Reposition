# L8 peak net-flow neighbourhoods (fig 19)

*Rendered by `11_l8_peak_hotspots.py`.*

## Method

For each three-hour peak (AM 07:00–09:59, PM 17:00–19:59):

1. accumulate departures and arrivals over the 60 weekdays and form the
   per-cell signed net inflow `arrivals − departures`, expressed as
   **trips per weekday**;
2. compute the **3 × 3 block net** for **every active in-region L8 cell as a centre**, then rank those blocks — descending for inflow, ascending for outflow;
3. keep ten blocks, requiring them to be pairwise disjoint.

The centre is the candidate cell that produced the block; the reported value is the **net flow of the nine-cell block**. Because the block is ranked directly, its centre need not be its dominant cell — the `dominant cell` column names the cell inside the block with the largest inflow or outflow. The table's `#` column ranks the selected blocks by that 3×3
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
| PM peak 17:00–19:59 · net inflow | `wx4eqtkv` | `wx4eqtku` | +271.3 | 33.3% | 7.0% |
| PM peak 17:00–19:59 · net outflow | `wx4eqw7r` | `wx4eqw7r` | -28.8 | 3.3% | 0.7% |

The AM net-outflow leader is the block centred on `wx4eqtku` (dominant cell `wx4eqtku`, -331.7/weekday, 29.5% of service net outflow). The PM net-inflow leader is the block centred on `wx4eqtkv` (dominant cell `wx4eqtku`, +271.3/weekday, 33.3% of service net inflow). Both blocks have `wx4eqtku` as their dominant cell, which is the L8-resolution role reversal in one cell.

The block ranking can promote a centre that is not itself the busiest cell, because its surrounding cells carry the imbalance. This is the intended difference from the single-cell ranking.

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
| 3 | 23 | `wx4eqtzt` | `wx4eqtzv` | 0.9 | 32.2 | +31.4 | 2.6% | 0.7% | 9 |
| 4 | 656 | `wx4eqwmk` | `wx4eqwm7` | 1.0 | 32.4 | +31.3 | 2.6% | 0.7% | 9 |
| 5 | 4 | `wx4eqwug` | `wx4eqwug` | 0.9 | 31.8 | +30.9 | 2.6% | 0.7% | 9 |
| 6 | 416 | `wx4eqwyg` | `wx4eqwzh` | 0.9 | 27.4 | +26.6 | 2.2% | 0.6% | 9 |
| 7 | 31 | `wx4eqwsb` | `wx4eqwmp` | 1.2 | 27.2 | +26.0 | 2.2% | 0.6% | 9 |
| 8 | 611 | `wx4ernmc` | `wx4ernm9` | 3.8 | 29.1 | +25.3 | 2.1% | 0.6% | 9 |
| 9 | 1585 | `wx4eqwwb` | `wx4eqwx0` | 1.0 | 25.2 | +24.2 | 2.0% | 0.6% | 9 |
| 10 | 331 | `wx4eqtyw` | `wx4eqtyq` | 1.8 | 25.8 | +24.0 | 2.0% | 0.6% | 9 |

### AM peak 07:00–09:59 · net outflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `wx4eqtku` | `wx4eqtku` | 385.8 | 54.0 | -331.7 | 29.5% | 7.7% | 9 |
| 2 | 2 | `wx4eqv7m` | `wx4eqv7m` | 110.4 | 21.2 | -89.2 | 7.9% | 2.1% | 9 |
| 3 | 4 | `wx4eqydw` | `wx4eqydw` | 78.1 | 17.9 | -60.3 | 5.4% | 1.4% | 9 |
| 4 | 5 | `wx4eqtkh` | `wx4eqtkh` | 46.4 | 8.4 | -38.0 | 3.4% | 0.9% | 9 |
| 5 | 8 | `wx4erje6` | `wx4erje6` | 26.8 | 5.9 | -20.9 | 1.9% | 0.5% | 9 |
| 6 | 144 | `wx4eqt6u` | `wx4eqt6s` | 22.5 | 6.1 | -16.4 | 1.5% | 0.4% | 9 |
| 7 | 223 | `wx4erj7n` | `wx4erj7m` | 53.2 | 37.0 | -16.2 | 1.4% | 0.4% | 9 |
| 8 | 101 | `wx4eqz48` | `wx4eqz49` | 29.1 | 14.9 | -14.1 | 1.3% | 0.3% | 9 |
| 9 | 27 | `wx4eqvkn` | `wx4eqvkn` | 19.2 | 8.2 | -11.0 | 1.0% | 0.3% | 9 |
| 10 | 532 | `wx4eqt5w` | `wx4eqt5y` | 13.0 | 4.3 | -8.7 | 0.8% | 0.2% | 9 |

### PM peak 17:00–19:59 · net inflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 8 | `wx4eqtkv` | `wx4eqtku` | 30.9 | 302.2 | +271.3 | 33.3% | 7.0% | 9 |
| 2 | 62 | `wx4eqt7h` | `wx4eqt7k` | 44.5 | 147.6 | +103.0 | 12.6% | 2.7% | 9 |
| 3 | 4 | `wx4eqydw` | `wx4eqydw` | 15.5 | 43.5 | +27.9 | 3.4% | 0.7% | 9 |
| 4 | 26 | `wx4eqz43` | `wx4eqz49` | 12.5 | 37.4 | +24.9 | 3.1% | 0.6% | 6 |
| 5 | 59 | `wx4eqtk5` | `wx4eqtkh` | 15.6 | 35.2 | +19.6 | 2.4% | 0.5% | 9 |
| 6 | 6554 | `wx4erjeh` | `wx4erje7` | 6.5 | 25.2 | +18.6 | 2.3% | 0.5% | 9 |
| 7 | 19 | `wx4eqexg` | `wx4eqexf` | 10.1 | 24.1 | +14.0 | 1.7% | 0.4% | 9 |
| 8 | 226 | `wx4erjkr` | `wx4erjs2` | 2.6 | 15.8 | +13.3 | 1.6% | 0.3% | 9 |
| 9 | 27 | `wx4erm7v` | `wx4erm7y` | 4.1 | 15.8 | +11.7 | 1.4% | 0.3% | 9 |
| 10 | 7762 | `wx4eqwx2` | `wx4eqwx0` | 26.5 | 35.6 | +9.1 | 1.1% | 0.2% | 9 |

### PM peak 17:00–19:59 · net outflow

| # | single rank | centre cell | dominant cell | 3×3 dep/day | 3×3 arr/day | 3×3 net/day | share of net | share of gross | active cells |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `wx4eqw7r` | `wx4eqw7r` | 39.8 | 10.9 | -28.8 | 3.3% | 0.7% | 9 |
| 2 | 11 | `wx4eqwm5` | `wx4eqwm5` | 27.5 | 3.1 | -24.4 | 2.8% | 0.6% | 9 |
| 3 | 24 | `wx4eqwmp` | `wx4eqwsb` | 25.3 | 4.8 | -20.5 | 2.4% | 0.5% | 9 |
| 4 | 5 | `wx4eqwj1` | `wx4eqwj1` | 20.5 | 1.0 | -19.6 | 2.3% | 0.5% | 9 |
| 5 | 41 | `wx4eqwnj` | `wx4eqwnj` | 15.4 | 0.9 | -14.4 | 1.7% | 0.4% | 9 |
| 6 | 142 | `wx4eqwe6` | `wx4eqwe4` | 14.7 | 1.5 | -13.2 | 1.5% | 0.3% | 9 |
| 7 | 51 | `wx4eqw7k` | `wx4eqw7m` | 14.5 | 1.7 | -12.8 | 1.5% | 0.3% | 9 |
| 8 | 33 | `wx4eqwtw` | `wx4eqwtw` | 13.5 | 1.1 | -12.4 | 1.4% | 0.3% | 9 |
| 9 | 4 | `wx4ernmd` | `wx4ernmd` | 14.6 | 2.8 | -11.8 | 1.4% | 0.3% | 9 |
| 10 | 78 | `wx4eqtyr` | `wx4eqtyq` | 13.8 | 2.1 | -11.6 | 1.3% | 0.3% | 9 |
