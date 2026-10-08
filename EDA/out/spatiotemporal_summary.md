# §E1 · §E2 · §E5 — Spatiotemporal joint analysis

*Rendered by `07_spatiotemporal.py`. Figures: **#10** cell × hour balance (lead),
**#11** anchor-hour net flow, **#12** role reversal.*

Population: the **top 30 L6 cells by volume**, of which **11 lie
inside the study area**. Fig 10 uses the 60 weekdays only (§C2 shows the
morning peak is a weekday structure); figs 11–12 use all 83 days.

## E1 — Cell × hour balance (the lead figure)

**What is plotted.** `balance = (departures − arrivals) / (departures + arrivals)`
per cell-hour, averaged over the 60 weekdays. It is a ratio rather than raw
net flow because the plan's "normalised per row" cannot be applied to a signed
quantity — a row sum of net flow is near zero and the ratio would explode. The
balance is bounded in [−1, +1], reads literally ("40% more departures than
arrivals that hour"), and lets a 200-trip cell be compared with a 20,000-trip
one. **Raw net flow per cell-hour is in `table_e1_cell_hour_netflow.csv`.**

**Grey is not zero.** 223 of 720 cell-hours
(31.0%) had under 2 trips/day of gross flow and are
left uncoloured. At that volume the sign of the balance is a coin flip; shading
it would paint noise as structure.

### The mirror-image pair

| | morning net | evening net | reading |
|---|---:|---:|---|
| `wx4eqt` | +86.2 | -101.5 | **workplace (drains AM, fills PM)** |
| `wx4eqw` | -159.7 | +132.6 | **residential (fills AM, drains PM)** |

These are the two largest opposing swings in the panel, and they are **adjacent
cells**. One fills through the morning and drains through the evening; the other
does the reverse, at a comparable magnitude. That is the imbalance the
assignment asks for, in one line: not too few bikes in the area, but bikes on
the wrong side of a cell boundary at the wrong hour.

## E2 — Net flow at the anchor hours

Fig 11 maps the same quantity geographically at **08:00 and 18:00**, on one
shared diverging scale. E1 says *which cells and when*; E11 says *where they
are*. Two panels is enough for that, which is why noon and 22:00 were dropped.

## E5 — Role reversal

A cell is counted as **reversed** when the sign of its mean net flow differs
between the morning (07:00–09:00) and the evening
(17:00–19:00) windows, and **only when both windows exceed
1 trip/day** — below that a sign is not a role. Of 30 cells:

| role | cells | |
|---|---:|---|
| **reversed** | **14** | `wx4eqw`, `wx4ern`, `wx4erp`, `wx4eqx`, `wx4ewb`, `wx4ex2`, `wx4erq`, `wx4eqm`, `wx4eqz`, `wx4eqs`, `wx4erh`, `wx4eqe`, `wx4erj`, `wx4eqt` |
| same role in both windows | 2 | |
| one-sided (active in one window only) | 9 | |
| flat in both | 5 | |

**14 of 30 cells (47%) swap roles between morning and evening.** This is a genuinely
non-obvious result and it is the strongest available argument for **dynamic**
rather than static repositioning: a fixed morning rule and a fixed evening rule
would each service these cells wrongly for half the day.

## Full role table

| cell | volume | morning net | evening net | role | reading |
|---|---:|---:|---:|---|---|
| `wx4eqw` | 177,392 | -159.7 | +132.6 | reversed | residential (fills AM, drains PM) |
| `wx4ern` | 72,445 | -24.3 | +23.1 | reversed | residential (fills AM, drains PM) |
| `wx4erp` | 11,708 | -11.3 | +4.8 | reversed | residential (fills AM, drains PM) |
| `wx4equ` | 70,383 | -3.7 | -2.6 | same | — |
| `wx4eqx` | 25,569 | -3.1 | +13.8 | reversed | residential (fills AM, drains PM) |
| `wx4eqd` | 4,362 | -1.7 | +0.2 | one-sided | — |
| `wx4ewb` | 5,398 | -1.4 | +1.2 | reversed | residential (fills AM, drains PM) |
| `wx4ex0` | 4,612 | -1.4 | +0.3 | one-sided | — |
| `wx4eqf` | 3,116 | -0.7 | +0.3 | flat | — |
| `wx4eqc` | 3,347 | -0.6 | -0.8 | flat | — |
| `wx4erm` | 29,297 | -0.5 | -3.3 | one-sided | — |
| `wx4er7` | 9,759 | -0.5 | +1.0 | one-sided | — |
| `wx4erk` | 16,007 | -0.2 | +1.7 | one-sided | — |
| `wx4ew8` | 4,808 | -0.1 | +2.4 | one-sided | — |
| `wx4eqy` | 156,974 | +0.2 | -5.5 | one-sided | — |
| `wx4eq9` | 4,070 | +0.3 | -0.3 | flat | — |
| `wx4ert` | 3,721 | +0.4 | -0.3 | flat | — |
| `wx4err` | 3,210 | +0.7 | -0.9 | flat | — |
| `wx4er5` | 13,785 | +1.1 | -0.2 | one-sided | — |
| `wx4ex2` | 3,340 | +1.2 | -2.5 | reversed | workplace (drains AM, fills PM) |
| `wx4erq` | 6,697 | +1.5 | -1.2 | reversed | workplace (drains AM, fills PM) |
| `wx4eqm` | 3,900 | +1.7 | -1.5 | reversed | workplace (drains AM, fills PM) |
| `wx4eqg` | 29,487 | +4.6 | -0.4 | one-sided | — |
| `wx4eqz` | 30,118 | +7.3 | -9.7 | reversed | workplace (drains AM, fills PM) |
| `wx4eqs` | 48,082 | +9.9 | -5.7 | reversed | workplace (drains AM, fills PM) |
| `wx4erh` | 43,582 | +13.8 | -8.1 | reversed | workplace (drains AM, fills PM) |
| `wx4eqe` | 43,679 | +15.7 | -9.5 | reversed | workplace (drains AM, fills PM) |
| `wx4eqv` | 131,053 | +18.9 | +8.2 | same | — |
| `wx4erj` | 84,246 | +28.9 | -14.6 | reversed | workplace (drains AM, fills PM) |
| `wx4eqt` | 210,022 | +86.2 | -101.5 | reversed | workplace (drains AM, fills PM) |
