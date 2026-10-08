# §D — Spatial patterns

*Rendered by `06_spatial.py`. Figures: **#5** departure/arrival maps, **#6** net-flow map.*

## What the numbers can and cannot say

The extract is *trips touching the study area* (§K0). So a cell inside the ring
has **complete** counts — every trip starting or ending there touches the area
and is therefore in the extract — while a cell outside it has only the trips
that also touch the area. Three classes follow, and the maps encode the
difference rather than hiding it:

| class | cells in the ±3 km window | how it is drawn | what it means |
|---|---:|---|---|
| wholly inside the ring | 6 | solid fill | counts **complete** |
| straddles the ring | 14 | hatched | counts **incomplete** — the outside sliver's trips are missing |
| outside the ring | 117 | faded | context only; its net flow estimates nothing |

**137 L6 cells** fall in the window; they carry **99.12% of all trip
endpoints** in the dataset. The remaining 551 cells lie
further out — the far ends of cross-district trips, one or two records each —
and are the subject of §E0, not of these maps.

## D1 — Departure / arrival volume

Both panels share **one log colour scale**. A linear scale is not an option
here: departures across L6 cells have a Gini of 0.969 (§B4), so on a linear ramp
each panel would render as a handful of dark cells on a flat field, and any
difference between them would be invisible.

**The two panels look nearly identical, and that is a finding, not a plotting
failure.** Departures and arrivals per cell agree closely — the gross flow
through every hotspot is close to balanced, and the imbalance is a *small
residual* on top of it. That is exactly why D2 needs a diverging scale centred
at zero (a shared sequential scale would show nothing) and why §E1 has to go to
the (cell, hour) resolution: the residual is real but it is an order of
magnitude smaller than the flow that carries it.

## D2 — Net flow (departures − arrivals)

Diverging scale, **centred at zero with symmetric limits** — with an asymmetric
scale a reader cannot tell which side dominates, which is the entire point of
the figure. Cells at the red extreme are where users will find no bike; blue
cells are where bikes accumulate and parking pressure builds.

Largest **drains** (positive net flow): `wx4eqv` (+12,387), `wx4eqx` (+4,629), `wx4erj` (+1,690), `wx4ern` (+1,341), `wx4eqg` (+897)

Largest **accumulations** (negative net flow): `wx4eqt` (-4,290), `wx4eqz` (-2,604), `wx4equ` (-2,165), `wx4eqw` (-1,714), `wx4erm` (-1,429)

> Note that the extreme values belong to cells that straddle or sit outside the
> ring, which is exactly the artefact the hatching warns about. The cells whose
> counts are complete are the ones in the table below, and it is those that
> should be read as the imbalance inside the service area.

### Cells with complete counts (wholly inside the ring)

| cell | departures | arrivals | net flow | share of all departures | inside fraction |
|---|---:|---:|---:|---:|---:|
| `wx4eqy` | 78,050 | 78,924 | -874 | 11.86% | 1.000 |
| `wx4eqv` | 71,720 | 59,333 | +12,387 | 10.89% | 1.000 |
| `wx4erj` | 42,968 | 41,278 | +1,690 | 6.53% | 1.000 |
| `wx4ern` | 36,893 | 35,552 | +1,341 | 5.60% | 1.000 |
| `wx4equ` | 34,109 | 36,274 | -2,165 | 5.18% | 1.000 |
| `wx4erh` | 21,550 | 22,032 | -482 | 3.27% | 1.000 |

## D3 — Hotspots

Top 5 L6 cells by departures, marked 1–5 on figures #5 and #6:

| # | cell (L6) | centroid (lat, lon) | departures | arrivals | net flow |
|---:|---|---|---:|---:|---:|
| 1 | `wx4eqt` | 39.9765, 116.3068 | 102,866 | 107,156 | -4,290 |
| 2 | `wx4eqw` | 39.9820, 116.3068 | 87,839 | 89,553 | -1,714 |
| 3 | `wx4eqy` | 39.9820, 116.3177 | 78,050 | 78,924 | -874 |
| 4 | `wx4eqv` | 39.9765, 116.3177 | 71,720 | 59,333 | +12,387 |
| 5 | `wx4erj` | 39.9765, 116.3287 | 42,968 | 41,278 | +1,690 |

**Read it as.** The hotspot list is the "gathering points" of the repositioning
problem, and the net-flow column already shows the asymmetry that §E1 resolves
in time: the busiest origin cell is also one of the largest accumulators, i.e.
its bikes come back faster than they leave across the day as a whole, while the
cell beside it drains. Aggregate net flow cancels opposite-signed hours — which
is precisely why E1 goes to the (cell, hour) resolution.
