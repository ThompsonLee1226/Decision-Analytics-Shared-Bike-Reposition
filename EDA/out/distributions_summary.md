# §B — Univariate distributions

*Rendered by `04_distributions.py`. Figures: **#1** trip profile, **#2** Lorenz curve.*

## B1 — Trip profile

| variable | n | mean | std | sample var | CV | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:---:|---:|---:|---:|---:|---:|---:|---:|
| duration (min) | 658,304 | 11.14 | 12.86 | 165.49 | 1.15 | 1.00 | 2.00 | 4.00 | 7.00 | 12.00 | 35.00 | 71.00 | 180.0 |
| crow-fly (km) | 658,304 | 1.04 | 1.12 | 1.25 | 1.07 | 0.02 | 0.17 | 0.47 | 0.71 | 1.20 | 2.98 | 6.02 | 29.4 |
| implied speed (km/h) | 658,304 | 6.77 | 2.99 | 8.94 | 0.44 | 0.13 | 0.97 | 5.11 | 6.99 | 8.65 | 11.37 | 13.79 | 30.0 |

**Read it as.** Median trip 7 min and 0.71 km, with a
**standard deviation (12.9 min) larger than the mean (11.1 min) — CV = 1.15**.
That is the signature of heavy right-skew and a standing warning against
assuming normality on raw duration: any method that treats duration as
Gaussian will be dominated by the tails. Distance agrees with published
bike-share profiles — the majority of trips fall inside 2–3 km — which is the
point of keeping it: it is a check that the geohash decode and the unit
conversions are sound. If this panel had looked wrong, every spatial result
downstream would have been wrong too.

Distance is reported as a **crow-fly** straight line, so it is a lower bound on
path length; the implied speed (7.0 km/h median) is correspondingly a
lower bound, and is consistent with assisted and unassisted urban cycling.

## B4 — Spatial concentration

**563 L6 cells receive at least one departure; 11 of them
(2.0%) carry 80% of all departures.
Gini = 0.969.**

| rank | cell (L6) | departures | share | cumulative |
|---:|---|---:|---:|---:|
| 1 | `wx4eqt` | 102,866 | 15.63% | 15.63% |
| 2 | `wx4eqw` | 87,839 | 13.34% | 28.97% |
| 3 | `wx4eqy` | 78,050 | 11.86% | 40.83% |
| 4 | `wx4eqv` | 71,720 | 10.89% | 51.72% |
| 5 | `wx4erj` | 42,968 | 6.53% | 58.25% |

**Why this is the highest-value item in §B.** It is the empirical basis for the
*gathering point* concept: a handful of sites hold nearly all the activity, so
repositioning is a problem about a short list of places, not about a uniform
grid. It also fixes the population for §E1 and §H — restricting the
spatiotemporal panel and the demand models to the top cells is a defensible
narrowing, not a convenience. See the plan's *fact 2*.

**Cells are not counted as zeros when they have no departure.** A cell that
never sees a trip is not a repositioning target and is not an observation of
"zero demand" in the same sense; padding the ranking with them would inflate the
Gini without adding information.
