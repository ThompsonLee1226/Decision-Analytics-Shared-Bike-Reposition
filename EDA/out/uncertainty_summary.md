# §F — Uncertainty: variance, quantiles, and the Poisson assumption

*Rendered by `08_uncertainty.py`. Figure: **#13** mean–variance dispersion.*

Panel: departures per (cell, hour, day) for the top 30 L6 cells over all
83 days — 720 cell-hours — i.e. exactly the panel §H1 predicts.

## F1 — Sample variance and sample quantiles

HW §1 asks for **sample variance and sample quantiles** by name. They are the
columns below, for the top 10 cells each at its own busiest hour:

| cell | peak hour | mean | **sample variance** | std | CV | **p50** | **p90** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wx4eqt` | 08:00 | 255.7 | **21,028.7** | 145.0 | 0.57 | **330** | **394** |
| `wx4eqw` | 18:00 | 210.3 | **10,649.5** | 103.2 | 0.49 | **248** | **324** |
| `wx4eqy` | 08:00 | 97.4 | **2,327.3** | 48.2 | 0.50 | **114** | **145** |
| `wx4erj` | 08:00 | 92.7 | **2,373.4** | 48.7 | 0.53 | **110** | **145** |
| `wx4eqv` | 08:00 | 89.9 | **2,133.4** | 46.2 | 0.51 | **109** | **141** |
| `wx4ern` | 18:00 | 53.9 | **643.1** | 25.4 | 0.47 | **65** | **80** |
| `wx4equ` | 17:00 | 39.9 | **188.3** | 13.7 | 0.34 | **40** | **58** |
| `wx4eqe` | 08:00 | 35.3 | **171.9** | 13.1 | 0.37 | **37** | **51** |
| `wx4erh` | 07:00 | 34.9 | **228.0** | 15.1 | 0.43 | **40** | **53** |
| `wx4eqs` | 17:00 | 25.2 | **92.8** | 9.6 | 0.38 | **25** | **35** |

**Read the p90 column operationally.** The mean is the wrong number to plan
against: a cell whose mean is 250 trips in its peak hour still has a 1-in-10
busy day at a much higher count, and it is the busy day that empties the
station. p90 is the inventory that keeps the stockout rate near one day in ten;
the mean would leave the cell short about half the time.

**Read the variance column as the modelling constraint.** The variance is
several times the mean everywhere in this table, so a method that assumes
variance = mean — a plain Poisson GLM — would be over-confident by construction.
F4 quantifies this.

### The five most variable cell-hours

| cell | hour | mean | variance | var / mean | CV |
|---|---:|---:|---:|---:|---:|
| `wx4eqt` | 08:00 | 255.7 | 21,028.7 | 82.24 | 0.57 |
| `wx4eqt` | 09:00 | 192.6 | 11,715.7 | 60.84 | 0.56 |
| `wx4eqw` | 18:00 | 210.3 | 10,649.5 | 50.63 | 0.49 |
| `wx4eqw` | 17:00 | 165.4 | 5,799.2 | 35.05 | 0.46 |
| `wx4erj` | 08:00 | 92.7 | 2,373.4 | 25.61 | 0.53 |

## F4 — Dispersion check

For each (cell, hour) with a mean of at least 0.5 trips/day
(**572 pairs**), the variance across the 83 days is plotted against the
mean on log-log axes, with the Poisson reference line `variance = mean` drawn:

| | |
|---|---|
| Pairs above the Poisson line | **87.9%** |
| Median variance / mean | **1.46** |
| Fitted log-log slope | **var ∝ mean^1.39** (Poisson = 1.00) |
| Robustness — **weekdays only**, 572 pairs | slope **1.33** |
| Robustness — **every cell**, 722 pairs | slope **1.36** |

**Two robustness checks, because the headline slope is not innocent.** The panel
mixes weekdays with weekends, and that mixture is itself a source of variance:
restricting to weekdays drops the slope to **1.33**. That is the honest
number for what a model must still explain *after* its day-of-week term is in —
some of the over-dispersion is a feature the model can absorb, and the rest is
not. Widening the population to all 563 cells gives **1.36**, so the
hotspot subset is not a special case.

**Conclusion: the data is over-dispersed, and the answer is **a negative-binomial (or quasi-Poisson) model, not a plain Poisson**.** A slope
of 1.39 against the Poisson value of 1.00 means the variance grows
faster than the mean, which is what a mixture of days with different latent
rates looks like — the extra variation is real and it is not sampling noise.

This is why §H4 fits the first of its two required methods as a **negative binomial** GLM rather than a Poisson GLM: the assumption has been tested here, before any model was fitted, and it failed. The decision is recorded in §H4 with this number as its justification.

**Why the zero-mean cell-hours are excluded from the fit.** Below a mean of
0.5 trips/day the variance is decided by whether a single trip happened;
those points sit on the reference line by construction and would dilute the
test rather than inform it. The exclusion is stated because it changes the
sample.
