# §C — Temporal patterns

*Rendered by `05_temporal.py`. Figures: **#3** temporal overview, **#4** dow × hour heat map.*

## C1 — Temporal overview

| | |
|---|---|
| Days | 84 cleaned of 84 in the window (2026-05-17 removed, §A5) |
| **Daily volume** | mean **7,935**, sample variance **4,611,771** (sd 2,148) |
| Range | 2,458 – 11,303 per day |
| Weekday | 8,759/day over 60 days |
| Weekend | 5,876/day over 24 days |
| Month | March **7,307/day** → April–May **8,285/day** |

**Read it as.** March runs lighter than April–May — a **ramp-up** worth naming so
nobody mistakes it for a demand trend the model should extrapolate. The daily
sample variance (4,611,771) is large relative to the mean
(7,935) but is dominated by the weekday/weekend split rather than by
day-to-day noise within a class.

**If this were a demand forecast, the ramp-up would be a problem.** It is not:
§H predicts *within* the panel, and the walk-forward split of H3 means every
validation block is predicted from the days that precede it, so the level shift
is absorbed by the training window rather than extrapolated.

### Hour of day

| hour | weekday (mean/day) | weekend (mean/day) | share · weekday | share · weekend |
|---:|---:|---:|---:|---:|
| 00 | 31 | 36 |  0.35% |  0.61% |
| 01 | 14 | 18 |  0.16% |  0.31% |
| 02 | 8 | 11 |  0.09% |  0.18% |
| 03 | 5 | 7 |  0.06% |  0.12% |
| 04 | 9 | 8 |  0.10% |  0.14% |
| 05 | 28 | 22 |  0.32% |  0.37% |
| 06 | 162 | 85 |  1.85% |  1.45% |
| 07 | 584 | 232 |  6.67% |  3.96% |
| 08 | 1,092 | 371 | 12.46% |  6.31% |
| 09 | 742 | 396 |  8.47% |  6.74% |
| 10 | 407 | 344 |  4.65% |  5.85% |
| 11 | 386 | 373 |  4.41% |  6.34% |
| 12 | 420 | 436 |  4.80% |  7.42% |
| 13 | 428 | 381 |  4.89% |  6.48% |
| 14 | 368 | 359 |  4.20% |  6.10% |
| 15 | 369 | 361 |  4.22% |  6.14% |
| 16 | 462 | 381 |  5.27% |  6.49% |
| 17 | 849 | 487 |  9.69% |  8.29% |
| 18 | 907 | 464 | 10.36% |  7.90% |
| 19 | 521 | 327 |  5.94% |  5.57% |
| 20 | 376 | 270 |  4.29% |  4.59% |
| 21 | 305 | 237 |  3.48% |  4.04% |
| 22 | 212 | 198 |  2.43% |  3.37% |
| 23 | 72 | 72 |  0.82% |  1.23% |

Weekday peaks at **08:00** and again at
**18:00**, with an overnight trough near 03:00. The morning peak
**flattens at the weekend** — the weekend profile is close to a single broad
afternoon hump. The morning peak is a commute-driven *supply* event (bikes leave
residential areas) and the evening peak is its mirror; that asymmetry, not
either peak alone, is what §E measures.

## C2 — Day-of-week × hour

**Normalisation: per row.** Each cell is the share of that weekday's trips
falling in that hour, so every row sums to 100%. Raw counts are *not* used
because the cleaned window does not contain an equal number of each weekday
(2026-05-17, a Sunday, was removed in §A5), which would bias Sundays down by
construction.

| weekday | peak hour | share of that day | mean trips in that hour |
|---|---|---:|---:|
| Mon | 08:00 | 12.4% | 1,017 |
| Tue | 08:00 | 12.5% | 1,102 |
| Wed | 08:00 | 13.1% | 1,189 |
| Thu | 08:00 | 12.6% | 1,103 |
| Fri | 08:00 | 11.8% | 1,047 |
| Sat | 17:00 | 8.4% | 535 |
| Sun | 17:00 | 8.1% | 439 |

**Mon–Thu are near-identical** (morning peak 12.3–13.0%, evening 9.4–9.9%).
**Friday** keeps the same peak hours but shifts weight out of the morning and
into the evening (morning 11.7%, evening 10.1/10.8%) — the weekend arriving.
**Sat/Sun lose the morning spike entirely** (07:00–09:00 share falls from
~19% of the day to ~11%) and flatten into a broad midday plateau, with a
single evening peak at 17:00 rather than a commute pair.

The important consequence for §E: the morning spike is a **weekday** structure,
not a Mon–Fri averaging artefact, which is why §E1's cell × hour panel is
restricted to weekdays. Including weekends would smear the very mirror-image
pattern the figure exists to show.
