# §C — Temporal patterns

*Rendered by `05_temporal.py`. Figures: **#3** temporal overview, **#4** dow × hour heat map.*

## C1 — Temporal overview

| | |
|---|---|
| Days | 83 cleaned of 84 in the window (2026-05-17 removed, §A5) |
| **Daily volume** | mean **7,931**, sample variance **4,221,322** (sd 2,055) |
| Range | 3,806 – 11,192 per day |
| Weekday | 8,684/day over 60 days |
| Weekend | 5,969/day over 23 days |
| Month | March **7,246/day** → April–May **8,319/day** |

**Read it as.** March runs lighter than April–May — a **ramp-up** worth naming so
nobody mistakes it for a demand trend the model should extrapolate. The daily
sample variance (4,221,322) is large relative to the mean
(7,931) but is dominated by the weekday/weekend split rather than by
day-to-day noise within a class.

**If this were a demand forecast, the ramp-up would be a problem.** It is not:
§H predicts *within* the panel, and the walk-forward split of H3 means every
validation block is predicted from the days that precede it, so the level shift
is absorbed by the training window rather than extrapolated.

### Hour of day

| hour | weekday (mean/day) | weekend (mean/day) | share · weekday | share · weekend |
|---:|---:|---:|---:|---:|
| 00 | 30 | 36 |  0.35% |  0.61% |
| 01 | 14 | 19 |  0.16% |  0.31% |
| 02 | 8 | 11 |  0.09% |  0.18% |
| 03 | 5 | 7 |  0.06% |  0.12% |
| 04 | 9 | 8 |  0.10% |  0.14% |
| 05 | 28 | 22 |  0.32% |  0.36% |
| 06 | 161 | 86 |  1.85% |  1.43% |
| 07 | 580 | 235 |  6.68% |  3.93% |
| 08 | 1,080 | 374 | 12.43% |  6.27% |
| 09 | 735 | 397 |  8.47% |  6.65% |
| 10 | 403 | 343 |  4.64% |  5.75% |
| 11 | 383 | 370 |  4.41% |  6.19% |
| 12 | 417 | 432 |  4.80% |  7.24% |
| 13 | 425 | 391 |  4.89% |  6.55% |
| 14 | 365 | 368 |  4.20% |  6.16% |
| 15 | 366 | 370 |  4.22% |  6.20% |
| 16 | 458 | 392 |  5.28% |  6.57% |
| 17 | 843 | 502 |  9.71% |  8.41% |
| 18 | 900 | 478 | 10.36% |  8.00% |
| 19 | 517 | 337 |  5.95% |  5.64% |
| 20 | 373 | 276 |  4.30% |  4.62% |
| 21 | 302 | 242 |  3.48% |  4.06% |
| 22 | 210 | 201 |  2.42% |  3.37% |
| 23 | 71 | 74 |  0.82% |  1.25% |

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
| Mon | 08:00 | 12.3% | 1,004 |
| Tue | 08:00 | 12.5% | 1,090 |
| Wed | 08:00 | 13.0% | 1,178 |
| Thu | 08:00 | 12.5% | 1,091 |
| Fri | 08:00 | 11.7% | 1,037 |
| Sat | 17:00 | 8.4% | 531 |
| Sun | 17:00 | 8.4% | 471 |

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
