# Why the folds are cut on Mondays

The short version: `walk_forward` divides the days a panel *contains* into equal
blocks. Once a filter removes some of them, equal blocks stop being equal weeks.

## What was wrong

`source.days: workday` drops every Saturday, Sunday and statutory holiday. The
shipped panel therefore has 57 distinct days spread over 82 calendar days — about
12 weeks. Dividing 57 by four puts the edges at days 14, 28 and 42, which is a
Thursday and two Fridays:

```
walk_forward   fold 1: 03-20 (Fri) .. 04-09 (Thu)   14d
               fold 2: 04-10 (Fri) .. 04-29 (Wed)   14d
               fold 3: 04-30 (Thu) .. 05-22 (Fri)   15d
```

Nothing leaks — no feature in this package uses a lag of the target — but the fold
stops meaning what a walk-forward fold is for. Fold 2 validates Friday 04-10
through Wednesday 04-29, while its training set contains Monday 03-02 through
Thursday 04-09. A validation day therefore shares most of its **week** with the
training set, and "predict next week" is not the claim the number supports.

The same defect shows up on the EDA comparison panel, which is worse because that
panel is *every* day: 77 days over four blocks gives 19/19/19/20, so fold 2 starts
on a Saturday and fold 3 on a Thursday.

## What the fix is

`fold.scheme: week_forward` places the interior cuts on Mondays and counts in whole
calendar weeks. Each fold trains on everything before a given week and validates
whole weeks:

```
week_forward   fold 1: 03-23 (Mon) .. 04-10 (Fri)   14d
               fold 2: 04-13 (Mon) .. 05-09 (Sat)   18d
               fold 3: 05-11 (Mon) .. 05-22 (Fri)   10d
```

`walk_forward` is kept, because the numbers published before this change were
produced with it and reproducing them has to stay possible.

## Two things that are not as tidy as they look

**The week counts are 3 / 4 / 2, not 3 / 3 / 3.** 04-06 and 05-04 are Mondays that
fall on statutory holidays. Neither is in the panel, so neither can carry a cut;
the nearest Monday takes over and one block ends up a week short. Forcing 3/3/3
would mean cutting inside a week, which is the thing the scheme exists to avoid.

**`week_forward` refuses instead of falling back.** If the panel has fewer usable
Mondays than it needs interior cuts, it raises and says so. Silently dropping to
equal-count blocks would produce a number labelled `week_forward` that came from a
different geometry — the failure mode this whole change is meant to remove.

## What it changed in the numbers

The shipped config, both targets, calibrated deviance:

| | walk_forward | week_forward |
|---|---|---|
| arrivals, fold 1 | 0.2313 | 0.2255 |
| arrivals, fold 2 | 0.2072 | 0.2074 |
| arrivals, fold 3 | 0.1973 | **0.1929** |
| arrivals, pooled | 0.2116 | 0.2100 |
| departures, pooled | 0.2470 | 0.2430 |

The last fold went from the **worst** to the **best** in both targets. Two things
changed together and both push that way: the last fold now validates ten working
days rather than fifteen, and those ten are two whole weeks rather than two and a
half. Fewer, more homogeneous days are easier to predict.

So the earlier story — "the last fold is worst because it validates the May demand
ramp" — was at least partly the geometry talking. The ramp is real, but the fold
that measured it was also the fold whose validation window straddled weeks.

## What it did **not** change

The comparison against the EDA report. On the 40,320 rows both splits validate,
scored against the EDA predictions on those same rows:

| scheme | n | EDA | ours | change |
|---|---|---|---|---|
| walk_forward | 40,320 | 1.7854 | 1.4892 | −16.6% |
| week_forward | 40,320 | 1.7854 | 1.4765 | −17.3% |

The gap survives the change, so it is a statement about the model and not about the
split.

## Reproducing

```bash
xgb-bikeflow check --config configs/horizon_am.yaml    # 40 s
xgb-bikeflow run   --config configs/horizon_am.yaml    # 42 min, both targets
xgb-bikeflow check --config configs/eda_comparison.yaml # 2 s
```

Switch `fold.scheme` in either file to `walk_forward` to get the old geometry; the
YAML comments carry both sets of numbers.
