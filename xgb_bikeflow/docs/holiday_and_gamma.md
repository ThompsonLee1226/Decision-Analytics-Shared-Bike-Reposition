# The eight holidays, and what `gamma` controls

Two questions that look like configuration details and are not.

## 1. The holidays are in the data. We were deleting them.

`2026-04-06` and `2026-05-04` are both Mondays and both statutory holidays. They
are in `data/orders_2026.csv.gz` like every other day:

| date | weekday | orders in the file |
|---|---|---|
| 2026-04-06 | Mon | 4,530 |
| 2026-05-04 | Mon | 5,081 |

They were absent from the panel because `source.days: workday` removes every date
in `source.holidays`, not because anything was missing. Inside the 07:30–10:30
horizon they are quiet but not empty:

| day type | days | mean events in horizon | vs workday |
|---|---|---|---|
| workday | 57 | 2,514 | 1.00x |
| make-up (2026-05-09) | 1 | 2,536 | 1.01x |
| weekend | 19 | 1,062 | 0.42x |
| **holiday** | **8** | **853** | **0.34x** |

| holiday | weekday | events | vs workday |
|---|---|---|---|
| 2026-04-04 | Sat | 952 | 0.38x |
| 2026-04-05 | Sun | 770 | 0.31x |
| **2026-04-06** | **Mon** | **749** | **0.30x** |
| 2026-05-01 | Fri | 849 | 0.34x |
| 2026-05-02 | Sat | 864 | 0.34x |
| 2026-05-03 | Sun | 869 | 0.35x |
| **2026-05-04** | **Mon** | **870** | **0.35x** |
| 2026-05-05 | Tue | 901 | 0.36x |

The eight days are 4.0% of horizon volume. Both holiday Mondays land on 0.30–0.35x,
so a Monday holiday is not a Monday and is not a weekend either.

## 2. What the shipped model does on a holiday

Fold 3, trained as shipped (`days: workday`), scored on its own validation window
with the holiday rows added back:

| target | arm | workday rows | holiday rows | holiday actual | holiday pred | calib |
|---|---|---|---|---|---|---|
| arrivals | A, no holidays in training | dev 0.1874 | dev 0.1449 | 0.0139 | 0.0425 | **3.05x** |
| arrivals | B, holidays in training | dev 0.1868 | **dev 0.1022** | 0.0139 | 0.0133 | **0.95x** |
| departures | A | dev 0.2229 | dev 0.1647 | 0.0138 | 0.0394 | **2.86x** |
| departures | B | dev 0.2203 | **dev 0.1251** | 0.0138 | 0.0133 | **0.96x** |

The shipped model over-predicts a holiday by **3x**. It has never seen
`is_holiday = 1`, so `is_holiday` cannot split anything, and all it has for those
days is the trend scale, which is fitted to a rising working-day series. A holiday
inherits a working day's trajectory.

Adding the eight days to training fixes exactly that, and costs almost nothing:

| target | holiday dev | workday dev |
|---|---|---|
| arrivals | **−29.5%** | −0.31% |
| departures | **−24.1%** | −1.14% |

Both arms share the fold edges, so the only difference is the eight days. (An
earlier attempt let each arm place its own edges and reported −1.24% on workday
rows; that number mixed two changes and is superseded by the table above.)

**So the answer to "should we label them instead of dropping them" is yes.** Keeping
them is `source.days: labelled`, which keeps working days *and* the eight holidays
and leaves weekends out — a holiday is a quiet working day, not a Saturday.

## 3. `gamma`: configured, not optimised

`gamma` weights the two group means:

    loss = gamma * mean(deviance over gathering points)
         + (1 - gamma) * mean(deviance over every other cell)

It is a number in the YAML (`loss.gamma: 0.5`), not a searched hyper-parameter, and
that is deliberate: it is a **policy** weight — how much a mistake at a
gathering point is worth relative to a mistake elsewhere — not something the data
can settle. What the data *can* settle is whether 0.5 is a sensible reading, and
that is measurable because `gamma` never enters the booster.

One fitted model, rescored at eleven gammas (fold 3, gathering rows 48,060 of
575,640):

| gamma | arrivals loss | departures loss |
|---|---|---|
| 0.0 | 0.16824 | 0.19758 |
| 0.1 | 0.19808 | 0.23149 |
| 0.2 | 0.22791 | 0.26541 |
| 0.3 | 0.25775 | 0.29933 |
| 0.4 | 0.28758 | 0.33324 |
| **0.5** | **0.31742** | **0.36716** |
| 0.6 | 0.34726 | 0.40108 |
| 0.7 | 0.37709 | 0.43499 |
| 0.8 | 0.40693 | 0.46891 |
| 0.9 | 0.43676 | 0.50283 |
| 1.0 | 0.46660 | 0.53675 |

The two group losses are constant down the column — gathering 0.4666, other 0.1682
for arrivals — because the model does not change. Only the mixture moves, and it
moves **monotonically**, so "the best gamma" is always 0.0 or 1.0 and says nothing.
An optimiser would return 0.0 here, which means "ignore the gathering points",
which is the opposite of what the brief asks for. That is why `gamma` is not tuned:
whatever number minimises this column is answering a different question.

**What `gamma` is not doing:** it does not change the fit. `loss.fit_weight` is the
knob that would, and it was swept (0 / 0.25 / 0.5 / 0.75, both targets) and moved
the loss by at most 0.53% with no consistent sign — noise. `count:poisson` already
fits the conditional mean; reweighting changes which rows push the gradient, not
what the model is asked to predict. It ships at `0.0`.

So the weights work like this:

| knob | what it does | shipped value | optimised? |
|---|---|---|---|
| `loss.gathering` | which cells are important (`data/gathering_points.json`, 801 of 9,594) | file | no — an input from the other group |
| `loss.gamma` | weight on the gathering group's **mean** | 0.5 | no — a policy choice, see above |
| `loss.fit_weight` | also weights training rows | 0.0 | swept, no effect, left off |
| `rows.off_target_weight` | weight for rows outside the scoring set | 1.0 | no — the gamma rule already balances groups |

## Reproducing

```
xgb-bikeflow check --config configs/horizon_am.yaml   # 40 s
```

The measurements above came from the scripts in `scripts/measure/`, run from the
repository root:

```
python scripts/measure/holiday_volume.py      # seconds
python scripts/measure/holiday_daytype.py     # ~20 min
python scripts/measure/holiday_ab.py          # ~35 min
python scripts/measure/holiday_gamma.py       # ~10 min
```

They each retrain, so they are slow; the JSON they write is checked in under
`docs/eda_comparison/` so the numbers can be read without rerunning anything.
`scripts/measure/holiday_ab_naive.py` is the version that let each arm choose its own
fold edges — kept as a negative example, see `scripts/measure/README.md`.
