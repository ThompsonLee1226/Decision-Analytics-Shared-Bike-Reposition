# scripts/measure/

Scripts that produce the numbers quoted in `docs/`. They are not part of the
package's import surface — nothing here is needed to train or predict — but each
one is the evidence behind a specific claim, and they are kept so the claims can
be re-derived instead of trusted.

Run them from the repository root, e.g.

```bash
python scripts/measure/holiday_volume.py
```

They write into `docs/eda_comparison/`.

| script | question it answers | output |
|---|---|---|
| `holiday_volume.py` | how much smaller is a holiday, in the horizon? | `holiday_volume.json` |
| `holiday_daytype.py` | what does a holiday row cost a model that never saw one? | `holiday_daytype.json` |
| `holiday_ab.py` | what are the eight holidays worth with fold edges held fixed? | `holiday_ab_same_folds.json` |
| `holiday_ab_naive.py` | the same question with each arm choosing its own edges | `holiday_ab.json` |
| `holiday_gamma.py` | is `gamma` an optimum or a policy knob? | `holiday_and_gamma.json` |

`holiday_ab_naive.py` is kept **as a negative example**. It lets each arm place
its own fold edges, so the arm with eight extra days also gets different folds,
and the resulting "−1.24% on shared workday rows" mixes the effect of the
holidays with the effect of a different split. `holiday_ab.py` holds the edges
fixed and reports −0.54%, which is the number in the docs. The pair is here
because the wrong method produced a plausible-looking number, and the only
defence against repeating it is to keep it visible next to the right one.

The `_v1_*` scripts that measured the v1.0 entity accuracy were one-off probes
and are not kept; their results are in `docs/network_v1.md`, and the entity
tables they read are written by `cli rollup`, so the measurements can be redone
from a finished run.
