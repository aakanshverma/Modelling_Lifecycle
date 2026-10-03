# Modelling Lifecycle

A reference implementation of the standard **NBFC / bank credit-risk customer
lifecycle** — the "ABCD" scorecard framework — as four linked, WOE-based
logistic-regression scorecards:

```
                 ┌─────────────────────┐
  New applicant → │  A — Acquisition    │ → approve / refer / reject
  (bureau data)   │     scorecard       │        │
                 └─────────────────────┘        │ approved → booked
                                                  ▼
                                      time passes on the books
                                                  ▼
                 ┌─────────────────────┐
                 │  B — Behavioral      │ → cross_sell route (performing)
                 │     scorecard        │ → collections route (underperforming)
                 └─────────────────────┘
                        │                              │
                        ▼                              ▼
             ┌─────────────────────┐        ┌─────────────────────┐
             │  C — Collections     │        │  D — Cross-sell      │
             │     scorecard         │        │     scorecard         │
             │  (self-cure           │        │  (offer acceptance    │
             │   propensity →        │        │   propensity →        │
             │   soft/tele/field)    │        │   top-tier/standard)  │
             └─────────────────────┘        └─────────────────────┘
```

This maps directly onto the journey described in the project brief: a
customer is scored at onboarding with bureau data (e.g. CIBIL), booked onto
the portfolio, re-scored on behavior after they've spent time with us, and
then forked into Collections if underperforming or Cross-sell if performing
well.

## Why one shared scorecard engine

All four stages (A/B/C/D) are the same statistical object — a binary bad-rate
model turned into a points-based score — just trained on a different
population and a different target. So the repo has **one** scorecard engine
(`lifecycle/scoring/`: WOE/IV binning + logistic regression + PDO points
scaling) and four thin stage modules (`lifecycle/stages/`) that each just
declare *which features*, *which target*, and *what decision rule* apply.
Swapping in real data means changing the feature lists in `stages/*.py`, not
the modeling code.

## Repo layout

```
src/lifecycle/
  data/synthetic.py      synthetic bureau + on-book + outcome data (swap for real feeds)
  scoring/woe.py          WOE/IV binning, shared by every scorecard
  scoring/scorecard.py    generic logistic-regression -> points scorecard
  stages/acquisition.py   Stage A: approve/refer/reject at origination
  stages/behavioral.py    Stage B: routes booked customers to collections/cross-sell
  stages/collections.py   Stage C: self-cure propensity -> collections strategy
  stages/cross_sell.py    Stage D: offer-acceptance propensity -> offer tier
  orchestrator.py         wires the four stages into one customer journey
scripts/run_lifecycle_demo.py   runs the whole thing end-to-end, prints a summary
tests/test_pipeline.py          smoke tests: pipeline runs, each stage learns signal
```

## Running it

```bash
pip install -r requirements.txt
python scripts/run_lifecycle_demo.py
```

This generates 20,000 synthetic applicants, runs them through all four
stages, prints a summary at each stage (decision counts, train AUC), and
writes the full per-customer journey to `lifecycle_journey.csv`.

Run the tests with:

```bash
pytest
```

## Plugging in real data

Replace `lifecycle/data/synthetic.py`'s generators with real extracts that
produce the same column names:

- **Acquisition**: bureau pull (CIBIL/Experian/Equifax) + application form →
  columns listed in `stages/acquisition.FEATURES`.
- **Behavioral**: core-banking/LMS repayment history for booked customers →
  columns listed in `stages/behavioral.FEATURES`.
- **Collections / Cross-sell outcomes**: historical collections/campaign
  results, used as the training target for stages C and D.

Everything downstream (`orchestrator.py`, the stage modules, the demo
script) is unchanged — it only depends on column names, not on the data
being synthetic.

## Extending

- Add a new lifecycle fork (e.g. a retention/attrition scorecard) by adding
  a new module under `stages/` following the same pattern (`FEATURES`,
  `TARGET`, `fit`, `score`, a decision function) and wiring it into
  `orchestrator.run_lifecycle`.
- Cutoffs (`approve_cutoff` in acquisition, the collections/cross-sell split
  in behavioral, decile bands in collections/cross_sell) are plain
  parameters — tune them against your own approval-rate / bad-rate targets
  rather than editing model internals.
