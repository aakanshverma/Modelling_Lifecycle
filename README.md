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
  data/synthetic.py        synthetic bureau + on-book + outcome data, incl. a monthly DPD panel
  scoring/woe.py            WOE/IV binning, shared by every scorecard
  scoring/scorecard.py      generic logistic-regression -> points scorecard (the "champion")
  scoring/challenger.py     XGBoost challenger + SHAP explainability, benchmarked vs. the champion
  scoring/reject_inference.py   parceling-based reject inference for the acquisition scorecard
  analytics/roll_rate.py   DPD transition matrix + vintage curve analysis
  monitoring/psi.py         Population Stability Index score-drift monitor
  stages/acquisition.py    Stage A: approve/refer/reject at origination
  stages/behavioral.py     Stage B: routes booked customers to collections/cross-sell
  stages/collections.py    Stage C: self-cure propensity -> collections strategy
  stages/cross_sell.py     Stage D: offer-acceptance propensity -> offer tier
  orchestrator.py          wires the four stages into one customer journey
scripts/
  run_lifecycle_demo.py           Phase 1: end-to-end ABCD chain, prints a summary
  run_reject_inference_demo.py    Phase 2: naive vs reject-inferred vs oracle acquisition model
  run_roll_rate_demo.py           Phase 2: roll-rate matrix + vintage curve by score band
  run_challenger_demo.py          Phase 2: XGBoost challenger vs champion, all 4 stages
  run_psi_monitoring_demo.py      Phase 2: PSI catching a simulated score drift
tests/
  test_pipeline.py   Phase 1 smoke tests: pipeline runs, each stage learns signal
  test_phase2.py      Phase 2 tests: reject inference, roll-rate invariants, challenger, PSI
```

## Running it

```bash
pip install -r requirements.txt
python scripts/run_lifecycle_demo.py
```

This generates 20,000 synthetic applicants, runs them through all four
stages, prints a summary at each stage (decision counts, train AUC), and
writes the full per-customer journey to `lifecycle_journey.csv`.

The Phase 2 scripts each stand alone and print their own report:
`run_reject_inference_demo.py`, `run_roll_rate_demo.py`,
`run_challenger_demo.py`, `run_psi_monitoring_demo.py`.

Run the tests with:

```bash
pytest
```

## Phase 2 — what each addition actually demonstrates

- **Reject inference** (`scoring/reject_inference.py`): simulates a prior,
  crude policy that only booked `cibil_score >= 700` applicants, so only
  they have an observed outcome. Fitting naively on that population alone
  reproduces the old policy's bias. Parceling infers labels for the
  never-booked population from the known population's bucketed bad rate,
  and refitting on the augmented set measurably closes the gap toward an
  (otherwise unreachable) oracle model fit on everyone's true outcome —
  0.70 → 0.73 AUC towards a 0.76 oracle in one run.
- **Roll-rate / vintage curves** (`data/synthetic.generate_dpd_panel`,
  `analytics/roll_rate.py`): simulates a monthly DPD-bucket panel per
  booked customer via a severity-blended Markov chain (DPD can only climb
  one bucket per month, but curing can jump back several at once — paying
  off the full overdue amount). The resulting vintage curve is the usual
  check that a lower acquisition score band reaches NPA faster than a
  higher one — i.e. that the acquisition cutoff is actually doing its job.
- **Challenger model** (`scoring/challenger.py`): an XGBoost model fit on
  the same raw features as each stage's champion, with SHAP for global
  feature attribution. Consistently edges out the WOE+logistic champion on
  holdout AUC/KS across all four stages in this synthetic data, which is
  the expected champion-challenger result — the gap is usually the
  business case for whether a more complex model is worth the loss of
  interpretability.
- **PSI monitoring** (`monitoring/psi.py`): bins a baseline score
  distribution and checks how a later population's scores land in those
  same bins. A fresh same-distribution cohort scores PSI ≈ 0.001 (stable);
  a simulated mild downturn (utilization and EMI burden both up) scores
  PSI ≈ 0.18 (moderate drift) — the standard signal that a scorecard needs
  to be revisited.

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
