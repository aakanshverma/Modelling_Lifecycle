# NBFC Customer Lifecycle Modelling — Plan

Status: **proposed, not executed.** Nothing beyond this document and an
unreviewed Phase 1 code scaffold has been built. No stage below starts
until it's explicitly approved.

## The framework

This project models the standard NBFC/bank credit-risk customer journey,
known industry-wide as the **"ABCD" scorecard framework** (this is not a
custom naming — bureaus like TransUnion CIBIL, CRIF, and Experian structure
their data products around it, and most Indian NBFCs run their risk stack
this way):

```
New applicant → [A: Acquisition scorecard] → approve / refer / reject
                        (bureau data)              │ approved → booked
                                                     ▼
                                     customer ages on the books
                                                     ▼
                           [B: Behavioral scorecard]
                        (on-us repayment behavior + bureau refresh)
                                 │                        │
                        underperforming              performing well
                                 ▼                        ▼
                  [C: Collections scorecard]   [D: Cross-sell scorecard]
                  (self-cure propensity →        (offer-acceptance propensity
                   soft/tele/field strategy)      → offer tier)
```

This is exactly the journey described in the project brief: score at the
door with bureau data, book the good ones, re-score on behavior after
they've spent time with us, then fork into Collections or Cross-sell.

## Phased plan

### Phase 1 — MVP: the ABCD chain on synthetic data
**Goal:** prove the end-to-end architecture works before touching real data.

- One shared scoring engine: WOE/IV binning + logistic regression + points
  scaling (the standard, regulator-friendly scorecard technique).
- Four thin stage modules (Acquisition, Behavioral, Collections, Cross-sell)
  that each just declare features, target, and a decision rule on top of
  the shared engine.
- Synthetic data generators standing in for real bureau/on-book feeds, so
  the pipeline is runnable immediately and swapping to real data later only
  means changing column sources, not model code.
- One orchestrator wiring all four stages into a single per-customer
  journey output (decision + score at every stage).
- A demo script + smoke tests.

**Status:** already scaffolded in the working tree (uncommitted) — see
`src/lifecycle/`, `scripts/run_lifecycle_demo.py`, `tests/`. Not yet
reviewed, run, or committed as working code.

### Phase 2 — Make it industry-realistic
**Goal:** close the gap between "demo" and "what a risk team would actually
ship."

- **Reject inference** on the Acquisition model — since you only observe
  outcomes for approved applicants, standard practice is to statistically
  infer performance of declines so the model isn't biased toward the
  population it already approved.
- **Roll-rate / vintage curve analysis** alongside the Behavioral score —
  a DPD transition matrix (Current → 30 → 60 → 90) used for loss
  forecasting, not just a single good/bad flag.
- **Challenger model**: XGBoost/LightGBM alongside logistic regression,
  with SHAP explainability — common practice now even where regulators
  require the final decision to come from an interpretable model.
- **PSI-based score monitoring** — detect when a live scorecard has
  drifted and needs refitting.

### Phase 3 — Make the fork smarter
**Goal:** move Collections and Cross-sell from "propensity" to "what
actually changes the outcome."

- **Collections:** uplift modeling — predict who actually responds to
  intervention (vs. who'd self-cure regardless), so field/legal effort
  isn't wasted on accounts that didn't need it.
- **Cross-sell:** Next-Best-Offer (NBO) ranking across the full product
  catalog instead of a single accept/reject propensity, weighted by
  Customer Lifetime Value (CLV) so high-value customers aren't offered
  low-margin products.
- Risk-adjusted offer sizing/pricing tied back to the current Behavioral
  score.

### Phase 4 — Production concerns
**Goal:** what's needed to actually run this against a live portfolio.

- Champion-challenger A/B testing framework for collection/cross-sell
  strategies per segment.
- Model governance documentation aligned to RBI's digital lending / model
  risk management expectations.
- Alternative-data features (UPI/account-aggregator transaction data,
  utility bills, GST data for MSME) for thin-file/new-to-credit segments —
  if such data sources are available.

## Open decisions (need your input before Phase 1 is finalized)

1. **Data**: stay on synthetic data for now, or point this at a real
   bureau/on-book extract? If real, what's the source and what fields are
   actually available?
2. **Scope**: execute Phase 1 only, or commit now to also doing Phase 2/3
   enhancements before calling it "built"?
3. **Stack**: Python + scikit-learn (as scaffolded) assumed fine — flag if
   a different tool/language is required (e.g. SAS, R, a specific MLOps
   platform already in use).
4. **Fork logic**: confirm the Behavioral-score cutoff deciding
   Collections vs. Cross-sell should be a simple threshold (as scaffolded)
   or needs additional business rules (e.g. minimum vintage, product
   eligibility).

## Next step

Nothing executes past this document until you say go. On approval, the
plan is: review/finalize the Phase 1 scaffold already in the working tree,
run it, verify output, then commit the working code as its own change.
