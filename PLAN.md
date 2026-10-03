# NBFC Customer Lifecycle Modelling — Plan

Status: **Phase 1, Phase 2, and Phase 3 executed, verified, and
committed.** Phase 4 has not started.

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

**Status: done.** Verified end-to-end (tests pass, demo script runs on
20k synthetic applicants) and committed. Along the way, fixed a
calibration bug where the synthetic default rate was unrealistically high
(43%), which pushed every applicant's score below the approve cutoff and
left the booked population empty — recalibrated to a ~13% bad rate and
re-tuned the acquisition cutoffs against the actual score distribution.

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

**Status: done.** All four pieces implemented, validated, and tested:
reject inference closes the AUC gap toward an oracle model (0.70 → 0.73
toward 0.76 in one run), the DPD panel + vintage curve correctly show a
lower acquisition score band reaching NPA faster, the XGBoost challenger
edges out the champion on holdout AUC/KS across all 4 stages, and PSI
correctly stays near 0 for a same-distribution cohort but flags ~0.18
(moderate drift) after a simulated downturn. See `scripts/run_*_demo.py`
and `tests/test_phase2.py`.

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

**Status: done.** A randomized collections-contact experiment with a
genuine heterogeneous treatment effect (persuadables, sleeping dogs) was
simulated; the average treatment effect alone is small/unremarkable
(~1-2pp), but a T-learner uplift model recovers the heterogeneity
underneath (0.80 Spearman correlation with ground truth; quantile
segments cleanly separate persuadable from sleeping-dog accounts). The
cross-sell catalog was expanded to 3 products (top-up loan, credit card,
insurance) each appealing to a different profile by design; NBO ranking
by expected value (propensity x margin x CLV) differentiates the catalog
sensibly instead of collapsing onto one product, and a risk-adjusted
offer-sizing function ties the top-up-loan amount back to the current
behavioral PD. See `scripts/run_uplift_demo.py`,
`scripts/run_next_best_offer_demo.py`, and `tests/test_phase3.py`.

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
