# NBFC Customer Lifecycle Modelling — Plan

Status: **All five phases executed, verified, and committed.**

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

**Status: done.** Built a generic, reusable champion/challenger A/B
significance test (`experimentation/ab_test.py`) and used it to evaluate
retiring Collections' blanket "contact everyone" policy for the Phase 3
uplift-targeted one: on a fresh randomized split, the targeted policy
cuts contact volume ~80% while giving up only ~2% relative cure rate
(~5x more cures per contact). Simulated bureau data being noisier for
thin-file applicants and blended in alternative-data features
(`generate_alt_data_features`, `fit_with_altdata`); the resulting AUC
lift is concentrated exactly where it should be (+1.6 points thin-file,
~0 thick-file). Wrote `GOVERNANCE.md` covering the model inventory,
validation approach, PSI-driven monitoring cadence, approval workflow,
and RBI digital-lending alignment. See `scripts/run_ab_test_demo.py`,
`scripts/run_altdata_demo.py`, and `tests/test_phase4.py`.

### Phase 5 — Front end
**Goal:** everything so far only speaks through terminal output
(`scripts/run_*_demo.py` print statements) and a CSV file. A risk team
or stakeholder needs a dashboard they can actually look at, not a
script they run and read off the console.

- **Stack: Streamlit.** The repo is all-Python (pandas/sklearn/XGBoost),
  the audience is a risk/analytics team, not web developers, and
  Streamlit is the standard tool for exactly this — turning a Python
  analytics pipeline into an interactive dashboard without a separate
  frontend build. No framework debate needed; this is the default choice
  for this kind of project.
- **Multipage app** under `app/`, one page per part of the lifecycle,
  all backed by the existing `src/lifecycle` library code (the dashboard
  presents what's already built; it doesn't reimplement any modeling
  logic):
  - **Overview** — the funnel: applicants -> approve/refer/reject ->
    booked -> behavioral route -> collections/cross-sell outcome, with
    stage AUCs.
  - **Acquisition** — score distribution by decision, champion vs.
    challenger (AUC/KS + SHAP importances), reject inference
    (baseline/augmented/oracle), alt-data lift (thin-file vs. thick-file).
    Interactive approve/refer cutoff sliders recompute the funnel live.
  - **Behavioral** — score distribution, routing split, roll-rate
    transition-matrix heatmap, vintage curves by acquisition score band.
  - **Collections** — self-cure propensity deciles, uplift model (true
    uplift by predicted decile, persuadable/sleeping-dog segments), the
    champion/challenger A/B test (blanket vs. targeted contact).
  - **Cross-sell** — per-product acceptance AUCs, Next-Best-Offer
    distribution and profile-by-offer, CLV distribution, risk-adjusted
    offer sizing.
  - **Monitoring** — PSI bin-level comparison (stable cohort vs.
    simulated downturn), with the stable/moderate/significant drift
    thresholds marked.
  - **Governance** — the model inventory and PSI monitoring cadence from
    `GOVERNANCE.md`, rendered rather than left in a file nobody opens.
- Heavy computations (fitting 10+ models) are cached per Streamlit
  session so navigating between pages doesn't refit everything.

**Status: done.** All 7 pages built, verified by actually launching the
app and checking each page renders with no exceptions (Playwright
screenshots). Run with `streamlit run app/Home.py`.

Building the Collections page's decile-validation chart caught a real,
pre-existing bug: the generic scorecard engine scales `score` assuming
the model's target is a *bad* outcome, which is backwards for
Collections' `self_cure_flag` and Cross-sell's `accepted_cross_sell`
(both *good* outcomes) — silently inverting `STRATEGY_BY_DECILE` and
`OFFER_BY_DECILE` since Phase 1 (best recovery prospects were getting
sent to field/legal, worst prospects to soft contact). Fixed at the root
in `scoring/scorecard.py` (`_decile` now built from `_pd`, not `_score`)
and covered by a regression test. See the README's "Phase 5 — dashboard"
section for the full writeup.

## Open decisions (resolved along the way)

1. **Data**: stayed on synthetic data throughout — no real bureau/on-book
   extract was provided. Every generator is isolated in
   `data/synthetic.py` specifically so a real extract can replace it
   without touching model code (see "Plugging in real data" in the
   README).
2. **Scope**: all five phases were executed, not just Phase 1.
3. **Stack**: Python + scikit-learn/XGBoost/SHAP/scipy for the modeling,
   Streamlit + Plotly for the dashboard — no objection was raised to
   change either.
4. **Fork logic**: the Behavioral-score cutoff deciding Collections vs.
   Cross-sell stayed a simple threshold; no additional business rules
   were requested.

## Next step

All 5 phases are built, tested (26 tests across `tests/test_pipeline.py`,
`tests/test_phase{2,3,4}.py`, and `tests/test_app.py`), and pushed. The
dashboard runs with `streamlit run app/Home.py`. What's left is entirely
about moving from "synthetic reference implementation" to "running
against your real portfolio": plugging in real bureau/on-book/alt-data
extracts, standing up the model risk committee process `GOVERNANCE.md`
describes, and deciding whether any Phase 2–4 enhancement (challenger
model, uplift targeting, NBO, alt data) is worth the added operational
complexity for your actual book — that's a business call, not a
modeling one.
