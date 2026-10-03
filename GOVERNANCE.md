# Model Risk Governance

How the models in this repo would actually be governed if deployed
against a live NBFC portfolio — model inventory, validation, monitoring
cadence, and the regulatory expectations (RBI's digital lending / model
risk management guidance) each piece is built to satisfy. This is a
template to adapt to your institution's actual committee structure and
risk appetite, not a substitute for your compliance/legal sign-off.

## Model inventory

| Model | Stage | Type | Target | Where |
|---|---|---|---|---|
| Acquisition champion | A | WOE + logistic regression | 12-month default | `stages/acquisition.py` |
| Acquisition challenger | A | XGBoost + SHAP | 12-month default | `scoring/challenger.py` |
| Acquisition (alt-data) | A | WOE + logistic regression | 12-month default | `stages/acquisition.fit_with_altdata` |
| Behavioral | B | WOE + logistic regression | forward 6m 90+ DPD | `stages/behavioral.py` |
| Collections propensity | C | WOE + logistic regression | self-cure | `stages/collections.py` |
| Collections uplift | C | T-learner (2x logistic) | contact treatment effect | `scoring/uplift.py` |
| Cross-sell propensity | D | WOE + logistic regression | offer acceptance | `stages/cross_sell.py` |
| Next-Best-Offer (x3) | D | WOE + logistic regression, per product | offer acceptance | `stages/next_best_offer.py` |

Every model above has a **champion** (interpretable: WOE + logistic
regression, scored to points) that is what actually drives a customer
decision, and in two places (Acquisition, and implicitly every stage via
`scoring/challenger.py`) a **challenger** used for benchmarking, not for
live decisions, unless and until it clears the promotion bar below.

## Validation approach

1. **Train/holdout separation.** Every model in this repo is validated on
   a population generated with a different random seed than training —
   never the training rows themselves. Any real deployment must do the
   same with an actual out-of-time holdout (a later vintage), not just an
   out-of-sample one, since population drift is the risk that actually
   bites in production.
2. **Discrimination metrics.** AUC and KS (`scoring/challenger.ks_statistic`)
   are computed on holdout for every champion/challenger pair. A
   challenger needs a material, consistent AUC/KS improvement — not a
   fluke on one seed — before it's even considered for promotion.
3. **Reject inference is itself validated**, not just applied —
   `scripts/run_reject_inference_demo.py` shows the augmented model's
   holdout AUC against an oracle benchmark. In production you don't have
   that oracle; the substitute is tracking the augmented model's live
   performance for several months before fully trusting it over the
   naive one.
4. **Independent review.** Whoever builds a scorecard should not be the
   sole sign-off on deploying it. At minimum: a second analyst
   reproduces the holdout metrics from the training code and data before
   a model moves from "challenger" to "champion."

## Monitoring cadence

Tied directly to `monitoring/psi.py`'s severity bands:

| PSI | Severity | Required action | Cadence |
|---|---|---|---|
| < 0.10 | stable | none | monthly PSI check |
| 0.10–0.25 | moderate drift | investigate root cause (macro shift? acquisition policy change? seasonality?) within 2 weeks | immediate review, then monitor weekly |
| > 0.25 | significant drift | freeze further reliance on the current scorecard for new decisions; begin a refit using recent vintage data | immediate escalation |

Behavioral, Collections, and Cross-sell scores should be monitored on the
same cadence as Acquisition — a drifted behavioral score silently breaks
the Collections/Cross-sell fork even if nobody touched those models
directly, since they run on behaviorally-routed populations you no longer
understand.

**Champion-challenger rollout** (`experimentation/ab_test.py`) is the
mechanism for changing a live policy, never a direct swap. Before
retiring a champion policy (e.g. "contact everyone" in Collections), run
a randomized split against the challenger policy and require statistical
significance (the module's `significant` flag) on the metric that
actually matters for the business decision — note that in
`scripts/run_ab_test_demo.py`, cure-rate lift and contact-volume lift are
each independently significant, and the deployment decision explicitly
weighs both rather than optimizing either alone.

## Approval workflow (template)

1. **Build** — model developer builds and documents the model (features,
   target definition, training window, exclusions).
2. **Independent validation** — a second reviewer reproduces holdout
   metrics and checks for an IV floor or feature leakage (e.g. is any
   feature only knowable *after* the outcome — see `scoring/woe.py`'s IV
   floor as the first automatic screen, not the only one).
3. **Business sign-off** — the stage owner (origination risk, collections
   ops, cross-sell/marketing) confirms the decision rule (cutoffs,
   segment definitions) matches what they're prepared to act on, and that
   expected approval/contact/offer volumes are operationally feasible.
4. **Champion-challenger pilot** — for any model replacing a *live*
   policy, run an A/B pilot (see above) before full rollout, not a
   backtest alone.
5. **Model risk committee sign-off** — formal approval recorded before
   production use, including the model's intended use, known
   limitations, and the monitoring plan that applies to it.
6. **Periodic re-validation** — annually at minimum, or immediately on a
   significant-drift PSI trigger, whichever comes first.

## Regulatory alignment (RBI digital lending expectations)

- **Explainability of the decision.** The customer-facing decision at
  every stage comes from the WOE + logistic-regression champion, which
  decomposes into a human-readable points breakdown per feature. The
  XGBoost challenger is explicitly benchmarking-only in this repo; if a
  black-box model is ever used for a live decision, RBI's digital lending
  guidelines expect you to be able to explain *why* a specific customer
  was declined or priced a certain way, not just a SHAP plot in a model
  validation deck.
- **Fair Practice Code.** Acquisition cutoffs, behavioral routing, and
  collections strategy assignment must all be disclosed in the loan
  agreement / customer communication in substance (what data is used, in
  general terms) even if the scorecard coefficients themselves aren't
  published.
- **Consent for data pulls.** Bureau pulls and — especially — the
  alternative data described in `data/synthetic.generate_alt_data_features`
  (UPI transaction history, utility bill payment) require explicit,
  specific customer consent, generally via the Account Aggregator
  framework for financial data. "We already had the data" is not
  sufficient; consent must be scoped to the actual use (credit
  underwriting), not blanket.
- **Grievance redressal.** A customer declined or routed to collections
  based on a model must have a route to question/appeal the decision —
  this is a process requirement outside the modeling code, but the model
  documentation above (feature list, cutoff, decision date) is exactly
  what a grievance-redressal review would need to pull first.
- **Non-discrimination.** None of the features used anywhere in this repo
  are protected attributes (religion, caste, gender, etc.) or close
  proxies for them. Before using any new feature — especially from
  alternative data sources, which can proxy for protected attributes in
  non-obvious ways (e.g. certain merchant categories correlating with
  religious practice) — run an explicit disparate-impact check before
  approval, not just an IV check.

## What this repo does *not* cover

This is a modeling and governance-process reference, not a compliance
sign-off. Real deployment additionally needs: legal review of the loan
agreement and consent language, data retention/deletion policy for
alternative data, an actual model risk committee charter, and IT/security
review of how bureau and Account Aggregator data is stored and accessed.
