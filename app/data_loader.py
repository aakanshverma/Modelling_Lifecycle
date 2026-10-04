"""Cached data-loading layer for the Streamlit dashboard.

Every function here just calls into `src/lifecycle` and packages the
result for display — none of the modeling logic lives here. Heavy
computations (fitting 10+ models across all four phases) are wrapped in
`st.cache_data` so switching between pages doesn't refit everything on
every rerun, which is how Streamlit scripts normally behave.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from lifecycle.analytics.clv import estimate_clv  # noqa: E402
from lifecycle.analytics.roll_rate import compute_transition_matrix, compute_vintage_curve  # noqa: E402
from lifecycle.data.synthetic import (  # noqa: E402
    degrade_bureau_for_thin_file,
    generate_alt_data_features,
    generate_applicants,
    generate_collections_experiment,
    generate_collections_outcomes,
    generate_cross_sell_outcomes,
    generate_dpd_panel,
    generate_nbo_outcomes,
    generate_onbook_behavior,
    simulate_collections_policy_outcome,
    true_collections_uplift,
)
from lifecycle.experimentation.ab_test import ABTestResult, assign_arm, evaluate_ab_test  # noqa: E402
from lifecycle.monitoring.psi import population_stability_index  # noqa: E402
from lifecycle.orchestrator import LifecycleResult, run_lifecycle  # noqa: E402
from lifecycle.scoring.challenger import compare_champion_challenger, fit_challenger  # noqa: E402
from lifecycle.scoring.reject_inference import run_reject_inference  # noqa: E402
from lifecycle.scoring.scorecard import fit_scorecard  # noqa: E402
from lifecycle.scoring.uplift import assign_uplift_segment, fit_uplift_model  # noqa: E402
from lifecycle.stages import acquisition, behavioral, collections, cross_sell  # noqa: E402
from lifecycle.stages.next_best_offer import (  # noqa: E402
    fit_product_models,
    rank_next_best_offer,
    risk_adjusted_offer_amount,
    score_products,
)

N_APPLICANTS = 20_000
SEED = 42


@st.cache_data(show_spinner="Running the full ABCD lifecycle...")
def get_lifecycle_result() -> LifecycleResult:
    return run_lifecycle(n_applicants=N_APPLICANTS, seed=SEED)


def recompute_acquisition_decision(approve_cutoff: int, refer_cutoff: int) -> pd.DataFrame:
    """Cheap recompute for the interactive cutoff sliders — no refit needed."""
    result = get_lifecycle_result()
    scored = acquisition.score(result.acquisition_model, result.applicants)
    decision = acquisition.decide(scored, approve_cutoff=approve_cutoff, refer_cutoff=refer_cutoff)
    return scored.assign(decision=decision.to_numpy())


@st.cache_data(show_spinner="Benchmarking XGBoost challengers against each stage's champion...")
def get_challenger_comparison() -> dict:
    applicants = generate_applicants(N_APPLICANTS, seed=SEED)
    onbook = generate_onbook_behavior(applicants, seed=43)
    collections_pop = generate_collections_outcomes(onbook, seed=44)
    cross_sell_pop = generate_cross_sell_outcomes(onbook, seed=45)

    holdout_applicants = generate_applicants(N_APPLICANTS, seed=142)
    holdout_onbook = generate_onbook_behavior(holdout_applicants, seed=143)
    holdout_collections = generate_collections_outcomes(holdout_onbook, seed=144)
    holdout_cross_sell = generate_cross_sell_outcomes(holdout_onbook, seed=145)

    stage_modules = {
        "acquisition": (acquisition, applicants, holdout_applicants),
        "behavioral": (behavioral, onbook, holdout_onbook),
        "collections": (collections, collections_pop, holdout_collections),
        "cross_sell": (cross_sell, cross_sell_pop, holdout_cross_sell),
    }

    results = {}
    for name, (module, train, holdout) in stage_modules.items():
        champion = fit_scorecard(name, train[module.FEATURES], train[module.TARGET])
        challenger = fit_challenger(f"{name}_xgb", train[module.FEATURES], train[module.TARGET])
        cmp = compare_champion_challenger(
            champion, challenger, holdout[module.FEATURES], holdout[module.TARGET]
        )
        importance = challenger.shap_importance(holdout[module.FEATURES])
        results[name] = {"comparison": cmp, "importance": importance}
    return results


@st.cache_data(show_spinner="Running reject inference...")
def get_reject_inference_result() -> dict:
    train = generate_applicants(N_APPLICANTS, seed=100)
    holdout = generate_applicants(N_APPLICANTS, seed=200)

    known_mask = train["cibil_score"] >= 700
    known, unknown = train[known_mask], train[~known_mask]

    result = run_reject_inference(known[acquisition.FEATURES], known[acquisition.TARGET], unknown[acquisition.FEATURES], seed=1)
    oracle = fit_scorecard("oracle", train[acquisition.FEATURES], train[acquisition.TARGET])

    aucs = {}
    for label, model in [
        ("baseline (naive)", result.baseline_model),
        ("augmented (reject-inferred)", result.augmented_model),
        ("oracle (full true labels)", oracle),
    ]:
        pd_hat = model.predict(holdout[acquisition.FEATURES])[f"{model.name}_pd"]
        aucs[label] = roc_auc_score(holdout[acquisition.TARGET], pd_hat)

    return {
        "known_frac": known_mask.mean(),
        "known_bad_rate": known[acquisition.TARGET].mean(),
        "unknown_true_bad_rate": unknown[acquisition.TARGET].mean(),
        "inferred_bad_rate": result.inferred_bad_rate,
        "aucs": aucs,
    }


@st.cache_data(show_spinner="Simulating the alt-data thin-file comparison...")
def get_altdata_lift() -> pd.DataFrame:
    train_applicants = generate_applicants(N_APPLICANTS, seed=42)
    train = degrade_bureau_for_thin_file(train_applicants, seed=49).merge(
        generate_alt_data_features(train_applicants, seed=48), on="applicant_id"
    )
    holdout_applicants = generate_applicants(N_APPLICANTS, seed=142)
    holdout = degrade_bureau_for_thin_file(holdout_applicants, seed=149).merge(
        generate_alt_data_features(holdout_applicants, seed=148), on="applicant_id"
    )

    bureau_model = acquisition.fit(train)
    altdata_model = acquisition.fit_with_altdata(train)

    rows = []
    for label, mask in [("thin-file", holdout["thin_file"]), ("thick-file", ~holdout["thin_file"])]:
        segment = holdout[mask]
        bureau_pd = bureau_model.predict(segment[acquisition.FEATURES])["acquisition_pd"]
        altdata_pd = altdata_model.predict(segment[acquisition.ALT_FEATURES])["acquisition_altdata_pd"]
        bureau_auc = roc_auc_score(segment["default_flag"], bureau_pd)
        altdata_auc = roc_auc_score(segment["default_flag"], altdata_pd)
        rows.append(
            {
                "segment": label,
                "n": len(segment),
                "bureau_only_auc": bureau_auc,
                "bureau_plus_altdata_auc": altdata_auc,
                "lift": altdata_auc - bureau_auc,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Simulating the DPD panel and vintage curves...")
def get_roll_rate_data() -> dict:
    result = get_lifecycle_result()
    booked = result.journey[result.journey["acquisition_decision"] == "approve"]
    panel = generate_dpd_panel(booked, risk_col="behavioral_pd", n_months=12, seed=55)

    transition_matrix = compute_transition_matrix(panel)
    score_band = pd.qcut(
        booked.set_index("applicant_id")["acquisition_score"],
        3,
        labels=["low_score", "mid_score", "high_score"],
    )
    vintage = compute_vintage_curve(panel, score_band)
    return {"transition_matrix": transition_matrix, "vintage_curve": vintage}


@st.cache_data(show_spinner="Training the uplift model and running the champion/challenger test...")
def get_uplift_and_ab_test() -> dict:
    applicants = generate_applicants(N_APPLICANTS, seed=42)
    onbook = generate_onbook_behavior(applicants, seed=43)
    history = generate_collections_experiment(onbook, seed=46)

    model = fit_uplift_model(
        history[collections.FEATURES], history["contacted"], history["self_cure_flag"]
    )
    predicted_uplift = model.predict_uplift(onbook[collections.FEATURES])
    true_uplift = true_collections_uplift(onbook)

    decile_df = pd.DataFrame({"predicted": predicted_uplift, "true": true_uplift})
    decile_df["decile"] = pd.qcut(decile_df["predicted"], 10, labels=False, duplicates="drop")
    decile_means = decile_df.groupby("decile")["true"].mean()

    segment = assign_uplift_segment(predicted_uplift)
    segment_means = pd.DataFrame({"segment": segment, "true_uplift": true_uplift}).groupby("segment")[
        "true_uplift"
    ].mean()
    segment_counts = segment.value_counts()

    # Champion/challenger rollout A/B test.
    arm = assign_arm(onbook.index, challenger_rate=0.5, seed=99)
    contacted_policy = pd.Series(0, index=onbook.index)
    contacted_policy[arm == "champion"] = 1
    contacted_policy[(arm == "challenger") & (segment == "persuadable")] = 1

    self_cure = simulate_collections_policy_outcome(onbook, contacted_policy, seed=46)
    contact_made = pd.Series(contacted_policy.to_numpy(), index=onbook.index)

    cure_result: ABTestResult = evaluate_ab_test(self_cure, arm)
    contact_result: ABTestResult = evaluate_ab_test(contact_made, arm)

    return {
        "decile_means": decile_means,
        "segment_means": segment_means,
        "segment_counts": segment_counts,
        "cure_result": cure_result,
        "contact_result": contact_result,
    }


@st.cache_data(show_spinner="Fitting Next-Best-Offer product models...")
def get_nbo_data() -> dict:
    applicants = generate_applicants(N_APPLICANTS, seed=42)
    acquisition_model = acquisition.fit(applicants)
    acquisition_scored = acquisition.score(acquisition_model, applicants)
    booked = applicants[acquisition.decide(acquisition_scored) == "approve"]

    onbook = generate_onbook_behavior(booked, seed=43)
    behavioral_model = behavioral.fit(onbook)
    behavioral_scored = behavioral.score(behavioral_model, onbook)
    onbook = onbook.merge(behavioral_scored[["applicant_id", "behavioral_pd"]], on="applicant_id")
    onbook["route"] = behavioral.route(behavioral_scored).to_numpy()

    cross_sell_pop = onbook[onbook["route"] == "cross_sell"].reset_index(drop=True)
    cross_sell_pop = generate_nbo_outcomes(cross_sell_pop, seed=47)

    models = fit_product_models(cross_sell_pop)
    scores = score_products(models, cross_sell_pop)
    clv = estimate_clv(cross_sell_pop, cross_sell_pop["behavioral_pd"])
    ranked = rank_next_best_offer(scores, clv)

    top_up_mask = ranked["next_best_offer"] == "top_up_loan"
    offer_amount = risk_adjusted_offer_amount(
        cross_sell_pop.loc[top_up_mask], cross_sell_pop.loc[top_up_mask, "behavioral_pd"]
    )

    profile = pd.concat(
        [cross_sell_pop[["age", "monthly_income", "existing_emi_to_income"]], ranked], axis=1
    )

    return {
        "model_aucs": {p: m.train_auc for p, m in models.items()},
        "clv": clv,
        "ranked": ranked,
        "profile": profile,
        "offer_amount": offer_amount,
        "segment_size": len(cross_sell_pop),
    }


@st.cache_data(show_spinner="Computing PSI drift comparison...")
def get_psi_data() -> dict:
    baseline = generate_applicants(N_APPLICANTS, seed=42)
    model = acquisition.fit(baseline)
    baseline_scored = acquisition.score(model, baseline)

    same_dist = generate_applicants(N_APPLICANTS, seed=142)
    same_dist_scored = acquisition.score(model, same_dist)
    stable_result = population_stability_index(
        baseline_scored["acquisition_score"], same_dist_scored["acquisition_score"]
    )

    downturn = generate_applicants(N_APPLICANTS, seed=242)
    downturn["credit_utilization"] = (downturn["credit_utilization"] * 1.25).clip(0, 1)
    downturn["existing_emi_to_income"] = (downturn["existing_emi_to_income"] * 1.3).clip(0, 1)
    downturn["dpd_30_plus_last_6m"] = downturn["dpd_30_plus_last_6m"] + 1
    downturn_scored = acquisition.score(model, downturn)
    drift_result = population_stability_index(
        baseline_scored["acquisition_score"], downturn_scored["acquisition_score"]
    )

    return {"stable": stable_result, "drift": drift_result}


@st.cache_data(show_spinner="Scoring the collections self-cure propensity model...")
def get_collections_propensity_data() -> dict:
    applicants = generate_applicants(N_APPLICANTS, seed=42)
    onbook = generate_onbook_behavior(applicants, seed=43)
    behavioral_model = behavioral.fit(onbook)
    behavioral_scored = behavioral.score(behavioral_model, onbook)
    onbook["route"] = behavioral.route(behavioral_scored).to_numpy()

    collections_pop = onbook[onbook["route"] == "collections"].reset_index(drop=True)
    collections_pop = generate_collections_outcomes(collections_pop, seed=44)

    model = collections.fit(collections_pop)
    scored = collections.score(model, collections_pop)
    strategy = collections.strategy(scored)

    decile_actual = pd.DataFrame(
        {"decile": scored["collections_decile"], "self_cure_flag": collections_pop["self_cure_flag"]}
    ).groupby("decile")["self_cure_flag"].mean()

    return {
        "segment_size": len(collections_pop),
        "train_auc": model.train_auc,
        "decile_actual_cure_rate": decile_actual,
        "strategy_counts": strategy.value_counts(),
    }
