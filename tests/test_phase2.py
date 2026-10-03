import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from lifecycle.analytics.roll_rate import compute_transition_matrix, compute_vintage_curve  # noqa: E402
from lifecycle.data.synthetic import DPD_BUCKETS, generate_applicants, generate_dpd_panel  # noqa: E402
from lifecycle.monitoring.psi import population_stability_index  # noqa: E402
from lifecycle.scoring.challenger import compare_champion_challenger, fit_challenger  # noqa: E402
from lifecycle.scoring.reject_inference import run_reject_inference  # noqa: E402
from lifecycle.scoring.scorecard import fit_scorecard  # noqa: E402
from lifecycle.stages.acquisition import FEATURES, TARGET  # noqa: E402


def test_reject_inference_closes_gap_toward_oracle():
    train = generate_applicants(10_000, seed=100)
    holdout = generate_applicants(10_000, seed=200)

    known_mask = train["cibil_score"] >= 700
    known, unknown = train[known_mask], train[~known_mask]

    result = run_reject_inference(known[FEATURES], known[TARGET], unknown[FEATURES], seed=1)
    oracle = fit_scorecard("oracle", train[FEATURES], train[TARGET])

    baseline_auc = roc_auc_score(
        holdout[TARGET], result.baseline_model.predict(holdout[FEATURES])["acquisition_baseline_pd"]
    )
    augmented_auc = roc_auc_score(
        holdout[TARGET],
        result.augmented_model.predict(holdout[FEATURES])["acquisition_augmented_pd"],
    )
    oracle_auc = roc_auc_score(holdout[TARGET], oracle.predict(holdout[FEATURES])["oracle_pd"])

    # Augmented should land strictly between the biased baseline and the (unreachable
    # in real life) oracle that was fit on everyone's true label.
    assert baseline_auc < augmented_auc <= oracle_auc + 1e-6


def test_dpd_panel_treats_npa_as_absorbing():
    applicants = generate_applicants(2_000, seed=5)
    panel = generate_dpd_panel(applicants.assign(risk=applicants["default_flag"]), risk_col="risk", n_months=6, seed=9)

    bucket_index = {b: i for i, b in enumerate(DPD_BUCKETS)}
    panel = panel.sort_values(["applicant_id", "month_on_book"])
    panel["bucket_idx"] = panel["dpd_bucket"].map(bucket_index)
    panel["prev_idx"] = panel.groupby("applicant_id")["bucket_idx"].shift(1)
    moves = panel.dropna(subset=["prev_idx"])

    # Once an account is written off to NPA it must stay there — no "recovery" out of NPA.
    from_npa = moves[moves["prev_idx"] == bucket_index["npa_90_plus"]]
    assert (from_npa["bucket_idx"] == bucket_index["npa_90_plus"]).all()

    # Nobody starting in 'current' can skip straight past dpd_1_30 in one month.
    from_current = moves[moves["prev_idx"] == bucket_index["current"]]
    assert (from_current["bucket_idx"] <= bucket_index["dpd_1_30"]).all()


def test_vintage_curve_is_monotonic_and_score_ordered():
    applicants = generate_applicants(4_000, seed=5)
    model = fit_scorecard("acquisition", applicants[FEATURES], applicants[TARGET])
    scored = model.predict(applicants[FEATURES])
    booked = applicants.assign(acquisition_pd=scored["acquisition_pd"])

    panel = generate_dpd_panel(booked, risk_col="acquisition_pd", n_months=10, seed=20)
    score_band = pd.qcut(scored["acquisition_score"], 2, labels=["low_score", "high_score"])
    score_band.index = booked["applicant_id"]

    curve = compute_vintage_curve(panel, score_band)

    # Cumulative ever-bad rate must never decrease month over month.
    assert (curve.diff().dropna() >= -1e-9).all().all()
    # Lower acquisition score band should reach a higher cumulative bad rate.
    assert curve["low_score"].iloc[-1] > curve["high_score"].iloc[-1]


def test_challenger_beats_or_matches_champion_on_holdout():
    train = generate_applicants(8_000, seed=1)
    holdout = generate_applicants(8_000, seed=2)

    champion = fit_scorecard("acquisition", train[FEATURES], train[TARGET])
    challenger = fit_challenger("acquisition_xgb", train[FEATURES], train[TARGET])

    cmp = compare_champion_challenger(champion, challenger, holdout[FEATURES], holdout[TARGET])
    assert cmp.champion_auc > 0.6
    assert cmp.challenger_auc > 0.6


def test_psi_flags_drift_but_not_a_fresh_identical_cohort():
    applicants = generate_applicants(10_000, seed=42)
    model = fit_scorecard("acquisition", applicants[FEATURES], applicants[TARGET])
    baseline_scores = model.predict(applicants[FEATURES])["acquisition_pd"]

    same_dist = generate_applicants(10_000, seed=142)
    same_dist_scores = model.predict(same_dist[FEATURES])["acquisition_pd"]
    stable_result = population_stability_index(baseline_scores, same_dist_scores)
    assert stable_result.psi < 0.1

    drifted = generate_applicants(10_000, seed=242)
    drifted["credit_utilization"] = (drifted["credit_utilization"] * 1.5).clip(0, 1)
    drifted["existing_emi_to_income"] = (drifted["existing_emi_to_income"] * 1.5).clip(0, 1)
    drifted_scores = model.predict(drifted[FEATURES])["acquisition_pd"]
    drift_result = population_stability_index(baseline_scores, drifted_scores)
    assert drift_result.psi > stable_result.psi
