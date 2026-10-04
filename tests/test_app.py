"""Smoke tests for the Streamlit dashboard's data-loading layer.

Doesn't test Streamlit rendering itself (that's verified by actually running
the app) — just that every cached loader function runs end-to-end and
returns the shape each page expects, so a page's data layer can't silently
break without a test catching it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data_loader as dl  # noqa: E402


def test_lifecycle_result_has_expected_shape():
    result = dl.get_lifecycle_result()
    assert len(result.journey) == dl.N_APPLICANTS
    assert {"acquisition_decision", "behavioral_route", "final_action"}.issubset(result.journey.columns)


def test_recompute_acquisition_decision_respects_cutoffs():
    scored = dl.recompute_acquisition_decision(approve_cutoff=600, refer_cutoff=570)
    assert set(scored["decision"].unique()) <= {"approve", "refer", "reject"}
    approved = scored[scored["decision"] == "approve"]
    assert (approved["acquisition_score"] >= 600).all()


def test_challenger_comparison_covers_all_four_stages():
    result = dl.get_challenger_comparison()
    assert set(result.keys()) == {"acquisition", "behavioral", "collections", "cross_sell"}
    for stage_result in result.values():
        assert stage_result["comparison"].challenger_auc > 0.5
        assert len(stage_result["importance"]) > 0


def test_reject_inference_result_shows_augmented_between_baseline_and_oracle():
    result = dl.get_reject_inference_result()
    aucs = result["aucs"]
    assert aucs["baseline (naive)"] < aucs["augmented (reject-inferred)"]
    assert aucs["augmented (reject-inferred)"] <= aucs["oracle (full true labels)"] + 1e-6


def test_altdata_lift_concentrated_in_thin_file():
    df = dl.get_altdata_lift()
    thin = df.set_index("segment").loc["thin-file", "lift"]
    thick = df.set_index("segment").loc["thick-file", "lift"]
    assert thin > thick


def test_roll_rate_data_has_transition_matrix_and_vintage_curve():
    result = dl.get_roll_rate_data()
    assert result["transition_matrix"].shape == (5, 5)
    assert result["vintage_curve"].shape[1] == 3  # low/mid/high score bands


def test_uplift_and_ab_test_result_is_internally_consistent():
    result = dl.get_uplift_and_ab_test()
    assert result["segment_means"]["persuadable"] > result["segment_means"]["sleeping_dog"]
    assert result["contact_result"].relative_lift < 0  # targeted policy contacts fewer people


def test_nbo_data_covers_all_products():
    result = dl.get_nbo_data()
    assert set(result["model_aucs"].keys()) == {"top_up_loan", "credit_card", "insurance"}
    assert result["ranked"]["next_best_offer"].nunique() >= 2


def test_psi_data_flags_drift_but_not_stable_cohort():
    result = dl.get_psi_data()
    assert result["stable"].psi < 0.1
    assert result["drift"].psi > result["stable"].psi


def test_collections_propensity_decile_direction_is_correct():
    """Guards the same decile-direction bug the dashboard caught in the library code:
    a higher collections score decile must mean a higher actual self-cure rate."""
    result = dl.get_collections_propensity_data()
    decile_cure = result["decile_actual_cure_rate"].sort_index()
    assert decile_cure.iloc[0] < decile_cure.iloc[-1]
    assert decile_cure.is_monotonic_increasing
