import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecycle.orchestrator import run_lifecycle  # noqa: E402

EXPECTED_JOURNEY_COLUMNS = {
    "applicant_id",
    "acquisition_score",
    "acquisition_pd",
    "acquisition_decision",
    "behavioral_score",
    "behavioral_pd",
    "behavioral_route",
    "collections_score",
    "cross_sell_score",
    "final_action",
}


def test_lifecycle_runs_end_to_end_and_each_stage_learns_signal():
    result = run_lifecycle(n_applicants=6_000, seed=7)

    assert EXPECTED_JOURNEY_COLUMNS.issubset(result.journey.columns)
    assert len(result.journey) == 6_000

    # Every stage should learn signal clearly better than a coin flip.
    assert result.acquisition_model.train_auc > 0.6
    assert result.behavioral_model.train_auc > 0.6
    assert result.collections_model.train_auc > 0.6
    assert result.cross_sell_model.train_auc > 0.6

    booked = result.journey[result.journey["acquisition_decision"] == "approve"]
    assert len(booked) > 0
    assert set(booked["behavioral_route"].dropna().unique()) <= {"collections", "cross_sell"}

    # Every booked customer lands in exactly one downstream action.
    assert booked["final_action"].notna().all()


def test_acquisition_decision_respects_score_cutoffs():
    result = run_lifecycle(n_applicants=4_000, seed=11)
    journey = result.journey

    approved = journey[journey["acquisition_decision"] == "approve"]
    rejected = journey[journey["acquisition_decision"] == "reject"]

    assert approved["acquisition_score"].min() >= rejected["acquisition_score"].max()
