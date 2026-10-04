import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecycle.data.synthetic import (  # noqa: E402
    generate_applicants,
    generate_collections_outcomes,
    generate_cross_sell_outcomes,
    generate_onbook_behavior,
)
from lifecycle.orchestrator import run_lifecycle  # noqa: E402
from lifecycle.stages import behavioral, collections, cross_sell  # noqa: E402

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


def test_decile_direction_matches_actual_propensity_for_good_outcome_targets():
    """Regression test for a real bug: `collections_score`/`cross_sell_score` are
    scaled so a HIGHER score means a LOWER P(target=1) — correct when target=1 is bad
    (Acquisition, Behavioral), but backwards when target=1 is good (Collections'
    self-cure, Cross-sell's acceptance). `_decile` must be built from `_pd` (which
    always correctly tracks P(target=1)), not from `_score`, or strategy/offer-tier
    assignment silently inverts: the worst recovery prospects get soft contact and the
    best get sent to field/legal, exactly backwards.
    """
    applicants = generate_applicants(8_000, seed=30)
    onbook = generate_onbook_behavior(applicants, seed=31)
    model = behavioral.fit(onbook)
    scored = behavioral.score(model, onbook)
    onbook["route"] = behavioral.route(scored).to_numpy()

    collections_pop = generate_collections_outcomes(
        onbook[onbook["route"] == "collections"].reset_index(drop=True), seed=44
    )
    collections_model = collections.fit(collections_pop)
    collections_scored = collections.score(collections_model, collections_pop)
    strategy = collections.strategy(collections_scored)
    cure_by_strategy = (
        collections_pop["self_cure_flag"].groupby(strategy.to_numpy()).mean()
    )
    assert cure_by_strategy["soft_contact"] > cure_by_strategy["tele_calling"]
    assert cure_by_strategy["tele_calling"] > cure_by_strategy["field_or_legal"]

    cross_sell_pop = generate_cross_sell_outcomes(
        onbook[onbook["route"] == "cross_sell"].reset_index(drop=True), seed=45
    )
    cross_sell_model = cross_sell.fit(cross_sell_pop)
    cross_sell_scored = cross_sell.score(cross_sell_model, cross_sell_pop)
    offer = cross_sell.offer_tier(cross_sell_scored)
    accept_by_offer = cross_sell_pop["accepted_cross_sell"].groupby(offer.to_numpy()).mean()
    assert accept_by_offer["top_tier_offer"] > accept_by_offer["standard_offer"]
    assert accept_by_offer["standard_offer"] > accept_by_offer["no_offer"]
