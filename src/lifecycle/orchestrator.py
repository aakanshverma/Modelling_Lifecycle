"""End-to-end customer journey: Acquisition -> Behavioral -> (Collections | Cross-sell).

This is the piece that encodes the actual lifecycle you run in production:
score at the door with bureau data, book the good ones, watch how they
behave on book, then fork underperformers into collections and
performers into cross-sell.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lifecycle.data.synthetic import (
    generate_applicants,
    generate_collections_outcomes,
    generate_cross_sell_outcomes,
    generate_onbook_behavior,
)
from lifecycle.scoring.scorecard import ScorecardModel
from lifecycle.stages import acquisition, behavioral, collections, cross_sell


@dataclass
class LifecycleResult:
    applicants: pd.DataFrame
    acquisition_model: ScorecardModel
    behavioral_model: ScorecardModel
    collections_model: ScorecardModel
    cross_sell_model: ScorecardModel
    journey: pd.DataFrame


def run_lifecycle(n_applicants: int = 20_000, seed: int = 42) -> LifecycleResult:
    # Stage 1: Acquisition — score every applicant with bureau + application data.
    applicants = generate_applicants(n_applicants, seed=seed)
    acquisition_model = acquisition.fit(applicants)
    acquisition_scored = acquisition.score(acquisition_model, applicants)
    acquisition_decision = acquisition.decide(acquisition_scored)

    booked_ids = acquisition_scored.loc[acquisition_decision == "approve", "applicant_id"]
    booked = applicants[applicants["applicant_id"].isin(booked_ids)].reset_index(drop=True)

    # Stage 2: Behavioral — booked customers age on the books, generate real behavior.
    onbook = generate_onbook_behavior(booked, seed=seed + 1)
    behavioral_model = behavioral.fit(onbook)
    behavioral_scored = behavioral.score(behavioral_model, onbook)
    behavioral_route = behavioral.route(behavioral_scored)

    onbook = onbook.merge(
        behavioral_scored[["applicant_id", "behavioral_pd", "behavioral_score"]],
        on="applicant_id",
    )
    onbook["behavioral_route"] = behavioral_route.to_numpy()

    # Stage 3a: Collections — underperforming behavioral segment.
    collections_pop = onbook[onbook["behavioral_route"] == "collections"].reset_index(drop=True)
    collections_pop = generate_collections_outcomes(collections_pop, seed=seed + 2)
    collections_model = collections.fit(collections_pop)
    collections_scored = collections.score(collections_model, collections_pop)
    collections_action = collections.strategy(collections_scored)
    collections_scored["final_action"] = collections_action.to_numpy()

    # Stage 3b: Cross-sell — performing behavioral segment.
    cross_sell_pop = onbook[onbook["behavioral_route"] == "cross_sell"].reset_index(drop=True)
    cross_sell_pop = generate_cross_sell_outcomes(cross_sell_pop, seed=seed + 3)
    cross_sell_model = cross_sell.fit(cross_sell_pop)
    cross_sell_scored = cross_sell.score(cross_sell_model, cross_sell_pop)
    cross_sell_action = cross_sell.offer_tier(cross_sell_scored)
    cross_sell_scored["final_action"] = cross_sell_action.to_numpy()

    # Consolidate the full journey into one row per applicant.
    journey = acquisition_scored[["applicant_id", "acquisition_score", "acquisition_pd"]].copy()
    journey["acquisition_decision"] = acquisition_decision.to_numpy()

    journey = journey.merge(
        onbook[["applicant_id", "behavioral_score", "behavioral_pd", "behavioral_route"]],
        on="applicant_id",
        how="left",
    )
    journey = journey.merge(
        collections_scored[["applicant_id", "collections_score", "final_action"]],
        on="applicant_id",
        how="left",
    )
    journey = journey.merge(
        cross_sell_scored[["applicant_id", "cross_sell_score", "final_action"]],
        on="applicant_id",
        how="left",
        suffixes=("_collections", "_cross_sell"),
    )
    journey["final_action"] = journey["final_action_collections"].fillna(
        journey["final_action_cross_sell"]
    )
    journey = journey.drop(columns=["final_action_collections", "final_action_cross_sell"])

    return LifecycleResult(
        applicants=applicants,
        acquisition_model=acquisition_model,
        behavioral_model=behavioral_model,
        collections_model=collections_model,
        cross_sell_model=cross_sell_model,
        journey=journey,
    )
