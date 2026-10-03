import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecycle.analytics.clv import estimate_clv  # noqa: E402
from lifecycle.data.synthetic import (  # noqa: E402
    generate_applicants,
    generate_collections_experiment,
    generate_nbo_outcomes,
    generate_onbook_behavior,
    true_collections_uplift,
)
from lifecycle.scoring.uplift import assign_uplift_segment, fit_uplift_model  # noqa: E402
from lifecycle.stages import acquisition, behavioral  # noqa: E402
from lifecycle.stages.collections import FEATURES as COLLECTIONS_FEATURES  # noqa: E402
from lifecycle.stages.next_best_offer import (  # noqa: E402
    PRODUCTS,
    fit_product_models,
    rank_next_best_offer,
    score_products,
)


def _booked_onbook(seed: int = 7, n: int = 8_000):
    applicants = generate_applicants(n, seed=seed)
    model = acquisition.fit(applicants)
    scored = acquisition.score(model, applicants)
    booked = applicants[acquisition.decide(scored) == "approve"]
    return generate_onbook_behavior(booked, seed=seed + 1)


def test_uplift_model_ranking_correlates_with_true_uplift():
    onbook = _booked_onbook(seed=10)
    experiment = generate_collections_experiment(onbook, seed=46)

    model = fit_uplift_model(
        experiment[COLLECTIONS_FEATURES], experiment["contacted"], experiment["self_cure_flag"]
    )
    predicted = model.predict_uplift(onbook[COLLECTIONS_FEATURES])
    true_uplift = true_collections_uplift(onbook)

    corr = predicted.corr(true_uplift, method="spearman")
    assert corr > 0.5  # a weak/no correlation would mean the model learned nothing useful

    decile = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    decile_means = true_uplift.groupby(decile).mean()
    # Top predicted-uplift decile should have the highest true uplift, bottom the lowest.
    assert decile_means.idxmax() == decile_means.index.max()
    assert decile_means.idxmin() == decile_means.index.min()


def test_uplift_segments_separate_persuadables_from_sleeping_dogs():
    onbook = _booked_onbook(seed=11)
    experiment = generate_collections_experiment(onbook, seed=46)

    model = fit_uplift_model(
        experiment[COLLECTIONS_FEATURES], experiment["contacted"], experiment["self_cure_flag"]
    )
    predicted = model.predict_uplift(onbook[COLLECTIONS_FEATURES])
    true_uplift = true_collections_uplift(onbook)

    segment = assign_uplift_segment(predicted)
    by_segment = pd.DataFrame({"segment": segment, "true_uplift": true_uplift}).groupby("segment")[
        "true_uplift"
    ].mean()

    assert by_segment["persuadable"] > by_segment["sure_thing_or_lost_cause"]
    assert by_segment["sure_thing_or_lost_cause"] > by_segment["sleeping_dog"]
    assert by_segment["sleeping_dog"] < 0  # contact should look actively harmful here


def test_randomized_experiment_has_near_zero_average_but_real_heterogeneity():
    onbook = _booked_onbook(seed=12)
    experiment = generate_collections_experiment(onbook, seed=46)

    ate = (
        experiment.loc[experiment["contacted"] == 1, "self_cure_flag"].mean()
        - experiment.loc[experiment["contacted"] == 0, "self_cure_flag"].mean()
    )
    true_uplift = true_collections_uplift(onbook)

    # The average treatment effect is small/unremarkable...
    assert abs(ate) < 0.08
    # ...even though real heterogeneity exists underneath (persuadables and sleeping dogs).
    assert true_uplift.max() > 0.1
    assert true_uplift.min() < -0.1


def test_next_best_offer_differentiates_the_catalog_by_customer_profile():
    onbook = _booked_onbook(seed=13)
    model = behavioral.fit(onbook)
    scored = behavioral.score(model, onbook)
    onbook = onbook.merge(scored[["applicant_id", "behavioral_pd"]], on="applicant_id")
    onbook["route"] = behavioral.route(scored).to_numpy()

    cross_sell_pop = onbook[onbook["route"] == "cross_sell"].reset_index(drop=True)
    cross_sell_pop = generate_nbo_outcomes(cross_sell_pop, seed=47)

    models = fit_product_models(cross_sell_pop)
    for product in PRODUCTS:
        assert models[product].train_auc > 0.6

    scores = score_products(models, cross_sell_pop)
    clv = estimate_clv(cross_sell_pop, cross_sell_pop["behavioral_pd"])
    ranked = rank_next_best_offer(scores, clv)

    # The catalog must not collapse onto a single product for everyone.
    assert ranked["next_best_offer"].nunique() >= 2
    assert (ranked["expected_value"] > 0).all()

    # Credit card should skew younger than the top-up loan segment (by design).
    profile = cross_sell_pop.assign(offer=ranked["next_best_offer"].to_numpy())
    if (profile["offer"] == "credit_card").sum() > 5 and (profile["offer"] == "top_up_loan").sum() > 5:
        assert (
            profile.loc[profile["offer"] == "credit_card", "age"].mean()
            < profile.loc[profile["offer"] == "top_up_loan", "age"].mean()
        )
