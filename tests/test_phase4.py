import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from lifecycle.data.synthetic import (  # noqa: E402
    degrade_bureau_for_thin_file,
    generate_alt_data_features,
    generate_applicants,
    generate_collections_experiment,
    generate_onbook_behavior,
    simulate_collections_policy_outcome,
)
from lifecycle.experimentation.ab_test import assign_arm, evaluate_ab_test  # noqa: E402
from lifecycle.scoring.uplift import assign_uplift_segment, fit_uplift_model  # noqa: E402
from lifecycle.stages import acquisition  # noqa: E402
from lifecycle.stages.collections import FEATURES as COLLECTIONS_FEATURES  # noqa: E402


def test_ab_test_detects_a_planted_difference():
    outcome = pd.Series([1] * 500 + [0] * 500)
    arm = pd.Series(["champion"] * 500 + ["challenger"] * 500)
    result = evaluate_ab_test(outcome, arm)

    assert result.champion_mean == 1.0
    assert result.challenger_mean == 0.0
    assert result.significant


def test_ab_test_does_not_flag_identical_arms_as_significant():
    import numpy as np

    rng = np.random.default_rng(0)
    outcome = pd.Series(rng.binomial(1, 0.5, 4000))
    arm = assign_arm(outcome.index, challenger_rate=0.5, seed=1)
    result = evaluate_ab_test(outcome, arm)

    assert not result.significant


def test_uplift_targeted_policy_cuts_contacts_with_small_cure_rate_cost():
    applicants = generate_applicants(10_000, seed=20)
    onbook = generate_onbook_behavior(applicants, seed=21)
    history = generate_collections_experiment(onbook, seed=46)

    model = fit_uplift_model(
        history[COLLECTIONS_FEATURES], history["contacted"], history["self_cure_flag"]
    )
    segment = assign_uplift_segment(model.predict_uplift(onbook[COLLECTIONS_FEATURES]))

    arm = assign_arm(onbook.index, challenger_rate=0.5, seed=99)
    contacted_policy = pd.Series(0, index=onbook.index)
    contacted_policy[arm == "champion"] = 1
    contacted_policy[(arm == "challenger") & (segment == "persuadable")] = 1

    self_cure = simulate_collections_policy_outcome(onbook, contacted_policy, seed=46)
    contact_made = pd.Series(contacted_policy.to_numpy(), index=onbook.index)

    cure_result = evaluate_ab_test(self_cure, arm)
    contact_result = evaluate_ab_test(contact_made, arm)

    # The targeted challenger should contact far fewer people...
    assert contact_result.relative_lift < -0.5
    # ...while giving up only a modest slice of the cure rate.
    assert -0.1 < cure_result.relative_lift < 0


def test_altdata_lift_is_concentrated_in_thin_file_segment():
    train_applicants = generate_applicants(10_000, seed=42)
    train = degrade_bureau_for_thin_file(train_applicants, seed=49).merge(
        generate_alt_data_features(train_applicants, seed=48), on="applicant_id"
    )
    holdout_applicants = generate_applicants(10_000, seed=142)
    holdout = degrade_bureau_for_thin_file(holdout_applicants, seed=149).merge(
        generate_alt_data_features(holdout_applicants, seed=148), on="applicant_id"
    )

    bureau_model = acquisition.fit(train)
    altdata_model = acquisition.fit_with_altdata(train)

    def auc_lift(mask):
        segment = holdout[mask]
        bureau_pd = bureau_model.predict(segment[acquisition.FEATURES])["acquisition_pd"]
        altdata_pd = altdata_model.predict(segment[acquisition.ALT_FEATURES])[
            "acquisition_altdata_pd"
        ]
        bureau_auc = roc_auc_score(segment["default_flag"], bureau_pd)
        altdata_auc = roc_auc_score(segment["default_flag"], altdata_pd)
        return altdata_auc - bureau_auc

    thin_lift = auc_lift(holdout["thin_file"])
    thick_lift = auc_lift(~holdout["thin_file"])

    assert thin_lift > 0.005
    assert thin_lift > thick_lift
