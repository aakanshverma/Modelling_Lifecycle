"""Two-model (T-learner) uplift model for the Collections stage.

Self-cure propensity alone tells you who's likely to pay without help —
it doesn't tell you whether contacting them *changes* that. This fits
separate propensity-to-cure models on the treated (contacted) and control
(not contacted) arms of a randomized contact test, and takes their
predicted-probability difference as the uplift score: the actual lift in
self-cure probability attributable to contact.

That's what collections effort should be prioritized on, not self-cure
propensity itself — a "sure thing" (high propensity regardless) doesn't
need contact, and a "sleeping dog" can get worse if agitated. Plain
propensity scoring can't tell those apart from a genuine "persuadable";
only a treated-vs-control comparison can.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class UpliftModel:
    treated_model: LogisticRegression
    control_model: LogisticRegression
    features: list[str]

    def predict_uplift(self, X: pd.DataFrame) -> pd.Series:
        p_treated = self.treated_model.predict_proba(X[self.features])[:, 1]
        p_control = self.control_model.predict_proba(X[self.features])[:, 1]
        return pd.Series(p_treated - p_control, index=X.index, name="uplift_score")


def fit_uplift_model(X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> UpliftModel:
    treated_mask = treatment.to_numpy() == 1
    treated_model = LogisticRegression(max_iter=1000).fit(X[treated_mask], outcome[treated_mask])
    control_model = LogisticRegression(max_iter=1000).fit(X[~treated_mask], outcome[~treated_mask])
    return UpliftModel(treated_model=treated_model, control_model=control_model, features=list(X.columns))


def assign_uplift_segment(
    uplift_score: pd.Series, persuadable_quantile: float = 0.8, sleeping_dog_quantile: float = 0.1
) -> pd.Series:
    """Segment by relative rank, not an absolute uplift threshold.

    A T-learner's estimated uplift scale is noisy and shifts with the
    population and the fit — but its *ranking* is what matters for
    prioritizing a fixed amount of collections capacity (contact the top
    of the ranking, avoid the bottom). `persuadable_quantile`/
    `sleeping_dog_quantile` control how much of the population falls into
    each actionable segment versus the untouched middle.
    """
    persuadable_cutoff = uplift_score.quantile(persuadable_quantile)
    sleeping_dog_cutoff = uplift_score.quantile(sleeping_dog_quantile)

    segment = pd.Series("sure_thing_or_lost_cause", index=uplift_score.index)
    segment[uplift_score >= persuadable_cutoff] = "persuadable"
    segment[uplift_score <= sleeping_dog_cutoff] = "sleeping_dog"
    return segment
