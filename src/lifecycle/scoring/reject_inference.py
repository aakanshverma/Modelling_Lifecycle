"""Parceling-based reject inference for the acquisition scorecard.

In production you only observe a default outcome for applicants you
actually approved under your *current* policy — anyone that policy
declined never gets booked, so you never see how they would have
performed. Fitting a new scorecard only on the known-outcome population
re-learns and reinforces whatever bias the old policy already had.

Parceling corrects for this: score the no-outcome population with a
baseline model trained on the known population, bucket both populations
by that score, and assign each no-outcome applicant a probabilistic
bad/good label drawn from the known bad rate in its bucket — inflated by
a risk multiplier, since applicants a prior policy declined are typically
riskier than approved applicants land in the same score band (a standard,
conservative assumption in scorecard reject inference).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lifecycle.scoring.scorecard import ScorecardModel, fit_scorecard


@dataclass
class RejectInferenceResult:
    baseline_model: ScorecardModel  # fit on the known-outcome population only
    augmented_model: ScorecardModel  # fit on known + inferred-label no-outcome applicants
    inferred_bad_rate: float  # mean inferred bad probability assigned to the no-outcome population


def _bucket_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def infer_labels(
    baseline_model: ScorecardModel,
    known_X: pd.DataFrame,
    known_y: pd.Series,
    unknown_X: pd.DataFrame,
    n_bins: int = 10,
    risk_multiplier: float = 1.5,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Assign the no-outcome population a probabilistic bad/good label via parceling.

    Returns (inferred_label, assigned_bad_rate) indexed like `unknown_X`.
    """
    rng = np.random.default_rng(seed)

    known_pd = baseline_model.predict(known_X)[f"{baseline_model.name}_pd"]
    edges = _bucket_edges(known_pd.to_numpy(), n_bins)

    known_bins = pd.cut(known_pd, bins=edges)
    bucket_bad_rate = pd.Series(known_y.to_numpy(), index=known_bins.to_numpy()).groupby(level=0).mean()

    unknown_pd = baseline_model.predict(unknown_X)[f"{baseline_model.name}_pd"]
    unknown_bins = pd.cut(unknown_pd, bins=edges)
    assigned_bad_rate = (
        unknown_bins.map(bucket_bad_rate).astype(float) * risk_multiplier
    ).clip(upper=0.95).fillna(bucket_bad_rate.max())
    assigned_bad_rate.index = unknown_X.index

    inferred_label = pd.Series(
        rng.binomial(1, assigned_bad_rate.to_numpy()), index=unknown_X.index, name="inferred_bad_flag"
    )
    return inferred_label, assigned_bad_rate


def run_reject_inference(
    known_X: pd.DataFrame,
    known_y: pd.Series,
    unknown_X: pd.DataFrame,
    risk_multiplier: float = 1.5,
    seed: int = 0,
) -> RejectInferenceResult:
    baseline_model = fit_scorecard("acquisition_baseline", known_X, known_y)

    inferred_label, assigned_bad_rate = infer_labels(
        baseline_model, known_X, known_y, unknown_X, risk_multiplier=risk_multiplier, seed=seed
    )

    augmented_X = pd.concat([known_X, unknown_X], ignore_index=True)
    augmented_y = pd.concat(
        [known_y.reset_index(drop=True), inferred_label.reset_index(drop=True)], ignore_index=True
    )
    augmented_model = fit_scorecard("acquisition_augmented", augmented_X, augmented_y)

    return RejectInferenceResult(
        baseline_model=baseline_model,
        augmented_model=augmented_model,
        inferred_bad_rate=float(assigned_bad_rate.mean()),
    )
