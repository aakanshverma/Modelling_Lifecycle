"""Population Stability Index — the standard scorecard drift monitor.

Bins the baseline ("expected") score distribution, then checks how the
current ("actual") population's scores fall into those same bins. The
severity thresholds below (< 0.1 stable, 0.1-0.25 moderate drift worth
investigating, > 0.25 significant drift that usually triggers a scorecard
refresh) are the standard industry convention, not something tuned here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_FLOOR = 1e-4  # avoids log(0)/div-by-0 when a bin is empty in one population


@dataclass
class PSIResult:
    psi: float
    detail: pd.DataFrame  # per-bin expected%/actual%/contribution breakdown

    @property
    def severity(self) -> str:
        if self.psi < 0.1:
            return "stable"
        if self.psi < 0.25:
            return "moderate_drift"
        return "significant_drift"


def population_stability_index(
    expected: pd.Series, actual: pd.Series, n_bins: int = 10
) -> PSIResult:
    """PSI of `actual` against the `expected` baseline, binned on the baseline's quantiles."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    expected_pct = pd.cut(expected, bins=edges).value_counts(normalize=True).sort_index()
    actual_pct = (
        pd.cut(actual, bins=edges)
        .value_counts(normalize=True)
        .reindex(expected_pct.index, fill_value=0.0)
        .sort_index()
    )
    expected_pct = expected_pct.clip(lower=_FLOOR)
    actual_pct = actual_pct.clip(lower=_FLOOR)

    contribution = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    detail = pd.DataFrame(
        {"expected_pct": expected_pct, "actual_pct": actual_pct, "psi_contribution": contribution}
    )
    return PSIResult(psi=float(contribution.sum()), detail=detail)
