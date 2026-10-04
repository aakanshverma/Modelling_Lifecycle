"""Generic WOE + logistic-regression scorecard, reused by every lifecycle stage.

Converts a binary bad-rate model into a traditional points-based score using the
standard PDO (points-to-double-odds) scaling, so every stage speaks the same
"score + PD + decile" language regardless of what it predicts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from lifecycle.scoring.woe import BinResult, fit_woe_frame, transform_woe_frame


@dataclass
class ScorecardConfig:
    n_bins: int = 5
    base_score: int = 600
    base_odds: float = 10.0  # odds of good:bad at base_score
    pdo: int = 20  # points to double the odds


@dataclass
class ScorecardModel:
    name: str
    bin_results: dict[str, BinResult]
    classifier: LogisticRegression
    config: ScorecardConfig
    train_auc: float

    @property
    def features(self) -> list[str]:
        return list(self.bin_results.keys())

    def _scaling_factors(self) -> tuple[float, float]:
        factor = self.config.pdo / np.log(2)
        offset = self.config.base_score - factor * np.log(self.config.base_odds)
        return factor, offset

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score a raw (un-transformed) feature frame. Returns pd, score, decile."""
        X_woe = transform_woe_frame(X, self.bin_results)
        pd_bad = self.classifier.predict_proba(X_woe)[:, 1]

        odds_good = (1 - pd_bad) / np.clip(pd_bad, 1e-6, None)
        factor, offset = self._scaling_factors()
        score = offset + factor * np.log(odds_good)

        result = pd.DataFrame(
            {
                f"{self.name}_pd": pd_bad,
                f"{self.name}_score": score.round(0),
            },
            index=X.index,
        )
        # Deciled on `pd` (ascending P(target=1)), not `score`. `score` is scaled so a
        # HIGHER score means a LOWER P(target=1) — correct when target=1 is the bad
        # outcome (Acquisition, Behavioral), but backwards when target=1 is the good
        # outcome (Collections' self-cure, Cross-sell's acceptance). Deciling on `pd`
        # directly is unambiguous either way: decile 9 always means "most likely to be
        # the modeled outcome," which is what collections.strategy() and
        # cross_sell.offer_tier() actually assume when they treat a high decile as
        # "high propensity."
        result[f"{self.name}_decile"] = pd.qcut(
            result[f"{self.name}_pd"], 10, labels=False, duplicates="drop"
        )
        return result


def fit_scorecard(
    name: str, X: pd.DataFrame, y: pd.Series, config: ScorecardConfig | None = None
) -> ScorecardModel:
    config = config or ScorecardConfig()
    bin_results = fit_woe_frame(X, y, n_bins=config.n_bins)
    if not bin_results:
        raise ValueError(f"[{name}] no feature cleared the IV floor — check inputs")

    X_woe = transform_woe_frame(X, bin_results)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_woe, y)

    train_auc = roc_auc_score(y, clf.predict_proba(X_woe)[:, 1])
    return ScorecardModel(
        name=name, bin_results=bin_results, classifier=clf, config=config, train_auc=train_auc
    )
