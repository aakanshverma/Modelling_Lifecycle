"""XGBoost challenger model with SHAP explainability, benchmarked against the
WOE + logistic-regression champion every stage ships with by default.

Regulators generally expect the shipped decision to come from an
interpretable model (hence the champion), but running a gradient-boosted
challenger alongside it is now standard practice: it tends to pick up
non-linear interactions the champion's bucketed WOE features miss, and
SHAP gives per-feature attribution so it isn't a black box when justifying
it in a model validation review.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from lifecycle.scoring.scorecard import ScorecardModel


@dataclass
class ChallengerModel:
    name: str
    classifier: XGBClassifier
    features: list[str]
    train_auc: float

    def predict_pd(self, X: pd.DataFrame) -> pd.Series:
        proba = self.classifier.predict_proba(X[self.features])[:, 1]
        return pd.Series(proba, index=X.index, name=f"{self.name}_pd")

    def shap_importance(self, X: pd.DataFrame, sample: int = 2000) -> pd.Series:
        """Mean |SHAP value| per feature — the standard global importance ranking."""
        import shap  # imported lazily: heavier dependency, only needed for explainability

        X_sample = X[self.features].sample(min(sample, len(X)), random_state=0)
        explainer = shap.TreeExplainer(self.classifier)
        shap_values = explainer.shap_values(X_sample)
        return pd.Series(np.abs(shap_values).mean(axis=0), index=self.features).sort_values(
            ascending=False
        )


def fit_challenger(name: str, X: pd.DataFrame, y: pd.Series) -> ChallengerModel:
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=0,
    )
    clf.fit(X, y)
    train_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return ChallengerModel(name=name, classifier=clf, features=list(X.columns), train_auc=train_auc)


def ks_statistic(y_true: pd.Series, y_score: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic: max separation between cumulative good/bad
    distributions — the other standard scorecard discrimination metric, alongside AUC/Gini."""
    df = pd.DataFrame({"y": y_true.to_numpy(), "score": y_score.to_numpy()}).sort_values("score")
    n_bad = max(int((df["y"] == 1).sum()), 1)
    n_good = max(int((df["y"] == 0).sum()), 1)
    cum_bad = (df["y"] == 1).cumsum() / n_bad
    cum_good = (df["y"] == 0).cumsum() / n_good
    return float((cum_bad - cum_good).abs().max())


@dataclass
class ChampionChallengerComparison:
    champion_auc: float
    challenger_auc: float
    champion_ks: float
    challenger_ks: float


def compare_champion_challenger(
    champion: ScorecardModel, challenger: ChallengerModel, X_raw: pd.DataFrame, y: pd.Series
) -> ChampionChallengerComparison:
    champion_pd = champion.predict(X_raw)[f"{champion.name}_pd"]
    challenger_pd = challenger.predict_pd(X_raw)
    return ChampionChallengerComparison(
        champion_auc=roc_auc_score(y, champion_pd),
        challenger_auc=roc_auc_score(y, challenger_pd),
        champion_ks=ks_statistic(y, champion_pd),
        challenger_ks=ks_statistic(y, challenger_pd),
    )
