"""Synthetic data generators standing in for real bureau / on-book feeds.

Swap these for your actual CIBIL pull and core-banking/LMS extracts — every
downstream stage module only assumes the column names defined here, so the
model code does not change when the data source does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# DPD bucket ordering used by the monthly panel, roll-rate matrix, and vintage curves.
DPD_BUCKETS = ["current", "dpd_1_30", "dpd_31_60", "dpd_61_90", "npa_90_plus"]

# Month-over-month transition matrices for the best- and worst-risk customers; every
# real customer's matrix is a severity-weighted blend of the two (see generate_dpd_panel).
# npa_90_plus is treated as absorbing: once written off as NPA, an account stays there.
_BEST_TRANSITION = np.array(
    [
        [0.93, 0.07, 0.00, 0.00, 0.00],
        [0.55, 0.30, 0.15, 0.00, 0.00],
        [0.25, 0.35, 0.25, 0.15, 0.00],
        [0.15, 0.20, 0.25, 0.25, 0.15],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ]
)
_WORST_TRANSITION = np.array(
    [
        [0.80, 0.20, 0.00, 0.00, 0.00],
        [0.20, 0.35, 0.35, 0.10, 0.00],
        [0.05, 0.15, 0.30, 0.35, 0.15],
        [0.02, 0.05, 0.13, 0.30, 0.50],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ]
)


def generate_dpd_panel(
    customers: pd.DataFrame, risk_col: str, n_months: int = 12, seed: int = 50
) -> pd.DataFrame:
    """Simulate a monthly DPD-bucket panel for booked customers via a severity-blended Markov chain.

    `risk_col` is any column where a higher value means higher risk (e.g. an
    acquisition PD) — customers are ranked on it to blend between the best-
    and worst-case transition matrices. Returns a long panel of
    (applicant_id, month_on_book, dpd_bucket), everyone starting 'current'
    in the month they're booked.
    """
    rng = np.random.default_rng(seed)
    severity = customers[risk_col].rank(pct=True).to_numpy()
    applicant_ids = customers["applicant_id"].to_numpy()
    state = np.zeros(len(customers), dtype=int)

    rows = []
    for month in range(1, n_months + 1):
        # Transition everyone off a single snapshot of this month's starting state —
        # updating `state` in place while iterating buckets would let a customer who
        # just rolled into e.g. dpd_1_30 get re-processed again in the same month.
        start_state = state.copy()
        next_state = state.copy()
        for bucket_idx in range(len(DPD_BUCKETS)):
            mask = start_state == bucket_idx
            if not mask.any():
                continue
            sev = severity[mask][:, None]
            probs = (1 - sev) * _BEST_TRANSITION[bucket_idx] + sev * _WORST_TRANSITION[bucket_idx]
            probs = probs / probs.sum(axis=1, keepdims=True)
            cum_probs = np.cumsum(probs, axis=1)
            draws = (rng.random((mask.sum(), 1)) < cum_probs).argmax(axis=1)
            next_state[mask] = draws
        state = next_state

        rows.append(
            pd.DataFrame(
                {
                    "applicant_id": applicant_ids,
                    "month_on_book": month,
                    "dpd_bucket": [DPD_BUCKETS[s] for s in state],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def generate_applicants(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Day-0 applicant pool: bureau (CIBIL-style) + application fields.

    `default_flag` is the 12-month bad outcome the acquisition scorecard predicts.
    """
    rng = np.random.default_rng(seed)

    cibil_score = rng.normal(700, 90, n).clip(300, 900)
    vintage_months = rng.gamma(4, 12, n).clip(1, 360)
    num_trades = rng.poisson(5, n).clip(0, 30)
    dpd_30_plus_6m = rng.poisson(0.4, n).clip(0, 10)
    credit_utilization = rng.beta(2, 5, n)
    monthly_income = rng.lognormal(10.8, 0.5, n).clip(8_000, 500_000)
    loan_amount = rng.lognormal(11.5, 0.6, n).clip(10_000, 2_000_000)
    existing_emi_to_income = rng.beta(2, 6, n)
    age = rng.normal(38, 10, n).clip(21, 65)

    latent = (
        -0.012 * (cibil_score - 650)
        + 0.55 * dpd_30_plus_6m
        + 2.0 * credit_utilization
        + 2.5 * existing_emi_to_income
        - 0.004 * vintage_months
        + rng.normal(0, 0.8, n)
    )
    default_prob = _sigmoid(latent - 3.3)
    default_flag = rng.binomial(1, default_prob)

    return pd.DataFrame(
        {
            "applicant_id": np.arange(1, n + 1),
            "cibil_score": cibil_score.round(0),
            "bureau_vintage_months": vintage_months.round(0),
            "num_trades": num_trades,
            "dpd_30_plus_last_6m": dpd_30_plus_6m,
            "credit_utilization": credit_utilization.round(3),
            "monthly_income": monthly_income.round(0),
            "loan_amount_requested": loan_amount.round(0),
            "existing_emi_to_income": existing_emi_to_income.round(3),
            "age": age.round(0),
            "default_flag": default_flag,
        }
    )


def generate_onbook_behavior(booked: pd.DataFrame, seed: int = 43) -> pd.DataFrame:
    """Simulate 6-12 months of on-book behavior for customers already booked.

    `behavioral_bad_flag` is the forward-looking bad outcome the behavioral
    scorecard predicts (90+ DPD in the next 6 months).
    """
    rng = np.random.default_rng(seed)
    n = len(booked)

    months_on_book = rng.integers(6, 24, n)
    max_dpd_last_3m = rng.poisson(booked["dpd_30_plus_last_6m"] * 0.6 + 0.3, n).clip(0, 90)
    num_bounces_6m = rng.poisson(0.3 + booked["existing_emi_to_income"] * 2, n).clip(0, 12)
    utilization_trend = rng.normal(0.0, 0.08, n) + 0.05 * (booked["credit_utilization"] - 0.4)
    payment_to_due_ratio = (1 - booked["existing_emi_to_income"] * 0.3 + rng.normal(0, 0.1, n)).clip(
        0.3, 1.3
    )
    bureau_refresh_score = (booked["cibil_score"] + rng.normal(0, 25, n)).clip(300, 900)

    latent = (
        0.09 * max_dpd_last_3m
        + 0.5 * num_bounces_6m
        + 3.0 * utilization_trend
        - 2.0 * (payment_to_due_ratio - 1)
        - 0.01 * (bureau_refresh_score - 650)
        + rng.normal(0, 0.7, n)
    )
    bad_prob = _sigmoid(latent - 1.5)
    behavioral_bad_flag = rng.binomial(1, bad_prob)

    behavior = pd.DataFrame(
        {
            "applicant_id": booked["applicant_id"].to_numpy(),
            "months_on_book": months_on_book,
            "max_dpd_last_3m": max_dpd_last_3m,
            "num_bounces_last_6m": num_bounces_6m,
            "utilization_trend": utilization_trend.round(3),
            "payment_to_due_ratio": payment_to_due_ratio.round(3),
            "bureau_refresh_score": bureau_refresh_score.round(0),
            "behavioral_bad_flag": behavioral_bad_flag,
        }
    )
    return booked.merge(behavior, on="applicant_id")


def _collections_baseline_latent(df: pd.DataFrame) -> np.ndarray:
    return (
        -0.07 * df["max_dpd_last_3m"]
        - 0.45 * df["num_bounces_last_6m"]
        + 3.5 * (df["payment_to_due_ratio"] - 0.8)
        + 0.01 * (df["bureau_refresh_score"] - 650)
    ).to_numpy()


def generate_collections_outcomes(collections_pop: pd.DataFrame, seed: int = 44) -> pd.DataFrame:
    """For behaviorally-bad accounts: did the account self-cure without hard collections?"""
    rng = np.random.default_rng(seed)
    n = len(collections_pop)

    latent = _collections_baseline_latent(collections_pop) + rng.normal(0, 0.5, n)
    cure_prob = _sigmoid(latent + 0.3)
    self_cure_flag = rng.binomial(1, cure_prob)

    out = collections_pop.copy()
    out["self_cure_flag"] = self_cure_flag
    return out


def _collections_uplift_latent(df: pd.DataFrame) -> np.ndarray:
    """True (noiseless) treatment effect of contacting an account — only computable
    because this is synthetic data. Contact helps "persuadable" accounts hovering
    near a near-cure payment ratio, does ~nothing for accounts that would cure or
    stay bad regardless, and backfires on severely delinquent "sleeping dog"
    accounts (aggressive contact triggers defensive/legal escalation instead of
    payment).
    """
    persuadable_bump = 0.8 * np.exp(-((df["payment_to_due_ratio"] - 0.80) ** 2) / 0.0025)
    sleeping_dog_penalty = 0.3 * np.clip(df["max_dpd_last_3m"] - 1.0, 0, None)
    return (persuadable_bump - sleeping_dog_penalty).to_numpy()


def true_collections_uplift(df: pd.DataFrame) -> pd.Series:
    """Ground-truth P(cure | contacted) - P(cure | not contacted), noiseless.

    Only available here because the data is synthetic — this is the benchmark an
    uplift model's predicted ranking is validated against, the same role the
    oracle model plays for reject inference.
    """
    baseline = _collections_baseline_latent(df)
    uplift = _collections_uplift_latent(df)
    p_treated = _sigmoid(baseline + uplift + 0.3)
    p_control = _sigmoid(baseline + 0.3)
    return pd.Series(p_treated - p_control, index=df.index, name="true_uplift")


def generate_collections_experiment(
    collections_pop: pd.DataFrame, treatment_rate: float = 0.5, seed: int = 46
) -> pd.DataFrame:
    """Randomized-contact experiment on the collections segment.

    This is what an uplift model is trained on in production: a genuine
    randomized (or quasi-randomized) contact test with an observed outcome,
    not just an observational self-cure rate. `contacted` is the randomized
    treatment; `self_cure_flag` is the outcome under that treatment.
    """
    rng = np.random.default_rng(seed)
    n = len(collections_pop)
    contacted = rng.binomial(1, treatment_rate, n)

    baseline_latent = _collections_baseline_latent(collections_pop)
    uplift_latent = _collections_uplift_latent(collections_pop)
    latent = baseline_latent + contacted * uplift_latent + rng.normal(0, 0.5, n)
    self_cure_flag = rng.binomial(1, _sigmoid(latent + 0.3))

    out = collections_pop.copy()
    out["contacted"] = contacted
    out["self_cure_flag"] = self_cure_flag
    return out


def generate_cross_sell_outcomes(cross_sell_pop: pd.DataFrame, seed: int = 45) -> pd.DataFrame:
    """For behaviorally-good accounts: did they take up an offered cross-sell product?"""
    rng = np.random.default_rng(seed)
    n = len(cross_sell_pop)

    latent = (
        0.00001 * cross_sell_pop["monthly_income"]
        + 0.02 * cross_sell_pop["months_on_book"]
        - 2.5 * cross_sell_pop["credit_utilization"]
        + 0.008 * (cross_sell_pop["bureau_refresh_score"] - 650)
        - 2.0 * cross_sell_pop["existing_emi_to_income"]
        + rng.normal(0, 0.5, n)
    )
    accept_prob = _sigmoid(latent - 1.2)
    accepted_cross_sell = rng.binomial(1, accept_prob)

    out = cross_sell_pop.copy()
    out["accepted_cross_sell"] = accepted_cross_sell
    return out


def generate_nbo_outcomes(cross_sell_pop: pd.DataFrame, seed: int = 47) -> pd.DataFrame:
    """Acceptance outcome for each product in the Next-Best-Offer catalog.

    Each product is deliberately built to appeal to a different customer
    profile: a top-up loan to higher-income, low-utilization, longer-tenure
    customers (the same signal as the single-product cross-sell model);
    a credit card to younger, higher-utilization-comfort customers; loan
    protection insurance to customers with a bigger loan and a higher
    existing EMI burden. Ranking by raw acceptance propensity alone would
    pick the wrong product for a given customer — NBO has to weigh it
    against expected value.
    """
    rng = np.random.default_rng(seed)
    n = len(cross_sell_pop)
    out = cross_sell_pop.copy()

    latent_top_up = (
        0.00001 * cross_sell_pop["monthly_income"]
        + 0.02 * cross_sell_pop["months_on_book"]
        - 2.5 * cross_sell_pop["credit_utilization"]
        + 0.008 * (cross_sell_pop["bureau_refresh_score"] - 650)
        - 2.0 * cross_sell_pop["existing_emi_to_income"]
        + rng.normal(0, 0.5, n)
    )
    out["accept_top_up_loan"] = rng.binomial(1, _sigmoid(latent_top_up - 1.2))

    latent_credit_card = (
        -0.05 * (cross_sell_pop["age"] - 30)
        + 0.000006 * cross_sell_pop["monthly_income"]
        + 1.2 * cross_sell_pop["credit_utilization"]
        - 1.0 * cross_sell_pop["existing_emi_to_income"]
        + rng.normal(0, 0.5, n)
    )
    out["accept_credit_card"] = rng.binomial(1, _sigmoid(latent_credit_card - 0.6))

    latent_insurance = (
        0.000004 * cross_sell_pop["loan_amount_requested"]
        + 2.4 * cross_sell_pop["existing_emi_to_income"]
        + 0.008 * (cross_sell_pop["bureau_refresh_score"] - 650)
        + 0.025 * (cross_sell_pop["age"] - 30)
        + rng.normal(0, 0.4, n)
    )
    out["accept_insurance"] = rng.binomial(1, _sigmoid(latent_insurance - 1.5))

    return out
