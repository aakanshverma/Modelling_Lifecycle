"""Shared color assignments, applied consistently across every page.

Categorical colors are assigned by fixed label, never auto-cycled, so the
same label always reads as the same color on every chart in the app.
Sequential scales (magnitude) and diverging scales (polarity, e.g. PSI
contribution or uplift) use Plotly's built-in perceptually-uniform scales
rather than a custom hand-picked palette.
"""

# Decision / outcome categories — fixed hue per label across every page.
DECISION_COLORS = {
    "approve": "#2E7D32",  # good
    "refer": "#F9A825",  # caution
    "reject": "#C62828",  # bad
}

ROUTE_COLORS = {
    "cross_sell": "#2E7D32",
    "collections": "#C62828",
}

UPLIFT_SEGMENT_COLORS = {
    "persuadable": "#2E7D32",
    "sure_thing_or_lost_cause": "#9E9E9E",
    "sleeping_dog": "#C62828",
}

ARM_COLORS = {
    "champion": "#546E7A",
    "challenger": "#1565C0",
}

MODEL_COLORS = {
    "champion": "#546E7A",
    "challenger": "#1565C0",
}

OFFER_COLORS = {
    "top_up_loan": "#1565C0",
    "credit_card": "#6A1B9A",
    "insurance": "#00897B",
    "no_offer": "#9E9E9E",
}

STRATEGY_COLORS = {
    "soft_contact": "#2E7D32",
    "tele_calling": "#F9A825",
    "field_or_legal": "#C62828",
}

SEQUENTIAL_SCALE = "Blues"
DIVERGING_SCALE = "RdBu"

PSI_SEVERITY_COLORS = {
    "stable": "#2E7D32",
    "moderate_drift": "#F9A825",
    "significant_drift": "#C62828",
}
