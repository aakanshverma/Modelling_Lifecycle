import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import get_lifecycle_result  # noqa: E402
from theme import DECISION_COLORS, OFFER_COLORS, ROUTE_COLORS, STRATEGY_COLORS  # noqa: E402

st.set_page_config(page_title="NBFC Customer Lifecycle", layout="wide", page_icon="📊")

st.title("NBFC Customer Lifecycle — Overview")
st.markdown(
    """
This dashboard presents the **"ABCD" credit-risk scorecard framework**: a
customer is scored at onboarding with bureau data (**A**cquisition), then
re-scored on actual repayment behavior once booked (**B**ehavioral), which
forks underperforming customers into **C**ollections and performing
customers into **D** (Cross-sell). Every number here comes from the
`src/lifecycle` library — this page only visualizes it.
"""
)

result = get_lifecycle_result()
journey = result.journey
booked = journey[journey["acquisition_decision"] == "approve"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Applicants scored", f"{len(journey):,}")
col2.metric("Approval rate", f"{(journey['acquisition_decision'] == 'approve').mean():.1%}")
col3.metric("Booked & behaviorally scored", f"{len(booked):,}")
col4.metric("Routed to Collections", f"{(booked['behavioral_route'] == 'collections').mean():.1%}")
col5.metric("Routed to Cross-sell", f"{(booked['behavioral_route'] == 'cross_sell').mean():.1%}")

st.subheader("Stage model quality (train AUC)")
auc_cols = st.columns(4)
for col, (label, model) in zip(
    auc_cols,
    [
        ("Acquisition", result.acquisition_model),
        ("Behavioral", result.behavioral_model),
        ("Collections", result.collections_model),
        ("Cross-sell", result.cross_sell_model),
    ],
):
    col.metric(label, f"{model.train_auc:.3f}")

st.subheader("The customer journey")

decision_counts = journey["acquisition_decision"].value_counts()
route_counts = booked["behavioral_route"].value_counts()
collections_actions = booked.loc[booked["behavioral_route"] == "collections", "final_action"].value_counts()
cross_sell_actions = booked.loc[booked["behavioral_route"] == "cross_sell", "final_action"].value_counts()

labels = (
    ["Applicants"]
    + list(decision_counts.index)
    + list(route_counts.index)
    + list(collections_actions.index)
    + list(cross_sell_actions.index)
)
# De-duplicate while preserving first-seen order (a label must be a single Sankey node).
seen = {}
node_labels = []
for lbl in labels:
    if lbl not in seen:
        seen[lbl] = len(node_labels)
        node_labels.append(lbl)
idx = seen

node_colors = []
for lbl in node_labels:
    node_colors.append(
        DECISION_COLORS.get(lbl)
        or ROUTE_COLORS.get(lbl)
        or STRATEGY_COLORS.get(lbl)
        or OFFER_COLORS.get(lbl)
        or "#90A4AE"
    )

sources, targets, values = [], [], []
for decision, count in decision_counts.items():
    sources.append(idx["Applicants"])
    targets.append(idx[decision])
    values.append(count)
for route, count in route_counts.items():
    sources.append(idx["approve"])
    targets.append(idx[route])
    values.append(count)
for action, count in collections_actions.items():
    sources.append(idx["collections"])
    targets.append(idx[action])
    values.append(count)
for action, count in cross_sell_actions.items():
    sources.append(idx["cross_sell"])
    targets.append(idx[action])
    values.append(count)

fig = go.Figure(
    go.Sankey(
        node=dict(label=node_labels, color=node_colors, pad=20, thickness=18, line=dict(width=0)),
        link=dict(source=sources, target=targets, value=values, color="rgba(144,164,174,0.35)"),
    )
)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), font_size=13)
st.plotly_chart(fig, width='stretch')

st.caption(
    "Applicants → Acquisition decision → (if approved) Behavioral route → final action. "
    "Use the pages in the sidebar to drill into each stage."
)
