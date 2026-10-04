import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import (  # noqa: E402
    get_altdata_lift,
    get_challenger_comparison,
    get_reject_inference_result,
    recompute_acquisition_decision,
)
from theme import DECISION_COLORS  # noqa: E402

st.set_page_config(page_title="Acquisition — NBFC Lifecycle", layout="wide", page_icon="📊")
st.title("Stage A — Acquisition")
st.caption("Scores a new applicant with bureau (CIBIL-style) data to decide approve / refer / reject.")

st.subheader("Approve / refer cutoffs (interactive)")
col_a, col_b = st.columns(2)
approve_cutoff = col_a.slider("Approve cutoff (score points)", 500, 700, 590, step=5)
refer_cutoff = col_b.slider("Refer cutoff (score points)", 450, approve_cutoff, 560, step=5)

scored = recompute_acquisition_decision(approve_cutoff, refer_cutoff)
counts = scored["decision"].value_counts()
bad_rate_by_decision = scored.groupby("decision")["acquisition_pd"].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Approve rate", f"{counts.get('approve', 0) / len(scored):.1%}")
m2.metric("Refer rate", f"{counts.get('refer', 0) / len(scored):.1%}")
m3.metric("Reject rate", f"{counts.get('reject', 0) / len(scored):.1%}")

left, right = st.columns(2)
with left:
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=DECISION_COLORS,
        labels={"x": "Decision", "y": "Applicants"},
        title="Decision volume at these cutoffs",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')
with right:
    fig = px.bar(
        x=bad_rate_by_decision.index,
        y=bad_rate_by_decision.values,
        color=bad_rate_by_decision.index,
        color_discrete_map=DECISION_COLORS,
        labels={"x": "Decision", "y": "Mean predicted PD"},
        title="Predicted bad rate by decision (sanity check: reject > refer > approve)",
    )
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width='stretch')

fig = px.histogram(
    scored,
    x="acquisition_score",
    color="decision",
    color_discrete_map=DECISION_COLORS,
    nbins=60,
    barmode="stack",
    title="Score distribution by decision",
)
fig.add_vline(x=approve_cutoff, line_dash="dash", line_color="#2E7D32")
fig.add_vline(x=refer_cutoff, line_dash="dash", line_color="#F9A825")
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Champion (WOE + logistic) vs. challenger (XGBoost)")
challenger_data = get_challenger_comparison()["acquisition"]
cmp = challenger_data["comparison"]

metric_df = pd.DataFrame(
    {
        "model": ["champion", "challenger", "champion", "challenger"],
        "metric": ["AUC", "AUC", "KS", "KS"],
        "value": [cmp.champion_auc, cmp.challenger_auc, cmp.champion_ks, cmp.challenger_ks],
    }
)
fig = px.bar(
    metric_df,
    x="metric",
    y="value",
    color="model",
    color_discrete_map={"champion": "#546E7A", "challenger": "#1565C0"},
    barmode="group",
    title="Holdout discrimination: champion vs. challenger",
)
fig.update_layout(yaxis_title="score")
st.plotly_chart(fig, width='stretch')

importance = challenger_data["importance"].sort_values()
fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h", marker_color="#1565C0"))
fig.update_layout(title="Challenger SHAP feature importance (mean |SHAP|)", height=350)
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Reject inference: closing the gap toward an oracle model")
ri = get_reject_inference_result()
st.markdown(
    f"Simulates a prior policy that only booked `cibil_score >= 700` applicants "
    f"({ri['known_frac']:.1%} of the population), so only they have an observed "
    f"outcome. The never-booked population's true bad rate is "
    f"**{ri['unknown_true_bad_rate']:.1%}** — parceling infers **{ri['inferred_bad_rate']:.1%}**."
)
auc_series = pd.Series(ri["aucs"])
fig = px.bar(
    x=auc_series.index,
    y=auc_series.values,
    color=auc_series.index,
    color_discrete_map={
        "baseline (naive)": "#9E9E9E",
        "augmented (reject-inferred)": "#1565C0",
        "oracle (full true labels)": "#2E7D32",
    },
    labels={"x": "", "y": "Holdout AUC"},
    title="Baseline (naive) vs. reject-inferred vs. oracle",
)
fig.update_layout(showlegend=False, yaxis_range=[0.6, 0.8])
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Alternative data: lift concentrated in the thin-file segment")
alt_df = get_altdata_lift()
plot_df = alt_df.melt(
    id_vars=["segment", "n"],
    value_vars=["bureau_only_auc", "bureau_plus_altdata_auc"],
    var_name="model",
    value_name="auc",
)
plot_df["model"] = plot_df["model"].map(
    {"bureau_only_auc": "bureau-only", "bureau_plus_altdata_auc": "bureau + alt-data"}
)
fig = px.bar(
    plot_df,
    x="segment",
    y="auc",
    color="model",
    color_discrete_map={"bureau-only": "#546E7A", "bureau + alt-data": "#1565C0"},
    barmode="group",
    title="AUC lift from blending in UPI/utility-bill alt-data features",
)
fig.update_layout(yaxis_range=[0.5, 0.8])
st.plotly_chart(fig, width='stretch')
st.caption(
    "Bureau data is simulated as noisier for thin-file applicants (little trade history) — "
    "the real-world reason alternative data is worth the integration cost. The lift should sit "
    "almost entirely in the thin-file bar."
)
