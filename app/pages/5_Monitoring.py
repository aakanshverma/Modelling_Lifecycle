import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import get_psi_data  # noqa: E402
from theme import PSI_SEVERITY_COLORS  # noqa: E402

st.set_page_config(page_title="Monitoring — NBFC Lifecycle", layout="wide", page_icon="📊")
st.title("Score monitoring — Population Stability Index")
st.caption(
    "Bins the baseline Acquisition score distribution and checks how a later "
    "population's scores land in those same bins. The standard early-warning signal "
    "that a scorecard needs to be revisited, before the bad rate actually moves."
)

psi = get_psi_data()

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Fresh same-distribution cohort",
        f"PSI = {psi['stable'].psi:.4f}",
        psi["stable"].severity,
        delta_color="off",
    )
with col2:
    st.metric(
        "After a simulated downturn",
        f"PSI = {psi['drift'].psi:.4f}",
        psi["drift"].severity,
        delta_color="off",
    )

st.subheader("PSI severity thresholds")
threshold_cols = st.columns(3)
for col, (label, rng) in zip(
    threshold_cols,
    [("stable", "< 0.10 — no action"), ("moderate_drift", "0.10–0.25 — investigate"), ("significant_drift", "> 0.25 — freeze & refit")],
):
    color = PSI_SEVERITY_COLORS[label]
    col.markdown(
        f"<div style='padding:10px;border-radius:6px;background:{color}22;border-left:4px solid {color}'>"
        f"<b>{label.replace('_', ' ')}</b><br>{rng}</div>",
        unsafe_allow_html=True,
    )

def _format_bin(interval) -> str:
    import math

    if math.isinf(interval.left):
        return f"≤ {interval.right:.0f}"
    if math.isinf(interval.right):
        return f"> {interval.left:.0f}"
    return f"{interval.left:.0f}–{interval.right:.0f}"


st.divider()
left, right = st.columns(2)
panels = [("Fresh cohort (no drift)", psi["stable"]), ("After downturn", psi["drift"])]
for col, (label, result) in zip([left, right], panels):
    with col:
        detail = result.detail.copy()
        bin_labels = [_format_bin(i) for i in detail.index]
        detail.index = bin_labels
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_labels, y=detail["expected_pct"], name="baseline"))
        fig.add_trace(go.Bar(x=bin_labels, y=detail["actual_pct"], name="current"))
        fig.update_layout(
            barmode="group",
            title=f"{label} — PSI {result.psi:.4f} ({result.severity})",
            yaxis_tickformat=".0%",
            xaxis_title="acquisition score bin",
            yaxis_title="share of population",
        )
        st.plotly_chart(fig, width='stretch')
        st.dataframe(detail.round(4), width='stretch')
