from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.charts import bar_chart, tradeoff_chart
from components.data import load_all
from components.text import DISCLAIMER

st.set_page_config(page_title="FairLens | Mitigation", layout="wide")
st.title("Mitigation Comparison")
try:
    performance, fairness, _ = load_all()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()
model = st.selectbox("Model", performance["model"].unique())
attribute = st.selectbox("Sensitive attribute", fairness["sensitive_attribute"].unique())
metric = st.selectbox("Fairness metric", ["demographic_parity_diff", "equal_opportunity_diff", "equalized_odds_diff", "disparate_impact_ratio"])
performance_view = performance[performance.model == model]
fairness_view = fairness[(fairness.model == model) & (fairness.sensitive_attribute == attribute)].drop_duplicates("mitigation_state")
comparison = performance_view.merge(fairness_view[["mitigation_state", metric]], on="mitigation_state")
st.dataframe(comparison, use_container_width=True)
st.pyplot(bar_chart(comparison, "mitigation_state", metric, title=f"{model}: {metric}"), clear_figure=True)
st.pyplot(tradeoff_chart(comparison, metric, "Actual accuracy–fairness trade-off"), clear_figure=True)
st.caption("Threshold adjustment was fit using a validation partition derived from training data. The test split was not used to choose thresholds or mitigation parameters.")
st.markdown(DISCLAIMER)
