from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.charts import bar_chart, format_metric
from components.data import load_all
from components.text import DISCLAIMER, METRIC_EXPLANATIONS

st.set_page_config(page_title="FairLens | Fairness", layout="wide")
st.title("Fairness Analysis")
try:
    performance, fairness, _ = load_all()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()
first, second, third = st.columns(3)
model = first.selectbox("Model", fairness["model"].unique())
state = second.selectbox("Experiment state", fairness["mitigation_state"].unique())
attribute = third.selectbox("Sensitive attribute", fairness["sensitive_attribute"].unique())
labels = {"Demographic Parity": "demographic_parity_diff", "Equal Opportunity": "equal_opportunity_diff", "Equalized Odds": "equalized_odds_diff", "Disparate Impact": "disparate_impact_ratio"}
choice = st.selectbox("Fairness metric", labels)
metric = labels[choice]
data = fairness[(fairness.model == model) & (fairness.mitigation_state == state) & (fairness.sensitive_attribute == attribute)]
overall = data.iloc[0][metric]
st.metric(choice, format_metric(overall))
st.caption(METRIC_EXPLANATIONS[metric])
if metric == "disparate_impact_ratio" and pd.notna(overall) and overall < 0.8:
    st.warning("Reference heuristic: this selection-rate ratio is below 0.80. It is a statistical indicator, not proof of unlawful discrimination or a universal determination of unfairness.")
if data["low_confidence_flag"].iloc[0]:
    st.warning("At least one group has fewer than the configured minimum of 30 test samples; interpret group estimates cautiously.")
st.dataframe(data[["group", "group_count", "selection_rate", "tpr", "fpr", "low_sample_warning"]], use_container_width=True)
st.pyplot(bar_chart(data, "group", "selection_rate", title="Selection rate by group"), clear_figure=True)
st.markdown(DISCLAIMER)
