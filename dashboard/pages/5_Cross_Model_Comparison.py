from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.charts import bar_chart, tradeoff_chart
from components.data import load_all
from components.text import DISCLAIMER

st.set_page_config(page_title="FairLens | Cross-model", layout="wide")
st.title("Cross-Model Comparison")
try:
    performance, fairness, _ = load_all()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()
attribute = st.selectbox("Sensitive attribute", fairness["sensitive_attribute"].unique())
fairness_metric = st.selectbox("Fairness metric", ["demographic_parity_diff", "equal_opportunity_diff", "equalized_odds_diff", "disparate_impact_ratio"])
performance_metric = st.selectbox("Performance metric", ["accuracy", "precision", "recall", "f1", "roc_auc"])
state = st.selectbox("Experiment state", performance["mitigation_state"].unique())
fairness_view = fairness[(fairness.sensitive_attribute == attribute) & (fairness.mitigation_state == state)].drop_duplicates("model")
merged = performance[performance.mitigation_state == state].merge(fairness_view[["model", fairness_metric]], on="model")
st.dataframe(merged[["model", performance_metric, fairness_metric]], use_container_width=True)
st.pyplot(bar_chart(merged, "model", performance_metric, title=f"{performance_metric} across models"), clear_figure=True)
st.pyplot(tradeoff_chart(merged, fairness_metric, "Accuracy and fairness indicator by model"), clear_figure=True)
st.markdown(DISCLAIMER)
