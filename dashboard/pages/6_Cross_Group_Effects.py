"""Show whether gender-targeted mitigation changes other group disparities."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from components.charts import bar_chart
from components.data import load_cross_group_effects
from components.text import DISCLAIMER

st.set_page_config(page_title="FairLens | Cross-Group Effects", layout="wide")
st.title("Cross-Group Mitigation Effects")
st.write(
    "Reweighing and threshold adjustment in this study target **gender**. This page checks "
    "whether those choices also improve or worsen measured disparities for age groups and "
    "gender × age-group intersections."
)
try:
    changes = load_cross_group_effects()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()

first, second, third = st.columns(3)
model = first.selectbox("Model", changes["model"].unique())
mitigation = second.selectbox("Mitigation", changes["mitigation_state"].unique())
metric = third.selectbox("Fairness metric", changes["fairness_metric"].unique())
view = changes[
    (changes["model"] == model)
    & (changes["mitigation_state"] == mitigation)
    & (changes["fairness_metric"] == metric)
].copy()

st.subheader("Baseline-to-mitigation comparison")
st.dataframe(
    view[[
        "sensitive_attribute", "baseline_value", "mitigated_value", "raw_change",
        "distance_to_ideal_change", "effect",
    ]],
    use_container_width=True,
    column_config={
        "baseline_value": st.column_config.NumberColumn(format="%.3f"),
        "mitigated_value": st.column_config.NumberColumn(format="%.3f"),
        "raw_change": st.column_config.NumberColumn(format="%.3f"),
        "distance_to_ideal_change": st.column_config.NumberColumn(format="%.3f"),
    },
)

age_or_intersection = view[view["sensitive_attribute"].isin(["age_group", "gender_age_group"])]
worsened = age_or_intersection[age_or_intersection["effect"] == "worsened"]
if not worsened.empty:
    st.warning(
        "Observed cross-group side effect: this gender-targeted mitigation moved at least one "
        "age or gender × age fairness statistic away from its ideal value. This is a descriptive "
        "result, not an overall fairness judgement."
    )
else:
    st.info("No selected age or intersectional statistic moved away from its ideal value in this comparison.")

st.pyplot(
    bar_chart(
        view, "sensitive_attribute", "distance_to_ideal_change", hue="effect",
        title="Change in distance to fairness-metric ideal (negative = improvement)",
    ),
    clear_figure=True,
)
st.caption(
    "For demographic parity, equal opportunity, and equalized odds, the ideal is 0. "
    "For disparate impact, the ideal is 1. Therefore, movement is classified using distance "
    "to the relevant ideal, rather than a simplistic positive/negative raw change."
)
st.markdown(DISCLAIMER)
