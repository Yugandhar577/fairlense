"""Reusable Matplotlib/Seaborn charts generated entirely from result-store data."""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def format_metric(value: float) -> str:
    """Render undefined metrics safely instead of exposing NaN."""
    return "undefined" if pd.isna(value) or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def bar_chart(data: pd.DataFrame, x: str, y: str, hue: str | None = None, title: str = ""):
    """Return a consistently styled bar chart."""
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=axis, errorbar=None)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


def tradeoff_chart(data: pd.DataFrame, fairness_column: str, title: str):
    """Plot actual accuracy against one selected disparity statistic."""
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x="accuracy", y=fairness_column, hue="model", style="mitigation_state", s=110, ax=axis)
    for _, row in data.iterrows():
        axis.annotate(row["mitigation_state"], (row["accuracy"], row[fairness_column]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_title(title)
    figure.tight_layout()
    return figure
