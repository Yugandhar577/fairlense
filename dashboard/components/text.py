"""Centralised responsible-AI wording and metric descriptions."""
DISCLAIMER = """
**Responsible AI notice:** FairLens is an educational, research-oriented benchmarking
framework, not a real-world decision system. Its metrics describe particular mathematical
group disparities; they do not prove that a model, organisation, or individual is universally
fair, unfair, lawful, or discriminatory. The 1994 US Adult dataset is US-specific, dated,
and records gender as binary. Intersectional results can have smaller group counts and must be interpreted cautiously. Results must not be generalized without additional validation.
"""

METRIC_EXPLANATIONS = {
    "demographic_parity_diff": "Difference between the highest and lowest group selection rates (ideal: 0).",
    "equal_opportunity_diff": "Difference between group true-positive rates (ideal: 0).",
    "equalized_odds_diff": "The larger disparity in group true-positive or false-positive rates (ideal: 0).",
    "disparate_impact_ratio": "Minimum selection rate divided by maximum selection rate (ideal: 1). Values below 0.80 are a commonly cited reference heuristic, not a universal verdict.",
}

METHOD_TEXT = """
**Accountability and oversight:** The project team is accountable for implementation,
testing, documentation, assumptions, metric choices, and limitations. FairLens collects no
personal user data and must remain subject to human interpretation; it is not an autonomous
decision-maker.
"""
