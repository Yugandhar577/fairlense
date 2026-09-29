# Functional Specification Document (FSD)

## FairLens: A Comparative Framework for Bias Detection and Mitigation in Machine Learning Models

| | |
|---|---|
| **Document Owner** | Team FairLens (Yugandhar Paulbudhe, Vedanti Raut, Palash Sahuji, Sara Ansingkar, Raunak Shah) |
| **Guide** | Prof. Dhananjay Bhagat |
| **Institution** | Vishwakarma Institute of Technology, Pune — Dept. of CS & AI |
| **Related Document** | prd.md (v1.0) |
| **Version** | 1.0 |
| **Status** | Draft |

---

## 1. Purpose & Scope of This Document

This FSD translates the requirements in `prd.md` into concrete functional behavior: modules, inputs/outputs, user flows, screen-by-screen dashboard behavior, data contracts between components, and edge-case/error handling. It is written to be directly usable as the basis for `System_Technical_Design.md` and for implementation.

This document does not prescribe internal algorithms or code structure (covered in the technical design doc) — it defines **what each part of the system must do, for whom, given what input, producing what output.**

---

## 2. System Overview

FairLens consists of two functional layers:

1. **Analysis Pipeline (backend)** — a Python-based batch pipeline that ingests the Adult Income dataset, preprocesses it, trains three classifiers, computes performance and fairness metrics, applies mitigation techniques, and recomputes metrics.
2. **Dashboard (frontend)** — a Streamlit application that lets a user interactively explore pipeline outputs: select a model, view metrics, apply mitigation, and compare baseline vs. mitigated results.

```
[Adult Income Dataset]
        │
        ▼
[Data Ingestion & Preprocessing Module]
        │
        ▼
[Model Training Module] ── trains 3 models in parallel
        │
        ▼
[Performance Evaluation Module]
        │
        ▼
[Fairness Evaluation Module]
        │
        ▼
[Bias Mitigation Module] ── reweighing / threshold adjustment
        │
        ▼
[Post-Mitigation Evaluation Module]
        │
        ▼
[Results Store] ──► [Streamlit Dashboard]
```

---

## 3. Functional Modules

### 3.1 Data Ingestion & Preprocessing Module

**Purpose:** Prepare a clean, model-ready dataset from the raw Adult Income data, and expose sensitive attributes for later fairness analysis.

**Inputs:**
- Raw Adult Income dataset (CSV, 48,842 rows × 14 columns, from UCI repository).

**Processing steps (functional):**
1. Load raw data; report initial row/column counts.
2. Identify and handle missing/unknown values (dataset uses `?` for missing entries) — rows dropped or imputed per a documented rule.
3. Encode categorical variables (e.g., one-hot or ordinal encoding, documented per column).
4. Scale/normalize numerical features where required by the model (e.g., for Logistic Regression).
5. Derive **age group** bins from the continuous `age` column (e.g., <25, 25–44, 45–64, 65+ — exact bins finalized in technical design).
6. Retain `gender` as the second sensitive attribute (as recorded in the dataset).
7. Separate sensitive attributes into a distinct reference structure so they can be excluded from or included in model features as configured (documented choice: whether sensitive attributes are used as predictive features or held out — must be explicit, not incidental).
8. Split data into training and test sets (e.g., 80/20 or 70/30 — ratio fixed in config, not per-run randomness beyond a fixed seed).
9. Report demographic representation (row counts per gender / age group) in both training and test splits, to catch severe imbalance before modeling.

**Outputs:**
- `X_train`, `X_test`, `y_train`, `y_test` (feature/label sets).
- `sensitive_train`, `sensitive_test` (gender, age group, aligned by index to the above).
- A preprocessing report (row counts, dropped rows, demographic split counts) surfaced to logs/report and optionally the dashboard.

**Edge cases:**
- If a demographic subgroup has too few samples in the test set (below a defined minimum threshold), fairness metrics for that subgroup are flagged as "low sample size — interpret with caution" rather than silently computed.
- If missing-value rate is unexpectedly high (data source changed), pipeline halts with a clear error rather than silently dropping large portions of data.

---

### 3.2 Model Training Module

**Purpose:** Train the three baseline classifiers on identical preprocessed data so results are directly comparable.

**Inputs:** `X_train`, `y_train` from 3.1.

**Functional behavior:**
- Train Logistic Regression, Decision Tree, and Random Forest using fixed, documented hyperparameters (or a documented light hyperparameter search) — identical data split and preprocessing across all three, so differences in results are attributable to model choice, not data handling.
- Persist each trained model (baseline) so it can later be compared against its mitigated counterpart.
- Fixed random seed used across all training runs for reproducibility.

**Outputs:**
- Three trained baseline model objects: `model_lr`, `model_dt`, `model_rf`.
- Training metadata (hyperparameters used, training time, random seed) logged for reproducibility.

**Edge cases:**
- If a model fails to converge (e.g., Logistic Regression convergence warning), this is logged and surfaced, not silently ignored.

---

### 3.3 Performance Evaluation Module

**Purpose:** Compute conventional ML performance metrics for each trained model.

**Inputs:** Trained model, `X_test`, `y_test`.

**Functional behavior:**
- For each model, generate predictions on the test set.
- Compute: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Store metrics keyed by model name and by mitigation state (baseline vs. post-mitigation), so before/after comparisons are structurally straightforward.

**Outputs:**
- Performance metrics table: `{model, mitigation_state, accuracy, precision, recall, f1, roc_auc}`.

---

### 3.4 Fairness Evaluation Module

**Purpose:** Quantify demographic disparities in model predictions.

**Inputs:** Model predictions, `y_test`, `sensitive_test` (gender, age group).

**Functional behavior — for each sensitive attribute, for each model:**
1. **Demographic Parity** — compare positive-prediction rates across groups.
2. **Equal Opportunity** — compare true-positive rates across groups.
3. **Equalized Odds** — compare both true-positive and false-positive rates across groups.
4. **Disparate Impact** — compute ratio of positive-outcome rates between groups (commonly, unprivileged/privileged group ratio).
5. Metrics computed per sensitive attribute independently (gender and age group are **not** combined/intersected in the baseline scope; intersectional analysis noted as a future extension, not required).

**Outputs:**
- Fairness metrics table: `{model, mitigation_state, sensitive_attribute, demographic_parity_diff, equal_opportunity_diff, equalized_odds_diff, disparate_impact_ratio}`.

**Edge cases:**
- Division-by-zero or undefined ratios (e.g., a group with zero positive predictions) handled explicitly — reported as "undefined" rather than crashing or silently returning 0/1.

---

### 3.5 Bias Mitigation Module

**Purpose:** Apply mitigation techniques to reduce measured disparities, and produce a mitigated counterpart for each baseline model.

**Inputs:** Baseline model(s), training data, sensitive attributes, fairness evaluation results.

**Functional behavior:**

**A. Reweighing (pre-processing)**
- Compute instance weights based on the joint distribution of the sensitive attribute and the label, such that underrepresented group/label combinations receive higher weight.
- Retrain each of the three models using these weights.
- Output: three "reweighed" model variants.

**B. Threshold Adjustment (post-processing)**
- For each baseline (or reweighed) model, adjust the decision threshold per demographic group to reduce a selected fairness metric's disparity, while tracking the resulting effect on accuracy.
- Output: adjusted prediction sets per model, per group.

**C. (Stretch) Sampling-based mitigation**
- If time permits: apply oversampling/undersampling of underrepresented group-label combinations prior to training, as an additional comparative technique.

**Outputs:**
- Mitigated model objects and/or mitigated prediction sets, tagged by technique (`reweighing`, `threshold_adjustment`, `sampling` if implemented).

**Edge cases:**
- If a mitigation technique produces a materially worse model on both accuracy and fairness simultaneously (a "lose-lose" outcome), this is still recorded and shown — not discarded — since it is itself a meaningful research finding.

---

### 3.6 Post-Mitigation Evaluation Module

**Purpose:** Recompute performance (3.3) and fairness (3.4) metrics on mitigated models/predictions, using identical logic to the baseline evaluation so results are directly comparable.

**Inputs:** Mitigated models/predictions, `X_test`, `y_test`, `sensitive_test`.

**Outputs:**
- Same schema as 3.3 and 3.4, with `mitigation_state` set to the applied technique instead of `baseline`.

---

### 3.7 Results Store

**Purpose:** Central, queryable store of all computed results so the dashboard does not need to re-run the pipeline on every interaction.

**Functional behavior:**
- Persist performance metrics table, fairness metrics table, and metadata (dataset stats, model configs, mitigation configs) to disk (e.g., CSV/JSON/Parquet — finalized in technical design).
- Dashboard reads from this store rather than triggering retraining live, except where a "re-run" action is explicitly provided (see 4.5).

---

## 4. Dashboard (Streamlit) — Functional Behavior

### 4.1 Screen: Overview / Home
- Displays project title, one-paragraph purpose statement, and the mandatory disclaimer (FR-15): *"FairLens is a research and educational tool. It is not intended for autonomous, high-stakes, real-world decision-making."*
- Displays dataset summary: total records, feature count, sensitive attributes used, target variable definition.
- Displays demographic distribution charts (gender, age group) for the dataset.

### 4.2 Screen: Model Selection & Performance
- User selects one of: Logistic Regression, Decision Tree, Random Forest (single-select control).
- Displays performance metrics (accuracy, precision, recall, F1, ROC-AUC) for the selected model, baseline state, as a table and/or bar chart.
- Allows switching between models without page reload delay beyond re-querying the Results Store.

### 4.3 Screen: Fairness Analysis
- For the selected model, user selects a sensitive attribute (gender or age group) via a control.
- Displays the four fairness metrics (demographic parity, equal opportunity, equalized odds, disparate impact) for the selected model + attribute, with plain-language one-line explanations next to each metric.
- Visual indicator (e.g., color-coded or threshold marker) showing whether a metric falls within a commonly-cited "acceptable" range (e.g., disparate impact between 0.8–1.25 — the "80% rule"), clearly labeled as a reference heuristic, not an absolute judgment.

### 4.4 Screen: Mitigation & Comparison
- User selects a mitigation technique to apply (Reweighing / Threshold Adjustment / Sampling if available) via a control.
- On selection, dashboard displays **side-by-side baseline vs. mitigated** comparison for the selected model:
  - Performance metrics (before/after).
  - Fairness metrics (before/after), per sensitive attribute.
- Displays a trade-off visualization (e.g., scatter or paired bar chart of accuracy vs. a fairness metric, baseline vs. mitigated, across all three models) so users can see the accuracy–fairness trade-off directly, not just per-model.

### 4.5 Screen: Cross-Model Comparison
- Displays a summary table/chart comparing all three models simultaneously (baseline and, optionally, a chosen mitigation state) across both performance and fairness metrics, to support the project's core comparative-framework goal.

### 4.6 Common Dashboard Behaviors
- All numeric outputs displayed to a consistent, documented precision (e.g., 3 decimal places).
- All charts labeled with axis titles and legends; no unlabeled visualizations.
- Any metric flagged in 3.1/3.4 as "low sample size" or "undefined" is visually distinguished (e.g., grey-out, footnote) rather than presented as a normal value.
- A persistent footer/disclaimer reiterating the tool's research/decision-support-only purpose (reinforces FR-15 on every screen, not just the home screen).

---

## 5. User Flows

### Flow A — First-Time Exploration
1. User opens dashboard → lands on Overview screen.
2. Reads dataset summary and disclaimer.
3. Navigates to Model Selection & Performance → selects Random Forest.
4. Navigates to Fairness Analysis → selects "gender" → views the four fairness metrics.
5. Navigates to Mitigation & Comparison → applies "Threshold Adjustment" → reviews before/after comparison.
6. Navigates to Cross-Model Comparison → compares all three models' baseline fairness/performance at a glance.

### Flow B — Guide/Evaluator Verification
1. Evaluator opens dashboard.
2. Directly jumps to Cross-Model Comparison to sanity-check that all 3 models and required metrics are present.
3. Spot-checks Fairness Analysis screen for one model/attribute combination against the technical report's reported numbers, confirming reproducibility.

---

## 6. Data Contracts (Functional-Level)

| Data Object | Produced By | Consumed By | Key Fields |
|---|---|---|---|
| Preprocessed dataset | 3.1 | 3.2 | X_train/test, y_train/test, sensitive_train/test |
| Baseline models | 3.2 | 3.3, 3.4, 3.5 | model object, model name |
| Performance metrics table | 3.3, 3.6 | Results Store, Dashboard | model, mitigation_state, accuracy, precision, recall, f1, roc_auc |
| Fairness metrics table | 3.4, 3.6 | Results Store, Dashboard | model, mitigation_state, sensitive_attribute, 4 fairness metrics |
| Mitigated models/predictions | 3.5 | 3.6 | model/prediction object, technique tag |

---

## 7. Error Handling & Validation (Functional Requirements)

| Scenario | Expected Behavior |
|---|---|
| Dataset file missing or unreadable at pipeline start | Pipeline halts with a clear, human-readable error; dashboard shows a "data not available" state rather than crashing. |
| A fairness metric is mathematically undefined for a group (e.g., zero denominator) | Reported as "undefined," not 0, NaN-silenced, or omitted without explanation. |
| User selects a mitigation technique before baseline results exist | Dashboard disables/greys out mitigation controls until baseline results are computed and available in the Results Store. |
| Model training fails/does not converge | Logged with a warning; dashboard flags the affected model's results as "trained with warnings." |
| Extremely small demographic subgroup in test split | Fairness result for that subgroup flagged as low-confidence in both report and dashboard. |

---

## 8. Non-Functional Notes Relevant to Functional Design

- All dashboard interactions (model select, attribute select, mitigation select) should read from the precomputed Results Store, keeping response times interactive (sub-second to a few seconds), rather than triggering full retraining on each click. Full pipeline re-runs are a separate, explicit action (not required for MVP but may be included as an "Advanced" option).
- All functional modules (3.1–3.6) must be independently runnable/testable (e.g., via a script or notebook) outside the dashboard, to support reproducibility and grading verification.

---

## 9. Traceability to PRD

| PRD Requirement | FSD Section(s) |
|---|---|
| FR-1 | 3.1 |
| FR-2 | 3.2 |
| FR-3 | 3.3 |
| FR-4 | 3.4 |
| FR-5 | 3.5 (A) |
| FR-6 | 3.5 (B) |
| FR-7 | 3.6 |
| FR-8 | 4.2 |
| FR-9 | 4.1 |
| FR-10 | 4.3 |
| FR-11 | 4.4 |
| FR-12 | 4.4, 4.5 |
| FR-13 | 3.5 (C) |
| FR-14 | 3.7 |
| FR-15 | 4.1, 4.6 |

---

## 10. Open Items for Technical Design Document

- Exact age-group bin boundaries.
- Exact hyperparameters / hyperparameter search strategy per model.
- File formats and schema for the Results Store.
- Exact reweighing algorithm and threshold-adjustment method (e.g., library-based via Fairlearn/AIF360 vs. custom implementation).
- Precision/rounding conventions and low-sample-size threshold value.
- Streamlit app file/module structure and caching strategy.
