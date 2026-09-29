# System Technical Design Document

> Implementation update: the original Streamlit presentation layer described below has been replaced by a React/TypeScript dashboard under `frontend/`. A read-only local service in `src/fairlens/dashboard_server.py`, launched with `python run_dashboard.py`, serves the saved metric files and built frontend. See `README.md` for current setup and API instructions. The batch experiment pipeline remains the source of dashboard measurements.

## FairLens: A Comparative Framework for Bias Detection and Mitigation in Machine Learning Models

| | |
|---|---|
| **Document Owner** | Team FairLens (Yugandhar Paulbudhe, Vedanti Raut, Palash Sahuji, Sara Ansingkar, Raunak Shah) |
| **Guide** | Prof. Dhananjay Bhagat |
| **Institution** | Vishwakarma Institute of Technology, Pune — Dept. of CS & AI |
| **Related Documents** | prd.md (v1.0), fsd.md (v1.0) |
| **Version** | 1.0 |
| **Status** | Draft |

---

## 1. Purpose

This document specifies **how** FairLens is built: architecture, technology choices, module/class design, algorithms and formulas, data schemas, folder structure, caching/reproducibility strategy, error handling implementation, and testing approach. It implements the functional behavior defined in `fsd.md`, which in turn satisfies the requirements in `prd.md`.

---

## 2. Architecture Overview

FairLens is a **local, batch-pipeline + interactive-dashboard** system with no external services, databases, or deployment infrastructure beyond a single machine.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FairLens System                         │
│                                                                   │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐ │
│  │ data/          │   │ pipeline/     │   │ dashboard/         │ │
│  │  adult.csv     │──▶│ (offline run) │──▶│ app.py (Streamlit) │ │
│  └───────────────┘   └───────┬───────┘   └─────────┬──────────┘ │
│                               │                      │            │
│                               ▼                      ▼            │
│                       ┌───────────────────────────────────┐      │
│                       │   results/ (Results Store)         │      │
│                       │   - performance_metrics.csv         │      │
│                       │   - fairness_metrics.csv            │      │
│                       │   - models/*.pkl                    │      │
│                       │   - run_config.json                 │      │
│                       └───────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

**Design principle:** the pipeline (training + evaluation) and the dashboard (presentation) are **decoupled**. The pipeline runs offline/on-demand and writes all results to a flat-file Results Store; the dashboard only reads from that store, keeping it fast and simple, and keeping the ML logic testable independently of the UI (per FSD §8).

---

## 3. Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| Language | Python 3.11 | Standard for ML/data science; team familiarity |
| Data handling | Pandas, NumPy | Standard tabular data manipulation |
| ML models | Scikit-learn | Logistic Regression, Decision Tree, Random Forest all natively supported |
| Fairness metrics & mitigation | Fairlearn (primary), AIF360 (reference/cross-check) | Both are peer-reviewed, widely cited open-source libraries (see literature survey); Fairlearn has simpler Scikit-learn-compatible API for `MetricFrame` and `ThresholdOptimizer` |
| Visualization (report/offline) | Matplotlib, Seaborn | Standard, sufficient for static comparison plots |
| Dashboard | Streamlit | Lightweight, Python-native, fast to build interactive demos without separate frontend stack |
| Persistence | CSV/JSON (metrics), `joblib`/`pickle` (models) | Simple, human-inspectable, sufficient for dataset/model scale (~49K rows, 3–5 models) |
| Dev environment | Jupyter Notebook (experimentation) + VS Code (module code) | Matches synopsis's stated tools |
| Version control | Git/GitHub | Reproducibility, collaboration across 5-member team |

No cloud infrastructure, database server, or containerization is required given the local-demo scope (per PRD §8 constraints). Docker packaging may be added later as an optional convenience, not a requirement.

---

## 4. Repository / Folder Structure

```
fairlens/
├── data/
│   ├── raw/
│   │   └── adult.csv                  # UCI Adult Income dataset (as downloaded)
│   └── processed/
│       └── adult_clean.parquet        # Cached preprocessed data
│
├── src/fairlens/
│   ├── __init__.py
│   ├── config.py                      # Central config: paths, seed, split ratio, bin edges
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # load_raw_data()
│   │   └── preprocess.py              # clean, encode, scale, bin age, split
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py                   # train_baseline_models()
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── performance.py             # accuracy/precision/recall/F1/ROC-AUC
│   │   └── fairness.py                # DP, EO, EOdds, DI via Fairlearn MetricFrame
│   ├── mitigation/
│   │   ├── __init__.py
│   │   ├── reweighing.py              # sample-weight computation + retraining
│   │   ├── threshold_adjustment.py    # Fairlearn ThresholdOptimizer wrapper
│   │   └── sampling.py                # (stretch) over/undersampling
│   ├── pipeline.py                    # orchestrates end-to-end run; writes Results Store
│   └── results_store.py               # read/write helpers for results/
│
├── dashboard/
│   ├── app.py                         # Streamlit entrypoint
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Model_Performance.py
│   │   ├── 3_Fairness_Analysis.py
│   │   ├── 4_Mitigation_Comparison.py
│   │   └── 5_Cross_Model_Comparison.py
│   └── components/
│       ├── charts.py                  # reusable Plotly/Matplotlib chart builders
│       └── text.py                    # plain-language metric explanations, disclaimers
│
├── results/                           # Results Store (generated, gitignored except schema sample)
│   ├── performance_metrics.csv
│   ├── fairness_metrics.csv
│   ├── run_config.json
│   └── models/
│       ├── baseline_lr.pkl / dt.pkl / rf.pkl
│       └── mitigated_reweigh_lr.pkl / ... / mitigated_threshold_lr.pkl / ...
│
├── notebooks/
│   └── exploration.ipynb              # EDA and prototyping only, not production logic
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_fairness_metrics.py
│   └── test_mitigation.py
│
├── reports/
│   └── FairLens_Technical_Report.pdf  # Final academic report/synopsis output
│
├── requirements.txt
├── run_pipeline.py                    # CLI entrypoint: `python run_pipeline.py`
└── README.md
```

---

## 5. Module Design

### 5.1 `config.py`

Centralizes all run parameters so the entire pipeline is reproducible from one file:

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
DATA_PATH = "data/raw/adult.csv"
AGE_BINS = [0, 25, 45, 65, 120]
AGE_LABELS = ["<25", "25-44", "45-64", "65+"]
SENSITIVE_ATTRIBUTES = ["gender", "age_group"]
TARGET_COLUMN = "income"          # binary: >50K vs <=50K
MODELS = ["logistic_regression", "decision_tree", "random_forest"]
MITIGATIONS = ["reweighing", "threshold_adjustment"]  # "sampling" optional/stretch
LOW_SAMPLE_THRESHOLD = 30          # min group size in test set before flagging as low-confidence
METRIC_PRECISION = 3               # decimal places for displayed metrics
```

### 5.2 `data/loader.py` and `data/preprocess.py`

- `load_raw_data(path) -> pd.DataFrame`: reads CSV, strips whitespace from string columns (a known quirk of the Adult dataset), replaces `?` with `NaN`.
- `preprocess(df, config) -> PreprocessedData`: a dataclass/namedtuple bundling `X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, report`.
  - Missing values: rows with `NaN` in categorical columns (`workclass`, `occupation`, `native-country`) dropped — documented rationale: dataset's missingness is a small fraction (~7%) and MCAR-like enough for a course-scope project; this is explicitly noted as a simplification, not a rigorous causal claim.
  - Categorical encoding: one-hot encoding via `pd.get_dummies` for nominal features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `native-country`); binary encoding for `gender` (kept as-is, since it doubles as a sensitive attribute).
  - Numerical scaling: `StandardScaler` from Scikit-learn applied to continuous features (`age`, `fnlwgt`, `capital-gain`, `capital-loss`, `hours-per-week`) — fit on train, applied to test only (no leakage).
  - Age group binning: `pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)`.
  - Sensitive attribute handling: `gender` and `age_group` are extracted into a separate `sensitive_df` **before** being (optionally) dropped from the model's feature matrix `X`. Configurable flag `INCLUDE_SENSITIVE_AS_FEATURE` (default `False`) — baseline experiments exclude sensitive attributes as direct model inputs (a common fairness-through-unawareness baseline), with the trade-off (proxy discrimination is still possible) explicitly documented in the technical report.
  - Split: `train_test_split(..., test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)`.
  - Report: dict of row counts, dropped-row count, demographic counts per split — written to `run_config.json` under `preprocessing_report`.

### 5.3 `models/train.py`

```python
def train_baseline_models(X_train, y_train, config) -> dict[str, BaseEstimator]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=config.RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=config.RANDOM_SEED),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
```
Hyperparameters are intentionally simple/fixed (not exhaustively tuned) — the project's goal is fairness comparison under consistent conditions, not maximizing raw accuracy. This is documented as a scope decision.

### 5.4 `evaluation/performance.py`

```python
def evaluate_performance(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
```

### 5.5 `evaluation/fairness.py`

Uses Fairlearn's `MetricFrame` to compute group-wise metrics, from which the four fairness metrics are derived:

```python
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

def evaluate_fairness(y_true, y_pred, sensitive_feature) -> dict:
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feature,
    )
    by_group = mf.by_group

    demographic_parity_diff = by_group["selection_rate"].max() - by_group["selection_rate"].min()
    equal_opportunity_diff = by_group["tpr"].max() - by_group["tpr"].min()
    equalized_odds_diff = max(
        by_group["tpr"].max() - by_group["tpr"].min(),
        by_group["fpr"].max() - by_group["fpr"].min(),
    )
    disparate_impact_ratio = (
        by_group["selection_rate"].min() / by_group["selection_rate"].max()
        if by_group["selection_rate"].max() > 0 else float("nan")
    )

    return {
        "demographic_parity_diff": demographic_parity_diff,
        "equal_opportunity_diff": equal_opportunity_diff,
        "equalized_odds_diff": equalized_odds_diff,
        "disparate_impact_ratio": disparate_impact_ratio,
        "group_counts": by_group_counts(sensitive_feature),  # for low-sample-size flagging
    }
```

**Metric definitions (formal):**

| Metric | Formula (per group *g*) | Ideal value |
|---|---|---|
| Demographic Parity Difference | `max(P(ŷ=1\|g)) − min(P(ŷ=1\|g))` | 0 |
| Equal Opportunity Difference | `max(TPR_g) − min(TPR_g)` | 0 |
| Equalized Odds Difference | `max(max(TPR_g)−min(TPR_g), max(FPR_g)−min(FPR_g))` | 0 |
| Disparate Impact Ratio | `min(P(ŷ=1\|g)) / max(P(ŷ=1\|g))` | 1 (commonly acceptable range: 0.8–1.25, the "80% rule") |

**Undefined-value handling:** if `max(selection_rate) == 0` (no group received a positive prediction), disparate impact is returned as `NaN` and flagged `"undefined"` at the results-store/dashboard layer, per FSD §7.

**Low-sample-size flagging:** `group_counts` compared against `config.LOW_SAMPLE_THRESHOLD`; any group below threshold is tagged, and the dashboard renders that row with a caution indicator (FSD §4.6).

### 5.6 `mitigation/reweighing.py`

Implements the classic Kamiran & Calders reweighing approach: each training instance is weighted by the ratio of its *expected* joint probability (if sensitive attribute and label were independent) to its *observed* joint probability.

```python
def compute_reweighing_weights(y_train, sensitive_train) -> np.ndarray:
    weights = np.ones(len(y_train))
    for group in sensitive_train.unique():
        for label in y_train.unique():
            mask = (sensitive_train == group) & (y_train == label)
            p_expected = (sensitive_train == group).mean() * (y_train == label).mean()
            p_observed = mask.mean()
            weights[mask] = p_expected / p_observed if p_observed > 0 else 1.0
    return weights

def train_reweighed_models(X_train, y_train, sensitive_train, config) -> dict:
    weights = compute_reweighing_weights(y_train, sensitive_train)
    models = build_model_instances(config)   # fresh, untrained instances
    for name, model in models.items():
        model.fit(X_train, y_train, sample_weight=weights)
    return models
```

### 5.7 `mitigation/threshold_adjustment.py`

Wraps Fairlearn's `ThresholdOptimizer`, applied post-hoc to a trained (baseline or reweighed) model:

```python
from fairlearn.postprocessing import ThresholdOptimizer

def apply_threshold_adjustment(base_estimator, X_train, y_train, sensitive_train, constraint="demographic_parity"):
    optimizer = ThresholdOptimizer(
        estimator=base_estimator,
        constraints=constraint,
        predict_method="predict_proba",
        prefit=True,
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    return optimizer
```
The `constraint` parameter is configurable per experiment run (`demographic_parity` or `equalized_odds`), and both are run and reported when time permits, so the trade-off between fairness *definitions* — not just accuracy vs. fairness — is also visible.

### 5.8 `mitigation/sampling.py` (stretch goal)
Implements simple oversampling of minority group–label combinations (e.g., via `imblearn.over_sampling.RandomOverSampler` applied within group-label strata) as a third, optional mitigation technique, following the same interface contract as reweighing so it can be dropped into the pipeline without touching orchestration code.

### 5.9 `pipeline.py` — Orchestration

```python
def run_full_pipeline(config):
    df = load_raw_data(config.DATA_PATH)
    data = preprocess(df, config)

    baseline_models = train_baseline_models(data.X_train, data.y_train, config)
    results = []
    for name, model in baseline_models.items():
        results.append(build_result_row(name, "baseline", model, data, config))

    reweighed_models = train_reweighed_models(data.X_train, data.y_train, data.sensitive_train, config)
    for name, model in reweighed_models.items():
        results.append(build_result_row(name, "reweighing", model, data, config))

    for name, model in baseline_models.items():
        adjusted = apply_threshold_adjustment(model, data.X_train, data.y_train, data.sensitive_train)
        results.append(build_result_row(name, "threshold_adjustment", adjusted, data, config, is_postprocessor=True))

    persist_results(results, config)   # writes performance_metrics.csv, fairness_metrics.csv, run_config.json, models/*.pkl
```

`build_result_row` internally calls `evaluate_performance` and, for each sensitive attribute in `config.SENSITIVE_ATTRIBUTES`, `evaluate_fairness`, flattening everything into row(s) matching the schemas in §6.

### 5.10 `results_store.py`
Thin read/write layer so both `pipeline.py` and the dashboard use a single, consistent interface (`save_results()`, `load_performance_metrics()`, `load_fairness_metrics()`, `load_model(name, mitigation_state)`), keeping the Results Store's on-disk format an implementation detail hidden from both callers.

---

## 6. Data Schemas (Results Store)

### 6.1 `performance_metrics.csv`

| Column | Type | Description |
|---|---|---|
| `model` | string | `logistic_regression` \| `decision_tree` \| `random_forest` |
| `mitigation_state` | string | `baseline` \| `reweighing` \| `threshold_adjustment` \| `sampling` |
| `accuracy` | float | |
| `precision` | float | |
| `recall` | float | |
| `f1` | float | |
| `roc_auc` | float | |

### 6.2 `fairness_metrics.csv`

| Column | Type | Description |
|---|---|---|
| `model` | string | |
| `mitigation_state` | string | |
| `sensitive_attribute` | string | `gender` \| `age_group` |
| `demographic_parity_diff` | float | |
| `equal_opportunity_diff` | float | |
| `equalized_odds_diff` | float | |
| `disparate_impact_ratio` | float or `NaN` | `NaN` → rendered "undefined" downstream |
| `min_group_size` | int | smallest subgroup count in test set, for low-sample flagging |
| `low_confidence_flag` | bool | `min_group_size < LOW_SAMPLE_THRESHOLD` |

### 6.3 `run_config.json`
Snapshot of `config.py` values used for the run, plus the `preprocessing_report` dict (§5.2), so every results file is traceable to the exact configuration that produced it — this is the mechanism satisfying FR-14 (reproducibility/export).

---

## 7. Dashboard Technical Design

- **Framework:** Streamlit multipage app (`dashboard/pages/`), matching the 5 screens defined in `fsd.md` §4.
- **Data access:** all pages call `results_store.load_*()` functions, wrapped in `@st.cache_data` to avoid re-reading CSVs on every widget interaction.
- **State:** Streamlit's `st.session_state` holds the currently selected model / sensitive attribute / mitigation technique so choices persist across page navigation within a session.
- **Charts:** Matplotlib/Seaborn figures rendered via `st.pyplot()` for static comparison plots (per PRD tool list); simple interactive elements (dropdowns, sliders for future extensions) via native Streamlit widgets.
- **Disclaimers:** `components/text.py` centralizes the FR-15 disclaimer string and per-metric plain-language explanations, imported into every page's header/footer, so wording changes happen in one place.
- **Undefined/low-confidence rendering:** a small helper `format_metric(value, flag)` in `components/charts.py` renders `NaN` as *"undefined"* and low-confidence rows with a caution icon/greyed style, implementing FSD §4.6 and §7 consistently across pages.

---

## 8. Reproducibility Strategy

- Single `RANDOM_SEED` (42) propagated to: train/test split, all three model constructors, and any stochastic mitigation step (e.g., sampling).
- `run_pipeline.py` is the single entrypoint (`python run_pipeline.py`) that regenerates the entire Results Store from raw data — no manual/notebook-only steps are required for the graded deliverable.
- `run_config.json` captures the exact configuration per run, enabling an evaluator to confirm dashboard numbers match a fresh pipeline run.
- `requirements.txt` pins library versions (Scikit-learn, Fairlearn, Pandas, Streamlit, etc.) to avoid version-drift changing results between team members' machines.

---

## 9. Testing Strategy

| Test file | Coverage |
|---|---|
| `tests/test_preprocess.py` | Missing-value handling, encoding correctness, age-bin boundaries, train/test split stratification, no data leakage (scaler fit only on train) |
| `tests/test_fairness_metrics.py` | Correctness of DP/EO/EOdds/DI formulas against hand-computed toy examples; undefined-ratio handling (division by zero) |
| `tests/test_mitigation.py` | Reweighing weights sum/behavior sanity checks (e.g., weighted joint distribution approx. independent); threshold optimizer produces valid probability-based decisions |

Tests run via `pytest`; a lightweight CI step (or manual pre-submission check) ensures `pytest` passes and `run_pipeline.py` completes without errors before each milestone.

---

## 10. Non-Functional Implementation Notes

| NFR (from PRD) | Implementation Approach |
|---|---|
| Reproducibility | §8 (seed propagation, run_config.json, pinned requirements) |
| Performance (pipeline runtime) | Dataset is small (~49K rows); Scikit-learn models on this scale train in seconds to low minutes on CPU — no distributed compute needed |
| Interpretability | `components/text.py` plain-language explanations shown alongside every metric |
| Usability | Streamlit multipage nav; consistent widget placement across pages |
| Maintainability | Clear module boundaries (`data/`, `models/`, `evaluation/`, `mitigation/`) with single-responsibility functions; mitigation techniques share a common interface for easy extension |
| Privacy | No PII; public anonymized dataset only; no user data collected by the dashboard itself |
| Transparency | `run_config.json` + dashboard Overview page explicitly list sensitive attributes, models, and techniques used |
| Portability | Pure Python + Streamlit; runs via `pip install -r requirements.txt` and `streamlit run dashboard/app.py` on any standard machine, no cloud dependency |

---

## 11. Deployment / Run Instructions (Local)

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the analysis pipeline (generates results/)
python run_pipeline.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

No external hosting is required for the course deliverable; the dashboard is demonstrated locally. Optional future extension: containerize via Docker or deploy to Streamlit Community Cloud for remote demo access.

---

## 12. Known Limitations (Technical)

- Dataset (Adult Income, 1994 US Census) is dated and US-specific; results demonstrate methodology, not generalizable real-world fairness claims.
- `gender` in the dataset is binary as recorded; the framework does not attempt to correct or extend this limitation, and this is explicitly documented, not hidden.
- Fairness metrics are computed per sensitive attribute independently; intersectional fairness (e.g., gender × age group jointly) is out of scope for this iteration (noted as future work).
- Hyperparameters are fixed/lightly tuned by design (comparative fairness study, not an accuracy-maximization exercise).
- Mitigation techniques (reweighing, threshold adjustment) are applied using library defaults/documented configurations rather than exhaustively tuned per technique.

---

## 13. Traceability to FSD / PRD

| FSD Module | Technical Design Section |
|---|---|
| 3.1 Data Ingestion & Preprocessing | §5.2 |
| 3.2 Model Training | §5.3 |
| 3.3 Performance Evaluation | §5.4 |
| 3.4 Fairness Evaluation | §5.5 |
| 3.5 Bias Mitigation | §5.6, §5.7, §5.8 |
| 3.6 Post-Mitigation Evaluation | §5.4, §5.5 (reused), §5.9 |
| 3.7 Results Store | §5.10, §6 |
| 4.x Dashboard | §7 |
| FR-14 (reproducibility/export) | §6.3, §8 |
| FR-15 (disclaimers) | §7 |
