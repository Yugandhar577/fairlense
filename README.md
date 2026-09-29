# FairLens

FairLens is an educational, reproducible framework for comparing machine-learning performance and group fairness measurements on the UCI Adult Income dataset. It trains Logistic Regression, Decision Tree, and Random Forest classifiers, evaluates baseline and mitigated predictions, and presents precomputed results in a React dashboard.

FairLens is a benchmarking study, not a new learning algorithm or a system for making real-world decisions.

## Research focus

The current experiment studies whether mitigation aimed at gender also changes measured disparities for age groups and gender-by-age intersections. It compares each baseline with:

- **Reweighing**, using training-split sample weights derived from gender and the income label.
- **Threshold adjustment**, using Fairlearn's `ThresholdOptimizer` with demographic parity as its constraint.

All three sensitive views—gender, age group, and gender × age group—are evaluated for every model and mitigation state. The intersection is an evaluation attribute; it is not used as a model feature or mitigation input.

## Experiment flow

```text
UCI Adult data
  → cleaning and age-group construction
  → stratified train/test split
  → preprocessing fitted on training data
  → baseline model training
  → performance and fairness evaluation
  → training-only mitigation / training-validation threshold selection
  → final evaluation on the untouched test split
  → CSV/JSON results store
  → read-only Python results service
  → React dashboard
```

The test split is not used to fit preprocessing, calculate reweighing weights, fit models, or select thresholds. For threshold adjustment, the training split is further divided into a model-fitting subset and a validation subset. The final test split is used only for evaluation.

## Dataset

Download the UCI Adult dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). Place the training data file at:

```text
data/raw/adult.csv
```

The loader accepts the standard headerless UCI `adult.data` format and a CSV with column headers. The project does not commit the dataset. It originates from US Census data from 1994, is US-specific, and represents gender as binary. Results are a methodological case study and should not be generalized to other populations without additional validation.

## Metrics

Predictive performance: accuracy, precision, recall, F1, and ROC-AUC.

Fairness measurements, computed with Fairlearn `MetricFrame`: demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio. Group selection rates, true-positive rates, false-positive rates, and group counts are also retained.

These are distinct mathematical definitions and can lead to different conclusions. No single metric establishes that a model is universally fair or unfair. A disparate-impact ratio near 1 is the metric ideal. The commonly cited 0.80 convention is a reference heuristic, not a universal or legal verdict. Small groups and undefined ratios are flagged for cautious interpretation.

## Requirements and installation

Python 3.11 is the project target. The current implementation has also been exercised with Python 3.13. Building the frontend requires Node.js 22.12+ and npm. The frontend uses React, TypeScript, Vite, Recharts, Lucide icons, and locally bundled DM Sans fonts.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
```

## Run the project

Run the offline benchmark to train models and regenerate the results:

```powershell
python run_pipeline.py
```

Launch the dashboard after the pipeline has completed and the frontend has been built:

```powershell
python run_dashboard.py
```

Open **http://127.0.0.1:8000**. One Python process serves both the built React app and its read-only results API. Choose another local port with `python run_dashboard.py --port 8001`. Node is only needed to install/build or develop the frontend.

The dashboard reads the saved results and does not retrain models when opened. Its pages cover the overview, model performance, fairness analysis, mitigation comparison, cross-model comparison, and cross-group mitigation effects, plus a methodology reference. Filters preserve their selections across pages. Charts are interactive; the accompanying tables expose exact values. The **Export results** menu downloads the generated CSVs, and **Refresh saved results** reloads the latest experiment.

If results are absent or malformed, the dashboard explains how to regenerate them. Undefined values are displayed as undefined rather than zero. The configurable low-sample threshold is shown for small demographic groups.

### Frontend development

Run these commands in two terminals from the repository root:

```powershell
# Terminal 1: saved-results API
python run_dashboard.py

# Terminal 2: React development server, with hot reload
npm --prefix frontend run dev
```

Open the URL printed by Vite (normally http://127.0.0.1:5173). Vite proxies `/api` requests to the Python service on port 8000. `npm --prefix frontend run build` performs a TypeScript check and creates the standalone frontend assets. To refresh the built dashboard after frontend edits, rebuild and reload the browser.

The local API provides `GET /api/results` and `GET /api/export/<filename>` for the three metric CSVs. It does not execute training, load model binaries, or accept file uploads. It binds to localhost; external hosting and authentication are outside this local research app's scope.

Run the tests with:

```powershell
pytest
```

Browser checks use the saved experiment files and a locally installed Google Chrome:

```powershell
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

They exercise navigation, filters, CSV downloads, mobile layout, undefined values, and missing-results recovery. The test runner starts the local Python service if it is not already running.

## Results files

The pipeline writes generated artifacts to `results/`:

| File | Contents |
|---|---|
| `performance_metrics.csv` | Performance per model and experiment state |
| `fairness_metrics.csv` | Fairness summaries and per-group measurements |
| `fairness_changes.csv` | Each mitigation result compared with its matching baseline, including movement toward/away from each metric's ideal |
| `run_config.json` | Configuration and preprocessing report for the run |
| `models/` | Serialized fitted model and postprocessor artifacts |

These generated outputs and model binaries are ignored by Git. Regenerate them locally with `python run_pipeline.py`.

## Project structure

```text
frontend/               React + TypeScript dashboard, styling, and interactive charts
data/raw/               Local UCI Adult CSV (not tracked)
data/processed/         Local derived data (not tracked)
src/fairlens/           Configuration, data, models, metrics, mitigation, pipeline
results/                Generated metrics, run configuration, and model artifacts
tests/                  Preprocessing, fairness, and mitigation tests
notebooks/              Exploratory work only
run_pipeline.py         Offline experiment entry point
run_dashboard.py        Local dashboard and results API entry point
requirements.txt        Python dependencies
```

The Streamlit implementation was replaced by React. The original PRD, FSD, and technical design remain in the repository as historical design documents; this README describes the current setup. Build outputs (`frontend/dist/`) and Node dependencies are ignored by Git; `frontend/package-lock.json` records the frontend dependency versions.

## Responsible AI and limitations

FairLens is for education and research. It is not intended for high-stakes decisions, does not establish that a real organization or person is discriminatory, and is not an autonomous decision-maker. The Adult dataset has historical and demographic limitations; its findings are not claims about present-day populations. Intersectional groups can be small, which increases uncertainty in their measured rates. The project team is responsible for documenting assumptions and metric choices, testing the implementation, and communicating limitations. The dashboard collects no personal user data.
