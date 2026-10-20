# FairLens

FairLens is a local, educational workspace for training and benchmarking supervised machine-learning models on tabular CSV datasets. Upload data, select a prediction target, choose classification or regression, select 1–7 compatible models, and train an isolated experiment from the React interface. It includes 14 registered estimators, task-appropriate performance metrics, optional sensitive-group analysis, and binary fairness mitigation.

The original reproducible UCI Adult Income study remains available unchanged through the **Results to view → Original Adult study** selector. New uploads and experiments never overwrite it.

FairLens is a benchmarking study, not a new learning algorithm or a system for making real-world decisions.

## Research focus

The original Adult experiment studies whether mitigation aimed at gender also changes measured disparities for age groups and gender-by-age intersections. It compares each baseline with:

- **Reweighing**, using training-split sample weights derived from gender and the income label.
- **Threshold adjustment**, using Fairlearn's `ThresholdOptimizer` with demographic parity as its constraint.

All three sensitive views—gender, age group, and gender × age group—are evaluated for every model and mitigation state. The intersection is an evaluation attribute; it is not used as a model feature or mitigation input.

## Train your own models

After installation/build, start `python run_dashboard.py` and open `http://127.0.0.1:8000/#train`. The training workspace works even if original Adult results have not been generated.

1. Upload a UTF-8, comma-separated CSV with a header row, or select a previous upload.
2. Choose **Classification** for categories or **Regression** for a numeric target. Choose the target column; classification also requires a positive/focus class.
3. Choose predictive features. Exclude identifiers and columns that reveal the target. Optionally select up to three sensitive columns and their intersection; these columns are excluded from predictive inputs.
4. Set **Number of models**, then select a different compatible model in each dropdown.
5. Choose the random seed and split sizes. For binary classification with sensitive groups, optionally select reweighing and/or threshold adjustment and the mitigation attribute.
6. Click **Train models**. Actual progress and failures appear in **Experiment history**. One experiment runs at a time; multiple selected estimators train sequentially within that experiment.
7. Click **View results** on a completed run. Explore its benchmark pages and export tables or the full run configuration. Use **Results to view** to switch between completed runs and the original Adult study.

The examples in `examples/` are explicitly synthetic CSVs for demonstrating both task types. See `examples/README.md` for the exact settings. A beginner-friendly teacher/presentation report is in [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md).

### Supported models and tasks

| Classification (binary or multiclass) | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree Classifier | Ridge Regression |
| Random Forest Classifier | Lasso Regression |
| Extra Trees Classifier | Decision Tree Regressor |
| Gradient Boosting Classifier | Random Forest Regressor |
| K-Nearest Neighbors Classifier | Gradient Boosting Regressor |
| Gaussian Naive Bayes | K-Nearest Neighbors Regressor |

This is an extensible registry, not support for every possible model or file format. Add a registry entry, constructor, compatible metrics, and tests to support another estimator. Uploaded Python, pickle/joblib models, neural-network weights, image/audio data, clustering, multilabel prediction, forecasting, and arbitrary user code are not supported. Linear regression and logistic regression require separate experiments because they solve different prediction tasks.

### Uploaded-data protocol and limits

- CSV: 20 MB maximum; 20–50,000 rows; 2–100 uniquely named columns. Classification supports 2–30 target classes. Each sensitive view must contain 2–40 groups; pre-bin continuous attributes such as age.
- Cleaning removes missing targets/sensitive values and exact duplicates across the selected columns. Missing features are imputed. Numeric scaling and one-hot encoding are learned **only on the training partition**. Categories are pooled to at most 32 encoded columns per categorical feature, with a 5-million-cell encoded-data memory limit.
- Defaults reserve 20% of cleaned rows for test, then 20% of the remainder for validation: approximately 64% training, 16% validation, 20% test. All selected models and mitigation arms share the same partitions and training-only preprocessor. Classification splits are stratified by class; threshold mitigation uses class × mitigation-group stratification and checks group/class coverage.
- Baselines fit only on training. Reweighing uses only training labels/groups. Threshold adjustment reuses the fitted baseline and learns thresholds only on validation. Test data is used only for final measurements, never to fit preprocessing, models, or thresholds.
- Classification reports accuracy, precision, recall, F1, and ROC-AUC. Multiclass precision/recall/F1 and one-vs-rest ROC-AUC are macro-averaged. Fairness measures the selected focus class versus the rest, not every class simultaneously.
- Regression reports MAE, RMSE, and R², plus per-group MAE/RMSE/mean predictions when sensitive columns are selected. Binary fairness metrics and mitigation are not misapplied to regression.
- Reweighing and threshold adjustment are binary-classification-only. K-nearest neighbors does not accept sample weights, so reweighing is unavailable when it is selected. Small/unsupported configurations fail clearly rather than fabricating results.
- Each local workspace is limited to 100 datasets and 200 runs. There is no automatic deletion; archive unused folders manually while the service is stopped. There is no cancellation/resume of an interrupted training run; restart marks it failed, and completed history is retained.
- The random row split is for ordinary independent tabular observations. It does not provide temporal or repeated-person separation. Do not repeatedly tune against displayed test results and treat them as independent confirmation.

All uploads, fitted preprocessors, estimators, configuration snapshots, and measurements are stored under ignored `workspace/`. No upload is sent to a cloud service. Only validated JSON requests from the local interface can start training; model artifacts cannot be uploaded or downloaded through the service. This is a localhost research app, not an authenticated production/multi-user service.

## Original Adult experiment flow

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

Launch the dashboard after the frontend has been built. The original Adult views require its pipeline outputs; the upload/training workspace works without them:

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

The legacy API provides `GET /api/results` and `GET /api/export/<filename>`. The workspace API adds `GET /api/workspace/catalog`, dataset listing/upload, experiment submission/history/status, completed-run results, and allowlisted CSV/configuration exports. Only an explicit `POST /api/workspace/runs` starts training; reading/filtering results never does. The service binds to localhost; external hosting and authentication are outside its scope.

Run the tests with:

```powershell
pytest
```

Browser checks use the saved experiment files and a locally installed Google Chrome:

```powershell
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

They exercise navigation, filters, CSV downloads, mobile layout, undefined values, missing-results recovery, real CSV upload/training for both task types, run reopening, and invalid-input rejection. Tests launch an isolated service on port 8010 with an ignored test workspace, not the user's uploaded datasets/runs.

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
examples/               Small, explicitly synthetic classification/regression demo CSVs
docs/PROJECT_EXPLAINED.md Beginner-friendly teacher/presentation report
data/raw/               Local UCI Adult CSV (not tracked)
data/processed/         Local derived data (not tracked)
src/fairlens/           Configuration, data, models, metrics, mitigation, pipeline
results/                Generated metrics, run configuration, and model artifacts
workspace/              Local uploads and isolated experiments (not tracked)
tests/                  Preprocessing, fairness, and mitigation tests
notebooks/              Exploratory work only
run_pipeline.py         Offline experiment entry point
run_dashboard.py        Local dashboard and results API entry point
requirements.txt        Python dependencies
```

The Streamlit implementation was replaced by React. The original PRD, FSD, and technical design remain in the repository as historical Adult-study design documents; this README describes the current setup. `experiments.py`, `workspace.py`, and `models/registry.py` implement the new configurable workflow separately from the legacy `pipeline.py`. Build outputs (`frontend/dist/`) and Node dependencies are ignored by Git; `frontend/package-lock.json` records the frontend dependency versions.

## Responsible AI and limitations

FairLens is for education and research. It is not intended for high-stakes decisions, does not establish that a real organization or person is discriminatory, and is not an autonomous decision-maker. The Adult dataset has historical and demographic limitations; its findings are not claims about present-day populations. Intersectional groups can be small, which increases uncertainty in their measured rates. The project team is responsible for documenting assumptions and metric choices, testing the implementation, and communicating limitations. Uploaded data may contain personal information: it stays on the local filesystem until manually removed and is not anonymized automatically. Obtain permission, exclude direct identifiers, and protect access to the machine and workspace.
