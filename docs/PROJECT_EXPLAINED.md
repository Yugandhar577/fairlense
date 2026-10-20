# FairLens: a beginner's guide and teacher-facing project report

Updated: 12 October 2026

## 1. What is this project?

FairLens is a local application for training, testing, and comparing machine-learning models on tabular datasets. A user uploads a CSV, chooses what to predict, selects suitable models, starts training, and inspects the resulting performance and demographic-group measurements in a React dashboard.

The project has two complementary parts:

1. An original, reproducible study using the UCI Adult Income dataset, three classifiers, and two fairness interventions.
2. A configurable training workspace supporting uploaded datasets, 14 registered models, classification and regression, optional group analysis, and separate saved experiments.

It is not a new machine-learning algorithm. Its engineering contribution is connecting data configuration, safe model/task selection, a consistent evaluation protocol, group analysis, and reproducible results in one usable workflow. It does not establish that any model is universally fair or legally compliant.

## 2. The problem it addresses

An overall score can hide very different outcomes for different groups. For example, a classifier might identify positive cases more often in one age group than another, even when its overall accuracy looks reasonable.

FairLens helps a student or researcher ask three questions:

- How well does each selected model predict the target?
- How do outcomes or errors differ across selected demographic groups?
- For supported classification tasks, what changes after a fairness intervention?

The application measures these differences. It does not guarantee that a disparity exists, make a legal judgement, or automatically choose a morally correct definition of fairness.

## 3. The basic vocabulary

| Term | Plain-language meaning | Example |
|---|---|---|
| Dataset | A table of observations | One row per record, one column per attribute |
| CSV | A plain-text representation of a table | A header followed by comma-separated rows |
| Feature | An input the model can use | Experience or qualification |
| Target | The answer the model learns to predict | An outcome category or a price |
| Model | A mathematical prediction method | Logistic Regression or Random Forest |
| Training | Learning a model's parameters from examples | Fitting a relationship between inputs and targets |
| Validation | Separate data reserved for decisions such as thresholds | Choosing group-specific decision thresholds |
| Test set | Held-out observations used to measure results | Checking predictions after fitting is complete |
| Benchmark | A consistent comparison of methods | Comparing two classifiers on identical test rows |
| Sensitive attribute | A column defining groups whose outcomes should be examined | A documented age-band or gender column |
| Intersectional group | A combination of sensitive attributes | Gender × age band |
| Mitigation | An intervention intended to address a specified disparity | Reweighing training examples |
| Baseline | A model without the selected mitigation | The ordinary fitted classifier |
| Experiment arm | One model/intervention combination | Random Forest with reweighing |
| API | The interface used by the browser to request backend operations | Submitting a training request |

The target must never be included as a feature. Otherwise, the model can effectively see the answer. Other columns can also reveal the answer indirectly, so the user must understand their dataset.

## 4. Classification and regression are different

Classification predicts a category. Binary classification has two possible categories, such as `yes` and `no`. Multiclass classification has more than two categories, such as three different outcome types. The workspace supports 2–30 classes.

Regression predicts a number, such as a price or a measured quantity. It requires a finite numeric target with at least two distinct values.

Despite its name, **Logistic Regression is a classifier**. Linear Regression is a regressor. They cannot be compared as interchangeable models in one task. For a category target, compare Logistic Regression with other classifiers. For a numeric target, compare Linear Regression with other regressors. Use separate experiments if investigating both formulations.

FairLens filters the model dropdowns by task type and also checks compatibility on the server. Changing browser code cannot bypass those checks.

## 5. How information moves through the system

```text
User opens the React application
          |
          v
Uploads a CSV or chooses an existing local upload
          |
          v
Chooses task, target, features, sensitive groups,
model count, models, split settings, and optional mitigation
          |
          v
Clicks Train: an explicit request goes to the Python service
          |
          v
Backend validates settings and creates an isolated run
          |
          v
Background worker cleans and splits selected observations
          |
          +--> Training rows: fit preprocessing and estimators
          |
          +--> Validation rows: fit selected threshold intervention
          |
          +--> Test rows: measure performance and group outcomes
          |
          v
Save real measurements, configuration, preprocessor, and estimators
          |
          v
Dashboard reads saved measurements
          |
          v
User compares models/groups/interventions and exports results
```

Only clicking Train starts a training operation. Changing a result filter, navigating between pages, refreshing a saved run, or downloading a table does not retrain models.

The server runs one experiment at a time. Within it, the selected estimators and interventions train sequentially. Selecting seven models means comparing seven models in that experiment, not starting seven independent simultaneous workers. Training continues while the user views another page, provided the Python process remains running.

## 6. What happens to uploaded data?

### Validation and profiling

The file must be a UTF-8, comma-separated CSV with a header row. The current limits are 20 MB, 20–50,000 rows, and 2–100 uniquely named columns. Binary files, malformed records, duplicate column names, and infinite numeric values are rejected. The interface receives column names, types, missing counts, distinct-value counts, and small lists of possible class values.

These limits intentionally describe the supported scope. “Different datasets” means compatible tabular CSVs—not arbitrary images, audio, spreadsheets, or every machine-learning problem.

### Choosing inputs and groups

The user selects the target and predictive features. Sensitive columns are optional; without them, performance benchmarking still works but demographic-group analysis is unavailable. Up to three sensitive columns can be selected, with an optional intersection. Each sensitive view must have 2–40 groups. A continuous attribute such as exact age should first be converted into meaningful bands.

Sensitive columns and the target are excluded from predictive features. This does not guarantee fairness: other inputs may be associated with those attributes or act as proxies.

### Cleaning

Rows missing the target or a selected sensitive value are removed. Exact duplicates across selected columns are also removed so identical selected observations cannot land in both training and test. Missing ordinary features are retained for training-based imputation.

The report records how many observations were removed. If too little usable data remains, the run fails with an explanation rather than producing invented measurements.

### Train/validation/test separation

With default settings, approximately 64% of cleaned observations are training, 16% validation, and 20% test. First, 20% is held out for test; then 20% of the remaining 80% becomes validation. Small datasets introduce rounding.

All selected models use the same rows and preprocessing. Classification is stratified by target class. Threshold mitigation additionally uses mitigation-group/class combinations and checks that training and validation contain both labels in every mitigation group.

This random row split assumes reasonably independent observations. It does not provide time-based or repeated-person separation. Such datasets need a suitable protocol before making generalization claims.

### Preprocessing

Numeric missing values are filled with the training median. Numeric features are standardized using training means and scales. Categorical missing values are filled with the training most-frequent category, then one-hot encoded into numeric indicator columns. Categories are pooled to at most 32 encoded columns per categorical feature. Unknown categories are handled without fitting on validation or test data.

The fitted preprocessor is reused on validation and test rows. It is never fitted on those partitions. A five-million-cell encoded-data limit bounds local memory use; very wide data may need fewer selected features or rows.

## 7. Which models can the user train?

### Classification models

| Model | What it does |
|---|---|
| Logistic Regression | Learns a weighted combination of inputs and turns it into class probabilities. |
| Decision Tree Classifier | Learns a sequence of feature-based questions that lead to a predicted class. |
| Random Forest Classifier | Combines many randomized trees to produce class predictions and probabilities. |
| Extra Trees Classifier | Combines trees with additional randomness in their feature split thresholds. |
| Gradient Boosting Classifier | Adds trees sequentially so later trees address errors in the current ensemble. |
| K-Nearest Neighbors Classifier | Predicts from the labels of the five nearest training examples in encoded feature space. |
| Gaussian Naive Bayes | Applies a probabilistic classification rule using Gaussian feature distributions and a conditional-independence assumption. |

### Regression models

| Model | What it does |
|---|---|
| Linear Regression | Fits a weighted linear relationship to predict a numeric target. |
| Ridge Regression | Fits a linear relationship with a penalty that shrinks large coefficients. |
| Lasso Regression | Fits a linear relationship with a penalty that can shrink some coefficients to zero. |
| Decision Tree Regressor | Divides feature space into regions and predicts a numeric value for each region. |
| Random Forest Regressor | Averages predictions from many randomized regression trees. |
| Gradient Boosting Regressor | Adds trees sequentially to improve numeric predictions. |
| K-Nearest Neighbors Regressor | Predicts from the numeric targets of the five nearest training examples. |

The user chooses 1–7 distinct models from the compatible list. Models use fixed defaults, not an automatic search for the best hyperparameters. For example, forests use 100 trees with maximum depth 10 in the uploaded-data workspace. Exact parameters are saved in the run configuration.

The registry can be extended by developers. It does not execute uploaded Python code, pickle files, or arbitrary neural-network weights. Clustering, forecasting, multilabel/multi-output tasks, and image/audio models are outside this version.

## 8. How performance is measured

### Classification

Think of a “positive” result as the outcome selected by the user for interpretation.

- Accuracy: the fraction of predictions that match their true class.
- Precision: among predicted positives, the fraction that are actually positive.
- Recall: among actual positives, the fraction correctly identified.
- F1: a harmonic mean combining precision and recall.
- ROC-AUC: how well continuous model scores rank classes, rather than performance at just one hard decision threshold.

For multiclass tasks, precision, recall, and F1 are macro-averaged: compute a measure for each class and average the class measures equally. ROC-AUC uses a macro one-vs-rest calculation. The table explicitly identifies this interpretation. Undefined AUC values are not replaced with zero.

### Regression

- MAE: mean absolute error—the average size of the prediction error, in target units. Lower is better.
- RMSE: root mean squared error—also in target units, but more affected by large errors. Lower is better.
- R²: performance relative to predicting the test target's mean. Higher is better; a negative value is possible and means the model performs worse under this measure than that reference.

It would be misleading to report classification accuracy for ordinary continuous regression predictions. FairLens instead chooses measurements appropriate to the task.

## 9. How group fairness is examined

Fairlearn's `MetricFrame` computes classification measurements separately for each selected group. FairLens retains the group sample counts and computes summary gaps and ratios.

| Measurement | Meaning | Mathematical reference |
|---|---|---|
| Selection rate | Fraction of a group's predictions assigned to the positive/focus class | No single rate is universally correct |
| True-positive rate | Fraction of a group's actual positive cases identified correctly | Used for equal opportunity |
| False-positive rate | Fraction of a group's actual negative cases predicted positive | Used for equalized odds |
| Demographic parity difference | Highest group selection rate minus lowest | Ideal difference: 0 |
| Equal opportunity difference | Highest group true-positive rate minus lowest | Ideal difference: 0 |
| Equalized odds difference | Larger of the true-positive-rate gap and false-positive-rate gap | Ideal difference: 0 |
| Disparate impact ratio | Lowest group selection rate divided by highest | Ideal ratio: 1 |

For multiple classes, this version evaluates the selected focus class versus all other outcomes. It does not establish fairness for every class simultaneously. Choosing the focus class is a substantive interpretation decision, not a decorative dashboard setting.

Groups below 30 test samples receive a caution. A true-positive rate with no actual positives has no denominator and remains undefined; the same applies to a false-positive rate with no actual negatives. If the required group rates are undefined, the corresponding summary gap is undefined too. A ratio is undefined when every group has a zero selection rate.

The commonly cited 0.80 selection-rate-ratio heuristic is a reference indicator. It is not proof of unlawful discrimination or a universal boundary between “fair” and “biased.” Different fairness definitions can disagree.

### Regression group analysis

Regression predictions are not silently converted into binary selections. Instead, FairLens reports each group's MAE, RMSE, mean prediction, and sample count. These help reveal different error patterns. They are descriptive group-error diagnostics, not the four classification fairness measures or a legal verdict.

## 10. What mitigation means in this version

Mitigation is optional, requires a binary classification target and a selected sensitive attribute, and adds arms alongside the baseline.

### Reweighing

Reweighing changes the importance of training examples using the training frequencies of group/label combinations. It does not change the held-out test distribution or rewrite the true test labels. A fresh estimator is fitted with these sample weights.

K-nearest neighbors does not support sample-weighted fitting, so the interface disables reweighing when that estimator is selected. The backend independently checks this restriction.

### Threshold adjustment

Threshold adjustment keeps the fitted baseline estimator and uses Fairlearn's `ThresholdOptimizer` to learn group-dependent decision rules on validation data, targeting demographic parity. It can randomize between threshold operations; a fixed prediction seed makes evaluation reproducible. The selected mitigation attribute is required when applying those rules.

The test set is not used to choose thresholds. Threshold-adjusted ROC-AUC still measures the underlying baseline estimator's continuous scores. The remaining measurements use adjusted hard predictions, so ROC-AUC can match baseline while accuracy and selection rates change.

### Cross-group outcomes

An intervention aimed at one sensitive attribute is measured on all selected sensitive views and their optional intersection. The interface compares each mitigated result with its matching baseline and labels movement toward or away from that metric's ideal. An improvement for one group view does not imply improvement for every other view or metric.

If the user chooses two models and both interventions, there are six arms: two baselines, two reweighed models, and two threshold-adjusted models. This is a count of experiment combinations, not six different learning algorithms.

## 11. What the user sees

| Screen | Purpose |
|---|---|
| Train models | Upload/select data, configure the task and models, start training, and inspect history/progress/errors |
| Overview | See dataset context, partition counts, baseline comparisons, and configuration |
| Model performance | Inspect the selected model and compare its intervention arms |
| Fairness analysis | Inspect classification group measurements or regression group errors |
| Mitigation comparison | Compare intervention results against the baseline for a selected sensitive view |
| Cross-model comparison | Compare compatible model families under the same experimental conditions |
| Cross-group effects | Examine how interventions change metrics across selected sensitive views |
| Methodology | Read the experiment protocol, interpretation limits, and reproducibility record |

The persistent **Results to view** dropdown switches between completed uploaded-data runs and the original Adult study. A completed run can be reopened after a browser reload or server restart. Each run has its own ID; a new run never overwrites another one.

The interface uses actual saved measurements, readable tables, meaningful charts, restrained colors, and responsive layouts. Keyboard focus, mobile navigation, and scrollable table access are supported. There are no fabricated live metrics or decorative research findings.

## 12. Technologies and responsibilities

| Technology | Its job in FairLens |
|---|---|
| Python | Runs validation, preprocessing, model fitting, measurement, and the local service. |
| pandas | Reads and cleans tabular data, profiles columns, and writes result tables. |
| NumPy | Represents numerical arrays and supports numerical operations and indexing. |
| scikit-learn | Provides estimators, splitting, preprocessing, and performance metrics. |
| Fairlearn | Computes group metrics and implements validation-fitted threshold optimization. |
| joblib | Saves fitted preprocessors and models as local artifacts. These files are never accepted as uploads. |
| Python HTTP server | Serves the React build and handles validated local API requests. It is not a production hosting/authentication system. |
| React | Builds the interactive browser interface from reusable components. |
| TypeScript | Describes frontend data shapes and checks many mistakes before building. |
| Vite and Node.js/npm | Install frontend dependencies, run development mode, and build browser assets. Node is not required to serve an already built dashboard. |
| Recharts | Displays real benchmark measurements as interactive charts. |
| CSS | Controls readable layout, spacing, focus styles, and mobile responsiveness. |
| Lucide | Supplies functional navigation/action icons. |
| pytest | Checks backend behavior, mathematical safeguards, and actual model fitting. |
| Playwright | Checks browser workflows, uploads, training, exports, navigation, and responsive behavior. |
| Git and GitHub | Track source changes and support sharing/review. Local datasets, runs, and model binaries are ignored. |

The current frontend is React, not Streamlit. Older design documents describe the original project plan, not all current features.

## 13. Files, persistence, and safety

```text
frontend/src/                 React interface and result views
src/fairlens/models/registry.py Supported models and constructors
src/fairlens/experiments.py    Generic preparation, fitting, and measurement
src/fairlens/workspace.py      Dataset persistence, run lifecycle, background worker
src/fairlens/dashboard_server.py Local API and frontend service
src/fairlens/pipeline.py       Original Adult study
workspace/datasets/<id>/       Uploaded CSV and column profile
workspace/runs/<id>/           Request, status, results, settings, and fitted artifacts
results/                      Original Adult measurements and models
examples/                     Clearly labelled synthetic teaching datasets
tests/ and frontend/e2e/      Backend and browser checks
```

Each uploaded-data run saves the request, final configuration, real measurement tables, complete JSON results, fitted preprocessor, and fitted estimators. Configuration includes selected columns, class mapping, seed, split settings/counts, model parameters, library versions, and a SHA-256 fingerprint of the normalized dataset. This helps reproduce and trace an experiment; identical environments and data remain important.

The browser exports allowlisted measurement CSVs and `run_config.json`, not executable model binaries or raw uploaded data. CSV exports escape formula-like text to reduce spreadsheet formula-injection risks. JSON snapshots use atomic replacement so readers do not receive half-written result files.

Uploads remain on the local filesystem until manually removed. They are not automatically anonymized, encrypted, or deleted. Upload only permitted data, remove direct identifiers, and protect the machine. The app binds to localhost, rejects inappropriate training requests, and is not designed for public multi-user deployment. Workspace limits are 100 datasets and 200 runs; archive unused folders with the service stopped.

A server restart marks interrupted runs failed because in-memory workers cannot resume. Completed results survive. There is no cancel/resume control in this version.

## 14. How to run and demonstrate it

For an already installed project with its frontend built:

```powershell
python run_dashboard.py
```

Open `http://127.0.0.1:8000/#train`. If frontend code has changed, first rebuild with `npm --prefix frontend run build`. Installation instructions are in the repository README.

### Demonstration A: two classifiers

1. Upload `examples/classification_demo.csv`.
2. Select Classification, target `outcome`, and positive class `yes`.
3. Select `region` as sensitive. Use `experience` and `qualification` as features. Optionally evaluate `age_band` as a second sensitive view and their intersection.
4. Choose two models: Logistic Regression and Decision Tree Classifier.
5. Optionally enable both mitigations targeting `region`. Keep the default seed/splits.
6. Start training, inspect actual progress, and open the completed run.
7. Show baseline performance, group counts/rates, and baseline-versus-mitigated changes. Explain that the tiny synthetic dataset is for teaching, not real-world evidence.

### Demonstration B: two regressors

1. Upload `examples/regression_demo.csv`.
2. Select Regression, target `price`, sensitive column `region`, and features `area` and `rooms`.
3. Choose two models: Linear Regression and Ridge Regression.
4. Train and show MAE, RMSE, R², and group errors.
5. Explain why classification accuracy and binary fairness mitigation are not offered for this task.

The demo files are invented, unitless teaching examples. Their small groups intentionally trigger cautions. Do not present their numerical results as research findings.

### Original Adult study

To regenerate the original study, supply the documented Adult CSV under `data/raw/adult.csv` and run `python run_pipeline.py`. Then choose **Original Adult study** in the results selector. That legacy study uses its documented 80/20 train/test protocol with a training-derived validation split for threshold fitting; uploaded-data experiments use the explicit shared three-way protocol described above.

## 15. What has been verified?

At handoff, the production frontend build, 41 Python tests, and 11 browser tests passed. Both documented demo configurations also trained successfully.

Backend checks cover every registered estimator, task compatibility, classification/regression/multiclass behavior, optional group analysis, disjoint partitions, training-only preprocessing statistics, reproducibility, real mitigation output, missing rates, upload validation, safe exports, history persistence, interrupted/failed runs, and the single-worker guard.

Browser checks cover original-study navigation and exports, responsive/keyboard behavior, real CSV upload, two-model classification with mitigation, two-model regression, saved-run reopening, result switching, and rejection of invalid uploads and cross-task model requests. Tests use an isolated workspace rather than the user's datasets and runs.

Tests establish implemented behavior, not scientific novelty, absence of all software defects, or validity for every dataset.

## 16. What to say to the teacher

### A short introduction

> “FairLens is a configurable machine-learning benchmarking and fairness-assessment workspace. Users bring a tabular dataset, select a prediction task and compatible models, and compare their held-out performance and group-level outcomes. For binary classification, they can also compare training-based reweighing and validation-based threshold adjustment. Every experiment is isolated and saves its settings and measurements so the comparison is traceable.”

### Likely questions

**Is this just training a few models?** Training is one stage. The application also validates datasets and task compatibility, applies training-only preprocessing, creates common evaluation partitions, measures group outcomes, evaluates compatible mitigation, preserves run history, and exposes reproducible comparisons and exports.

**Can it train any model on any dataset?** No. It currently supports 14 registered supervised estimators on compatible tabular CSVs. The architecture is extensible, but arbitrary algorithms, unstructured inputs, and unsupported tasks are not claimed.

**What checks fairness?** Fairlearn computes classification group metrics; the project defines the comparison protocol, selects meaningful groups/outcomes, handles undefined rates, and reports the results. Regression uses group-error diagnostics instead.

**Why not always choose the highest-accuracy model?** Overall accuracy does not describe all error types, group outcomes, uncertainty, or operational priorities. The project exposes several measurements so trade-offs can be inspected.

**Does a fairness intervention always help?** No. It may improve one metric and worsen another, affect different group views differently, or reduce performance. FairLens must report the observations without forcing a desired result.

**What makes the comparison reproducible?** Common partitions, fixed seeds/defaults, a training-only preprocessor, saved model settings and library versions, a dataset fingerprint, and separately stored fitted artifacts/results. Reproducibility does not remove uncertainty.

**Is this already publishable research?** Not merely because the configurable workspace exists. This is a useful engineering foundation. A publication would need a specific research question, a justified experimental protocol, comparison with prior work, and supported findings. Repeated-seed inference, confidence intervals, distribution-shift stress tests, and automatic subgroup discovery remain future work—not current claims.

## 17. Important limitations

FairLens is for education and research, not autonomous high-stakes decision-making. Dataset labels may themselves encode historical decisions. Sensitive-group membership and favorable outcomes require contextual interpretation. Small groups can make measured disparities unstable. This version does not add confidence intervals or significance testing, establish causality, guarantee fairness, select a universally best model, or prove a new research contribution. It intentionally reports what it measures and makes unsupported configurations explicit.

## References

- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn: model evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [Fairlearn: fairness assessment and mitigation](https://fairlearn.org/main/user_guide/index.html).
- [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).
