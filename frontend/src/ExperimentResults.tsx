import { useEffect, useState } from "react";
import {
  Bars,
  Effect,
  GroupTable,
  Legend,
  Panel,
  Select,
  Stat,
} from "./components";
import {
  format,
  metricNames,
  metricDescriptions,
  signed,
  stateNames,
  type FairnessKey,
  type MetricValue,
} from "./data";
import type { ExperimentResult } from "./workspace-data";

export default function ExperimentResults({
  data,
  page,
}: {
  data: ExperimentResult;
  page: string;
}) {
  const c = data.config;
  const regression = c.task === "regression";
  const names = c.model_names;
  const [model, setModel] = useState(c.models[0]);
  const [state, setState] = useState("baseline");
  const [attribute, setAttribute] = useState("");
  const [metric, setMetric] = useState<FairnessKey>("demographic_parity_diff");
  const views = [
    ...new Set(
      (regression ? data.group_errors : data.fairness).map(
        (r) => r.sensitive_attribute,
      ),
    ),
  ];
  const selectedAttribute = views.includes(attribute)
    ? attribute
    : views[0] || "";
  const selectedModel = c.models.includes(model) ? model : c.models[0];
  const selectedState = ["baseline", ...c.mitigations].includes(state)
    ? state
    : "baseline";
  useEffect(() => {
    setModel(c.models[0]);
    setState("baseline");
    setAttribute("");
  }, [c.created_at_utc]);
  const performances = data.performance.filter(
    (r) => r.model === selectedModel && r.mitigation_state === selectedState,
  );
  const metrics = regression
    ? { mae: "MAE", rmse: "RMSE", r2: "R²" }
    : {
        accuracy: "Accuracy",
        precision: "Precision",
        recall: "Recall",
        f1: "F1 score",
        roc_auc: "ROC-AUC",
      };
  const value = (row: Record<string, unknown> | undefined, key: string) =>
    row?.[key] as MetricValue | undefined;
  const comparisonPage = ["mitigation", "effects"].includes(page);
  const filters = (
    <div className="filter-bar">
      {page !== "models" && (
        <Select
          label="Model"
          value={selectedModel}
          options={names}
          onChange={setModel}
        />
      )}
      <Select
        label="Experiment state"
        value={selectedState}
        options={Object.fromEntries(
          ["baseline", ...c.mitigations].map((s) => [
            s,
            comparisonPage && s === "baseline"
              ? "All interventions"
              : stateNames[s],
          ]),
        )}
        onChange={setState}
      />
      {views.length > 0 && ["fairness", "mitigation"].includes(page) && (
        <Select
          label="Sensitive attribute"
          value={selectedAttribute}
          options={Object.fromEntries(views.map((v) => [v, v]))}
          onChange={setAttribute}
        />
      )}
      {!regression &&
        views.length > 0 &&
        ["fairness", "mitigation", "effects"].includes(page) && (
          <Select
            label="Fairness metric"
            value={metric}
            options={metricNames}
            onChange={(v) => setMetric(v as FairnessKey)}
          />
        )}
    </div>
  );
  function performanceTable(
    rows: ExperimentResult["performance"],
    title = "Measured performance",
  ) {
    return (
      <Panel
        title={title}
        subtitle={
          regression
            ? "MAE and RMSE are lower-is-better, in target units. R² is higher-is-better and can be negative."
            : c.classes.length > 2
              ? "Precision, recall, F1, and one-vs-rest ROC-AUC are macro-averaged across classes."
              : "Precision, recall, and F1 refer to the selected positive class. ROC-AUC uses continuous scores."
        }
      >
        <div
          className="table-scroll"
          role="region"
          aria-label={title}
          tabIndex={0}
        >
          <table>
            <caption className="sr-only">
              Held-out performance measurements for the selected experiment
            </caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Experiment state</th>
                {Object.entries(metrics).map(([key, label]) => (
                  <th scope="col" key={key}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.model}-${row.mitigation_state}`}>
                  <th scope="row">{names[row.model]}</th>
                  <td>{stateNames[row.mitigation_state]}</td>
                  {Object.keys(metrics).map((key) => (
                    <td key={key}>{format(value(row, key))}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    );
  }
  const classificationGroups = data.fairness.filter(
    (r) =>
      r.model === selectedModel &&
      r.mitigation_state === selectedState &&
      r.sensitive_attribute === selectedAttribute,
  );
  const errorGroups = data.group_errors.filter(
    (r) =>
      r.model === selectedModel &&
      r.mitigation_state === selectedState &&
      r.sensitive_attribute === selectedAttribute,
  );

  if (page === "methodology")
    return (
      <div className="page-content">
        <Panel
          title="Experiment protocol"
          subtitle={`${c.dataset_name} · ${c.task} · target: ${c.target}`}
        >
          <ol className="protocol">
            <li>
              <h3>Clean selected observations</h3>
              <p>
                Rows with missing target or sensitive values and exact
                duplicates of the selected columns are removed. Other missing
                feature values are filled using training-only medians or
                most-frequent categories.
              </p>
            </li>
            <li>
              <h3>Separate training, validation, and test</h3>
              <p>
                {c.split_protocol} Seed: {c.random_seed}. Every model uses the
                same partitions. This random row protocol is not appropriate for
                time-series or repeated-person data without an appropriate
                upstream split.
              </p>
            </li>
            <li>
              <h3>Fit the selected models</h3>
              <p>
                Numeric features are standardized using training statistics.{" "}
                {c.categorical_encoding} Sensitive columns and the target are
                not predictive inputs. Models use fixed defaults, not a
                hyperparameter search.
              </p>
            </li>
            <li>
              <h3>Apply compatible mitigation</h3>
              <p>
                {c.mitigations.length
                  ? `${c.mitigations.map((m) => stateNames[m]).join(" and ")} target ${c.mitigation_attribute}. Reweighing uses training group/label frequencies. Threshold adjustment uses the baseline estimator and learns demographic-parity thresholds only on validation rows.`
                  : "No mitigation was selected for this experiment."}
              </p>
            </li>
            <li>
              <h3>Evaluate once and save</h3>
              <p>
                Performance and group measurements are computed on the held-out
                test rows, then saved. Navigating or filtering results does not
                retrain anything. Re-running experiments after inspecting test
                results is exploratory—not independent confirmation.
              </p>
            </li>
          </ol>
        </Panel>
        <Panel title="Measurement scope">
          <div className="prose">
            <p>
              {regression
                ? "Regression uses MAE, RMSE, R², and group-level errors. Differences in group errors are diagnostic; they are not demographic parity, equalized odds, or a universal fairness judgement."
                : `Classification fairness is measured for ${c.positive_label} versus all other outcomes. ${c.classes.length > 2 ? "This focus-class analysis does not establish fairness for every class." : "Both supported mitigations apply only to binary classification."}`}
            </p>
            <p>
              Small groups below 30 test observations receive a caution. Rates
              without a valid denominator remain undefined. The 0.80
              selection-rate ratio convention is only a reference heuristic—not
              a legal or universal fairness judgement. One run does not
              establish statistical significance, causality, or robustness.
            </p>
            <p>
              Uploaded data is stored locally in the project workspace and is
              not automatically deleted. Do not upload identifying or sensitive
              information without permission. No cloud training or arbitrary
              uploaded model execution is supported.
            </p>
          </div>
        </Panel>
        <Panel title="Reproducibility record">
          <dl className="run-details">
            <div>
              <dt>Predictive features</dt>
              <dd>{c.features.join(", ")}</dd>
            </div>
            <div>
              <dt>Sensitive columns</dt>
              <dd>{c.sensitive.join(", ") || "None"}</dd>
            </div>
            <div>
              <dt>Library versions</dt>
              <dd>
                {Object.entries(c.versions)
                  .map(([k, v]) => `${k} ${v}`)
                  .join(" · ")}
              </dd>
            </div>
            <div>
              <dt>Dataset fingerprint</dt>
              <dd className="fingerprint">{c.dataset_sha256}</dd>
            </div>
          </dl>
          <p className="field-help">
            Export run_config.json for the full settings, class mapping, and
            model parameters.
          </p>
        </Panel>
      </div>
    );
  if (page === "overview") {
    const r = c.preprocessing_report;
    return (
      <div className="page-content">
        <dl className="experiment-context" aria-label="Experiment context">
          <div>
            <dt>Dataset / task</dt>
            <dd>
              {c.dataset_name} · {c.task}
            </dd>
          </div>
          <div>
            <dt>Training / validation / test</dt>
            <dd>
              {r.training_samples.toLocaleString()} /{" "}
              {r.validation_samples.toLocaleString()} /{" "}
              {r.test_samples.toLocaleString()} rows
            </dd>
          </div>
          <div>
            <dt>Target / scope</dt>
            <dd>
              {c.target} · {c.models.length} models · {data.performance.length}{" "}
              model–intervention combinations
            </dd>
          </div>
        </dl>
        {performanceTable(
          data.performance.filter((r) => r.mitigation_state === "baseline"),
          "Baseline benchmark",
        )}
        <Panel title="Experiment configuration">
          <dl className="run-details">
            <div>
              <dt>Random seed</dt>
              <dd>{c.random_seed}</dd>
            </div>
            <div>
              <dt>Rows removed</dt>
              <dd>{r.rows_removed.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Encoded features</dt>
              <dd>{r.feature_count}</dd>
            </div>
            <div>
              <dt>Sensitive views</dt>
              <dd>{views.join(", ") || "None selected"}</dd>
            </div>
            <div>
              <dt>Mitigation</dt>
              <dd>
                {c.mitigations.map((m) => stateNames[m]).join(", ") || "None"}
              </dd>
            </div>
            {!regression && (
              <div>
                <dt>Positive / focus outcome</dt>
                <dd>{c.positive_label}</dd>
              </div>
            )}
          </dl>
        </Panel>
        <p className="notice">
          Explore performance, group measurements, and mitigation outcomes using
          the navigation. These are results for this dataset and split, not a
          universal model ranking.
        </p>
      </div>
    );
  }
  if (page === "performance" || page === "models") {
    const rows =
      page === "models"
        ? data.performance.filter((r) => r.mitigation_state === selectedState)
        : data.performance.filter((r) => r.model === selectedModel);
    const chartMetric = regression ? "rmse" : "accuracy";
    return (
      <div className="page-content">
        {filters}
        {page === "performance" && (
          <dl className={`metric-summary ${regression ? "three" : "five"}`}>
            {Object.entries(metrics).map(([key, label]) => (
              <Stat
                key={key}
                label={label}
                value={format(value(performances[0], key))}
                detail={stateNames[selectedState]}
              />
            ))}
          </dl>
        )}
        <Panel
          title={regression ? "RMSE comparison" : "Accuracy comparison"}
          subtitle="Actual held-out measurements; see the table for all metrics."
        >
          <Bars
            data={rows.map((row) => ({
              name:
                page === "models"
                  ? names[row.model]
                  : stateNames[row.mitigation_state],
              value: row[chartMetric],
            }))}
            series={[
              {
                key: "value",
                name: regression ? "RMSE" : "Accuracy",
                color: "#286f65",
              },
            ]}
            percent={!regression}
          />
        </Panel>
        {performanceTable(
          rows,
          page === "models" ? "Model comparison" : "Measured performance",
        )}
        {c.mitigations.includes("threshold_adjustment") && (
          <p className="notice">
            Threshold-adjusted ROC-AUC measures the underlying baseline
            estimator's scores, not the adjusted hard decisions. Its ROC-AUC
            therefore matches the baseline.
          </p>
        )}
      </div>
    );
  }
  if (page === "fairness")
    return (
      <div className="page-content">
        {filters}
        {!views.length ? (
          <p>
            No sensitive columns were selected. Start another experiment with
            group columns to enable group analysis.
          </p>
        ) : regression ? (
          <>
            <p className="notice">
              These are regression group-error diagnostics, not binary
              classification fairness metrics. Differences can reflect data
              composition and do not establish discrimination.
            </p>
            <Panel title="Error by group">
              <Bars
                data={errorGroups.map((r) => ({ name: r.group, value: r.mae }))}
                series={[{ key: "value", name: "MAE", color: "#286f65" }]}
              />
            </Panel>
            <Panel title="Group-level regression measurements">
              <div
                className="table-scroll"
                role="region"
                aria-label="Group-level regression measurements"
                tabIndex={0}
              >
                <table>
                  <caption className="sr-only">
                    Regression error and sample counts by group
                  </caption>
                  <thead>
                    <tr>
                      <th>Group</th>
                      <th>Test samples</th>
                      <th>MAE</th>
                      <th>RMSE</th>
                      <th>Mean prediction</th>
                      <th>Sample note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {errorGroups.map((r) => (
                      <tr key={r.group}>
                        <th scope="row">{r.group}</th>
                        <td>{r.group_count}</td>
                        <td>{format(r.mae)}</td>
                        <td>{format(r.rmse)}</td>
                        <td>{format(r.mean_prediction)}</td>
                        <td>{r.low_sample_warning ? "Below 30" : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        ) : (
          <>
            <p className="notice">
              Focus outcome: {c.positive_label}
              {c.classes.length > 2
                ? " versus every other class"
                : " (positive class)"}
              . {metricDescriptions[metric]}
            </p>
            <dl className="metric-summary">
              {Object.entries(metricNames).map(([key, label]) => (
                <Stat
                  key={key}
                  label={label}
                  value={format(classificationGroups[0]?.[key as FairnessKey])}
                  detail={`Metric ideal: ${key === "disparate_impact_ratio" ? 1 : 0}`}
                  accent={key === metric}
                />
              ))}
            </dl>
            <Panel title="Outcomes by group">
              <Legend
                items={[
                  { name: "Selection rate", color: "#286f65" },
                  { name: "True-positive rate", color: "#8a81bd" },
                  { name: "False-positive rate", color: "#b77b37" },
                ]}
              />
              <Bars
                data={classificationGroups.map((r) => ({
                  name: r.group,
                  selection: r.selection_rate,
                  tpr: r.tpr,
                  fpr: r.fpr,
                }))}
                series={[
                  {
                    key: "selection",
                    name: "Selection rate",
                    color: "#286f65",
                  },
                  { key: "tpr", name: "True-positive rate", color: "#8a81bd" },
                  { key: "fpr", name: "False-positive rate", color: "#b77b37" },
                ]}
              />
            </Panel>
            <Panel
              title="Group-level measurements"
              subtitle="Groups below 30 test samples receive a caution; denominator-free rates are undefined."
            >
              <GroupTable rows={classificationGroups} threshold={30} />
            </Panel>
          </>
        )}
      </div>
    );
  const changes = data.changes.filter(
    (r) =>
      r.model === selectedModel &&
      (selectedState === "baseline" || r.mitigation_state === selectedState) &&
      r.fairness_metric === metric &&
      (page !== "mitigation" || r.sensitive_attribute === selectedAttribute),
  );
  return (
    <div className="page-content">
      {filters}
      {!data.changes.length ? (
        <p className="notice">
          {regression
            ? "Binary fairness mitigation is not applicable to a regression task. Use Fairness analysis to inspect group errors."
            : "This run has no mitigation results. Select a binary target, a sensitive attribute, and a mitigation when configuring a new experiment."}
        </p>
      ) : (
        <>
          <p className="notice">
            Mitigation targets {c.mitigation_attribute}. Changes in other
            sensitive views are evaluated independently on the same test set.
            Movement toward one metric's ideal is not an overall fairness
            judgement.
          </p>
          <Panel
            title={
              page === "effects"
                ? "Cross-group comparison"
                : "Changes from baseline"
            }
            subtitle={metricNames[metric]}
          >
            <div
              className="table-scroll"
              role="region"
              aria-label="Mitigation changes"
              tabIndex={0}
            >
              <table>
                <caption className="sr-only">
                  Baseline and mitigated values by sensitive view
                </caption>
                <thead>
                  <tr>
                    <th>Intervention / group view</th>
                    <th>Baseline</th>
                    <th>Mitigated</th>
                    <th>Raw change</th>
                    <th>Distance change</th>
                    <th>Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {changes.map((r, i) => (
                    <tr key={i}>
                      <th scope="row">
                        {stateNames[r.mitigation_state]}
                        <span className="cell-secondary">
                          {r.sensitive_attribute}
                        </span>
                      </th>
                      <td>{format(r.baseline_value)}</td>
                      <td>{format(r.mitigated_value)}</td>
                      <td>{signed(r.raw_change)}</td>
                      <td>{signed(r.distance_to_ideal_change)}</td>
                      <td>
                        <Effect value={r.effect} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          {performanceTable(
            data.performance.filter((r) => r.model === selectedModel),
            "Performance context",
          )}
        </>
      )}
    </div>
  );
}
