import { useEffect, useState, type FormEvent } from "react";
import { Panel, Select } from "./components";
import {
  workspaceRequest,
  type Dataset,
  type ModelOption,
  type Run,
} from "./workspace-data";

export default function Workbench({
  onOpenRun,
  onRunsChange,
  refreshKey,
}: {
  onOpenRun: (id: string) => void;
  onRunsChange: (runs: Run[]) => void;
  refreshKey: number;
}) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [catalog, setCatalog] = useState<ModelOption[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [task, setTask] = useState("classification");
  const [target, setTarget] = useState("");
  const [positive, setPositive] = useState("");
  const [sensitive, setSensitive] = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>([]);
  const [intersection, setIntersection] = useState(false);
  const [models, setModels] = useState(["logistic_regression"]);
  const [mitigations, setMitigations] = useState<string[]>([]);
  const [mitigationAttribute, setMitigationAttribute] = useState("");
  const [seed, setSeed] = useState("42");
  const [testSize, setTestSize] = useState("0.2");
  const [validationSize, setValidationSize] = useState("0.2");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [ready, setReady] = useState(false);
  const dataset = datasets.find((d) => d.id === datasetId);
  const compatible = catalog.filter((m) => m.task === task);
  const targetColumn = dataset?.columns.find((c) => c.name === target);
  const binary = task === "classification" && targetColumn?.unique === 2;
  const weightCompatible = models.every(
    (id) => catalog.find((m) => m.id === id)?.sample_weight,
  );
  const active = runs.some((r) => ["queued", "running"].includes(r.status));

  function configure(d: Dataset) {
    setDatasetId(d.id);
    setTarget("");
    setPositive("");
    setSensitive([]);
    setFeatures(d.columns.map((c) => c.name));
    setMitigations([]);
    setMitigationAttribute("");
    setIntersection(false);
    setError("");
  }
  function refreshRuns(signal?: AbortSignal) {
    return workspaceRequest<{ runs: Run[] }>("runs", undefined, signal).then(
      (body) => {
        setRuns(body.runs);
        onRunsChange(body.runs);
      },
    );
  }
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      workspaceRequest<{ datasets: Dataset[] }>(
        "datasets",
        undefined,
        controller.signal,
      ),
      workspaceRequest<{ models: ModelOption[] }>(
        "catalog",
        undefined,
        controller.signal,
      ),
      refreshRuns(controller.signal),
    ])
      .then(([data, options]) => {
        setDatasets(data.datasets);
        setCatalog(options.models);
        setReady(true);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [refreshKey]);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const timer = window.setInterval(
      () =>
        refreshRuns(controller.signal).catch((e) => {
          if (!controller.signal.aborted) setError(e.message);
        }),
      1500,
    );
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, [active]);
  useEffect(() => {
    if (!binary) setMitigations([]);
  }, [binary]);
  useEffect(() => {
    if (!weightCompatible)
      setMitigations((previous) => previous.filter((m) => m !== "reweighing"));
  }, [weightCompatible]);

  async function upload(file?: File) {
    if (!file) return;
    setError("");
    setNotice("");
    if (
      file.size > 20 * 1024 * 1024 ||
      !file.name.toLowerCase().endsWith(".csv")
    ) {
      setError("Choose a UTF-8 CSV file no larger than 20 MB.");
      return;
    }
    setBusy(true);
    try {
      const text = new TextDecoder("utf-8", { fatal: true }).decode(
        await file.arrayBuffer(),
      );
      const profile = await workspaceRequest<Dataset>("datasets", {
        name: file.name,
        csv: text,
      });
      setDatasets((previous) => [...previous, profile]);
      configure(profile);
      setNotice(
        `${profile.name} uploaded locally. Choose the target column next.`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }
  async function train(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await workspaceRequest<Run>("runs", {
        dataset_id: datasetId,
        task,
        target,
        positive_label: positive,
        models,
        features: features.filter(
          (c) => c !== target && !sensitive.includes(c),
        ),
        sensitive,
        intersection,
        mitigations,
        mitigation_attribute: mitigationAttribute,
        random_seed: Number(seed),
        test_size: Number(testSize),
        validation_size: Number(validationSize),
      });
      await refreshRuns();
      setNotice(
        "Training started. Progress appears in Experiment history below; you can leave this page.",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start training.");
    } finally {
      setBusy(false);
    }
  }
  const toggle = (values: string[], value: string) =>
    values.includes(value)
      ? values.filter((v) => v !== value)
      : [...values, value];
  return (
    <div className="page-content workbench">
      <div className="notice">
        <p>
          Train selected models on your own tabular data. Classification
          predicts a category; regression predicts a number. Each experiment
          uses the same training, validation, and test rows for every selected
          model.
        </p>
      </div>
      {error && (
        <div className="notice amber-notice" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <p className="form-feedback" role="status">
          {notice}
        </p>
      )}
      {!ready && !error && <p role="status">Loading training workspace…</p>}
      <form onSubmit={train}>
        <Panel
          title="Dataset"
          subtitle="UTF-8, comma-separated CSV with a header row. 20–50,000 rows; 2–100 columns; up to 20 MB."
        >
          <div className="form-row">
            <label className="input-field">
              Upload dataset
              <input
                type="file"
                accept=".csv,text/csv"
                disabled={busy || !ready}
                onChange={(event) => {
                  void upload(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </label>
            {datasets.length > 0 && (
              <Select
                label="Uploaded dataset"
                value={datasetId}
                options={{
                  "": "Choose a dataset",
                  ...Object.fromEntries(
                    datasets.map((d) => [
                      d.id,
                      `${d.name} (${d.rows.toLocaleString()} rows)`,
                    ]),
                  ),
                }}
                onChange={(id) => {
                  const d = datasets.find((d) => d.id === id);
                  if (d) configure(d);
                  else setDatasetId("");
                }}
              />
            )}
          </div>
          <p className="field-help">
            Files and trained artifacts stay in this project's local workspace
            folder until you remove them. Upload only data you have permission
            to use; exclude direct personal identifiers.
          </p>
        </Panel>
        {dataset && (
          <>
            <Panel
              title="Prediction task"
              subtitle="Choose what the model should predict. A single experiment compares models for one task and target."
            >
              <div className="form-row">
                <Select
                  label="Task type"
                  value={task}
                  options={{
                    classification: "Classification — predict a category",
                    regression: "Regression — predict a number",
                  }}
                  onChange={(value) => {
                    setTask(value);
                    setModels([catalog.find((m) => m.task === value)!.id]);
                    setMitigations([]);
                  }}
                />
                <Select
                  label="Target column"
                  value={target}
                  options={{
                    "": "Choose the prediction target",
                    ...Object.fromEntries(
                      dataset.columns.map((c) => [c.name, c.name]),
                    ),
                  }}
                  onChange={(value) => {
                    setTarget(value);
                    const column = dataset.columns.find(
                      (c) => c.name === value,
                    );
                    setPositive(column?.values.at(-1) || "");
                    setSensitive((previous) =>
                      previous.filter((c) => c !== value),
                    );
                    setFeatures(
                      dataset.columns
                        .filter(
                          (c) =>
                            c.name !== value && !sensitive.includes(c.name),
                        )
                        .map((c) => c.name),
                    );
                    setMitigations([]);
                  }}
                />
                {task === "classification" && target && (
                  <Select
                    label="Positive / focus class"
                    value={positive}
                    options={Object.fromEntries(
                      (targetColumn?.values || []).map((v) => [v, v]),
                    )}
                    onChange={setPositive}
                  />
                )}
              </div>
              {task === "classification" && target && (
                <p className="field-help">
                  Classification supports 2–30 classes. For multiple classes,
                  performance uses macro averages; fairness compares the focus
                  class against all other classes. Choose the outcome whose
                  selection rate you want to inspect.
                </p>
              )}
              {task === "regression" && (
                <p className="field-help">
                  Regression requires a numeric target and uses MAE, RMSE, and
                  R². Classification fairness metrics and binary mitigation are
                  not applied to continuous predictions.
                </p>
              )}
            </Panel>
            <Panel
              title="Features and sensitive groups"
              subtitle="Features are model inputs. Sensitive columns define evaluation groups and are excluded from predictive inputs."
            >
              <div className="column-choices">
                <fieldset>
                  <legend>Sensitive columns (optional, up to 3)</legend>
                  {dataset.columns
                    .filter((c) => c.name !== target)
                    .map((c) => (
                      <label className="check-field" key={c.name}>
                        <input
                          type="checkbox"
                          checked={sensitive.includes(c.name)}
                          disabled={
                            !sensitive.includes(c.name) && sensitive.length >= 3
                          }
                          onChange={() => {
                            const next = toggle(sensitive, c.name);
                            setSensitive(next);
                            setMitigationAttribute(next[0] || "");
                            if (!next.length) setMitigations([]);
                          }}
                        />
                        {c.name}
                        <span className="muted">{c.unique} groups</span>
                      </label>
                    ))}
                </fieldset>
                <fieldset>
                  <legend>Predictive features</legend>
                  {dataset.columns
                    .filter(
                      (c) => c.name !== target && !sensitive.includes(c.name),
                    )
                    .map((c) => (
                      <label className="check-field" key={c.name}>
                        <input
                          type="checkbox"
                          checked={features.includes(c.name)}
                          onChange={() => setFeatures(toggle(features, c.name))}
                        />
                        {c.name}
                        <span className="muted">
                          {c.kind}; {c.missing} missing
                        </span>
                      </label>
                    ))}
                </fieldset>
              </div>
              {sensitive.length > 1 && (
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={intersection}
                    onChange={(e) => setIntersection(e.target.checked)}
                  />
                  Also evaluate the intersection of selected sensitive columns
                </label>
              )}
              <p className="field-help">
                Each sensitive view must have 2–40 groups. For continuous age,
                create an age-band column before uploading. Remove ID columns
                and any features that reveal the answer.
              </p>
            </Panel>
            <Panel
              title="Models"
              subtitle="Choose the number of distinct models to benchmark. Only models compatible with this task are offered."
            >
              <div className="form-row">
                <Select
                  label="Number of models"
                  value={String(models.length)}
                  options={Object.fromEntries(
                    Array.from(
                      { length: Math.min(7, compatible.length) },
                      (_, i) => [String(i + 1), String(i + 1)],
                    ),
                  )}
                  onChange={(value) => {
                    const count = Number(value);
                    const next = [...models];
                    for (const item of compatible)
                      if (!next.includes(item.id) && next.length < count)
                        next.push(item.id);
                    setModels(next.slice(0, count));
                  }}
                />
                {models.map((model, index) => (
                  <Select
                    key={index}
                    label={`Model ${index + 1}`}
                    value={model}
                    options={Object.fromEntries(
                      compatible
                        .filter((m) => m.id === model || !models.includes(m.id))
                        .map((m) => [m.id, m.name]),
                    )}
                    onChange={(value) =>
                      setModels(models.map((m, i) => (i === index ? value : m)))
                    }
                  />
                ))}
              </div>
              <p className="field-help">
                Models use fixed defaults rather than automatic hyperparameter
                search. Linear regression and logistic regression belong to
                different task types; benchmark them in separate experiments.
              </p>
            </Panel>
            <Panel
              title="Mitigation and reproducibility"
              subtitle="Mitigation is optional and adds experiments alongside each baseline model."
            >
              <div className="form-row">
                <label className="input-field">
                  Random seed
                  <input
                    type="number"
                    min="0"
                    max="4294967295"
                    step="1"
                    value={seed}
                    required
                    onChange={(e) => setSeed(e.target.value)}
                  />
                </label>
                <Select
                  label="Test share of all rows"
                  value={testSize}
                  options={{
                    "0.1": "10%",
                    "0.2": "20%",
                    "0.3": "30%",
                    "0.4": "40%",
                  }}
                  onChange={setTestSize}
                />
                <Select
                  label="Validation share of remaining rows"
                  value={validationSize}
                  options={{
                    "0.1": "10%",
                    "0.2": "20%",
                    "0.3": "30%",
                    "0.4": "40%",
                  }}
                  onChange={setValidationSize}
                />
              </div>
              {binary && sensitive.length > 0 ? (
                <>
                  <Select
                    label="Mitigation target attribute"
                    value={mitigationAttribute}
                    options={Object.fromEntries(sensitive.map((c) => [c, c]))}
                    onChange={setMitigationAttribute}
                  />
                  <label className="check-field">
                    <input
                      type="checkbox"
                      disabled={!weightCompatible}
                      checked={mitigations.includes("reweighing")}
                      onChange={() =>
                        setMitigations(toggle(mitigations, "reweighing"))
                      }
                    />
                    Reweighing
                    {!weightCompatible &&
                      " — unavailable for K-nearest neighbors"}
                  </label>
                  <label className="check-field">
                    <input
                      type="checkbox"
                      checked={mitigations.includes("threshold_adjustment")}
                      onChange={() =>
                        setMitigations(
                          toggle(mitigations, "threshold_adjustment"),
                        )
                      }
                    />
                    Threshold adjustment — demographic parity, learned on
                    validation only
                  </label>
                </>
              ) : (
                <p className="field-help">
                  Binary mitigation becomes available when you select a
                  two-class classification target and at least one sensitive
                  attribute.
                </p>
              )}
              <p className="field-help">
                With the default split: 64% training, 16% validation, 20% test,
                subject to row rounding. Cleaning removes missing targets/groups
                and exact duplicate selected observations. Preprocessing is
                fitted only on training rows.
              </p>
              <button
                className="button primary"
                type="submit"
                disabled={
                  busy ||
                  active ||
                  !target ||
                  !features.filter(
                    (c) => c !== target && !sensitive.includes(c),
                  ).length ||
                  (task === "classification" && !positive)
                }
              >
                {active
                  ? "An experiment is training"
                  : busy
                    ? "Saving…"
                    : `Train ${models.length} model${models.length === 1 ? "" : "s"}`}
              </button>
            </Panel>
          </>
        )}
      </form>
      <Panel
        title="Experiment history"
        subtitle="Runs are saved separately. Training never overwrites the original Adult benchmark or another experiment."
      >
        {active && (
          <p className="form-feedback" role="status">
            {
              runs.find((run) => ["queued", "running"].includes(run.status))
                ?.message
            }
          </p>
        )}
        {!runs.length ? (
          <p>
            No experiments yet. Upload a CSV and configure your first run above.
          </p>
        ) : (
          <div
            className="table-scroll"
            role="region"
            aria-label="Experiment history"
            tabIndex={0}
          >
            <table>
              <caption className="sr-only">
                Saved experiments and current training progress
              </caption>
              <thead>
                <tr>
                  <th>Dataset / run</th>
                  <th>Task</th>
                  <th>Models</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td className="strong-cell">
                      {run.dataset_name}
                      <span className="cell-secondary">
                        {new Date(run.created_at_utc).toLocaleString()} ·{" "}
                        {run.id.slice(0, 8)}
                      </span>
                    </td>
                    <td>{run.task}</td>
                    <td>{run.models.length}</td>
                    <td className="run-status">
                      {run.status}
                      <span className="cell-secondary">{run.message}</span>
                    </td>
                    <td>
                      {run.status === "completed" && (
                        <button
                          type="button"
                          className="text-link"
                          onClick={() => onOpenRun(run.id)}
                        >
                          View results
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
