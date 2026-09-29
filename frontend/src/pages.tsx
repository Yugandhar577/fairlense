import {
  ArrowRight,
  Database,
  GitCompareArrows,
  Layers3,
  ScanLine,
  FlaskConical,
  Info,
} from "lucide-react";
import {
  Bars,
  Effect,
  Empty,
  GroupTable,
  Legend,
  Panel,
  Select,
  Stat,
  Tradeoff,
} from "./components";
import {
  type FairnessKey,
  type PerformanceKey,
  type Results,
  format,
  ideal,
  metricDescriptions,
  metricNames,
  modelNames,
  performanceNames,
  signed,
  stateColors,
  stateNames,
  summarize,
  valid,
  viewNames,
} from "./data";

export interface Selection {
  model: string;
  state: string;
  attribute: string;
  metric: FairnessKey;
  performanceMetric: PerformanceKey;
  setModel: (value: string) => void;
  setState: (value: string) => void;
  setAttribute: (value: string) => void;
  setMetric: (value: FairnessKey) => void;
  setPerformanceMetric: (value: PerformanceKey) => void;
}
type Props = { data: Results; selection: Selection };
export const titles: Record<string, [string, string]> = {
  overview: [
    "Experiment overview",
    "Explore the balance between predictive performance and demographic fairness.",
  ],
  performance: [
    "Model performance",
    "Five performance measures. Three models. One shared test set.",
  ],
  fairness: [
    "Fairness analysis",
    "Look closer at outcomes across demographic and intersectional groups.",
  ],
  mitigation: [
    "Mitigation comparison",
    "Understand how each intervention changes performance and group disparities.",
  ],
  models: [
    "Cross-model comparison",
    "Compare model families under the same experimental conditions.",
  ],
  effects: [
    "Cross-group effects",
    "See how gender-targeted mitigation affects other demographic groups.",
  ],
  methodology: [
    "Study methodology",
    "The experimental protocol, metric definitions, and boundaries of this study.",
  ],
};

function Filters({
  data,
  selection: s,
  fields,
  noBaseline = false,
}: Props & { fields: string[]; noBaseline?: boolean }) {
  const models = Object.fromEntries(
    [...new Set(data.performance.map((r) => r.model))].map((key) => [
      key,
      modelNames[key] || key,
    ]),
  );
  const views = Object.fromEntries(
    [...new Set(data.fairness.map((r) => r.sensitive_attribute))].map((key) => [
      key,
      viewNames[key] || key,
    ]),
  );
  const states = Object.fromEntries(
    [...new Set(data.performance.map((r) => r.mitigation_state))]
      .filter((key) => !noBaseline || key !== "baseline")
      .map((key) => [key, stateNames[key] || key]),
  );
  return (
    <div className="filter-bar">
      {fields.includes("model") && (
        <Select
          label="Model"
          value={s.model}
          options={models}
          onChange={s.setModel}
        />
      )}
      {fields.includes("state") && (
        <Select
          label="Experiment state"
          value={noBaseline && s.state === "baseline" ? "reweighing" : s.state}
          options={states}
          onChange={s.setState}
        />
      )}
      {fields.includes("attribute") && (
        <Select
          label="Sensitive attribute"
          value={s.attribute}
          options={views}
          onChange={s.setAttribute}
        />
      )}
      {fields.includes("metric") && (
        <Select
          label="Fairness metric"
          value={s.metric}
          options={metricNames}
          onChange={(value) => s.setMetric(value as FairnessKey)}
        />
      )}
      {fields.includes("performance") && (
        <Select
          label="Performance metric"
          value={s.performanceMetric}
          options={performanceNames}
          onChange={(value) => s.setPerformanceMetric(value as PerformanceKey)}
        />
      )}
    </div>
  );
}

function RateNote({ data, selection: s }: Props) {
  const groups = data.fairness.filter(
    (row) =>
      row.model === s.model &&
      row.mitigation_state === s.state &&
      row.sensitive_attribute === s.attribute,
  );
  const small = groups.filter(
    (row) => row.group_count < data.config.low_sample_threshold,
  );
  return small.length ? (
    <div className="notice amber-notice">
      <Info size={17} />
      <span>
        {small.length} group{small.length > 1 ? "s have" : " has"} fewer than{" "}
        {data.config.low_sample_threshold} test samples:{" "}
        {small.map((g) => g.group).join(", ")}. Interpret these estimates
        cautiously.
      </span>
    </div>
  ) : null;
}

export function Overview({
  data,
  selection: s,
  navigate,
}: Props & { navigate: (page: string) => void }) {
  const report = data.config.preprocessing_report;
  const baselines = data.performance.filter(
    (row) => row.mitigation_state === "baseline",
  );
  const selectedView = {
    ...data,
    fairness: data.fairness.filter(
      (row) => row.sensitive_attribute === s.attribute,
    ),
  };
  const worsened = data.changes.filter(
    (row) => row.sensitive_attribute !== "gender" && row.effect === "worsened",
  ).length;
  const comparisons = data.changes.filter(
    (row) => row.sensitive_attribute !== "gender",
  ).length;
  return (
    <>
      <div className="stats-grid">
        <Stat
          label="Dataset records"
          value={report.input_rows.toLocaleString()}
          detail="UCI Adult · 1994 US Census"
          icon={<Database size={18} />}
        />
        <Stat
          label="Model families"
          value={String(baselines.length).padStart(2, "0")}
          detail="A shared evaluation protocol"
          icon={<Layers3 size={18} />}
        />
        <Stat
          label="Sensitive views"
          value={String(
            new Set(data.fairness.map((row) => row.sensitive_attribute)).size,
          ).padStart(2, "0")}
          detail="Including gender × age"
          icon={<ScanLine size={18} />}
        />
        <Stat
          label="Experiment arms"
          value={String(data.performance.length).padStart(2, "0")}
          detail="Baseline + two interventions"
          icon={<GitCompareArrows size={18} />}
          accent
        />
      </div>
      <div className="overview-columns">
        <Panel
          title="Performance meets fairness"
          subtitle="Each point is a model and mitigation combination."
          action={<span className="badge">Test-set results</span>}
        >
          <div className="compact-filters">
            <Select
              label="Sensitive attribute"
              value={s.attribute}
              options={viewNames}
              onChange={s.setAttribute}
            />
            <Select
              label="Fairness metric"
              value={s.metric}
              options={metricNames}
              onChange={(v) => s.setMetric(v as FairnessKey)}
            />
          </div>
          <Tradeoff
            rows={data.performance}
            data={selectedView}
            metric={s.metric}
          />
        </Panel>
        <Panel
          title="Experiment at a glance"
          subtitle="A reproducible Adult Income case study."
          className="run-panel"
        >
          <div className="dataset-emblem">
            <Database size={23} />
            <div>
              <strong>Adult Income</strong>
              <span>Binary classification · Income &gt;50K</span>
            </div>
          </div>
          <div className="split-labels">
            <span>
              Training <b>{(100 * (1 - data.config.test_size)).toFixed(0)}%</b>
            </span>
            <span>
              Test <b>{(100 * data.config.test_size).toFixed(0)}%</b>
            </span>
          </div>
          <div className="split-bar">
            <span style={{ width: `${(1 - data.config.test_size) * 100}%` }} />
          </div>
          <div className="split-counts">
            <span>{report.training_samples.toLocaleString()} samples</span>
            <span>{report.test_samples.toLocaleString()} samples</span>
          </div>
          <dl className="run-details">
            <div>
              <dt>Random seed</dt>
              <dd>{data.config.random_seed}</dd>
            </div>
            <div>
              <dt>Encoded features</dt>
              <dd>{report.feature_count}</dd>
            </div>
            <div>
              <dt>Mitigation target</dt>
              <dd>Gender</dd>
            </div>
            <div>
              <dt>Threshold constraint</dt>
              <dd>
                {
                  metricNames[
                    data.config.threshold_constraint === "equalized_odds"
                      ? "equalized_odds_diff"
                      : "demographic_parity_diff"
                  ]
                }
              </dd>
            </div>
          </dl>
          <button className="text-link" onClick={() => navigate("methodology")}>
            Explore the methodology <ArrowRight size={15} />
          </button>
        </Panel>
      </div>
      <Panel
        title="Baseline benchmark"
        subtitle="The starting point before applying mitigation."
        action={
          <button className="text-link" onClick={() => navigate("models")}>
            Compare models <ArrowRight size={15} />
          </button>
        }
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 score</th>
                <th>ROC-AUC</th>
                <th>Gender DP difference</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {baselines.map((row, index) => (
                <tr key={row.model}>
                  <td className="model-cell">
                    <span className="model-index">0{index + 1}</span>
                    {modelNames[row.model]}
                  </td>
                  <td>
                    <div className="mini-value">
                      {format(row.accuracy, true)}
                      <i
                        style={{
                          width: valid(row.accuracy)
                            ? `${row.accuracy * 70}px`
                            : 0,
                        }}
                      />
                    </div>
                  </td>
                  <td>{format(row.f1)}</td>
                  <td>{format(row.roc_auc)}</td>
                  <td>
                    {format(
                      data.fairness.find(
                        (f) =>
                          f.model === row.model &&
                          f.mitigation_state === "baseline" &&
                          f.sensitive_attribute === "gender",
                      )?.demographic_parity_diff,
                    )}
                  </td>
                  <td>
                    <span className="badge">Baseline</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="research-banner">
        <div className="research-icon">
          <FlaskConical size={24} />
        </div>
        <div>
          <span className="eyebrow">A CLOSER LOOK</span>
          <h3>What happens beyond the target group?</h3>
          <p>
            {comparisons
              ? `${worsened} of ${comparisons} age and intersectional metric comparisons moved away from their ideal in this run.`
              : "Explore the age and intersectional outcomes of gender-targeted mitigation."}
          </p>
        </div>
        <button className="button" onClick={() => navigate("effects")}>
          Explore cross-group effects <ArrowRight size={16} />
        </button>
      </div>
    </>
  );
}

export function PerformancePage({ data, selection: s }: Props) {
  const row = data.performance.find(
    (row) => row.model === s.model && row.mitigation_state === s.state,
  );
  const rows = data.performance.filter((row) => row.model === s.model);
  const series = Object.entries(stateColors).map(([key, color]) => ({
    key,
    name: stateNames[key],
    color,
  }));
  const bars = Object.entries(performanceNames).map(([key, name]) => ({
    name,
    ...Object.fromEntries(
      rows.map((row) => [row.mitigation_state, row[key as PerformanceKey]]),
    ),
  }));
  return (
    <>
      <Filters data={data} selection={s} fields={["model", "state"]} />
      <div className="stats-grid five">
        {Object.entries(performanceNames).map(([key, name]) => (
          <Stat
            key={key}
            label={name}
            value={format(row?.[key as PerformanceKey], key !== "roc_auc")}
            detail={stateNames[s.state]}
          />
        ))}
      </div>
      <Panel
        title="Performance across interventions"
        subtitle={modelNames[s.model]}
      >
        <Legend items={series} />
        <Bars data={bars} series={series} percent />
      </Panel>
      <Panel
        title="Measured performance"
        subtitle="All measures are calculated from saved experiment predictions."
      >
        <PerformanceTable rows={rows} />
      </Panel>
      <div className="notice">
        <Info size={17} />
        <span>
          For threshold adjustment, ROC-AUC measures the base estimator’s
          continuous scores. Accuracy, precision, recall, and F1 use the
          adjusted decisions.
        </span>
      </div>
    </>
  );
}
function PerformanceTable({ rows }: { rows: Results["performance"] }) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Experiment state</th>
            {Object.values(performanceNames).map((name) => (
              <th key={name}>{name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.model}-${row.mitigation_state}`}>
              <td className="strong-cell">{modelNames[row.model]}</td>
              <td>{stateNames[row.mitigation_state]}</td>
              {Object.keys(performanceNames).map((key) => (
                <td key={key}>{format(row[key as PerformanceKey])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function FairnessPage({ data, selection: s }: Props) {
  const rows = data.fairness.filter(
    (row) =>
      row.model === s.model &&
      row.mitigation_state === s.state &&
      row.sensitive_attribute === s.attribute,
  );
  const summary = rows[0];
  return (
    <>
      <Filters
        data={data}
        selection={s}
        fields={["model", "state", "attribute", "metric"]}
      />
      <div className="stats-grid">
        {Object.entries(metricNames).map(([key, name]) => (
          <Stat
            key={key}
            label={name}
            value={format(summary?.[key as FairnessKey])}
            detail={`Metric ideal: ${ideal(key as FairnessKey)}`}
            accent={key === s.metric}
          />
        ))}
      </div>
      <div className="notice">
        <Info size={17} />
        <span>
          {metricDescriptions[s.metric]}{" "}
          {s.metric === "disparate_impact_ratio" &&
          valid(summary?.disparate_impact_ratio) &&
          summary.disparate_impact_ratio < 0.8
            ? "The observed ratio is below the 0.80 reference heuristic; this is not a legal finding or an overall fairness verdict."
            : ""}
        </span>
      </div>
      <RateNote data={data} selection={s} />
      <Panel
        title="Outcomes by group"
        subtitle={`${viewNames[s.attribute]} · ${modelNames[s.model]} · ${stateNames[s.state]}`}
      >
        <Legend
          items={[
            { name: "Selection rate", color: "#286f65" },
            { name: "True-positive rate", color: "#9b8cbd" },
            { name: "False-positive rate", color: "#cda16a" },
          ]}
        />
        <Bars
          data={rows.map((row) => ({
            ...row,
            name: row.group.replace(" | ", " · "),
          }))}
          series={[
            { key: "selection_rate", name: "Selection rate", color: "#286f65" },
            { key: "tpr", name: "True-positive rate", color: "#9b8cbd" },
            { key: "fpr", name: "False-positive rate", color: "#cda16a" },
          ]}
        />
      </Panel>
      <Panel
        title="Group-level measurements"
        subtitle={`Low-sample threshold: ${data.config.low_sample_threshold} test records`}
      >
        <GroupTable rows={rows} threshold={data.config.low_sample_threshold} />
      </Panel>
    </>
  );
}

export function MitigationPage({ data, selection: s }: Props) {
  const rows = data.performance.filter((row) => row.model === s.model);
  const view = {
    ...data,
    fairness: data.fairness.filter(
      (row) => row.sensitive_attribute === s.attribute,
    ),
  };
  const summaries = summarize(
    view.fairness.filter((row) => row.model === s.model),
  );
  const changes = data.changes.filter(
    (row) =>
      row.model === s.model &&
      row.sensitive_attribute === s.attribute &&
      row.fairness_metric === s.metric,
  );
  return (
    <>
      <Filters
        data={data}
        selection={s}
        fields={["model", "attribute", "metric"]}
      />
      <div className="two-columns">
        <Panel
          title="Fairness by intervention"
          subtitle={`${metricNames[s.metric]} · ideal ${ideal(s.metric)}`}
        >
          <Bars
            data={summaries.map((row) => ({
              name: stateNames[row.mitigation_state],
              value: row[s.metric],
            }))}
            series={[
              { key: "value", name: metricNames[s.metric], color: "#286f65" },
            ]}
          />
        </Panel>
        <Panel
          title="Performance–fairness trade-off"
          subtitle="Compare the position of each intervention."
        >
          <Tradeoff data={view} rows={rows} metric={s.metric} />
        </Panel>
      </div>
      <Panel
        title="Changes from baseline"
        subtitle={`${modelNames[s.model]} · ${viewNames[s.attribute]}`}
      >
        <ChangeTable rows={changes} />
      </Panel>
      <Panel
        title="Performance context"
        subtitle="Interpret fairness changes alongside prediction quality."
      >
        <PerformanceTable rows={rows} />
      </Panel>
    </>
  );
}

export function ModelsPage({ data, selection: s }: Props) {
  const rows = data.performance.filter(
    (row) => row.mitigation_state === s.state,
  );
  const view = {
    ...data,
    fairness: data.fairness.filter(
      (row) =>
        row.sensitive_attribute === s.attribute &&
        row.mitigation_state === s.state,
    ),
  };
  return (
    <>
      <Filters
        data={data}
        selection={s}
        fields={["state", "attribute", "metric", "performance"]}
      />
      <div className="two-columns">
        <Panel
          title={`${performanceNames[s.performanceMetric]} across models`}
          subtitle={stateNames[s.state]}
        >
          <Bars
            data={rows.map((row) => ({
              name: modelNames[row.model],
              value: row[s.performanceMetric],
            }))}
            series={[
              {
                key: "value",
                name: performanceNames[s.performanceMetric],
                color: "#286f65",
              },
            ]}
            percent
          />
        </Panel>
        <Panel
          title="Compare the trade-off"
          subtitle={`${metricNames[s.metric]} · ${viewNames[s.attribute]}`}
        >
          <Tradeoff
            data={view}
            rows={rows}
            metric={s.metric}
            performanceMetric={s.performanceMetric}
          />
        </Panel>
      </div>
      <Panel
        title="Model comparison"
        subtitle="A consistent test split across all three model families."
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>{performanceNames[s.performanceMetric]}</th>
                <th>{metricNames[s.metric]}</th>
                <th>Smallest test group</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const f = view.fairness.find((f) => f.model === row.model);
                return (
                  <tr key={row.model}>
                    <td className="strong-cell">{modelNames[row.model]}</td>
                    <td>{format(row[s.performanceMetric])}</td>
                    <td>{format(f?.[s.metric])}</td>
                    <td>
                      {f?.min_group_size?.toLocaleString() ?? "Unavailable"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

function ChangeTable({ rows }: { rows: Results["changes"] }) {
  if (!rows.length)
    return <Empty>No mitigation changes are stored for this selection.</Empty>;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Experiment / group view</th>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Mitigated</th>
            <th>Raw change</th>
            <th>Distance change</th>
            <th>Direction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td className="strong-cell">
                {stateNames[row.mitigation_state]}
                <span className="cell-secondary">
                  {viewNames[row.sensitive_attribute]}
                </span>
              </td>
              <td>{metricNames[row.fairness_metric]}</td>
              <td>{format(row.baseline_value)}</td>
              <td>{format(row.mitigated_value)}</td>
              <td>{signed(row.raw_change)}</td>
              <td>{signed(row.distance_to_ideal_change)}</td>
              <td>
                <Effect value={row.effect} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EffectsPage({ data, selection: s }: Props) {
  const state = s.state === "baseline" ? "reweighing" : s.state;
  const rows = data.changes.filter(
    (row) =>
      row.model === s.model &&
      row.mitigation_state === state &&
      row.fairness_metric === s.metric,
  );
  const other = rows.filter((row) => row.sensitive_attribute !== "gender");
  const worsening = other.filter((row) => row.effect === "worsened");
  const undefinedRows = rows.filter((row) => row.effect === "undefined");
  return (
    <>
      <Filters
        data={data}
        selection={s}
        fields={["model", "state", "metric"]}
        noBaseline
      />
      <div className="notice">
        <ScanLine size={18} />
        <span>
          Both interventions target <strong>gender</strong>. These comparisons
          examine the resulting age-group and gender × age-group measurements on
          the same test data.
        </span>
      </div>
      {rows.length ? (
        <>
          <div className="stats-grid three">
            {rows.map((row) => (
              <div className="stat-card" key={row.sensitive_attribute}>
                <div className="stat-top">
                  {viewNames[row.sensitive_attribute]}
                  <span className="badge">
                    {row.sensitive_attribute === "gender"
                      ? "Target"
                      : "Observed view"}
                  </span>
                </div>
                <strong>{signed(row.distance_to_ideal_change)}</strong>
                <Effect value={row.effect} />
                <span className="stat-detail">
                  {format(row.baseline_value)} → {format(row.mitigated_value)}
                </span>
              </div>
            ))}
          </div>
          <Panel
            title="Where do the effects appear?"
            subtitle="Change in distance to the metric ideal. Negative values indicate movement closer to the ideal."
          >
            <Bars
              data={rows.map((row) => ({
                name: viewNames[row.sensitive_attribute],
                change: row.distance_to_ideal_change,
              }))}
              series={[
                {
                  key: "change",
                  name: "Distance-to-ideal change",
                  color: "#286f65",
                },
              ]}
              diverging
            />
          </Panel>
          <div className={`notice ${worsening.length ? "amber-notice" : ""}`}>
            <Info size={18} />
            <span>
              {worsening.length
                ? `${worsening.map((row) => viewNames[row.sensitive_attribute]).join(" and ")} moved farther from the selected metric’s ideal in this run.`
                : other.length
                  ? "No defined age or intersectional comparison moved farther from the selected metric’s ideal in this run."
                  : "Age and intersectional comparisons are unavailable."}{" "}
              {undefinedRows.length
                ? `${undefinedRows.length} comparison(s) are undefined.`
                : ""}{" "}
              These are descriptive changes from a single split, without
              significance estimates.
            </span>
          </div>
          {rows.map((row) => (
            <RateNote
              key={row.sensitive_attribute}
              data={data}
              selection={{ ...s, state, attribute: row.sensitive_attribute }}
            />
          ))}
          <Panel
            title="Cross-group comparison"
            subtitle="Direction refers to this metric only, not an overall fairness judgement."
          >
            <ChangeTable rows={rows} />
          </Panel>
        </>
      ) : (
        <Empty>
          Run the pipeline to generate the cross-group comparison file.
        </Empty>
      )}
    </>
  );
}

export function MethodologyPage({ data }: { data: Results }) {
  return (
    <>
      <div className="method-intro">
        <FlaskConical size={30} />
        <div>
          <h2>One protocol. Multiple perspectives.</h2>
          <p>
            FairLens compares established classifiers and mitigation methods on
            a historical income dataset. Different fairness definitions can
            disagree; each measurement describes one aspect of group disparity.
          </p>
        </div>
      </div>
      <Panel
        title="Experimental protocol"
        subtitle={`Seed ${data.config.random_seed} · ${(data.config.test_size * 100).toFixed(0)}% held-out test data`}
      >
        <ol className="protocol">
          {[
            [
              "Prepare the data",
              "Clean Adult Income records and create age bands: <25, 25–44, 45–64, and 65+. Record gender and the gender × age intersection for analysis.",
            ],
            [
              "Fit baseline models",
              "Fit preprocessing and the three classifiers using the training split. The default feature set excludes gender and the derived sensitive groups; other features can still act as proxies.",
            ],
            [
              "Apply mitigation",
              `Reweighing uses gender and training labels. Threshold adjustment fits its estimator on a training subset and learns thresholds on a validation subset (${(data.config.validation_size * 100).toFixed(0)}% of the original training split).`,
            ],
            [
              "Measure and compare",
              "Evaluate predictions on the common held-out test split. Compare metrics across gender, age groups, and their intersection. Save actual results before opening the dashboard.",
            ],
          ].map(([title, detail]) => (
            <li key={title}>
              <h3>{title}</h3>
              <p>{detail}</p>
            </li>
          ))}
        </ol>
      </Panel>
      <Panel
        title="Four definitions of fairness"
        subtitle="No arbitrary composite fairness score."
      >
        <div className="definition-grid">
          {Object.entries(metricDescriptions).map(([key, text]) => (
            <article key={key}>
              <span className="badge">Ideal {ideal(key as FairnessKey)}</span>
              <h3>{metricNames[key as FairnessKey]}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </Panel>
      <Panel
        title="Interpretation and limitations"
        subtitle="The boundaries of the current experiment."
      >
        <div className="prose">
          <p>
            The Adult dataset originates from 1994 US Census data and encodes
            gender as binary. These observations should not be generalized to
            contemporary populations without further validation.
          </p>
          <p>
            The 0.80 disparate-impact convention is a reference heuristic. It
            does not establish unlawful discrimination or determine whether a
            model is universally fair. Undefined measurements remain undefined;
            groups below {data.config.low_sample_threshold} test samples receive
            a caution.
          </p>
          <p>
            Cross-group results describe changes under the configured
            experiment. Repeated splits, confidence intervals, and composition
            robustness experiments are not part of this version. A change in a
            metric is not evidence of statistical significance or a causal
            effect.
          </p>
          <p>
            The team is accountable for implementation, evaluation, assumptions,
            and documentation. Human interpretation is required. FairLens is for
            educational research and must not make real-world high-stakes
            decisions. No personal user data is collected.
          </p>
          <p>
            Threshold-adjusted ROC-AUC is calculated from the underlying
            estimator’s continuous scores; the other performance and fairness
            measures use the adjusted hard decisions.
          </p>
        </div>
      </Panel>
    </>
  );
}
