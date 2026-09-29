import { useEffect, useId, useRef, type ReactNode } from "react";
import {
  ArrowDownRight,
  ArrowUpRight,
  CircleHelp,
  Download,
  Minus,
} from "lucide-react";
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  type Change,
  type Fairness,
  type FairnessKey,
  type Performance,
  type PerformanceKey,
  type Results,
  format,
  ideal,
  metricNames,
  modelColors,
  modelNames,
  performanceNames,
  stateNames,
  valid,
} from "./data";

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
export function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Record<string, string>;
  onChange: (value: string) => void;
}) {
  const labelId = useId();
  return (
    <div className="select-field">
      <span id={labelId}>{label}</span>
      <select
        aria-labelledby={labelId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {Object.entries(options).map(([key, name]) => (
          <option key={key} value={key}>
            {name}
          </option>
        ))}
      </select>
    </div>
  );
}
export function Stat({
  label,
  value,
  detail,
  icon,
  accent = false,
}: {
  label: string;
  value: string;
  detail: string;
  icon?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className={`stat-card ${accent ? "stat-accent" : ""}`}>
      <div className="stat-top">
        <span>{label}</span>
        {icon}
      </div>
      <strong>{value}</strong>
      <span className="stat-detail">{detail}</span>
    </div>
  );
}
export function Effect({ value }: { value: Change["effect"] }) {
  const labels = {
    improved: "Closer to ideal",
    worsened: "Farther from ideal",
    unchanged: "Unchanged",
    undefined: "Undefined",
  };
  const Icon =
    value === "improved"
      ? ArrowDownRight
      : value === "worsened"
        ? ArrowUpRight
        : Minus;
  return (
    <span className={`effect ${value}`}>
      <Icon size={13} />
      {labels[value] || labels.undefined}
    </span>
  );
}
export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty-inline">
      <CircleHelp size={20} />
      <p>{children}</p>
    </div>
  );
}
export function DownloadMenu() {
  const menu = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const closeOnOutside = (event: PointerEvent) => {
      if (menu.current && !menu.current.contains(event.target as Node))
        menu.current.open = false;
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && menu.current) menu.current.open = false;
    };
    document.addEventListener("pointerdown", closeOnOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, []);
  return (
    <details className="download-menu" ref={menu}>
      <summary className="button">
        <Download size={15} />
        Export results
      </summary>
      <div className="download-options">
        {[
          ["performance_metrics.csv", "Performance metrics"],
          ["fairness_metrics.csv", "Fairness measurements"],
          ["fairness_changes.csv", "Mitigation changes"],
        ].map(([file, label]) => (
          <a
            key={file}
            href={`/api/export/${file}`}
            download={file}
            onClick={() => {
              if (menu.current) menu.current.open = false;
            }}
          >
            <Download size={14} />
            {label}
          </a>
        ))}
      </div>
    </details>
  );
}

const axis = { fontSize: 11, fill: "#7b8581" };
const tooltipStyle = {
  borderRadius: 10,
  border: "1px solid #e4e8e4",
  fontSize: 12,
  boxShadow: "0 8px 32px #14232112",
};
export function Bars({
  data,
  series,
  diverging = false,
  percent = false,
}: {
  data: Record<string, unknown>[];
  series: { key: string; name: string; color: string }[];
  diverging?: boolean;
  percent?: boolean;
}) {
  const hasData = data.some((row) =>
    series.some((item) => valid(row[item.key])),
  );
  if (!hasData)
    return <Empty>No defined values are available for this selection.</Empty>;
  const horizontal =
    data.length > 5 || data.some((row) => String(row.name).length > 16);
  const numberTick = (value: number) =>
    percent ? `${Math.round(value * 100)}%` : value.toFixed(2);
  return (
    <div
      className="chart"
      style={
        horizontal ? { height: Math.max(280, data.length * 56) } : undefined
      }
      role="img"
      aria-label={`${series.map((s) => s.name).join(", ")} by category`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout={horizontal ? "vertical" : "horizontal"}
          margin={{
            top: 12,
            right: 12,
            left: horizontal ? 5 : -20,
            bottom: 20,
          }}
          barGap={5}
        >
          <CartesianGrid
            vertical={horizontal}
            horizontal={!horizontal}
            stroke="#edf0ed"
            strokeDasharray="4 4"
          />
          <XAxis
            dataKey={horizontal ? undefined : "name"}
            type={horizontal ? "number" : "category"}
            tickFormatter={horizontal ? numberTick : undefined}
            tick={axis}
            axisLine={false}
            tickLine={false}
            interval={0}
            tickMargin={13}
          />
          <YAxis
            type={horizontal ? "category" : "number"}
            dataKey={horizontal ? "name" : undefined}
            width={horizontal ? 132 : 60}
            interval={horizontal ? 0 : undefined}
            tick={axis}
            axisLine={false}
            tickLine={false}
            tickFormatter={horizontal ? undefined : numberTick}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            cursor={{ fill: "#f4f7f3" }}
            formatter={(value) =>
              format(typeof value === "number" ? value : null, percent)
            }
          />
          {diverging && (
            <ReferenceLine
              {...(horizontal ? { x: 0 } : { y: 0 })}
              stroke="#9aa69f"
            />
          )}
          {series.map((item) => (
            <Bar
              key={item.key}
              dataKey={item.key}
              name={item.name}
              fill={item.color}
              radius={[4, 4, 0, 0]}
              maxBarSize={42}
              isAnimationActive={false}
            >
              {diverging &&
                data.map((row, index) => (
                  <Cell
                    key={index}
                    fill={
                      valid(row[item.key]) && Number(row[item.key]) > 0
                        ? "#c88957"
                        : "#368477"
                    }
                  />
                ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
export function Legend({
  items,
}: {
  items: { name: string; color: string }[];
}) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span key={item.name}>
          <i style={{ background: item.color }} />
          {item.name}
        </span>
      ))}
    </div>
  );
}
export function Tradeoff({
  rows,
  data,
  metric,
  performanceMetric = "accuracy",
}: {
  rows: Performance[];
  data: Results;
  metric: FairnessKey;
  performanceMetric?: PerformanceKey;
}) {
  // Caller supplies only a single fairness view, so each arm has exactly one summary.
  const points = rows
    .map((row) => ({
      ...row,
      disparity: data.fairness.find(
        (f) =>
          f.model === row.model && f.mitigation_state === row.mitigation_state,
      )?.[metric],
    }))
    .filter((row) => valid(row[performanceMetric]) && valid(row.disparity));
  if (!points.length)
    return <Empty>No defined measurements for this trade-off.</Empty>;
  const yMax = Math.min(
    1,
    Math.ceil(Math.max(0.1, ...points.map((p) => p.disparity! * 1.15)) * 10) /
      10,
  );
  return (
    <>
      <div
        className="chart scatter-chart"
        role="img"
        aria-label={`${performanceNames[performanceMetric]} versus ${metricNames[metric]}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 25, bottom: 20, left: -15 }}>
            <CartesianGrid strokeDasharray="4 4" stroke="#e9eeea" />
            <XAxis
              type="number"
              dataKey={performanceMetric}
              name={performanceNames[performanceMetric]}
              domain={[
                (min: number) =>
                  Math.max(0, Math.floor((min - 0.012) * 100) / 100),
                (max: number) =>
                  Math.min(1, Math.ceil((max + 0.012) * 100) / 100),
              ]}
              tick={axis}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              label={{
                value: performanceNames[performanceMetric],
                position: "insideBottom",
                offset: -14,
                ...axis,
              }}
            />
            <YAxis
              type="number"
              dataKey="disparity"
              name={metricNames[metric]}
              domain={[0, ideal(metric) === 1 ? 1 : yMax]}
              tick={axis}
              axisLine={false}
              tickLine={false}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <ReferenceLine
              y={ideal(metric)}
              stroke="#a8b5ac"
              strokeDasharray="3 3"
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                const point = payload?.[0]?.payload;
                if (!active || !point) return null;
                return (
                  <div className="chart-tooltip">
                    <strong>{modelNames[point.model]}</strong>
                    <span>{stateNames[point.mitigation_state]}</span>
                    <span>
                      {performanceNames[performanceMetric]}:{" "}
                      {format(point[performanceMetric], true)}
                    </span>
                    <span>
                      {metricNames[metric]}: {format(point.disparity)}
                    </span>
                  </div>
                );
              }}
            />
            {points.map((point) => (
              <Scatter
                key={`${point.model}-${point.mitigation_state}`}
                data={[point]}
                fill={modelColors[point.model] || "#286f65"}
                shape={
                  point.mitigation_state === "baseline"
                    ? "circle"
                    : point.mitigation_state === "reweighing"
                      ? "diamond"
                      : "triangle"
                }
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <Legend
        items={Object.entries(modelColors).map(([key, color]) => ({
          name: modelNames[key],
          color,
        }))}
      />
      <p className="chart-caption">
        ● Baseline <span>◆ Reweighing</span> ▲ Threshold adjustment ·{" "}
        {metric === "disparate_impact_ratio"
          ? "Higher ratio is closer to 1"
          : "Lower disparity is closer to 0"}
      </p>
    </>
  );
}

export function GroupTable({
  rows,
  threshold,
}: {
  rows: Fairness[];
  threshold: number;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Demographic group</th>
            <th>Test samples</th>
            <th>Selection rate</th>
            <th>True-positive rate</th>
            <th>False-positive rate</th>
            <th>Sample note</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.group}>
              <td className="strong-cell">{row.group.replace(" | ", " · ")}</td>
              <td>{row.group_count.toLocaleString()}</td>
              <td>{format(row.selection_rate)}</td>
              <td>{format(row.tpr)}</td>
              <td>{format(row.fpr)}</td>
              <td>
                {row.group_count < threshold ? (
                  <span className="badge amber">Below {threshold}</span>
                ) : (
                  <span className="muted">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
