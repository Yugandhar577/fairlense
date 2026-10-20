import { useEffect, useRef, useState } from "react";
import {
  Activity,
  BookOpen,
  ChartNoAxesCombined,
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  Menu,
  RefreshCw,
  ScanLine,
  X,
  SlidersHorizontal,
} from "lucide-react";
import { DownloadMenu, Select } from "./components";
import Workbench from "./Workbench";
import ExperimentResults from "./ExperimentResults";
import {
  workspaceRequest,
  type ExperimentResult,
  type Run,
} from "./workspace-data";
import {
  type FairnessKey,
  type PerformanceKey,
  type Results,
  getResults,
} from "./data";
import {
  EffectsPage,
  FairnessPage,
  MethodologyPage,
  MitigationPage,
  ModelsPage,
  Overview,
  PerformancePage,
  titles,
} from "./pages";

const navigation = [
  { key: "train", label: "Train models", icon: SlidersHorizontal },
  { key: "overview", label: "Overview", icon: LayoutDashboard },
  { key: "performance", label: "Model performance", icon: Activity },
  { key: "fairness", label: "Fairness analysis", icon: ScanLine },
  { key: "mitigation", label: "Mitigation comparison", icon: GitCompareArrows },
  { key: "models", label: "Cross-model comparison", icon: ChartNoAxesCombined },
  { key: "effects", label: "Cross-group effects", icon: FlaskConical },
];
const currentPage = () =>
  Object.hasOwn(titles, location.hash.slice(1))
    ? location.hash.slice(1)
    : "overview";
function Mark() {
  return (
    <svg viewBox="0 0 44 36" aria-hidden="true">
      <circle cx="16" cy="18" r="12" />
      <circle cx="28" cy="18" r="12" />
    </svg>
  );
}

export default function App() {
  const [page, setPage] = useState(currentPage);
  const [menuOpen, setMenuOpen] = useState(false);
  const sidebar = useRef<HTMLElement>(null);
  const [data, setData] = useState<Results | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState(
    () => localStorage.getItem("fairlens.selectedRun") || "",
  );
  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [experimentError, setExperimentError] = useState("");
  const [model, setModel] = useState("logistic_regression");
  const [state, setState] = useState("baseline");
  const [attribute, setAttribute] = useState("gender");
  const [metric, setMetric] = useState<FairnessKey>("demographic_parity_diff");
  const [performanceMetric, setPerformanceMetric] =
    useState<PerformanceKey>("accuracy");
  const selection = {
    model,
    state,
    attribute,
    metric,
    performanceMetric,
    setModel,
    setState,
    setAttribute,
    setMetric,
    setPerformanceMetric,
  };

  useEffect(() => {
    const onHash = () => {
      setPage(currentPage());
      setMenuOpen(false);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    document.title = `${titles[page][0]} · FairLens`;
  }, [page]);
  useEffect(() => {
    if (!menuOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const links = () =>
      Array.from(
        sidebar.current?.querySelectorAll<HTMLElement>("a[href], button") || [],
      );
    links()[0]?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
      if (event.key !== "Tab") return;
      const targets = links();
      const first = targets[0];
      const last = targets[targets.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    const onResize = () => {
      if (window.innerWidth > 760) setMenuOpen(false);
    };
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
      previousFocus?.focus();
    };
  }, [menuOpen]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    getResults(controller.signal)
      .then(setData)
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(
            reason instanceof Error
              ? reason.message
              : "Unable to load experiment results.",
          );
          setData(null);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reload]);
  useEffect(() => {
    const controller = new AbortController();
    workspaceRequest<{ runs: Run[] }>("runs", undefined, controller.signal)
      .then((body) => setRuns(body.runs))
      .catch(() => {});
    return () => controller.abort();
  }, [reload]);
  useEffect(() => {
    localStorage.setItem("fairlens.selectedRun", runId);
    setExperiment(null);
    setExperimentError("");
    if (!runId) {
      setExperimentLoading(false);
      return;
    }
    const controller = new AbortController();
    setExperimentLoading(true);
    workspaceRequest<ExperimentResult>(
      `runs/${runId}/results`,
      undefined,
      controller.signal,
    )
      .then(setExperiment)
      .catch((e) => {
        if (!controller.signal.aborted) setExperimentError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setExperimentLoading(false);
      });
    return () => controller.abort();
  }, [runId, reload]);
  const navigate = (key: string) => {
    location.hash = key;
    setMenuOpen(false);
  };
  const date = data?.config.created_at_utc
    ? new Intl.DateTimeFormat("en", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }).format(new Date(data.config.created_at_utc))
    : null;

  return (
    <div className="app-shell">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        Skip to content
      </a>
      {menuOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        />
      )}
      <aside
        id="workspace-navigation"
        ref={sidebar}
        className={`sidebar ${menuOpen ? "open" : ""}`}
        aria-label="Workspace navigation"
      >
        <button
          className="icon-button navigation-close"
          aria-label="Close navigation menu"
          onClick={() => setMenuOpen(false)}
        >
          <X size={20} />
        </button>
        <a
          className="brand"
          href="#overview"
          onClick={() => setMenuOpen(false)}
        >
          <Mark />
          <span>FairLens</span>
        </a>
        <p className="workspace-context">
          Model training & fairness benchmarks
        </p>
        <nav aria-label="Benchmark analysis">
          {navigation.map(({ key, label, icon: Icon }) => (
            <a
              key={key}
              href={`#${key}`}
              onClick={() => setMenuOpen(false)}
              aria-current={page === key ? "page" : undefined}
              className={page === key ? "nav-active" : ""}
            >
              <Icon size={18} aria-hidden="true" />
              <span>{label}</span>
            </a>
          ))}
        </nav>
        <nav className="resource-nav" aria-label="Study reference">
          <a
            href="#methodology"
            onClick={() => setMenuOpen(false)}
            aria-current={page === "methodology" ? "page" : undefined}
            className={page === "methodology" ? "nav-active" : ""}
          >
            <BookOpen size={18} aria-hidden="true" />
            Methodology
          </a>
        </nav>
        <p className="sidebar-note">
          Train explicitly. Compare saved results without retraining.
        </p>
      </aside>
      <div className="main-shell" inert={menuOpen}>
        <header className="topbar">
          <div className="workspace-heading">
            <button
              className="icon-button menu-toggle"
              onClick={() => setMenuOpen(!menuOpen)}
              aria-label={
                menuOpen ? "Close navigation menu" : "Open navigation menu"
              }
              aria-expanded={menuOpen}
              aria-controls="workspace-navigation"
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span>FairLens · Local research workspace</span>
          </div>
          <span className="result-source" role="status">
            {(runId ? experimentLoading : loading)
              ? "Loading saved results…"
              : (runId ? experimentError : error)
                ? "Results unavailable"
                : (runId ? experiment?.config.created_at_utc : date)
                  ? `Saved ${runId ? new Date(experiment!.config.created_at_utc).toLocaleDateString() : date}`
                  : ""}
          </span>
        </header>
        <main id="main-content" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <h1>{titles[page][0]}</h1>
              <p>
                {runId && page !== "train"
                  ? `${experiment?.config.dataset_name || "Selected experiment"} · ${page === "effects" ? "Inspect mitigation effects across sensitive groups." : page === "performance" ? "Compare task-appropriate measurements on held-out test rows." : titles[page][1]}`
                  : titles[page][1]}
              </p>
            </div>
            <div className="heading-actions">
              <button
                className="icon-button refresh"
                disabled={runId ? experimentLoading : loading}
                onClick={() => setReload((n) => n + 1)}
                aria-label="Refresh saved results"
                title="Refresh saved results"
              >
                <RefreshCw size={18} aria-hidden="true" />
              </button>
              {page !== "train" &&
                (runId
                  ? experiment && (
                      <DownloadMenu
                        runId={runId}
                        regression={experiment.config.task === "regression"}
                      />
                    )
                  : data && <DownloadMenu />)}
            </div>
          </div>
          <div className="results-picker">
            <Select
              label="Results to view"
              value={runId}
              options={{
                "": "Original Adult study",
                ...Object.fromEntries(
                  runs
                    .filter((r) => r.status === "completed")
                    .map((r) => [
                      r.id,
                      `${r.dataset_name} · ${r.task} · ${r.id.slice(0, 8)}`,
                    ]),
                ),
                ...(runId && !runs.some((r) => r.id === runId)
                  ? { [runId]: "Selected experiment" }
                  : {}),
              }}
              onChange={(id) => {
                setRunId(id);
                if (page === "train") navigate("overview");
              }}
            />
          </div>
          {page === "train" ? (
            <Workbench
              refreshKey={reload}
              onRunsChange={setRuns}
              onOpenRun={(id) => {
                setRunId(id);
                navigate("overview");
              }}
            />
          ) : runId ? (
            experimentLoading ? (
              <p role="status">Loading experiment results…</p>
            ) : experimentError ? (
              <div role="alert" className="error-state">
                <h2>Experiment results unavailable</h2>
                <p>{experimentError}</p>
                <button className="button" onClick={() => setRunId("")}>
                  View original Adult study
                </button>
              </div>
            ) : (
              experiment && <ExperimentResults data={experiment} page={page} />
            )
          ) : loading && !data ? (
            <div className="loading-state" role="status">
              <h2>Loading saved results</h2>
              <p>Loading the saved benchmark measurements.</p>
            </div>
          ) : error ? (
            <div className="error-state" role="alert">
              <h2>Experiment results unavailable</h2>
              <p>{error}</p>
              <code>python run_pipeline.py</code>
              <button
                className="button primary"
                onClick={() => setReload((n) => n + 1)}
              >
                <RefreshCw size={16} />
                Try again
              </button>
            </div>
          ) : data ? (
            <div className="page-content" key={page}>
              {page === "overview" && (
                <Overview
                  data={data}
                  selection={selection}
                  navigate={navigate}
                />
              )}
              {page === "performance" && (
                <PerformancePage data={data} selection={selection} />
              )}
              {page === "fairness" && (
                <FairnessPage data={data} selection={selection} />
              )}
              {page === "mitigation" && (
                <MitigationPage data={data} selection={selection} />
              )}
              {page === "models" && (
                <ModelsPage data={data} selection={selection} />
              )}
              {page === "effects" && (
                <EffectsPage data={data} selection={selection} />
              )}
              {page === "methodology" && <MethodologyPage data={data} />}
            </div>
          ) : null}
          <footer className="main-footer">
            <span>
              For educational research. Metrics describe disparities, not
              universal fairness.
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
