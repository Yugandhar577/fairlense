import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  BookOpen,
  ChartNoAxesCombined,
  ChevronRight,
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  Menu,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  X,
} from "lucide-react";
import { DownloadMenu } from "./components";
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
  const [data, setData] = useState<Results | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
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
        className={`sidebar ${menuOpen ? "open" : ""}`}
        aria-label="Workspace navigation"
      >
        <a
          className="brand"
          href="#overview"
          onClick={() => setMenuOpen(false)}
        >
          <Mark />
          <span>
            FairLens<span className="brand-dot">.</span>
          </span>
        </a>
        <div className="workspace-tag">
          <span className="workspace-avatar">F</span>
          <div>
            <strong>Research workspace</strong>
            <small>Adult Income benchmark</small>
          </div>
        </div>
        <span className="nav-section-label">ANALYSIS</span>
        <nav>
          {navigation.map(({ key, label, icon: Icon }) => (
            <a
              key={key}
              href={`#${key}`}
              onClick={() => setMenuOpen(false)}
              aria-current={page === key ? "page" : undefined}
              className={page === key ? "nav-active" : ""}
            >
              <Icon size={18} />
              <span>{label}</span>
              {page === key && <span className="active-dot" />}
            </a>
          ))}
        </nav>
        <span className="nav-section-label resources-label">RESOURCES</span>
        <nav>
          <a
            href="#methodology"
            onClick={() => setMenuOpen(false)}
            aria-current={page === "methodology" ? "page" : undefined}
            className={page === "methodology" ? "nav-active" : ""}
          >
            <BookOpen size={18} />
            Methodology
          </a>
        </nav>
        <div className="sidebar-bottom">
          <div className="study-card">
            <ShieldCheck size={20} />
            <strong>Measure. Compare. Understand.</strong>
            <p>A transparent perspective on responsible machine learning.</p>
            <button onClick={() => navigate("methodology")}>
              About this study <ArrowRight size={14} />
            </button>
          </div>
          <div className="sidebar-footer">
            <span className="status-dot" />
            Local research environment<span>v1.0</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button menu-toggle"
              onClick={() => setMenuOpen(!menuOpen)}
              aria-label={
                menuOpen ? "Close navigation menu" : "Open navigation menu"
              }
              aria-expanded={menuOpen}
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{titles[page][0]}</strong>
          </div>
          <div className="topbar-right">
            <span className="saved-status">
              <span className={`status-dot ${!data ? "inactive" : ""}`} />
              {data
                ? "Saved experiment"
                : loading
                  ? "Loading results"
                  : "Results unavailable"}
            </span>
            <span className="avatar">FL</span>
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                FAIRNESS BENCHMARKING <span>/</span> UCI ADULT
              </div>
              <h1>{titles[page][0]}</h1>
              <p>{titles[page][1]}</p>
            </div>
            <div className="heading-actions">
              <button
                className={`icon-button refresh ${loading ? "spinning" : ""}`}
                disabled={loading}
                onClick={() => setReload((n) => n + 1)}
                aria-label="Refresh saved results"
                title="Refresh saved results"
              >
                <RefreshCw size={16} />
              </button>
              {data && <DownloadMenu />}
            </div>
          </div>
          {loading && !data ? (
            <div className="loading-state" role="status">
              <div className="loader" />
              <h2>Opening your experiment</h2>
              <p>Loading the saved benchmark measurements.</p>
            </div>
          ) : error ? (
            <div className="error-state" role="alert">
              <DatabaseIcon />
              <h2>Let’s bring in your results</h2>
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
              <ShieldCheck size={14} />
              For educational research. Metrics describe disparities, not
              universal fairness.
            </span>
            <span>
              {date
                ? `Experiment saved ${date}`
                : "FairLens research workspace"}
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
function DatabaseIcon() {
  return (
    <div className="empty-icon">
      <FlaskConical size={30} />
    </div>
  );
}
