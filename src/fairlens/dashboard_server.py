"""Local, read-only HTTP bridge between the results store and React dashboard.

Only metric files are exposed. This module does not import the training pipeline
or deserialize fitted models. The built frontend is served from its own directory.
"""
from __future__ import annotations

import csv
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

RESULT_FILES = {
    "performance": "performance_metrics.csv",
    "fairness": "fairness_metrics.csv",
    "changes": "fairness_changes.csv",
}
NUMBER_COLUMNS = {
    "accuracy", "precision", "recall", "f1", "roc_auc", "demographic_parity_diff",
    "equal_opportunity_diff", "equalized_odds_diff", "disparate_impact_ratio",
    "min_group_size", "group_count", "selection_rate", "tpr", "fpr",
    "baseline_value", "mitigated_value", "raw_change", "distance_to_ideal_change",
}
BOOLEAN_COLUMNS = {"low_confidence_flag", "low_sample_warning"}
REQUIRED_COLUMNS = {
    "performance": {"model", "mitigation_state", "accuracy", "precision", "recall", "f1", "roc_auc"},
    "fairness": {"model", "mitigation_state", "sensitive_attribute", "group", "group_count",
                 "selection_rate", "tpr", "fpr", "demographic_parity_diff", "equal_opportunity_diff",
                 "equalized_odds_diff", "disparate_impact_ratio", "low_sample_warning"},
    "changes": {"model", "mitigation_state", "sensitive_attribute", "fairness_metric",
                "baseline_value", "mitigated_value", "raw_change", "distance_to_ideal_change", "effect"},
}


def finite_json(value):
    """Convert non-finite metrics to JSON null, preserving genuinely missing values."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_results(directory: Path) -> dict:
    payload = {}
    for key, filename in RESULT_FILES.items():
        with (directory / filename).open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not REQUIRED_COLUMNS[key].issubset(reader.fieldnames or []):
                raise ValueError(f"{filename} has an incompatible schema")
            rows = []
            for record in reader:
                for column, value in record.items():
                    if column in NUMBER_COLUMNS:
                        record[column] = float(value) if value and value.strip() else None
                    elif column in BOOLEAN_COLUMNS:
                        record[column] = str(value).lower() == "true"
                rows.append(record)
            if not rows:
                raise ValueError(f"{filename} is empty")
            payload[key] = rows
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("preprocessing_report"), dict):
        raise ValueError("The run configuration is incomplete")
    report = config["preprocessing_report"]
    for section, fields in [
        (config, ("random_seed", "test_size", "validation_size", "low_sample_threshold")),
        (report, ("input_rows", "training_samples", "test_samples", "feature_count", "rows_removed")),
    ]:
        if any(not isinstance(section.get(field), (int, float)) or not math.isfinite(section[field]) for field in fields):
            raise ValueError("The run configuration contains missing or invalid numeric fields")
    if not isinstance(config.get("created_at_utc"), str):
        raise ValueError("The run timestamp is missing")
    datetime.fromisoformat(config["created_at_utc"])
    # Machine-specific paths are unnecessary for the dashboard.
    payload["config"] = {k: v for k, v in config.items() if k not in {"data_path", "results_dir"}}
    return finite_json(payload)


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, results_dir: Path, frontend_dir: Path, **kwargs):
        self.results_dir = results_dir
        self.frontend_dir = frontend_dir.resolve()
        super().__init__(*args, **kwargs)

    def _send(self, status: int, body: bytes, content_type: str, filename: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict):
        self._send(status, json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/results":
            try:
                self._json(200, read_results(self.results_dir))
            except FileNotFoundError:
                self._json(404, {"error": "results_missing", "message": "Run python run_pipeline.py to generate the experiment results, then refresh."})
            except (ValueError, TypeError, OSError, csv.Error):
                self._json(422, {"error": "results_invalid", "message": "The results store is incomplete or unreadable. Regenerate it with python run_pipeline.py."})
            return
        if path.startswith("/api/export/"):
            filename = path.removeprefix("/api/export/")
            if filename not in RESULT_FILES.values():
                self._json(404, {"error": "not_found"})
                return
            try:
                self._send(200, (self.results_dir / filename).read_bytes(), "text/csv; charset=utf-8", filename)
            except OSError:
                self._json(404, {"error": "results_missing", "message": "Generate the results before downloading."})
            return
        if path.startswith("/api/"):
            self._json(404, {"error": "not_found"})
            return
        target = (self.frontend_dir / path.lstrip("/")).resolve()
        if not target.is_relative_to(self.frontend_dir):
            self._json(404, {"error": "not_found"})
            return
        if path == "/":
            target = self.frontend_dir / "index.html"
        if not target.is_file():
            self._json(404, {"error": "frontend_missing", "message": "Build the React dashboard with npm --prefix frontend run build."})
            return
        content_type = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(target.suffix)
        self._send(200, target.read_bytes(), content_type or mimetypes.guess_type(target.name)[0] or "application/octet-stream")


def make_server(results_dir: Path, frontend_dir: Path, port: int = 8000) -> ThreadingHTTPServer:
    handler = partial(DashboardHandler, results_dir=results_dir, frontend_dir=frontend_dir)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
