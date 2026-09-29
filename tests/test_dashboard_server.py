"""Exercise the React-facing boundary without importing or fitting any model."""
import csv
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from fairlens.dashboard_server import REQUIRED_COLUMNS, RESULT_FILES, make_server


@pytest.fixture
def dashboard(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for key, filename in RESULT_FILES.items():
        with (results / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=sorted(REQUIRED_COLUMNS[key]))
            writer.writeheader()
            row = {column: "0" for column in REQUIRED_COLUMNS[key]}
            row.update(model="logistic_regression", mitigation_state="baseline")
            if key == "fairness":
                row.update(disparate_impact_ratio="", low_sample_warning="False", group="Female")
            writer.writerow(row)
    (results / "run_config.json").write_text(json.dumps({
        "preprocessing_report": {"input_rows": 100, "training_samples": 80, "test_samples": 20, "feature_count": 4, "rows_removed": 0},
        "random_seed": 42, "test_size": 0.2, "validation_size": 0.2, "low_sample_threshold": 30,
        "created_at_utc": "2026-09-29T07:15:48+00:00", "data_path": "private-local-path",
    }))
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<html>FairLens</html>")
    (tmp_path / "private.txt").write_text("not served")
    server = make_server(results, frontend, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", results
    server.shutdown()
    server.server_close()
    thread.join()


def test_api_preserves_undefined_metrics_and_boolean_flags(dashboard):
    base, _ = dashboard
    with urlopen(base + "/api/results") as response:
        body = json.load(response)
    assert body["fairness"][0]["disparate_impact_ratio"] is None
    assert body["fairness"][0]["low_sample_warning"] is False
    assert "data_path" not in body["config"]


def test_missing_and_invalid_results_return_actionable_errors(dashboard):
    base, results = dashboard
    (results / "performance_metrics.csv").write_text("broken\nrecord\n")
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/results")
    assert error.value.code == 422
    (results / "performance_metrics.csv").unlink()
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/results")
    assert error.value.code == 404
    assert json.load(error.value)["error"] == "results_missing"


def test_only_metric_exports_are_downloadable(dashboard):
    base, results = dashboard
    filename = "fairness_metrics.csv"
    with urlopen(base + "/api/export/" + filename) as response:
        assert response.read() == (results / filename).read_bytes()
        assert "attachment" in response.headers["Content-Disposition"]
    for path in ["/api/export/run_config.json", "/api/export/../models/model.joblib", "/%2e%2e/private.txt"]:
        with pytest.raises(HTTPError) as error:
            urlopen(base + path)
        assert error.value.code == 404


def test_frontend_is_served_independently_of_results(dashboard):
    base, results = dashboard
    (results / "performance_metrics.csv").unlink()
    with urlopen(base) as response:
        assert response.status == 200
        assert b"FairLens" in response.read()


def test_incomplete_run_metadata_is_reported_before_frontend_render(dashboard):
    base, results = dashboard
    (results / "run_config.json").write_text('{"preprocessing_report": {}}')
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/results")
    assert error.value.code == 422
    assert json.load(error.value)["error"] == "results_invalid"
