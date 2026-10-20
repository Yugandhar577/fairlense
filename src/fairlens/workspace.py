"""Persistent local datasets and isolated asynchronous experiment runs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from threading import Lock, RLock, Thread
import time
from uuid import uuid4

_JSON_LOCK = RLock()

def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict):
    # Windows cannot replace a file while another thread holds a read handle.
    # Serialize local readers/writers; retry brief external antivirus/indexer locks.
    with _JSON_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.01 * (attempt + 1))


def read_json(path: Path) -> dict:
    with _JSON_LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def checked_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid dataset or experiment ID.")
    return value


class Workspace:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.lock = Lock()
        self.active = False
        # A terminated process cannot complete its in-memory worker. Never display it as running forever.
        for path in (self.directory / "runs").glob("*/status.json"):
            status = read_json(path)
            if status["status"] in {"queued", "running"}:
                write_json(path, {**status, "status": "failed", "message": "Training was interrupted by a server restart. Start a new experiment."})

    def datasets(self) -> list[dict]:
        return [read_json(path)
                for path in sorted((self.directory / "datasets").glob("*/profile.json"))]

    def profile(self, dataset_id: str) -> dict:
        return read_json(self.directory / "datasets" / checked_id(dataset_id) / "profile.json")

    def upload(self, payload: dict) -> dict:
        from fairlens.experiments import dataset_profile, parse_csv
        if not isinstance(payload, dict):
            raise ValueError("Upload settings must be an object.")
        if len(self.datasets()) >= 100:
            raise ValueError("This local workspace already has 100 datasets. Archive unused dataset folders before uploading more.")
        name = payload.get("name")
        if not isinstance(name, str) or not name.lower().endswith(".csv") or len(name) > 160:
            raise ValueError("Choose a CSV file with a filename of at most 160 characters.")
        frame = parse_csv(payload.get("csv"))
        dataset_id = uuid4().hex
        directory = self.directory / "datasets" / dataset_id
        directory.mkdir(parents=True)
        profile = dataset_profile(frame, dataset_id, name.replace("\\", "/").split("/")[-1])
        frame.to_csv(directory / "data.csv", index=False)
        write_json(directory / "profile.json", profile)
        return profile

    def runs(self) -> list[dict]:
        return sorted([read_json(path)
                       for path in (self.directory / "runs").glob("*/status.json")],
                      key=lambda status: status["created_at_utc"], reverse=True)

    def status(self, run_id: str) -> dict:
        return read_json(self.directory / "runs" / checked_id(run_id) / "status.json")

    def result(self, run_id: str) -> dict:
        if self.status(run_id)["status"] != "completed":
            raise ValueError("Results are available only after training completes successfully.")
        return read_json(self.directory / "runs" / checked_id(run_id) / "result.json")

    def export(self, run_id: str, filename: str) -> bytes:
        if filename not in {"performance_metrics.csv", "fairness_metrics.csv", "fairness_changes.csv", "group_errors.csv", "run_config.json"}:
            raise ValueError("Only benchmark tables and run configuration can be downloaded.")
        if self.status(run_id)["status"] != "completed":
            raise ValueError("Wait for this experiment to complete before exporting.")
        return (self.directory / "runs" / checked_id(run_id) / filename).read_bytes()

    def submit(self, spec: dict) -> dict:
        from fairlens.experiments import validate_spec
        if not isinstance(spec, dict):
            raise ValueError("Experiment settings must be an object.")
        profile = self.profile(spec.get("dataset_id"))
        spec = validate_spec(spec, [column["name"] for column in profile["columns"]])
        spec["dataset_name"] = profile["name"]
        with self.lock:
            if self.active:
                raise RuntimeError("An experiment is already training. Wait for it to finish before starting another.")
            if len(self.runs()) >= 200:
                raise ValueError("This local workspace already has 200 runs. Archive unused run folders before starting another.")
            run_id = uuid4().hex
            status = dict(id=run_id, dataset_id=profile["id"], dataset_name=profile["name"], task=spec["task"],
                          models=spec["models"], status="queued", message="Experiment queued",
                          created_at_utc=datetime.now(timezone.utc).isoformat())
            directory = self.directory / "runs" / run_id
            write_json(directory / "status.json", status)
            write_json(directory / "request.json", spec)
            self.active = True
            Thread(target=self._train, args=(status, spec, directory), daemon=True).start()
            return status

    def _train(self, status: dict, spec: dict, directory: Path):
        def progress(message):
            write_json(directory / "status.json", {**status, "status": "running", "message": message})
        try:
            from fairlens.experiments import parse_csv, run_experiment
            frame = parse_csv((self.directory / "datasets" / spec["dataset_id"] / "data.csv").read_text(encoding="utf-8"))
            result = run_experiment(frame, spec, directory, progress)
            write_json(directory / "result.json", result)
            write_json(directory / "run_config.json", result["config"])
            write_json(directory / "status.json", {**status, "status": "completed", "message": "Benchmark complete",
                                                   "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        except Exception as error:
            # Data/schema failures should be actionable; unexpected errors must not expose local filesystem paths.
            message = str(error) if isinstance(error, ValueError) else "Training failed. Check the local server log for details."
            import logging
            logging.getLogger(__name__).exception("Experiment %s failed", status["id"])
            write_json(directory / "status.json", {**status, "status": "failed", "message": message})
        finally:
            with self.lock:
                self.active = False
