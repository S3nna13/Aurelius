"""Experiment tracker for Aurelius training runs (stdlib-only, JSON backend)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Metadata and metrics for a single experiment run."""

    run_id: str
    run_name: str
    experiment_name: str = "default"
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    status: str = "RUNNING"
    start_time: str = ""
    end_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrackingBackend(ABC):
    """Abstract backend for experiment tracking."""

    @abstractmethod
    def start_run(
        self, run_name: str, experiment_name: str, params: dict[str, Any]
    ) -> str: ...

    @abstractmethod
    def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int
    ) -> None: ...

    @abstractmethod
    def log_artifact(self, run_id: str, path: str, name: str) -> None: ...

    @abstractmethod
    def end_run(self, run_id: str, status: str) -> None: ...

    @abstractmethod
    def get_runs(self, experiment_name: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...


class JSONBackend(TrackingBackend):
    """JSON-file backed experiment tracker."""

    def __init__(self, tracking_dir: str = "experiments/") -> None:
        self._root = Path(tracking_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._active_runs: dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    def start_run(
        self, run_name: str, experiment_name: str, params: dict[str, Any]
    ) -> str:
        with self._lock:
            run_id = uuid.uuid4().hex[:12]
            run_dir = self._root / experiment_name / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            record = RunRecord(
                run_id=run_id,
                run_name=run_name,
                experiment_name=experiment_name,
                params=dict(params),
                start_time=datetime.now(timezone.utc).isoformat(),
            )
            self._active_runs[run_id] = record
            (run_dir / "params.json").write_text(json.dumps(params, indent=2))
            self._save_record(record)
            return run_id

    def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int
    ) -> None:
        with self._lock:
            record = self._active_runs.get(run_id)
            if record is None:
                logger.error("Run '%s' not found", run_id)
                return
            for key, value in metrics.items():
                record.metrics.setdefault(key, []).append(
                    {"step": step, "value": value}
                )
            self._atomic_write(
                self._root / record.experiment_name / run_id / "metrics.json",
                json.dumps(record.metrics, indent=2),
            )
            self._save_record(record)

    def log_artifact(self, run_id: str, path: str, name: str) -> None:
        with self._lock:
            record = self._active_runs.get(run_id)
            if record is None:
                logger.error("Run '%s' not found", run_id)
                return
            artifact_dir = (
                self._root / record.experiment_name / run_id / "artifacts"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            src = Path(path)
            if not src.exists():
                logger.warning("Artifact '%s' does not exist", path)
                return
            shutil.copy2(src, artifact_dir / name)
            record.artifacts.append(name)
            self._save_record(record)

    def end_run(self, run_id: str, status: str) -> None:
        with self._lock:
            record = self._active_runs.pop(run_id, None)
            if record is None:
                logger.error("Run '%s' not found", run_id)
                return
            record.status = status
            record.end_time = datetime.now(timezone.utc).isoformat()
            self._save_record(record)

    def get_runs(self, experiment_name: str) -> list[dict[str, Any]]:
        exp_dir = self._root / experiment_name
        if not exp_dir.exists():
            return []
        runs: list[dict[str, Any]] = []
        for run_dir in sorted(exp_dir.iterdir()):
            rp = run_dir / "run.json"
            if rp.exists():
                runs.append(json.loads(rp.read_text()))
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not self._root.exists():
            return None
        for exp_dir in self._root.iterdir():
            if not exp_dir.is_dir():
                continue
            rp = exp_dir / run_id / "run.json"
            if rp.exists():
                return json.loads(rp.read_text())
        return None

    def _save_record(self, record: RunRecord) -> None:
        run_dir = self._root / record.experiment_name / record.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(
            run_dir / "run.json", json.dumps(record.to_dict(), indent=2)
        )

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=target.parent, prefix=".tmp_", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(content)
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class ExperimentTracker:
    """High-level wrapper around a ``TrackingBackend``."""

    def __init__(
        self,
        backend: TrackingBackend | None = None,
        tracking_dir: str = "experiments/",
    ) -> None:
        self._backend = backend if backend is not None else JSONBackend(tracking_dir)
        self._current_run_id: str | None = None
        self._current_experiment: str = "default"

    @property
    def backend(self) -> TrackingBackend:
        return self._backend

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    def start_run(
        self,
        run_name: str,
        experiment_name: str = "default",
        params: dict[str, Any] | None = None,
    ) -> str:
        self._current_experiment = experiment_name
        self._current_run_id = self._backend.start_run(
            run_name, experiment_name, params or {}
        )
        return self._current_run_id

    def log_metric(self, name: str, value: float, step: int) -> None:
        if self._current_run_id is None:
            logger.error("No active run")
            return
        self._backend.log_metrics(self._current_run_id, {name: value}, step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._current_run_id is None:
            logger.error("No active run")
            return
        self._backend.log_metrics(self._current_run_id, metrics, step)

    def log_artifact(self, path: str, name: str) -> None:
        if self._current_run_id is None:
            logger.error("No active run")
            return
        self._backend.log_artifact(self._current_run_id, path, name)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._current_run_id is None:
            return
        self._backend.end_run(self._current_run_id, status)
        self._current_run_id = None

    @contextmanager
    def run(
        self,
        run_name: str,
        experiment_name: str = "default",
        params: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        run_id = self.start_run(run_name, experiment_name, params)
        try:
            yield run_id
        except Exception:
            self.end_run("FAILED")
            raise
        else:
            self.end_run("FINISHED")

    def best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "max",
    ) -> dict[str, Any] | None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        best: dict[str, Any] | None = None
        best_val: float | None = None
        for run in self._backend.get_runs(experiment_name):
            entries = run.get("metrics", {}).get(metric)
            if not entries:
                continue
            last = entries[-1]
            value = last.get("value") if isinstance(last, dict) else last
            if value is None:
                continue
            if best_val is None:
                best, best_val = run, float(value)
                continue
            if mode == "max" and float(value) > best_val:
                best, best_val = run, float(value)
            elif mode == "min" and float(value) < best_val:
                best, best_val = run, float(value)
        return best


EXPERIMENT_TRACKER_REGISTRY = {"json": ExperimentTracker}
