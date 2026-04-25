"""Tests for src.training.experiment_tracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.experiment_tracker import (
    EXPERIMENT_TRACKER_REGISTRY,
    ExperimentTracker,
    JSONBackend,
    RunRecord,
    TrackingBackend,
)


@pytest.fixture()
def backend(tmp_path: Path) -> JSONBackend:
    return JSONBackend(str(tmp_path / "exp"))


@pytest.fixture()
def tracker(tmp_path: Path) -> ExperimentTracker:
    return ExperimentTracker(tracking_dir=str(tmp_path / "t"))


def test_runrecord_defaults():
    r = RunRecord(run_id="abc", run_name="n")
    assert r.status == "RUNNING"
    assert r.params == {} and r.metrics == {} and r.artifacts == []


def test_runrecord_to_dict_roundtrip():
    r = RunRecord(run_id="x", run_name="n", params={"lr": 1e-3})
    d = r.to_dict()
    assert d["run_id"] == "x"
    assert d["params"] == {"lr": 1e-3}


def test_registry_has_json_entry():
    assert "json" in EXPERIMENT_TRACKER_REGISTRY
    assert EXPERIMENT_TRACKER_REGISTRY["json"] is ExperimentTracker


def test_backend_is_abstract():
    assert issubclass(JSONBackend, TrackingBackend)


def test_json_backend_creates_root(tmp_path: Path):
    d = tmp_path / "missing"
    JSONBackend(str(d))
    assert d.exists()


def test_json_backend_start_run_returns_id(backend: JSONBackend):
    run_id = backend.start_run("run1", "exp", {"lr": 0.1})
    assert isinstance(run_id, str) and len(run_id) == 12


def test_json_backend_start_run_writes_params(backend: JSONBackend, tmp_path: Path):
    rid = backend.start_run("r", "exp", {"lr": 0.1})
    params_file = tmp_path / "exp" / "exp" / rid / "params.json"
    assert params_file.exists()
    assert json.loads(params_file.read_text())["lr"] == 0.1


def test_json_backend_unique_run_ids(backend: JSONBackend):
    ids = {backend.start_run(f"r{i}", "exp", {}) for i in range(10)}
    assert len(ids) == 10


def test_log_metrics_appends(backend: JSONBackend):
    rid = backend.start_run("r", "exp", {})
    backend.log_metrics(rid, {"loss": 1.0}, 0)
    backend.log_metrics(rid, {"loss": 0.5}, 1)
    run = backend.get_run(rid)
    assert run is not None
    assert len(run["metrics"]["loss"]) == 2
    assert run["metrics"]["loss"][1] == {"step": 1, "value": 0.5}


def test_log_metrics_missing_run_noop(backend: JSONBackend):
    backend.log_metrics("nope", {"a": 1.0}, 0)  # should not raise


def test_log_artifact_copies_file(backend: JSONBackend, tmp_path: Path):
    src = tmp_path / "art.txt"
    src.write_text("hi")
    rid = backend.start_run("r", "exp", {})
    backend.log_artifact(rid, str(src), "art.txt")
    run = backend.get_run(rid)
    assert "art.txt" in run["artifacts"]


def test_log_artifact_missing_source(backend: JSONBackend):
    rid = backend.start_run("r", "exp", {})
    backend.log_artifact(rid, "/does/not/exist", "x")
    run = backend.get_run(rid)
    assert run["artifacts"] == []


def test_log_artifact_missing_run(backend: JSONBackend, tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text("x")
    backend.log_artifact("nope", str(src), "a")  # no-op


def test_end_run_sets_status(backend: JSONBackend):
    rid = backend.start_run("r", "exp", {})
    backend.end_run(rid, "FINISHED")
    run = backend.get_run(rid)
    assert run["status"] == "FINISHED"
    assert run["end_time"] != ""


def test_end_run_missing_noop(backend: JSONBackend):
    backend.end_run("nope", "FAILED")


def test_get_runs_empty(backend: JSONBackend):
    assert backend.get_runs("nonexistent") == []


def test_get_runs_multiple(backend: JSONBackend):
    for i in range(3):
        rid = backend.start_run(f"r{i}", "exp", {"i": i})
        backend.end_run(rid, "FINISHED")
    runs = backend.get_runs("exp")
    assert len(runs) == 3


def test_get_run_none(backend: JSONBackend):
    assert backend.get_run("missing") is None


def test_tracker_start_and_end(tracker: ExperimentTracker):
    rid = tracker.start_run("r", "exp", {"x": 1})
    assert tracker.current_run_id == rid
    tracker.end_run("FINISHED")
    assert tracker.current_run_id is None


def test_tracker_log_metric_before_start(tracker: ExperimentTracker):
    tracker.log_metric("loss", 0.1, 0)  # no active run -> no-op


def test_tracker_log_metrics(tracker: ExperimentTracker):
    tracker.start_run("r", "exp", {})
    tracker.log_metrics({"loss": 0.5, "acc": 0.9}, 0)
    run = tracker.backend.get_run(tracker.current_run_id)
    assert "loss" in run["metrics"] and "acc" in run["metrics"]


def test_tracker_log_metric_single(tracker: ExperimentTracker):
    tracker.start_run("r", "exp", {})
    tracker.log_metric("loss", 0.3, 5)
    run = tracker.backend.get_run(tracker.current_run_id)
    assert run["metrics"]["loss"][0] == {"step": 5, "value": 0.3}


def test_tracker_log_artifact(tracker: ExperimentTracker, tmp_path: Path):
    f = tmp_path / "f.txt"
    f.write_text("a")
    tracker.start_run("r", "exp", {})
    tracker.log_artifact(str(f), "f.txt")
    run = tracker.backend.get_run(tracker.current_run_id)
    assert "f.txt" in run["artifacts"]


def test_tracker_context_manager_success(tracker: ExperimentTracker):
    with tracker.run("r", "exp", {"a": 1}) as rid:
        assert rid is not None
        tracker.log_metric("loss", 0.1, 0)
    run = tracker.backend.get_run(rid)
    assert run["status"] == "FINISHED"


def test_tracker_context_manager_failure(tracker: ExperimentTracker):
    with pytest.raises(RuntimeError):
        with tracker.run("r", "exp", {}) as rid:
            captured = rid
            raise RuntimeError("boom")
    run = tracker.backend.get_run(captured)
    assert run["status"] == "FAILED"


def test_best_run_max(tracker: ExperimentTracker):
    for v in [0.5, 0.9, 0.7]:
        with tracker.run(f"r-{v}", "exp", {}):
            tracker.log_metric("acc", v, 0)
    best = tracker.best_run("exp", "acc", mode="max")
    assert best is not None
    assert best["metrics"]["acc"][-1]["value"] == 0.9


def test_best_run_min(tracker: ExperimentTracker):
    for v in [0.5, 0.2, 0.7]:
        with tracker.run(f"r-{v}", "exp", {}):
            tracker.log_metric("loss", v, 0)
    best = tracker.best_run("exp", "loss", mode="min")
    assert best["metrics"]["loss"][-1]["value"] == 0.2


def test_best_run_no_runs(tracker: ExperimentTracker):
    assert tracker.best_run("missing", "loss") is None


def test_best_run_invalid_mode(tracker: ExperimentTracker):
    with pytest.raises(ValueError):
        tracker.best_run("exp", "loss", mode="bogus")


def test_best_run_ignores_runs_missing_metric(tracker: ExperimentTracker):
    with tracker.run("a", "exp", {}):
        tracker.log_metric("loss", 0.5, 0)
    with tracker.run("b", "exp", {}):
        tracker.log_metric("other", 0.1, 0)
    best = tracker.best_run("exp", "loss", mode="min")
    assert best is not None
    assert best["run_name"] == "a"


def test_end_run_without_start_is_safe(tracker: ExperimentTracker):
    tracker.end_run("FINISHED")


def test_tracker_custom_backend(tmp_path: Path):
    be = JSONBackend(str(tmp_path / "cb"))
    t = ExperimentTracker(backend=be)
    assert t.backend is be
