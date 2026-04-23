from __future__ import annotations

import pytest

from src.training.elastic_coordinator import (
    ElasticConfig,
    ElasticCoordinator,
    WorkerInfo,
    WorkerState,
    ELASTIC_REGISTRY,
)


def make_coordinator(**kwargs) -> ElasticCoordinator:
    cfg = ElasticConfig(**kwargs)
    return ElasticCoordinator(cfg)


def test_register_worker_returns_info():
    coord = make_coordinator()
    info = coord.register_worker(0)
    assert isinstance(info, WorkerInfo)
    assert info.worker_id == 0
    assert info.state == WorkerState.INIT
    assert info.restart_count == 0


def test_register_multiple_workers():
    coord = make_coordinator()
    for i in range(3):
        coord.register_worker(i)
    assert len(coord.workers) == 3


def test_mark_failed_restarts_within_limit():
    coord = make_coordinator(max_restarts=3)
    coord.register_worker(0)
    result = coord.mark_failed(0)
    assert result is True
    assert coord.workers[0].state == WorkerState.INIT
    assert coord.workers[0].restart_count == 1


def test_mark_failed_exhausts_restarts():
    coord = make_coordinator(max_restarts=2)
    coord.register_worker(0)
    coord.mark_failed(0)
    coord.mark_failed(0)
    result = coord.mark_failed(0)
    assert result is False
    assert coord.workers[0].state == WorkerState.FAILED


def test_mark_succeeded():
    coord = make_coordinator()
    coord.register_worker(0)
    coord.mark_succeeded(0)
    assert coord.workers[0].state == WorkerState.SUCCEEDED


def test_active_workers_init_and_running():
    coord = make_coordinator()
    coord.register_worker(0)
    coord.register_worker(1)
    coord.workers[1].state = WorkerState.RUNNING
    coord.register_worker(2)
    coord.mark_succeeded(2)
    active = coord.active_workers()
    assert set(active) == {0, 1}


def test_should_restart_true():
    coord = make_coordinator(max_restarts=3)
    coord.register_worker(0)
    coord.workers[0].state = WorkerState.FAILED
    coord.workers[0].restart_count = 2
    assert coord.should_restart() is True


def test_should_restart_false_when_exhausted():
    coord = make_coordinator(max_restarts=2)
    coord.register_worker(0)
    coord.workers[0].state = WorkerState.FAILED
    coord.workers[0].restart_count = 2
    assert coord.should_restart() is False


def test_world_size_excludes_failed_and_stopped():
    coord = make_coordinator()
    for i in range(4):
        coord.register_worker(i)
    coord.workers[1].state = WorkerState.FAILED
    coord.workers[2].state = WorkerState.STOPPED
    assert coord.world_size() == 2


def test_is_elastic_valid_within_bounds():
    coord = make_coordinator(min_workers=2, max_workers=4)
    for i in range(3):
        coord.register_worker(i)
    assert coord.is_elastic_valid() is True


def test_is_elastic_valid_below_min():
    coord = make_coordinator(min_workers=3, max_workers=8)
    coord.register_worker(0)
    assert coord.is_elastic_valid() is False


def test_is_elastic_valid_above_max():
    coord = make_coordinator(min_workers=1, max_workers=2)
    for i in range(5):
        coord.register_worker(i)
    assert coord.is_elastic_valid() is False


def test_reset_clears_workers():
    coord = make_coordinator()
    coord.register_worker(0)
    coord.register_worker(1)
    coord.reset()
    assert len(coord.workers) == 0


def test_elastic_registry_key():
    assert "default" in ELASTIC_REGISTRY
    assert ELASTIC_REGISTRY["default"] is ElasticCoordinator


def test_default_config():
    coord = ElasticCoordinator()
    assert coord.config.min_workers == 1
    assert coord.config.max_workers == 8
    assert coord.config.rdzv_backend == "c10d"
    assert coord.config.max_restarts == 3
    assert coord.config.monitor_interval_s == 5.0
