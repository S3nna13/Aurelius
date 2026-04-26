import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from serving.session_router import ConsistentHashRouter, Session, SessionConfig, SessionManager


@pytest.fixture
def config():
    return SessionConfig(n_workers=4, max_sessions_per_worker=32, eviction_policy="lru")


@pytest.fixture
def router(config):
    return ConsistentHashRouter(config)


@pytest.fixture
def manager(config):
    return SessionManager(config)


def test_consistent_hash_router_instantiates(config):
    r = ConsistentHashRouter(config)
    assert r is not None


def test_route_returns_int_in_range(router, config):
    worker_id = router.route("test-session-abc")
    assert isinstance(worker_id, int)
    assert 0 <= worker_id < config.n_workers


def test_same_session_id_always_routes_to_same_worker(router):
    sid = "stable-session-id-123"
    results = {router.route(sid) for _ in range(10)}
    assert len(results) == 1


def test_different_session_ids_can_route_to_different_workers(router, config):
    ids = [f"session-{i}" for i in range(50)]
    workers = {router.route(sid) for sid in ids}
    assert len(workers) > 1


def test_session_manager_instantiates(config):
    m = SessionManager(config)
    assert m is not None


def test_create_session_returns_session(manager):
    s = manager.create_session()
    assert isinstance(s, Session)


def test_session_has_valid_worker_id(manager, config):
    s = manager.create_session()
    assert 0 <= s.worker_id < config.n_workers


def test_get_session_retrieves_created_session(manager):
    s = manager.create_session()
    retrieved = manager.get_session(s.session_id)
    assert retrieved is not None
    assert retrieved.session_id == s.session_id


def test_get_session_returns_none_for_unknown(manager):
    result = manager.get_session("nonexistent-session-id")
    assert result is None


def test_update_session_increments_n_turns(manager):
    s = manager.create_session()
    assert s.n_turns == 0
    manager.update_session(s.session_id)
    assert s.n_turns == 1
    manager.update_session(s.session_id)
    assert s.n_turns == 2


def test_list_sessions_contains_created_session(manager):
    s = manager.create_session()
    sessions = manager.list_sessions()
    ids = [x.session_id for x in sessions]
    assert s.session_id in ids


def test_session_count_increases_after_create(manager):
    count_before = manager.session_count()
    manager.create_session()
    assert manager.session_count() == count_before + 1
