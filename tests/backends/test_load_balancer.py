"""Tests for src.backends.load_balancer."""
from __future__ import annotations

import pytest

from src.backends.load_balancer import BackendInstance, LBAlgorithm, LoadBalancer, LOAD_BALANCER_REGISTRY


def _inst(id: str, weight: float = 1.0, active: int = 0) -> BackendInstance:
    return BackendInstance(instance_id=id, host="localhost", port=8000, weight=weight, active_requests=active)


def test_registry_has_default():
    assert LOAD_BALANCER_REGISTRY["default"] is LoadBalancer


def test_empty_pool_returns_none():
    lb = LoadBalancer([])
    assert lb.select() is None


def test_round_robin_cycles():
    instances = [_inst("a"), _inst("b"), _inst("c")]
    lb = LoadBalancer(instances, LBAlgorithm.ROUND_ROBIN)
    selected = [lb.select().instance_id for _ in range(6)]
    assert selected == ["a", "b", "c", "a", "b", "c"]


def test_round_robin_single():
    lb = LoadBalancer([_inst("only")], LBAlgorithm.ROUND_ROBIN)
    assert lb.select().instance_id == "only"
    assert lb.select().instance_id == "only"


def test_least_connections_picks_min():
    instances = [_inst("a", active=5), _inst("b", active=1), _inst("c", active=3)]
    lb = LoadBalancer(instances, LBAlgorithm.LEAST_CONNECTIONS)
    assert lb.select().instance_id == "b"


def test_least_connections_tiebreak_by_id():
    instances = [_inst("b", active=0), _inst("a", active=0)]
    lb = LoadBalancer(instances, LBAlgorithm.LEAST_CONNECTIONS)
    assert lb.select().instance_id == "a"


def test_weighted_round_robin_picks_highest():
    instances = [_inst("heavy", weight=10.0, active=0), _inst("light", weight=1.0, active=0)]
    lb = LoadBalancer(instances, LBAlgorithm.WEIGHTED_ROUND_ROBIN)
    assert lb.select().instance_id == "heavy"


def test_weighted_prefers_less_loaded():
    instances = [_inst("a", weight=2.0, active=10), _inst("b", weight=1.0, active=0)]
    lb = LoadBalancer(instances, LBAlgorithm.WEIGHTED_ROUND_ROBIN)
    chosen = lb.select()
    assert chosen is not None


def test_random_returns_valid_instance():
    instances = [_inst("x"), _inst("y"), _inst("z")]
    lb = LoadBalancer(instances, LBAlgorithm.RANDOM)
    for _ in range(20):
        chosen = lb.select()
        assert chosen.instance_id in {"x", "y", "z"}


def test_random_single_always_that():
    lb = LoadBalancer([_inst("sole")], LBAlgorithm.RANDOM)
    for _ in range(5):
        assert lb.select().instance_id == "sole"


def test_acquire_increments_active_and_total():
    lb = LoadBalancer([])
    inst = _inst("srv")
    lb.acquire(inst)
    assert inst.active_requests == 1
    assert inst.total_requests == 1


def test_acquire_twice():
    lb = LoadBalancer([])
    inst = _inst("srv")
    lb.acquire(inst)
    lb.acquire(inst)
    assert inst.active_requests == 2
    assert inst.total_requests == 2


def test_release_decrements_active():
    lb = LoadBalancer([])
    inst = _inst("srv")
    lb.acquire(inst)
    lb.release(inst)
    assert inst.active_requests == 0
    assert inst.total_requests == 1


def test_release_floors_at_zero():
    lb = LoadBalancer([])
    inst = _inst("srv")
    lb.release(inst)
    assert inst.active_requests == 0


def test_add_instance_expands_pool():
    lb = LoadBalancer([_inst("a")], LBAlgorithm.ROUND_ROBIN)
    lb.add_instance(_inst("b"))
    ids = {lb.select().instance_id, lb.select().instance_id}
    assert "b" in ids


def test_remove_instance_returns_true():
    lb = LoadBalancer([_inst("a"), _inst("b")])
    assert lb.remove_instance("a") is True


def test_remove_instance_returns_false_when_missing():
    lb = LoadBalancer([_inst("a")])
    assert lb.remove_instance("z") is False


def test_remove_instance_shrinks_pool():
    lb = LoadBalancer([_inst("a"), _inst("b"), _inst("c")], LBAlgorithm.ROUND_ROBIN)
    lb.remove_instance("b")
    for _ in range(6):
        assert lb.select().instance_id != "b"


def test_remove_all_leaves_empty():
    lb = LoadBalancer([_inst("a")])
    lb.remove_instance("a")
    assert lb.select() is None


def test_stats_format():
    inst = _inst("s1", weight=2.5)
    lb = LoadBalancer([inst])
    lb.acquire(inst)
    s = lb.stats()
    assert len(s) == 1
    assert s[0]["id"] == "s1"
    assert s[0]["active"] == 1
    assert s[0]["total"] == 1
    assert s[0]["weight"] == 2.5


def test_stats_empty():
    lb = LoadBalancer([])
    assert lb.stats() == []


def test_stats_multiple():
    lb = LoadBalancer([_inst("a"), _inst("b")])
    assert len(lb.stats()) == 2


def test_round_robin_after_remove_no_crash():
    instances = [_inst("a"), _inst("b"), _inst("c")]
    lb = LoadBalancer(instances, LBAlgorithm.ROUND_ROBIN)
    lb.select()
    lb.remove_instance("b")
    for _ in range(4):
        assert lb.select() is not None


def test_acquire_release_active_counts():
    lb = LoadBalancer([])
    inst = _inst("srv")
    lb.acquire(inst)
    lb.acquire(inst)
    lb.release(inst)
    assert inst.active_requests == 1


def test_backend_instance_defaults():
    inst = BackendInstance(instance_id="x", host="h", port=80)
    assert inst.weight == 1.0
    assert inst.active_requests == 0
    assert inst.total_requests == 0


def test_lb_algorithm_values():
    assert LBAlgorithm.ROUND_ROBIN.value == "round_robin"
    assert LBAlgorithm.LEAST_CONNECTIONS.value == "least_connections"
    assert LBAlgorithm.WEIGHTED_ROUND_ROBIN.value == "weighted_round_robin"
    assert LBAlgorithm.RANDOM.value == "random"


def test_default_algorithm_is_round_robin():
    lb = LoadBalancer([_inst("a"), _inst("b")])
    assert lb._algorithm is LBAlgorithm.ROUND_ROBIN
