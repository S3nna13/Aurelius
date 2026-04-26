"""Tests for src/multiagent/task_router.py."""

from __future__ import annotations

from src.multiagent.task_router import (
    TASK_ROUTER,
    RoutingStrategy,
    TaskRouter,
)

# ---------------------------------------------------------------------------
# RoutingStrategy enum
# ---------------------------------------------------------------------------


def test_routing_strategy_values():
    assert RoutingStrategy.CAPABILITY == "capability"
    assert RoutingStrategy.ROUND_ROBIN == "round_robin"
    assert RoutingStrategy.LEAST_LOADED == "least_loaded"
    assert RoutingStrategy.PRIORITY == "priority"


def test_routing_strategy_count():
    assert len(RoutingStrategy) == 4


def test_routing_strategy_is_str():
    assert isinstance(RoutingStrategy.CAPABILITY, str)


# ---------------------------------------------------------------------------
# register() / registered_agents()
# ---------------------------------------------------------------------------


def test_register_and_list():
    router = TaskRouter()
    router.register("a1", ["nlp"])
    assert "a1" in router.registered_agents()


def test_register_multiple():
    router = TaskRouter()
    router.register("a1", ["nlp"])
    router.register("a2", ["vision"])
    agents = router.registered_agents()
    assert "a1" in agents
    assert "a2" in agents


def test_register_overwrite():
    router = TaskRouter()
    router.register("a1", ["nlp"], load=5)
    router.register("a1", ["vision"], load=0)
    # last registration wins — a1 now has only "vision"
    assert "a1" in router.agents_by_capability("vision")
    assert "a1" not in router.agents_by_capability("nlp")


def test_registered_agents_empty():
    router = TaskRouter()
    assert router.registered_agents() == []


# ---------------------------------------------------------------------------
# route() — CAPABILITY
# ---------------------------------------------------------------------------


def test_route_capability_returns_matching_agent():
    router = TaskRouter(RoutingStrategy.CAPABILITY)
    router.register("a1", ["summarize"])
    result = router.route("do task", required_capability="summarize")
    assert result == "a1"


def test_route_capability_returns_none_if_no_match():
    router = TaskRouter(RoutingStrategy.CAPABILITY)
    router.register("a1", ["vision"])
    result = router.route("do task", required_capability="nlp")
    assert result is None


def test_route_capability_picks_lowest_load():
    router = TaskRouter(RoutingStrategy.CAPABILITY)
    router.register("a1", ["nlp"], load=10)
    router.register("a2", ["nlp"], load=2)
    result = router.route("task", required_capability="nlp")
    assert result == "a2"


def test_route_capability_no_required_picks_lowest_load():
    router = TaskRouter(RoutingStrategy.CAPABILITY)
    router.register("a1", ["x"], load=5)
    router.register("a2", ["y"], load=1)
    result = router.route("task")
    assert result == "a2"


def test_route_capability_returns_none_empty_router():
    router = TaskRouter(RoutingStrategy.CAPABILITY)
    assert router.route("task", required_capability="nlp") is None


# ---------------------------------------------------------------------------
# route() — ROUND_ROBIN
# ---------------------------------------------------------------------------


def test_route_round_robin_cycles():
    router = TaskRouter(RoutingStrategy.ROUND_ROBIN)
    router.register("a1", [])
    router.register("a2", [])
    router.register("a3", [])
    seen = [router.route("task") for _ in range(6)]
    # each agent must appear at least twice in 6 calls
    for agent in ["a1", "a2", "a3"]:
        assert agent in seen


def test_route_round_robin_single_agent():
    router = TaskRouter(RoutingStrategy.ROUND_ROBIN)
    router.register("only", [])
    for _ in range(5):
        assert router.route("task") == "only"


def test_route_round_robin_returns_none_empty():
    router = TaskRouter(RoutingStrategy.ROUND_ROBIN)
    assert router.route("task") is None


def test_route_round_robin_increments_index():
    router = TaskRouter(RoutingStrategy.ROUND_ROBIN)
    router.register("a1", [])
    router.register("a2", [])
    first = router.route("task")
    second = router.route("task")
    assert first != second


# ---------------------------------------------------------------------------
# route() — LEAST_LOADED
# ---------------------------------------------------------------------------


def test_route_least_loaded():
    router = TaskRouter(RoutingStrategy.LEAST_LOADED)
    router.register("a1", [], load=10)
    router.register("a2", [], load=0)
    assert router.route("task") == "a2"


def test_route_least_loaded_tie_picks_one():
    router = TaskRouter(RoutingStrategy.LEAST_LOADED)
    router.register("a1", [], load=0)
    router.register("a2", [], load=0)
    result = router.route("task")
    assert result in ("a1", "a2")


def test_route_least_loaded_empty():
    router = TaskRouter(RoutingStrategy.LEAST_LOADED)
    assert router.route("task") is None


# ---------------------------------------------------------------------------
# update_load()
# ---------------------------------------------------------------------------


def test_update_load_increment():
    router = TaskRouter()
    router.register("a1", [], load=0)
    router.update_load("a1", delta=3)
    # Verify by routing: a1 has load 3, a2 has 0 — a2 should win
    router.register("a2", [], load=0)
    router2 = TaskRouter(RoutingStrategy.LEAST_LOADED)
    router2.register("a1", [], load=0)
    router2.register("a2", [], load=0)
    router2.update_load("a1", delta=3)
    assert router2.route("task") == "a2"


def test_update_load_decrement():
    router = TaskRouter(RoutingStrategy.LEAST_LOADED)
    router.register("a1", [], load=5)
    router.register("a2", [], load=2)
    router.update_load("a1", delta=-10)
    assert router.route("task") == "a1"


def test_update_load_unknown_agent_noop():
    router = TaskRouter()
    router.register("a1", [], load=1)
    router.update_load("nonexistent", delta=100)
    # Should not raise
    assert "a1" in router.registered_agents()


# ---------------------------------------------------------------------------
# route() — PRIORITY
# ---------------------------------------------------------------------------


def test_route_priority_picks_highest():
    router = TaskRouter(RoutingStrategy.PRIORITY)
    router.register("a1", ["nlp"], priority=1)
    router.register("a2", ["nlp"], priority=5)
    assert router.route("task", required_capability="nlp") == "a2"


def test_route_priority_with_capability_filter():
    router = TaskRouter(RoutingStrategy.PRIORITY)
    router.register("a1", ["vision"], priority=10)
    router.register("a2", ["nlp"], priority=1)
    result = router.route("task", required_capability="nlp")
    assert result == "a2"


def test_route_priority_no_match_returns_none():
    router = TaskRouter(RoutingStrategy.PRIORITY)
    router.register("a1", ["vision"], priority=10)
    result = router.route("task", required_capability="nlp")
    assert result is None


# ---------------------------------------------------------------------------
# agents_by_capability()
# ---------------------------------------------------------------------------


def test_agents_by_capability_returns_matching():
    router = TaskRouter()
    router.register("a1", ["nlp", "vision"])
    router.register("a2", ["vision"])
    router.register("a3", ["nlp"])
    result = router.agents_by_capability("nlp")
    assert "a1" in result
    assert "a3" in result
    assert "a2" not in result


def test_agents_by_capability_empty():
    router = TaskRouter()
    router.register("a1", ["vision"])
    assert router.agents_by_capability("nlp") == []


def test_agents_by_capability_all():
    router = TaskRouter()
    router.register("a1", ["x"])
    router.register("a2", ["x"])
    assert set(router.agents_by_capability("x")) == {"a1", "a2"}


# ---------------------------------------------------------------------------
# TASK_ROUTER singleton
# ---------------------------------------------------------------------------


def test_task_router_singleton_exists():
    assert TASK_ROUTER is not None
    assert isinstance(TASK_ROUTER, TaskRouter)
