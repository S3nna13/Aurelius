"""Integration tests for BackgroundExecutor: registry wiring and end-to-end flow."""

from __future__ import annotations


def test_background_executor_registered():
    """BackgroundExecutor must appear in AGENT_LOOP_REGISTRY."""
    import src.agent as agent
    from src.agent.background_executor import BackgroundExecutor

    assert "background_executor" in agent.AGENT_LOOP_REGISTRY
    assert agent.AGENT_LOOP_REGISTRY["background_executor"] is BackgroundExecutor


def test_prior_registries_still_intact():
    """Existing registry entries must not be clobbered by the new module."""
    import src.agent as agent

    assert "react" in agent.AGENT_LOOP_REGISTRY
    assert "dispatch_task" in agent.AGENT_LOOP_REGISTRY
    assert "budget_bounded" in agent.AGENT_LOOP_REGISTRY


def test_full_lifecycle_priority_order():
    """Submit 3 tasks with different priorities, run all, verify completion order."""
    from src.agent.background_executor import BackgroundExecutor, TaskStatus

    execution_order: list[str] = []

    def make_fn(label: str):
        def fn():
            execution_order.append(label)
            return label

        return fn

    ex = BackgroundExecutor()
    ex.submit("mid", make_fn("mid"), priority=5)
    ex.submit("high", make_fn("high"), priority=1)
    ex.submit("low", make_fn("low"), priority=3)

    # Run all three tasks in sequence
    t1 = ex.run_next()
    t2 = ex.run_next()
    t3 = ex.run_next()

    # Priority order: 1 → 3 → 5
    assert execution_order == ["high", "low", "mid"]

    # All completed
    assert t1.status == TaskStatus.COMPLETED  # type: ignore[union-attr]
    assert t2.status == TaskStatus.COMPLETED  # type: ignore[union-attr]
    assert t3.status == TaskStatus.COMPLETED  # type: ignore[union-attr]

    # Results match labels
    assert t1.result == "high"  # type: ignore[union-attr]
    assert t2.result == "low"  # type: ignore[union-attr]
    assert t3.result == "mid"  # type: ignore[union-attr]

    # Queue exhausted
    assert ex.pending_count() == 0
    assert ex.run_next() is None


def test_registry_class_is_instantiable():
    """The registered class can be instantiated via the registry key."""
    import src.agent as agent

    cls = agent.AGENT_LOOP_REGISTRY["background_executor"]
    instance = cls()
    # Verify basic contract
    assert instance.pending_count() == 0
    assert instance.running_count() == 0
    assert instance.queue_snapshot() == []
