"""Unit tests for src/agent/tutorial_engine.py."""

from __future__ import annotations

import pytest

from src.agent.tutorial_engine import (
    DEFAULT_TUTORIAL_ENGINE,
    TUTORIAL_ENGINE_REGISTRY,
    Tutorial,
    TutorialEngine,
    TutorialEngineError,
    TutorialProgress,
    TutorialStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_engine() -> TutorialEngine:
    """Return a clean engine to avoid test pollution."""
    return TutorialEngine()


def _step(
    step_id: str = "s1",
    check_fn=None,
    hints=None,
    is_checkpoint: bool = False,
) -> TutorialStep:
    return TutorialStep(
        step_id=step_id,
        title=f"Step {step_id}",
        description=f"Description for {step_id}",
        check_fn=check_fn,
        hints=hints or [],
        is_checkpoint=is_checkpoint,
    )


def _tutorial(
    tid: str = "t1",
    name: str = "Test Tutorial",
    prerequisites=None,
    steps=None,
) -> Tutorial:
    if steps is None:
        steps = [_step("s1"), _step("s2")]
    return Tutorial(
        tutorial_id=tid,
        name=name,
        description="A tutorial for testing",
        steps=steps,
        prerequisites=prerequisites or [],
        estimated_minutes=5,
    )


# ---------------------------------------------------------------------------
# 1. Register / basic lifecycle
# ---------------------------------------------------------------------------


def test_register_tutorial():
    engine = _fresh_engine()
    tutorial = _tutorial()
    engine.register_tutorial(tutorial)
    assert engine.list_tutorials() == [tutorial]


def test_register_duplicate_raises():
    engine = _fresh_engine()
    tutorial = _tutorial()
    engine.register_tutorial(tutorial)
    with pytest.raises(TutorialEngineError):
        engine.register_tutorial(tutorial)


# ---------------------------------------------------------------------------
# 2. Start tutorial
# ---------------------------------------------------------------------------


def test_start_tutorial():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    progress = engine.start_tutorial("u1", "t1")
    assert progress.tutorial_id == "t1"
    assert progress.user_id == "u1"
    assert progress.current_step_id == "s1"
    assert progress.started_at is not None


def test_start_tutorial_not_found():
    engine = _fresh_engine()
    with pytest.raises(TutorialEngineError):
        engine.start_tutorial("u1", "missing")


def test_start_tutorial_already_started():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    with pytest.raises(TutorialEngineError):
        engine.start_tutorial("u1", "t1")


def test_start_tutorial_no_steps():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial(steps=[]))
    with pytest.raises(TutorialEngineError):
        engine.start_tutorial("u1", "t1")


# ---------------------------------------------------------------------------
# 3. Step progression
# ---------------------------------------------------------------------------


def test_get_current_step():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    step = engine.get_current_step("u1", "t1")
    assert step.step_id == "s1"


def test_get_current_step_not_started():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    with pytest.raises(TutorialEngineError):
        engine.get_current_step("u1", "t1")


def test_submit_step_without_check_fn():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    result = engine.submit_step("u1", "t1", {})
    assert result["success"] is True
    assert result["next_step"] is not None
    assert result["next_step"].step_id == "s2"
    assert result["progress_percent"] == 50.0


def test_step_progression_to_completion():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    engine.submit_step("u1", "t1", {})
    result = engine.submit_step("u1", "t1", {})
    assert result["success"] is True
    assert result["next_step"] is None
    progress = engine.get_progress("u1", "t1")
    assert progress.completed_at is not None


def test_submit_after_completion_raises():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    engine.submit_step("u1", "t1", {})
    engine.submit_step("u1", "t1", {})
    with pytest.raises(TutorialEngineError):
        engine.submit_step("u1", "t1", {})


# ---------------------------------------------------------------------------
# 4. check_fn validation
# ---------------------------------------------------------------------------


def test_submit_step_with_check_fn_success():
    engine = _fresh_engine()
    step = _step("s1", check_fn=lambda r: r.get("ok") is True)
    engine.register_tutorial(_tutorial(steps=[step, _step("s2")]))
    engine.start_tutorial("u1", "t1")
    result = engine.submit_step("u1", "t1", {"ok": True})
    assert result["success"] is True
    assert result["next_step"].step_id == "s2"


def test_submit_step_with_check_fn_failure():
    engine = _fresh_engine()
    step = _step("s1", check_fn=lambda r: r.get("ok") is True)
    engine.register_tutorial(_tutorial(steps=[step, _step("s2")]))
    engine.start_tutorial("u1", "t1")
    result = engine.submit_step("u1", "t1", {"ok": False})
    assert result["success"] is False
    assert result["next_step"] is None


def test_attempts_per_step_tracked():
    engine = _fresh_engine()
    step = _step("s1", check_fn=lambda r: r.get("ok") is True)
    engine.register_tutorial(_tutorial(steps=[step, _step("s2")]))
    engine.start_tutorial("u1", "t1")
    engine.submit_step("u1", "t1", {"ok": False})
    engine.submit_step("u1", "t1", {"ok": False})
    engine.submit_step("u1", "t1", {"ok": True})
    progress = engine.get_progress("u1", "t1")
    assert progress.attempts_per_step["s1"] == 3


# ---------------------------------------------------------------------------
# 5. Hint cycling
# ---------------------------------------------------------------------------


def test_hint_cycle():
    engine = _fresh_engine()
    step = _step("s1", hints=["hint-a", "hint-b", "hint-c"])
    engine.register_tutorial(_tutorial(steps=[step]))
    engine.start_tutorial("u1", "t1")
    assert engine.request_hint("u1", "t1") == "hint-a"
    assert engine.request_hint("u1", "t1") == "hint-b"
    assert engine.request_hint("u1", "t1") == "hint-c"
    assert engine.request_hint("u1", "t1") == "hint-a"


def test_hint_no_hints_available():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    assert engine.request_hint("u1", "t1") == "No hints available for this step."


# ---------------------------------------------------------------------------
# 6. Prerequisites & availability
# ---------------------------------------------------------------------------


def test_list_available_tutorials_prerequisites():
    engine = _fresh_engine()
    t1 = _tutorial("t1", steps=[_step("s1")])
    t2 = _tutorial("t2", prerequisites=["t1"])
    engine.register_tutorial(t1)
    engine.register_tutorial(t2)

    available = engine.list_available_tutorials("u1")
    assert len(available) == 1
    assert available[0].tutorial_id == "t1"

    engine.start_tutorial("u1", "t1")
    engine.submit_step("u1", "t1", {})
    available = engine.list_available_tutorials("u1")
    assert len(available) == 2


def test_list_tutorials():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial("a"))
    engine.register_tutorial(_tutorial("b"))
    assert len(engine.list_tutorials()) == 2


# ---------------------------------------------------------------------------
# 7. Progress tracking
# ---------------------------------------------------------------------------


def test_get_progress():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    progress = engine.get_progress("u1", "t1")
    assert isinstance(progress, TutorialProgress)


def test_get_progress_not_found():
    engine = _fresh_engine()
    with pytest.raises(TutorialEngineError):
        engine.get_progress("u1", "t1")


# ---------------------------------------------------------------------------
# 8. Reset
# ---------------------------------------------------------------------------


def test_reset_progress():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial())
    engine.start_tutorial("u1", "t1")
    engine.reset_progress("u1", "t1")
    with pytest.raises(TutorialEngineError):
        engine.get_progress("u1", "t1")


def test_reset_progress_not_found():
    engine = _fresh_engine()
    with pytest.raises(TutorialEngineError):
        engine.reset_progress("u1", "t1")


def test_reset_clears_hint_index():
    engine = _fresh_engine()
    step = _step("s1", hints=["h1", "h2"])
    engine.register_tutorial(_tutorial(steps=[step]))
    engine.start_tutorial("u1", "t1")
    engine.request_hint("u1", "t1")
    engine.reset_progress("u1", "t1")
    engine.start_tutorial("u1", "t1")
    assert engine.request_hint("u1", "t1") == "h1"


# ---------------------------------------------------------------------------
# 9. Stats
# ---------------------------------------------------------------------------


def test_stats():
    engine = _fresh_engine()
    engine.register_tutorial(_tutorial("t1"))
    engine.register_tutorial(_tutorial("t2"))
    engine.start_tutorial("u1", "t1")
    engine.submit_step("u1", "t1", {})
    engine.submit_step("u1", "t1", {})
    engine.start_tutorial("u2", "t2")
    engine.submit_step("u2", "t2", {})

    stats = engine.stats()
    assert stats["total_tutorials"] == 2
    assert stats["total_completions"] == 1
    assert stats["avg_completion_rate"] == 75.0


# ---------------------------------------------------------------------------
# 10. Registry / singleton
# ---------------------------------------------------------------------------


def test_default_engine_singleton():
    assert "default" in TUTORIAL_ENGINE_REGISTRY
    assert TUTORIAL_ENGINE_REGISTRY["default"] is DEFAULT_TUTORIAL_ENGINE
    assert isinstance(DEFAULT_TUTORIAL_ENGINE, TutorialEngine)
