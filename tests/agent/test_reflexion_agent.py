"""Unit and integration tests for src.agent.reflexion_agent.

Tests 1-14 cover unit behaviour; test 15 is the integration test that drives a
full reflexion loop end-to-end.
"""

from __future__ import annotations

import pytest

from src.agent.reflexion_agent import (
    ReflexionAgent,
    ReflexionAttempt,
    ReflexionConfig,
    ReflexionMemory,
    ReflexionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scorer_fail(output: str) -> float:
    """Score function that always returns 0.0."""
    return 0.0


def _scorer_succeed(output: str) -> float:
    """Score function that always returns 1.0."""
    return 1.0


def _noop_attempt(context: str) -> str:
    """attempt_fn that returns its context unchanged."""
    return context


def _noop_reflect(prompt: str) -> str:
    """reflect_fn that returns a canned string."""
    return "I should try harder."


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ReflexionConfig()
    assert cfg.max_attempts == 4
    assert cfg.max_reflection_tokens == 256
    assert cfg.reflection_decay == 0.9
    assert cfg.early_stop_on_success is True
    assert cfg.memory_size == 10


# ---------------------------------------------------------------------------
# 2. test_memory_add_and_get
# ---------------------------------------------------------------------------


def test_memory_add_and_get():
    mem = ReflexionMemory(max_size=5)
    mem.add("first")
    mem.add("second")
    mem.add("third")
    items = mem.get_all()
    assert items == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# 3. test_memory_eviction
# ---------------------------------------------------------------------------


def test_memory_eviction():
    mem = ReflexionMemory(max_size=3)
    for i in range(5):
        mem.add(f"r{i}")
    # Only the three most-recent should remain.
    assert mem.get_all() == ["r2", "r3", "r4"]


# ---------------------------------------------------------------------------
# 4. test_memory_len
# ---------------------------------------------------------------------------


def test_memory_len():
    mem = ReflexionMemory(max_size=10)
    assert len(mem) == 0
    mem.add("a")
    mem.add("b")
    assert len(mem) == 2


# ---------------------------------------------------------------------------
# 5. test_memory_build_context
# ---------------------------------------------------------------------------


def test_memory_build_context():
    mem = ReflexionMemory(max_size=5)
    # Empty memory should return empty string.
    assert mem.build_context() == ""

    mem.add("missed the target")
    mem.add("ran out of time")
    ctx = mem.build_context(decay=0.9)
    assert isinstance(ctx, str)
    assert len(ctx) > 0
    assert "missed the target" in ctx
    assert "ran out of time" in ctx
    # Should have [Reflection N]: prefixes.
    assert "[Reflection 1]:" in ctx
    assert "[Reflection 2]:" in ctx


# ---------------------------------------------------------------------------
# 6. test_reflect_calls_fn
# ---------------------------------------------------------------------------


def test_reflect_calls_fn():
    """reflect() must call generate_fn with task and attempt in the prompt."""
    agent = ReflexionAgent()
    received: list[str] = []

    def gen(prompt: str) -> str:
        received.append(prompt)
        return "reflection text"

    result = agent.reflect(
        attempt_output="my attempt",
        task_description="solve X",
        score=0.3,
        generate_fn=gen,
    )
    assert result == "reflection text"
    assert len(received) == 1
    assert "solve X" in received[0]
    assert "my attempt" in received[0]


# ---------------------------------------------------------------------------
# 7. test_reflect_includes_score
# ---------------------------------------------------------------------------


def test_reflect_includes_score():
    """The score must appear verbatim in the prompt passed to generate_fn."""
    agent = ReflexionAgent()
    captured: list[str] = []

    def gen(prompt: str) -> str:
        captured.append(prompt)
        return "ok"

    agent.reflect(
        attempt_output="output",
        task_description="task",
        score=0.42,
        generate_fn=gen,
    )
    assert "0.42" in captured[0]


# ---------------------------------------------------------------------------
# 8. test_run_success_early_stop
# ---------------------------------------------------------------------------


def test_run_success_early_stop():
    """A perfect score on the first attempt should stop the loop immediately."""
    agent = ReflexionAgent(ReflexionConfig(max_attempts=4, early_stop_on_success=True))
    result = agent.run(
        task_description="trivial",
        attempt_fn=lambda ctx: "answer",
        score_fn=_scorer_succeed,
        reflect_fn=_noop_reflect,
        success_threshold=1.0,
    )
    assert result.success is True
    assert result.n_attempts == 1
    assert len(result.attempts) == 1


# ---------------------------------------------------------------------------
# 9. test_run_max_attempts
# ---------------------------------------------------------------------------


def test_run_max_attempts():
    """When score always fails, the agent should run exactly max_attempts."""
    cfg = ReflexionConfig(max_attempts=3)
    agent = ReflexionAgent(cfg)
    result = agent.run(
        task_description="impossible",
        attempt_fn=_noop_attempt,
        score_fn=_scorer_fail,
        reflect_fn=_noop_reflect,
    )
    assert result.n_attempts == 3
    assert len(result.attempts) == 3
    assert result.success is False


# ---------------------------------------------------------------------------
# 10. test_run_result_type
# ---------------------------------------------------------------------------


def test_run_result_type():
    """run() must return a ReflexionResult instance."""
    agent = ReflexionAgent()
    result = agent.run(
        task_description="check type",
        attempt_fn=_noop_attempt,
        score_fn=_scorer_fail,
        reflect_fn=_noop_reflect,
    )
    assert isinstance(result, ReflexionResult)


# ---------------------------------------------------------------------------
# 11. test_run_reflections_accumulated
# ---------------------------------------------------------------------------


def test_run_reflections_accumulated():
    """Each failed attempt must add one entry to reflection_history."""
    cfg = ReflexionConfig(max_attempts=3)
    agent = ReflexionAgent(cfg)

    call_count = [0]

    def counting_reflect(prompt: str) -> str:
        call_count[0] += 1
        return f"reflection {call_count[0]}"

    result = agent.run(
        task_description="accumulate",
        attempt_fn=_noop_attempt,
        score_fn=_scorer_fail,
        reflect_fn=counting_reflect,
    )
    # All 3 attempts fail -> 3 reflections.
    assert len(result.reflection_history) == 3
    assert result.reflection_history[0] == "reflection 1"
    assert result.reflection_history[2] == "reflection 3"


# ---------------------------------------------------------------------------
# 12. test_best_attempt
# ---------------------------------------------------------------------------


def test_best_attempt():
    """best_attempt() must return the attempt with the highest score."""
    scores = [0.2, 0.7, 0.5]
    idx = [0]

    def varied_score(output: str) -> float:
        s = scores[idx[0] % len(scores)]
        idx[0] += 1
        return s

    cfg = ReflexionConfig(max_attempts=3, early_stop_on_success=False)
    agent = ReflexionAgent(cfg)
    result = agent.run(
        task_description="find best",
        attempt_fn=_noop_attempt,
        score_fn=varied_score,
        reflect_fn=_noop_reflect,
        success_threshold=2.0,  # Never triggers early stop.
    )
    best = agent.best_attempt(result)
    assert isinstance(best, ReflexionAttempt)
    assert best.score == 0.7
    assert best.attempt_idx == 1


# ---------------------------------------------------------------------------
# 13. test_statistics_keys
# ---------------------------------------------------------------------------


def test_statistics_keys():
    """statistics() dict must contain exactly the required keys."""
    agent = ReflexionAgent()
    result = agent.run(
        task_description="keys test",
        attempt_fn=_noop_attempt,
        score_fn=_scorer_fail,
        reflect_fn=_noop_reflect,
    )
    stats = agent.statistics(result)
    required_keys = {"success_rate", "n_attempts", "best_score", "mean_score", "improvement"}
    assert required_keys.issubset(stats.keys())
    # All values must be numeric.
    for k, v in stats.items():
        assert isinstance(v, float), f"stats[{k!r}] is not float: {v!r}"


# ---------------------------------------------------------------------------
# 14. test_statistics_improvement
# ---------------------------------------------------------------------------


def test_statistics_improvement():
    """improvement should be positive when score increases over attempts."""
    scores_seq = [0.1, 0.3, 0.8]
    pos = [0]

    def rising_score(output: str) -> float:
        s = scores_seq[min(pos[0], len(scores_seq) - 1)]
        pos[0] += 1
        return s

    cfg = ReflexionConfig(max_attempts=3, early_stop_on_success=False)
    agent = ReflexionAgent(cfg)
    result = agent.run(
        task_description="improve",
        attempt_fn=_noop_attempt,
        score_fn=rising_score,
        reflect_fn=_noop_reflect,
        success_threshold=2.0,  # Never triggers.
    )
    stats = agent.statistics(result)
    assert stats["improvement"] > 0.0, f"Expected improvement > 0; got {stats['improvement']}"


# ---------------------------------------------------------------------------
# 15. Integration test
# ---------------------------------------------------------------------------


def test_integration_compute_2_plus_2():
    """Integration: agent solves '2+2=4' through reflection.

    attempt_fn echoes back the context it receives.
    score_fn returns 1.0 iff '4' appears in the output.
    The task deliberately omits '4', so the first attempt fails, the
    reflection is prepended, and a subsequent attempt that includes
    '4' in the reflection context must pass.
    """
    # A reflect_fn that always hints at the answer.
    def helpful_reflect(prompt: str) -> str:
        return "The answer is 4. Include '4' in your next attempt."

    # attempt_fn echoes the full context (which will contain the reflection
    # hint after the first failure).
    def echo_attempt(context: str) -> str:
        return context

    # score_fn: succeed iff '4' is in the output.
    def four_score(output: str) -> float:
        return 1.0 if "4" in output else 0.0

    cfg = ReflexionConfig(max_attempts=3, early_stop_on_success=True)
    agent = ReflexionAgent(cfg)
    result = agent.run(
        task_description="compute 2+2",
        attempt_fn=echo_attempt,
        score_fn=four_score,
        reflect_fn=helpful_reflect,
        success_threshold=1.0,
    )

    # Result structure assertions.
    assert isinstance(result, ReflexionResult)
    assert result.n_attempts >= 1
    assert isinstance(result.attempts, list)
    assert all(isinstance(a, ReflexionAttempt) for a in result.attempts)
    assert isinstance(result.final_output, str)
    assert isinstance(result.success, bool)
    assert isinstance(result.best_score, float)
    assert isinstance(result.reflection_history, list)

    # First attempt context does not contain '4' -> must fail.
    assert result.attempts[0].score == 0.0

    # After the first reflection, context will contain '4'.
    # The agent must succeed on attempt 2 (idx 1) at the latest.
    assert result.success is True
    assert result.n_attempts <= 3
    assert "4" in result.final_output
