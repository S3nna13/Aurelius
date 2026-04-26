"""Integration tests for src.eval — WebArena scorer integration.

WebArena benchmark (Zhou et al. 2307.13854, Apache-2.0), GitHub Actions (MIT spec),
clean-room implementation.
"""

from __future__ import annotations

from src.eval.webarena_scorer import (
    WEBARENA_TASK_REGISTRY,
    WebArenaResult,
    WebArenaScorer,
)

# ---------------------------------------------------------------------------
# WebArena scorer integration
# ---------------------------------------------------------------------------


def test_webarena_scorer() -> None:
    """Score 3 results against WEBARENA_TASK_REGISTRY tasks and verify metrics structure."""
    scorer = WebArenaScorer()

    # Pick 3 tasks from the registry
    task_ids = list(WEBARENA_TASK_REGISTRY.keys())[:3]
    tasks_subset = {tid: WEBARENA_TASK_REGISTRY[tid] for tid in task_ids}

    # Two successful results (match the first reference answer), one failure
    results = [
        WebArenaResult(
            task_id=task_ids[0],
            predicted_answer=WEBARENA_TASK_REGISTRY[task_ids[0]].reference_answers[0],
            steps_taken=3,
        ),
        WebArenaResult(
            task_id=task_ids[1],
            predicted_answer=WEBARENA_TASK_REGISTRY[task_ids[1]].reference_answers[0],
            steps_taken=5,
        ),
        WebArenaResult(
            task_id=task_ids[2],
            predicted_answer="completely wrong answer xyz",
            steps_taken=1,
        ),
    ]

    metrics = scorer.score_batch(results, tasks_subset)

    # Structural checks
    assert metrics.n_tasks == 3
    assert isinstance(metrics.success_rate, float)
    assert 0.0 <= metrics.success_rate <= 1.0
    assert isinstance(metrics.by_category, dict)
    assert isinstance(metrics.by_difficulty, dict)
    assert metrics.avg_steps > 0

    # At least the first two should succeed
    assert metrics.n_success >= 2

    # format_report should produce a non-empty string
    report = scorer.format_report(metrics)
    assert len(report) > 0
    assert "WebArena" in report


# ---------------------------------------------------------------------------
# OSWorld scorer integration (additive)
# OSWorld: Xie et al. 2024 (2406.14800, Apache-2.0), clean-room reimplementation.
# ---------------------------------------------------------------------------

from src.eval.osworld_scorer import (  # noqa: E402
    OSWORLD_TASK_REGISTRY,
    OSWorldResult,
    OSWorldScorer,
)


def test_osworld_scorer_batch() -> None:
    """Score 3 results against OSWORLD_TASK_REGISTRY and verify metrics."""
    scorer = OSWorldScorer()

    task_ids = list(OSWORLD_TASK_REGISTRY.keys())[:3]
    tasks_subset = {tid: OSWORLD_TASK_REGISTRY[tid] for tid in task_ids}

    results = [
        OSWorldResult(
            task_id=task_ids[0],
            app=OSWORLD_TASK_REGISTRY[task_ids[0]].app,
            completed=True,
            steps_taken=2,
        ),
        OSWorldResult(
            task_id=task_ids[1],
            app=OSWORLD_TASK_REGISTRY[task_ids[1]].app,
            completed=True,
            steps_taken=4,
        ),
        OSWorldResult(
            task_id=task_ids[2],
            app=OSWORLD_TASK_REGISTRY[task_ids[2]].app,
            completed=False,
            steps_taken=1,
        ),
    ]

    metrics = scorer.score_batch(results, tasks_subset)

    assert metrics.n_tasks == 3
    assert metrics.n_completed == 2
    assert abs(metrics.completion_rate - 2 / 3) < 1e-9
    assert isinstance(metrics.by_app, dict)
    assert isinstance(metrics.by_difficulty, dict)
    assert metrics.avg_steps > 0

    report = scorer.format_report(metrics)
    assert len(report) > 0
