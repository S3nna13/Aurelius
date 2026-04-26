"""Unit tests for src.eval.osworld_scorer (16 tests).

OSWorld: Xie et al. 2024 (2406.14800, Apache-2.0), clean-room reimplementation.
"""

from __future__ import annotations

from src.eval.osworld_scorer import (
    OSWORLD_TASK_REGISTRY,
    OSWorldResult,
    OSWorldScorer,
    OSWorldTask,
)

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_osworld_task_defaults() -> None:
    """OSWorldTask must default difficulty='medium' and subtasks=[]."""
    task = OSWorldTask(
        task_id="t1",
        app="chrome",
        instruction="Do something",
        reference_answers=["answer"],
    )
    assert task.difficulty == "medium"
    assert task.subtasks == []


def test_osworld_task_all_difficulties() -> None:
    for diff in ("easy", "medium", "hard"):
        task = OSWorldTask(
            task_id="t",
            app="a",
            instruction="i",
            reference_answers=[],
            difficulty=diff,  # type: ignore[arg-type]
        )
        assert task.difficulty == diff


def test_osworld_result_defaults() -> None:
    """OSWorldResult defaults: completed=False, error=None, final_screen_state=None."""
    result = OSWorldResult(
        task_id="t1",
        app="chrome",
        completed=False,
        steps_taken=0,
    )
    assert result.completed is False
    assert result.error is None
    assert result.final_screen_state is None


def test_osworld_result_with_screen_state() -> None:
    result = OSWorldResult(
        task_id="t1",
        app="chrome",
        completed=True,
        steps_taken=3,
        final_screen_state="New Tab opened",
    )
    assert result.final_screen_state == "New Tab opened"


# ---------------------------------------------------------------------------
# score_result()
# ---------------------------------------------------------------------------


def _make_task(refs: list[str], difficulty: str = "medium") -> OSWorldTask:
    return OSWorldTask(
        task_id="tid",
        app="chrome",
        instruction="Do it",
        reference_answers=refs,
        difficulty=difficulty,  # type: ignore[arg-type]
    )


def test_score_result_completed_no_screen_true() -> None:
    """completed=True with no final_screen_state -> True."""
    scorer = OSWorldScorer()
    task = _make_task(["ref"])
    result = OSWorldResult(task_id="tid", app="chrome", completed=True, steps_taken=1)
    assert scorer.score_result(result, task) is True


def test_score_result_not_completed_false() -> None:
    """completed=False -> always False regardless of screen state."""
    scorer = OSWorldScorer()
    task = _make_task(["ref"])
    result = OSWorldResult(
        task_id="tid",
        app="chrome",
        completed=False,
        steps_taken=1,
        final_screen_state="ref visible here",
    )
    assert scorer.score_result(result, task) is False


def test_score_result_completed_screen_contains_ref_true() -> None:
    """completed=True, screen contains a reference answer -> True."""
    scorer = OSWorldScorer()
    task = _make_task(["New Tab", "newtab"])
    result = OSWorldResult(
        task_id="tid",
        app="chrome",
        completed=True,
        steps_taken=2,
        final_screen_state="Browser showed New Tab page",
    )
    assert scorer.score_result(result, task) is True


def test_score_result_completed_screen_missing_ref_false() -> None:
    """completed=True but screen does NOT contain any reference -> False."""
    scorer = OSWorldScorer()
    task = _make_task(["Expected Output", "correct result"])
    result = OSWorldResult(
        task_id="tid",
        app="chrome",
        completed=True,
        steps_taken=2,
        final_screen_state="Something completely different",
    )
    assert scorer.score_result(result, task) is False


def test_score_result_completed_any_ref_matches() -> None:
    """If any one reference answer matches, result is True."""
    scorer = OSWorldScorer()
    task = _make_task(["alpha", "beta", "gamma"])
    result = OSWorldResult(
        task_id="tid",
        app="chrome",
        completed=True,
        steps_taken=1,
        final_screen_state="screen shows gamma text",
    )
    assert scorer.score_result(result, task) is True


# ---------------------------------------------------------------------------
# score_batch()
# ---------------------------------------------------------------------------


def _make_tasks() -> dict[str, OSWorldTask]:
    return {
        "t1": OSWorldTask("t1", "chrome", "i1", ["ok"], difficulty="easy"),
        "t2": OSWorldTask("t2", "vscode", "i2", ["ok"], difficulty="medium"),
        "t3": OSWorldTask("t3", "chrome", "i3", ["ok"], difficulty="hard"),
        "t4": OSWorldTask("t4", "terminal", "i4", ["ok"], difficulty="easy"),
    }


def test_score_batch_completion_rate() -> None:
    """3 of 4 completed -> completion_rate ~= 0.75."""
    scorer = OSWorldScorer()
    tasks = _make_tasks()
    results = [
        OSWorldResult("t1", "chrome", True, 1),
        OSWorldResult("t2", "vscode", True, 2),
        OSWorldResult("t3", "chrome", True, 3),
        OSWorldResult("t4", "terminal", False, 4),
    ]
    metrics = scorer.score_batch(results, tasks)
    assert metrics.n_tasks == 4
    assert metrics.n_completed == 3
    assert abs(metrics.completion_rate - 0.75) < 1e-9


def test_score_batch_by_app_groups_correctly() -> None:
    """by_app should group results per app correctly."""
    scorer = OSWorldScorer()
    tasks = _make_tasks()
    results = [
        OSWorldResult("t1", "chrome", True, 1),  # chrome pass
        OSWorldResult("t2", "vscode", False, 2),  # vscode fail
        OSWorldResult("t3", "chrome", True, 3),  # chrome pass
        OSWorldResult("t4", "terminal", True, 4),  # terminal pass
    ]
    metrics = scorer.score_batch(results, tasks)
    assert "chrome" in metrics.by_app
    assert abs(metrics.by_app["chrome"] - 1.0) < 1e-9
    assert abs(metrics.by_app["vscode"] - 0.0) < 1e-9
    assert abs(metrics.by_app["terminal"] - 1.0) < 1e-9


def test_score_batch_avg_steps() -> None:
    scorer = OSWorldScorer()
    tasks = _make_tasks()
    results = [
        OSWorldResult("t1", "chrome", True, 4),
        OSWorldResult("t2", "vscode", False, 6),
    ]
    metrics = scorer.score_batch(results, tasks)
    assert abs(metrics.avg_steps - 5.0) < 1e-9


def test_score_batch_empty_results() -> None:
    scorer = OSWorldScorer()
    metrics = scorer.score_batch([], {})
    assert metrics.n_tasks == 0
    assert metrics.n_completed == 0
    assert metrics.completion_rate == 0.0


# ---------------------------------------------------------------------------
# format_report()
# ---------------------------------------------------------------------------


def test_format_report_non_empty() -> None:
    scorer = OSWorldScorer()
    tasks = _make_tasks()
    results = [
        OSWorldResult("t1", "chrome", True, 2),
        OSWorldResult("t2", "vscode", False, 1),
    ]
    metrics = scorer.score_batch(results, tasks)
    report = scorer.format_report(metrics)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_report_contains_key_info() -> None:
    scorer = OSWorldScorer()
    tasks = _make_tasks()
    results = [OSWorldResult("t1", "chrome", True, 3)]
    metrics = scorer.score_batch(results, tasks)
    report = scorer.format_report(metrics)
    assert "chrome" in report
    assert "100" in report or "1.0" in report or "100.0" in report


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_osworld_task_registry_has_six_entries() -> None:
    """OSWORLD_TASK_REGISTRY must contain exactly 6 pre-registered tasks."""
    assert len(OSWORLD_TASK_REGISTRY) == 6


def test_osworld_task_registry_covers_three_apps() -> None:
    apps = {task.app for task in OSWORLD_TASK_REGISTRY.values()}
    assert "chrome" in apps
    assert "vscode" in apps
    assert "terminal" in apps


def test_osworld_task_registry_covers_all_difficulties() -> None:
    difficulties = {task.difficulty for task in OSWORLD_TASK_REGISTRY.values()}
    assert {"easy", "medium", "hard"} == difficulties


def test_benchmark_registry_contains_osworld() -> None:
    """BENCHMARK_REGISTRY in src.eval must have 'osworld' key after import."""
    # osworld_scorer registers itself on import; trigger it.
    import src.eval.osworld_scorer  # noqa: F401, PLC0415
    from src.eval import BENCHMARK_REGISTRY  # noqa: PLC0415

    assert "osworld" in BENCHMARK_REGISTRY
