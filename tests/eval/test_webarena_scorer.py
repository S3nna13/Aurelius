"""Tests for src.eval.webarena_scorer — WebArena task-completion scorer.

WebArena benchmark (Zhou et al. 2307.13854, Apache-2.0), GitHub Actions (MIT spec),
clean-room implementation.
"""

from __future__ import annotations

from src.eval.webarena_scorer import (
    WEBARENA_SCORER_REGISTRY,
    WEBARENA_TASK_REGISTRY,
    WebArenaMetrics,
    WebArenaResult,
    WebArenaScorer,
    WebArenaTask,
)

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_webarena_task_default_difficulty() -> None:
    """WebArenaTask defaults difficulty to 'medium'."""
    task = WebArenaTask(
        task_id="t1",
        category="shopping",
        intent="Buy something",
        start_url="http://example.com",
        reference_answers=["bought"],
    )
    assert task.difficulty == "medium"


def test_webarena_task_explicit_difficulty() -> None:
    """WebArenaTask accepts explicit difficulty values."""
    for diff in ("easy", "medium", "hard"):
        task = WebArenaTask(
            task_id="t1",
            category="shopping",
            intent="Buy something",
            start_url="http://example.com",
            reference_answers=["bought"],
            difficulty=diff,  # type: ignore[arg-type]
        )
        assert task.difficulty == diff


def test_webarena_result_default_success() -> None:
    """WebArenaResult defaults success to False."""
    result = WebArenaResult(
        task_id="t1",
        predicted_answer="some answer",
        steps_taken=5,
    )
    assert result.success is False


def test_webarena_result_default_trajectory_len() -> None:
    """WebArenaResult defaults trajectory_len to 0."""
    result = WebArenaResult(
        task_id="t1",
        predicted_answer="some answer",
        steps_taken=5,
    )
    assert result.trajectory_len == 0


# ---------------------------------------------------------------------------
# score_result
# ---------------------------------------------------------------------------


def test_score_result_match_returns_true() -> None:
    """score_result returns True when predicted contains a reference answer."""
    scorer = WebArenaScorer()
    task = WebArenaTask(
        task_id="t1",
        category="shopping",
        intent="Add item to cart",
        start_url="http://shop.example/",
        reference_answers=["added to cart"],
    )
    result = WebArenaResult(
        task_id="t1", predicted_answer="Item added to cart successfully.", steps_taken=3
    )
    assert scorer.score_result(result, task) is True


def test_score_result_no_match_returns_false() -> None:
    """score_result returns False when predicted does not contain any reference answer."""
    scorer = WebArenaScorer()
    task = WebArenaTask(
        task_id="t2",
        category="shopping",
        intent="Add item to cart",
        start_url="http://shop.example/",
        reference_answers=["added to cart"],
    )
    result = WebArenaResult(task_id="t2", predicted_answer="Nothing happened.", steps_taken=2)
    assert scorer.score_result(result, task) is False


def test_score_result_none_predicted_returns_false() -> None:
    """score_result returns False when predicted_answer is None."""
    scorer = WebArenaScorer()
    task = WebArenaTask(
        task_id="t3",
        category="reddit",
        intent="Post comment",
        start_url="http://reddit.example/",
        reference_answers=["comment posted"],
    )
    result = WebArenaResult(task_id="t3", predicted_answer=None, steps_taken=0)
    assert scorer.score_result(result, task) is False


def test_score_result_case_insensitive() -> None:
    """score_result performs case-insensitive substring matching."""
    scorer = WebArenaScorer()
    task = WebArenaTask(
        task_id="t4",
        category="gitlab",
        intent="Open issue",
        start_url="http://gitlab.example/",
        reference_answers=["Issue Created"],
    )
    result = WebArenaResult(
        task_id="t4", predicted_answer="issue created successfully", steps_taken=4
    )
    assert scorer.score_result(result, task) is True


def test_score_result_partial_match_in_reference() -> None:
    """score_result matches when predicted contains a partial reference as substring."""
    scorer = WebArenaScorer()
    task = WebArenaTask(
        task_id="t5",
        category="shopping",
        intent="Find item",
        start_url="http://shop.example/",
        reference_answers=["ThinkPad X1", "MacBook Pro"],
    )
    result = WebArenaResult(
        task_id="t5", predicted_answer="The best laptop is MacBook Pro Carbon.", steps_taken=6
    )
    assert scorer.score_result(result, task) is True


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


def test_score_batch_success_rate() -> None:
    """score_batch computes correct success_rate (2/3 ≈ 0.667)."""
    scorer = WebArenaScorer()
    tasks = {
        "a": WebArenaTask("a", "shopping", "Buy", "http://s.example/", ["bought"], "easy"),
        "b": WebArenaTask("b", "reddit", "Post", "http://r.example/", ["posted"], "medium"),
        "c": WebArenaTask("c", "gitlab", "Issue", "http://g.example/", ["issue #"], "hard"),
    }
    results = [
        WebArenaResult("a", "Item bought.", 3),
        WebArenaResult("b", "Comment posted here.", 5),
        WebArenaResult("c", "No action taken.", 1),
    ]
    metrics = scorer.score_batch(results, tasks)
    assert metrics.n_tasks == 3
    assert metrics.n_success == 2
    assert abs(metrics.success_rate - 2 / 3) < 0.001


def test_score_batch_missing_task_skipped() -> None:
    """score_batch skips results whose task_id is not in tasks dict."""
    scorer = WebArenaScorer()
    tasks = {
        "real": WebArenaTask("real", "shopping", "Buy", "http://s.example/", ["bought"], "easy"),
    }
    results = [
        WebArenaResult("real", "bought it", 2),
        WebArenaResult("ghost", "this task is missing", 1),  # not in tasks
    ]
    metrics = scorer.score_batch(results, tasks)
    assert metrics.n_tasks == 1  # ghost was skipped
    assert metrics.n_success == 1


def test_score_batch_by_category_is_dict() -> None:
    """WebArenaMetrics.by_category is a dict."""
    scorer = WebArenaScorer()
    tasks = {
        "a": WebArenaTask("a", "shopping", "Buy", "http://s.example/", ["ok"], "easy"),
    }
    results = [WebArenaResult("a", "ok done", 1)]
    metrics = scorer.score_batch(results, tasks)
    assert isinstance(metrics.by_category, dict)


def test_score_batch_by_difficulty_is_dict() -> None:
    """WebArenaMetrics.by_difficulty is a dict."""
    scorer = WebArenaScorer()
    tasks = {
        "a": WebArenaTask("a", "shopping", "Buy", "http://s.example/", ["ok"], "easy"),
    }
    results = [WebArenaResult("a", "ok done", 1)]
    metrics = scorer.score_batch(results, tasks)
    assert isinstance(metrics.by_difficulty, dict)


def test_score_batch_avg_steps() -> None:
    """score_batch correctly computes average steps."""
    scorer = WebArenaScorer()
    tasks = {
        "a": WebArenaTask("a", "shopping", "Buy", "http://s.example/", ["ok"], "easy"),
        "b": WebArenaTask("b", "reddit", "Post", "http://r.example/", ["ok"], "medium"),
    }
    results = [
        WebArenaResult("a", "ok done", 4),
        WebArenaResult("b", "ok done", 6),
    ]
    metrics = scorer.score_batch(results, tasks)
    assert abs(metrics.avg_steps - 5.0) < 0.001


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_non_empty() -> None:
    """format_report returns a non-empty string."""
    scorer = WebArenaScorer()
    metrics = WebArenaMetrics(
        n_tasks=5,
        n_success=3,
        success_rate=0.6,
        by_category={"shopping": 0.5, "reddit": 1.0},
        by_difficulty={"easy": 1.0, "hard": 0.0},
        avg_steps=4.2,
    )
    report = scorer.format_report(metrics)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_report_contains_key_fields() -> None:
    """format_report includes n_tasks, n_success, and success_rate info."""
    scorer = WebArenaScorer()
    metrics = WebArenaMetrics(
        n_tasks=10,
        n_success=7,
        success_rate=0.7,
        by_category={"shopping": 0.7},
        by_difficulty={"medium": 0.7},
        avg_steps=3.0,
    )
    report = scorer.format_report(metrics)
    assert "10" in report
    assert "7" in report
    assert "70.0%" in report or "70%" in report


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------


def test_webarena_task_registry_has_five_entries() -> None:
    """WEBARENA_TASK_REGISTRY contains exactly 5 stub tasks."""
    assert len(WEBARENA_TASK_REGISTRY) == 5


def test_webarena_task_registry_covers_all_categories() -> None:
    """WEBARENA_TASK_REGISTRY covers shopping, reddit, and gitlab categories."""
    categories = {t.category for t in WEBARENA_TASK_REGISTRY.values()}
    assert "shopping" in categories
    assert "reddit" in categories
    assert "gitlab" in categories


def test_webarena_task_registry_covers_all_difficulties() -> None:
    """WEBARENA_TASK_REGISTRY covers easy, medium, and hard difficulties."""
    difficulties = {t.difficulty for t in WEBARENA_TASK_REGISTRY.values()}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties


def test_webarena_scorer_registry_has_default() -> None:
    """WEBARENA_SCORER_REGISTRY contains the 'default' key."""
    assert "default" in WEBARENA_SCORER_REGISTRY
    assert WEBARENA_SCORER_REGISTRY["default"] is WebArenaScorer


def test_benchmark_registry_contains_webarena() -> None:
    """BENCHMARK_REGISTRY contains 'webarena' after importing src.eval."""
    import src.eval as eval_module

    registry = eval_module.BENCHMARK_REGISTRY
    assert "webarena" in registry
