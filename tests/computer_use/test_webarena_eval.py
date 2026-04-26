"""Unit tests for src/computer_use/webarena_eval.py.

Covers WebTask, TaskResult, SuccessEvaluator, WebArenaHarness, and the
WEBARENA_DEFAULT_TASKS / WEBARENA_HARNESS_REGISTRY module-level constants.
"""

from __future__ import annotations

import pytest

from src.computer_use.browser_driver import StubBrowserDriver
from src.computer_use.webarena_eval import (
    WEBARENA_DEFAULT_TASKS,
    WEBARENA_HARNESS_REGISTRY,
    SuccessEvaluator,
    TaskResult,
    WebArenaError,
    WebArenaHarness,
    WebTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> WebTask:
    """Return a minimal valid WebTask with optional field overrides."""
    defaults: dict = {
        "task_id": "test_task",
        "description": "A test task",
        "start_url": "https://example.com",
        "success_criteria": ["example.com/success"],
        "max_steps": 30,
        "tags": [],
    }
    defaults.update(overrides)
    return WebTask(**defaults)


def _make_result(**overrides) -> TaskResult:
    """Return a minimal valid TaskResult with optional field overrides."""
    defaults: dict = {
        "task_id": "test_task",
        "success": False,
        "steps_taken": 0,
        "final_url": None,
        "final_state_html": None,
    }
    defaults.update(overrides)
    return TaskResult(**defaults)


# ---------------------------------------------------------------------------
# WebTask dataclass
# ---------------------------------------------------------------------------


class TestWebTask:
    def test_task_id_field(self):
        task = _make_task(task_id="navigate_to_url")
        assert task.task_id == "navigate_to_url"

    def test_description_field(self):
        task = _make_task(description="Fill out a form.")
        assert task.description == "Fill out a form."

    def test_max_steps_default(self):
        task = WebTask(
            task_id="t1",
            description="d",
            start_url="https://example.com",
            success_criteria=["ok"],
        )
        assert task.max_steps == 30

    def test_max_steps_override(self):
        task = _make_task(max_steps=10)
        assert task.max_steps == 10

    def test_tags_default_empty(self):
        task = WebTask(
            task_id="t2",
            description="d",
            start_url="https://example.com",
            success_criteria=["ok"],
        )
        assert task.tags == []

    def test_tags_field(self):
        task = _make_task(tags=["navigation", "form"])
        assert "navigation" in task.tags


# ---------------------------------------------------------------------------
# TaskResult dataclass
# ---------------------------------------------------------------------------


class TestTaskResult:
    def test_success_field_true(self):
        result = _make_result(success=True)
        assert result.success is True

    def test_success_field_false(self):
        result = _make_result(success=False)
        assert result.success is False

    def test_error_default_none(self):
        result = _make_result()
        assert result.error is None

    def test_trajectory_id_default_none(self):
        result = _make_result()
        assert result.trajectory_id is None

    def test_steps_taken_field(self):
        result = _make_result(steps_taken=7)
        assert result.steps_taken == 7


# ---------------------------------------------------------------------------
# SuccessEvaluator
# ---------------------------------------------------------------------------


class TestSuccessEvaluator:
    def setup_method(self):
        self.evaluator = SuccessEvaluator()
        self.task = _make_task(success_criteria=["example.com/success", "thank-you"])

    # evaluate_url_match
    def test_url_match_criterion_in_url(self):
        result = _make_result(final_url="https://example.com/success")
        assert self.evaluator.evaluate_url_match(result, self.task) is True

    def test_url_match_criterion_not_in_url(self):
        result = _make_result(final_url="https://example.com/other")
        assert self.evaluator.evaluate_url_match(result, self.task) is False

    def test_url_match_none_url(self):
        result = _make_result(final_url=None)
        assert self.evaluator.evaluate_url_match(result, self.task) is False

    def test_url_match_second_criterion_matches(self):
        result = _make_result(final_url="https://example.com/thank-you")
        assert self.evaluator.evaluate_url_match(result, self.task) is True

    # evaluate_content_match
    def test_content_match_criterion_in_html(self):
        result = _make_result(final_state_html="<html><body>thank-you</body></html>")
        assert self.evaluator.evaluate_content_match(result, self.task) is True

    def test_content_match_criterion_not_in_html(self):
        result = _make_result(final_state_html="<html><body>nothing here</body></html>")
        assert self.evaluator.evaluate_content_match(result, self.task) is False

    def test_content_match_none_html(self):
        result = _make_result(final_state_html=None)
        assert self.evaluator.evaluate_content_match(result, self.task) is False

    # evaluate (combined)
    def test_evaluate_returns_tuple(self):
        result = _make_result(final_url="https://example.com/other")
        outcome = self.evaluator.evaluate(result, self.task)
        assert isinstance(outcome, tuple)
        assert len(outcome) == 2

    def test_evaluate_url_match_wins(self):
        result = _make_result(final_url="https://example.com/success")
        success, reason = self.evaluator.evaluate(result, self.task)
        assert success is True
        assert "url_match" in reason

    def test_evaluate_content_match_fallback(self):
        result = _make_result(
            final_url="https://example.com/other",
            final_state_html="<html>thank-you</html>",
        )
        success, reason = self.evaluator.evaluate(result, self.task)
        assert success is True
        assert "content_match" in reason

    def test_evaluate_no_match(self):
        result = _make_result(
            final_url="https://example.com/other",
            final_state_html="<html>nothing</html>",
        )
        success, reason = self.evaluator.evaluate(result, self.task)
        assert success is False
        assert "no_match" in reason


# ---------------------------------------------------------------------------
# WebArenaHarness — registry
# ---------------------------------------------------------------------------


class TestWebArenaHarnessRegistry:
    def test_register_task_accessible(self):
        harness = WebArenaHarness()
        task = _make_task(task_id="unique_task_abc")
        harness.register_task(task)
        assert "unique_task_abc" in harness.tasks
        assert harness.tasks["unique_task_abc"] is task

    def test_register_duplicate_raises_error(self):
        harness = WebArenaHarness()
        task = _make_task(task_id="dup_task")
        harness.register_task(task)
        with pytest.raises(WebArenaError):
            harness.register_task(_make_task(task_id="dup_task"))


# ---------------------------------------------------------------------------
# WebArenaHarness — run_task
# ---------------------------------------------------------------------------


class TestWebArenaHarnessRunTask:
    def test_run_task_none_agent_zero_steps(self):
        """Agent that returns None immediately → steps_taken == 0."""
        harness = WebArenaHarness()
        task = _make_task(
            task_id="early_stop",
            start_url="https://example.com",
            success_criteria=["IMPOSSIBLE_TOKEN_12345"],
        )
        harness.register_task(task)
        driver = StubBrowserDriver()

        def none_agent(state, task):
            return None

        result = harness.run_task("early_stop", driver, none_agent)
        assert isinstance(result, TaskResult)
        assert result.steps_taken == 0
        assert result.task_id == "early_stop"

    def test_run_task_unknown_task_id_returns_error(self):
        harness = WebArenaHarness()
        driver = StubBrowserDriver()
        result = harness.run_task("nonexistent", driver, lambda s, t: None)
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# WebArenaHarness — score
# ---------------------------------------------------------------------------


class TestWebArenaHarnessScore:
    def _build_tasks_and_results(self):
        tasks = {
            "t1": _make_task(task_id="t1", tags=["nav"]),
            "t2": _make_task(task_id="t2", tags=["nav"]),
            "t3": _make_task(task_id="t3", tags=["form"]),
        }
        results = [
            _make_result(task_id="t1", success=True),
            _make_result(task_id="t2", success=True),
            _make_result(task_id="t3", success=False),
        ]
        return tasks, results

    def test_success_rate_two_of_three(self):
        tasks, results = self._build_tasks_and_results()
        report = WebArenaHarness.score(results, tasks)
        assert report["n_tasks"] == 3
        assert report["n_success"] == 2
        assert abs(report["success_rate"] - 2 / 3) < 1e-9

    def test_by_tag_nav_group(self):
        tasks, results = self._build_tasks_and_results()
        report = WebArenaHarness.score(results, tasks)
        # "nav" tag: 2 tasks, both success → rate = 1.0
        assert "nav" in report["by_tag"]
        assert abs(report["by_tag"]["nav"] - 1.0) < 1e-9

    def test_by_tag_form_group(self):
        tasks, results = self._build_tasks_and_results()
        report = WebArenaHarness.score(results, tasks)
        # "form" tag: 1 task, 0 success → rate = 0.0
        assert "form" in report["by_tag"]
        assert abs(report["by_tag"]["form"] - 0.0) < 1e-9

    def test_score_empty_results(self):
        report = WebArenaHarness.score([], {})
        assert report["n_tasks"] == 0
        assert report["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_webarena_default_tasks_has_three(self):
        assert len(WEBARENA_DEFAULT_TASKS) == 3

    def test_webarena_default_tasks_are_web_tasks(self):
        for task in WEBARENA_DEFAULT_TASKS:
            assert isinstance(task, WebTask)

    def test_webarena_default_task_ids(self):
        ids = {t.task_id for t in WEBARENA_DEFAULT_TASKS}
        assert "navigate_to_url" in ids
        assert "find_element" in ids
        assert "fill_form" in ids

    def test_webarena_harness_registry_is_dict(self):
        assert isinstance(WEBARENA_HARNESS_REGISTRY, dict)
