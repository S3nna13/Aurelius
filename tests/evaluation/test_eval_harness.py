"""Tests for src/evaluation/eval_harness.py — ≥28 test cases."""

import dataclasses

import pytest

from src.evaluation.eval_harness import (
    EVAL_HARNESS_REGISTRY,
    EvalHarness,
    EvalResult,
    EvalTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(task_id="t1", prompt="hello", reference="world", category="general"):
    return EvalTask(task_id=task_id, prompt=prompt, reference=reference, category=category)


def _identity_predict(prompt: str) -> str:
    """Returns the prompt unchanged — useful for exact-match tests."""
    return prompt


def _always_wrong(prompt: str) -> str:
    return "__never_matches__"


def _always_right_predict(reference: str):
    """Returns a predict function that always returns `reference`."""

    def predict(prompt: str) -> str:
        return reference

    return predict


# ---------------------------------------------------------------------------
# EvalTask dataclass
# ---------------------------------------------------------------------------


class TestEvalTask:
    def test_task_creation(self):
        task = EvalTask(task_id="t1", prompt="What?", reference="42")
        assert task.task_id == "t1"
        assert task.prompt == "What?"
        assert task.reference == "42"

    def test_task_default_category(self):
        task = EvalTask(task_id="t2", prompt="p", reference="r")
        assert task.category == "general"

    def test_task_custom_category(self):
        task = EvalTask(task_id="t3", prompt="p", reference="r", category="math")
        assert task.category == "math"

    def test_task_default_metadata(self):
        task = EvalTask(task_id="t4", prompt="p", reference="r")
        assert task.metadata == {}

    def test_task_custom_metadata(self):
        task = EvalTask(task_id="t5", prompt="p", reference="r", metadata={"k": "v"})
        assert task.metadata["k"] == "v"

    def test_task_frozen_immutable(self):
        task = EvalTask(task_id="t6", prompt="p", reference="r")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            task.task_id = "new"  # type: ignore[misc]

    def test_task_frozen_reference_immutable(self):
        task = EvalTask(task_id="t7", prompt="p", reference="r")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            task.reference = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_result_frozen(self):
        result = EvalResult(
            task_id="t1", predicted="a", reference="a", correct=True, score=1.0, latency_ms=5.0
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.score = 0.0  # type: ignore[misc]

    def test_result_default_latency(self):
        result = EvalResult(task_id="t1", predicted="a", reference="a", correct=True, score=1.0)
        assert result.latency_ms == 0.0


# ---------------------------------------------------------------------------
# EvalHarness.run — exact match (score_fn=None)
# ---------------------------------------------------------------------------


class TestEvalHarnessRunExactMatch:
    def test_run_exact_match_correct(self):
        task = EvalTask(task_id="t1", prompt="42", reference="42")
        harness = EvalHarness([task])
        results = harness.run(_identity_predict)
        assert len(results) == 1
        assert results[0].correct is True
        assert results[0].score == 1.0

    def test_run_exact_match_wrong(self):
        task = EvalTask(task_id="t1", prompt="hello", reference="world")
        harness = EvalHarness([task])
        results = harness.run(_identity_predict)
        assert results[0].correct is False
        assert results[0].score == 0.0

    def test_run_exact_match_strips_whitespace(self):
        task = EvalTask(task_id="t1", prompt=" ans ", reference="ans")
        harness = EvalHarness([task])
        results = harness.run(_identity_predict)
        assert results[0].score == 1.0
        assert results[0].correct is True

    def test_run_preserves_task_id(self):
        task = EvalTask(task_id="my-task-99", prompt="42", reference="42")
        harness = EvalHarness([task])
        results = harness.run(_identity_predict)
        assert results[0].task_id == "my-task-99"

    def test_run_preserves_reference(self):
        task = EvalTask(task_id="t1", prompt="p", reference="correct_answer")
        harness = EvalHarness([task])
        results = harness.run(_always_wrong)
        assert results[0].reference == "correct_answer"

    def test_run_records_predicted(self):
        task = EvalTask(task_id="t1", prompt="x", reference="y")
        harness = EvalHarness([task])
        results = harness.run(_identity_predict)
        assert results[0].predicted == "x"

    def test_run_latency_non_negative(self):
        task = EvalTask(task_id="t1", prompt="p", reference="r")
        harness = EvalHarness([task])
        results = harness.run(_always_wrong)
        assert results[0].latency_ms >= 0.0

    def test_run_multiple_tasks(self):
        tasks = [
            EvalTask(task_id="t1", prompt="42", reference="42"),
            EvalTask(task_id="t2", prompt="hello", reference="world"),
        ]
        harness = EvalHarness(tasks)
        results = harness.run(_identity_predict)
        assert len(results) == 2
        assert results[0].correct is True
        assert results[1].correct is False

    def test_run_correct_flag_at_threshold_05(self):
        """score_fn returning exactly 0.5 should mark correct=True."""
        task = EvalTask(task_id="t1", prompt="p", reference="r")
        harness = EvalHarness([task])
        results = harness.run(lambda p: p, score_fn=lambda pred, ref: 0.5)
        assert results[0].correct is True

    def test_run_correct_flag_below_threshold(self):
        task = EvalTask(task_id="t1", prompt="p", reference="r")
        harness = EvalHarness([task])
        results = harness.run(lambda p: p, score_fn=lambda pred, ref: 0.49)
        assert results[0].correct is False

    def test_run_correct_flag_above_threshold(self):
        task = EvalTask(task_id="t1", prompt="p", reference="r")
        harness = EvalHarness([task])
        results = harness.run(lambda p: p, score_fn=lambda pred, ref: 0.8)
        assert results[0].correct is True


# ---------------------------------------------------------------------------
# EvalHarness.run — custom score_fn
# ---------------------------------------------------------------------------


class TestEvalHarnessCustomScoreFn:
    def test_custom_score_fn_used(self):
        task = EvalTask(task_id="t1", prompt="p", reference="r")
        harness = EvalHarness([task])
        results = harness.run(_always_wrong, score_fn=lambda pred, ref: 0.75)
        assert results[0].score == 0.75

    def test_custom_score_fn_partial_credit(self):
        task = EvalTask(task_id="t1", prompt="p", reference="ref")
        harness = EvalHarness([task])

        # Score based on character overlap fraction
        def overlap(pred, ref):
            return len(set(pred) & set(ref)) / max(len(set(ref)), 1)

        results = harness.run(lambda p: "r", score_fn=overlap)
        assert 0.0 <= results[0].score <= 1.0


# ---------------------------------------------------------------------------
# EvalHarness.summary
# ---------------------------------------------------------------------------


class TestEvalHarnessSummary:
    def test_summary_total(self):
        tasks = [_make_task(f"t{i}") for i in range(5)]
        harness = EvalHarness(tasks)
        results = harness.run(_always_wrong)
        s = harness.summary(results)
        assert s["total"] == 5

    def test_summary_correct_count(self):
        tasks = [
            EvalTask(task_id="t1", prompt="42", reference="42"),
            EvalTask(task_id="t2", prompt="x", reference="y"),
        ]
        harness = EvalHarness(tasks)
        results = harness.run(_identity_predict)
        s = harness.summary(results)
        assert s["correct"] == 1

    def test_summary_accuracy(self):
        tasks = [
            EvalTask(task_id="t1", prompt="42", reference="42"),
            EvalTask(task_id="t2", prompt="x", reference="y"),
        ]
        harness = EvalHarness(tasks)
        results = harness.run(_identity_predict)
        s = harness.summary(results)
        assert s["accuracy"] == pytest.approx(0.5)

    def test_summary_mean_score_all_correct(self):
        tasks = [
            EvalTask(task_id="t1", prompt="a", reference="a"),
            EvalTask(task_id="t2", prompt="b", reference="b"),
        ]
        harness = EvalHarness(tasks)
        results = harness.run(_identity_predict)
        s = harness.summary(results)
        assert s["mean_score"] == pytest.approx(1.0)

    def test_summary_mean_score_all_wrong(self):
        tasks = [_make_task(f"t{i}") for i in range(3)]
        harness = EvalHarness(tasks)
        results = harness.run(_always_wrong)
        s = harness.summary(results)
        assert s["mean_score"] == pytest.approx(0.0)

    def test_summary_keys_present(self):
        harness = EvalHarness([_make_task()])
        results = harness.run(_always_wrong)
        s = harness.summary(results)
        assert set(s.keys()) == {"total", "correct", "accuracy", "mean_score"}


# ---------------------------------------------------------------------------
# EvalHarness.run — empty tasks
# ---------------------------------------------------------------------------


class TestEvalHarnessEmpty:
    def test_empty_tasks_run(self):
        harness = EvalHarness([])
        results = harness.run(_identity_predict)
        assert results == []

    def test_empty_tasks_summary(self):
        harness = EvalHarness([])
        s = harness.summary([])
        assert s["total"] == 0
        assert s["correct"] == 0
        assert s["accuracy"] == 0.0
        assert s["mean_score"] == 0.0


# ---------------------------------------------------------------------------
# EvalHarness.filter_by_category
# ---------------------------------------------------------------------------


class TestEvalHarnessFilterByCategory:
    def test_filter_by_category_returns_subset(self):
        tasks = [
            EvalTask(task_id="t1", prompt="p", reference="r", category="math"),
            EvalTask(task_id="t2", prompt="p", reference="r", category="science"),
            EvalTask(task_id="t3", prompt="p", reference="r", category="math"),
        ]
        harness = EvalHarness(tasks)
        results = harness.run(_always_wrong)
        math_results = harness.filter_by_category(results, "math")
        assert len(math_results) == 2
        assert all(r.task_id in {"t1", "t3"} for r in math_results)

    def test_filter_by_category_no_match(self):
        tasks = [EvalTask(task_id="t1", prompt="p", reference="r", category="math")]
        harness = EvalHarness(tasks)
        results = harness.run(_always_wrong)
        filtered = harness.filter_by_category(results, "physics")
        assert filtered == []

    def test_filter_by_category_all_match(self):
        tasks = [
            EvalTask(task_id="t1", prompt="p", reference="r", category="lang"),
            EvalTask(task_id="t2", prompt="p", reference="r", category="lang"),
        ]
        harness = EvalHarness(tasks)
        results = harness.run(_always_wrong)
        filtered = harness.filter_by_category(results, "lang")
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in EVAL_HARNESS_REGISTRY

    def test_registry_default_is_eval_harness(self):
        assert EVAL_HARNESS_REGISTRY["default"] is EvalHarness
