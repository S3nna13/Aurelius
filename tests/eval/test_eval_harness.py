"""Tests for src/eval/eval_harness.py"""

import pytest
from src.eval.eval_harness import EvalTask, EvalMetrics, EvalHarness


# ---------------------------------------------------------------------------
# EvalTask
# ---------------------------------------------------------------------------

def test_eval_task_fields():
    t = EvalTask(task_id="t1", prompt="What is 2+2?", reference="4")
    assert t.task_id == "t1"
    assert t.prompt == "What is 2+2?"
    assert t.reference == "4"


def test_eval_task_default_metadata():
    t = EvalTask(task_id="t1", prompt="p", reference="r")
    assert t.metadata == {}


def test_eval_task_custom_metadata():
    t = EvalTask(task_id="t2", prompt="p", reference="r", metadata={"source": "test"})
    assert t.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# EvalMetrics
# ---------------------------------------------------------------------------

def test_eval_metrics_fields():
    m = EvalMetrics(
        task_id="t1",
        prediction="hello",
        exact_match=True,
        token_f1=1.0,
        char_overlap=1.0,
        passed=True,
    )
    assert m.task_id == "t1"
    assert m.prediction == "hello"
    assert m.exact_match is True
    assert m.token_f1 == 1.0
    assert m.char_overlap == 1.0
    assert m.passed is True


# ---------------------------------------------------------------------------
# EvalHarness construction
# ---------------------------------------------------------------------------

def test_harness_init_no_tasks():
    h = EvalHarness()
    assert h._tasks == []


def test_harness_init_with_tasks():
    tasks = [EvalTask("t1", "p", "r"), EvalTask("t2", "p2", "r2")]
    h = EvalHarness(tasks=tasks)
    assert len(h._tasks) == 2


def test_harness_add_task():
    h = EvalHarness()
    t = EvalTask("t1", "p", "r")
    h.add_task(t)
    assert len(h._tasks) == 1
    assert h._tasks[0].task_id == "t1"


def test_harness_load_tasks_replaces():
    h = EvalHarness(tasks=[EvalTask("old", "p", "r")])
    new_tasks = [EvalTask("new1", "p", "r"), EvalTask("new2", "p", "r")]
    h.load_tasks(new_tasks)
    assert len(h._tasks) == 2
    assert h._tasks[0].task_id == "new1"


# ---------------------------------------------------------------------------
# _token_f1
# ---------------------------------------------------------------------------

def test_token_f1_exact_match():
    h = EvalHarness()
    assert h._token_f1("hello world", "hello world") == 1.0


def test_token_f1_no_overlap():
    h = EvalHarness()
    assert h._token_f1("foo bar", "baz qux") == 0.0


def test_token_f1_partial_overlap():
    h = EvalHarness()
    score = h._token_f1("the cat sat", "the dog sat")
    assert 0.0 < score < 1.0


def test_token_f1_empty_pred():
    h = EvalHarness()
    assert h._token_f1("", "hello") == 0.0


def test_token_f1_empty_ref():
    h = EvalHarness()
    assert h._token_f1("hello", "") == 0.0


def test_token_f1_both_empty():
    h = EvalHarness()
    assert h._token_f1("", "") == 1.0


def test_token_f1_range():
    h = EvalHarness()
    score = h._token_f1("the quick brown fox", "the lazy brown dog")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# evaluate_one
# ---------------------------------------------------------------------------

def test_evaluate_one_exact_match():
    h = EvalHarness()
    task = EvalTask("t1", "p", "answer")
    m = h.evaluate_one(task, "answer")
    assert m.exact_match is True
    assert m.passed is True
    assert m.token_f1 == 1.0


def test_evaluate_one_mismatch():
    h = EvalHarness()
    task = EvalTask("t1", "p", "answer")
    m = h.evaluate_one(task, "wrong")
    assert m.exact_match is False
    assert m.passed is False


def test_evaluate_one_task_id_preserved():
    h = EvalHarness()
    task = EvalTask("my-task", "p", "r")
    m = h.evaluate_one(task, "r")
    assert m.task_id == "my-task"


def test_evaluate_one_prediction_preserved():
    h = EvalHarness()
    task = EvalTask("t1", "p", "r")
    m = h.evaluate_one(task, "my prediction")
    assert m.prediction == "my prediction"


def test_evaluate_one_char_overlap_exact():
    h = EvalHarness()
    task = EvalTask("t1", "p", "hello")
    m = h.evaluate_one(task, "hello")
    assert m.char_overlap == 1.0


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

def test_evaluate_all_returns_list():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    results = h.evaluate_all(["a", "b"])
    assert isinstance(results, list)
    assert len(results) == 2


def test_evaluate_all_correct_answers():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    results = h.evaluate_all(["a", "b"])
    assert all(r.exact_match for r in results)


def test_evaluate_all_wrong_answers():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    results = h.evaluate_all(["x", "y"])
    assert not any(r.exact_match for r in results)


def test_evaluate_all_mixed():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    results = h.evaluate_all(["a", "wrong"])
    assert results[0].exact_match is True
    assert results[1].exact_match is False


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    h = EvalHarness()
    agg = h.aggregate([])
    assert agg["n_tasks"] == 0
    assert agg["exact_match_rate"] == 0.0
    assert agg["mean_token_f1"] == 0.0
    assert agg["mean_char_overlap"] == 0.0
    assert agg["pass_rate"] == 0.0


def test_aggregate_all_correct():
    tasks = [EvalTask(f"t{i}", "p", "answer") for i in range(4)]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["answer"] * 4)
    agg = h.aggregate(metrics)
    assert agg["exact_match_rate"] == 1.0
    assert agg["pass_rate"] == 1.0
    assert agg["n_tasks"] == 4


def test_aggregate_all_wrong():
    tasks = [EvalTask(f"t{i}", "p", "answer") for i in range(3)]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["wrong"] * 3)
    agg = h.aggregate(metrics)
    assert agg["exact_match_rate"] == 0.0
    assert agg["pass_rate"] == 0.0


def test_aggregate_keys_present():
    h = EvalHarness(tasks=[EvalTask("t1", "p", "r")])
    metrics = h.evaluate_all(["r"])
    agg = h.aggregate(metrics)
    assert set(agg.keys()) == {
        "exact_match_rate", "mean_token_f1", "mean_char_overlap", "pass_rate", "n_tasks"
    }


def test_aggregate_partial():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["a", "wrong"])
    agg = h.aggregate(metrics)
    assert agg["exact_match_rate"] == 0.5
    assert agg["pass_rate"] == 0.5
    assert agg["n_tasks"] == 2


# ---------------------------------------------------------------------------
# filter_failed
# ---------------------------------------------------------------------------

def test_filter_failed_none_fail():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["a", "b"])
    assert h.filter_failed(metrics) == []


def test_filter_failed_all_fail():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["x", "y"])
    failed = h.filter_failed(metrics)
    assert len(failed) == 2


def test_filter_failed_partial():
    tasks = [EvalTask("t1", "p", "a"), EvalTask("t2", "p", "b")]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["a", "wrong"])
    failed = h.filter_failed(metrics)
    assert len(failed) == 1
    assert failed[0].task_id == "t2"


def test_filter_failed_returns_eval_metrics():
    tasks = [EvalTask("t1", "p", "a")]
    h = EvalHarness(tasks=tasks)
    metrics = h.evaluate_all(["wrong"])
    failed = h.filter_failed(metrics)
    assert all(isinstance(m, EvalMetrics) for m in failed)
