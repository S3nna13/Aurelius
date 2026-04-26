"""Tests for commonsense reasoning evaluation module."""

import pytest
import torch

from src.eval.commonsense_eval import (
    CommonsenseEvaluator,
    CommonsenseTask,
    compute_accuracy,
    generate_arc_tasks,
    generate_hellaswag_tasks,
    generate_winogrande_tasks,
    length_normalize_scores,
    score_task_by_likelihood,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def byte_encode(text: str) -> list[int]:
    """Simple byte-level encoder that maps each character to its ASCII/Unicode byte value mod 256."""  # noqa: E501
    return [b % 256 for b in text.encode("utf-8")]


@pytest.fixture(scope="module")
def evaluator(small_model):
    return CommonsenseEvaluator(small_model, byte_encode)


# ---------------------------------------------------------------------------
# CommonsenseTask dataclass tests
# ---------------------------------------------------------------------------


def test_commonsense_task_fields():
    """CommonsenseTask should store all fields correctly."""
    task = CommonsenseTask(
        task_type="hellaswag",
        context="A person is cooking.",
        choices=["stirs the pot", "drives away", "reads a book", "waters plants"],
        correct_idx=0,
    )
    assert task.task_type == "hellaswag"
    assert task.context == "A person is cooking."
    assert len(task.choices) == 4
    assert task.correct_idx == 0


def test_commonsense_task_winogrande_fields():
    """CommonsenseTask should support winogrande type."""
    task = CommonsenseTask(
        task_type="winogrande",
        context="Sarah gave her jacket to Emma because _ was cold.",
        choices=["Emma", "Sarah"],
        correct_idx=0,
    )
    assert task.task_type == "winogrande"
    assert "_" in task.context
    assert len(task.choices) == 2


# ---------------------------------------------------------------------------
# generate_hellaswag_tasks
# ---------------------------------------------------------------------------


def test_generate_hellaswag_tasks_count():
    """generate_hellaswag_tasks should return exactly n tasks."""
    tasks = generate_hellaswag_tasks(7)
    assert len(tasks) == 7


def test_generate_hellaswag_tasks_type():
    """All hellaswag tasks should have task_type='hellaswag'."""
    tasks = generate_hellaswag_tasks(5)
    assert all(t.task_type == "hellaswag" for t in tasks)


def test_generate_hellaswag_tasks_four_choices():
    """Each hellaswag task should have exactly 4 choices."""
    tasks = generate_hellaswag_tasks(5)
    assert all(len(t.choices) == 4 for t in tasks)


def test_generate_hellaswag_tasks_correct_idx_in_range():
    """correct_idx must be within [0, n_choices)."""
    tasks = generate_hellaswag_tasks(10)
    for t in tasks:
        assert 0 <= t.correct_idx < len(t.choices)


def test_generate_hellaswag_tasks_deterministic():
    """generate_hellaswag_tasks with same seed should be deterministic."""
    tasks_a = generate_hellaswag_tasks(5, seed=7)
    tasks_b = generate_hellaswag_tasks(5, seed=7)
    for a, b in zip(tasks_a, tasks_b):
        assert a.correct_idx == b.correct_idx
        assert a.choices == b.choices


# ---------------------------------------------------------------------------
# generate_winogrande_tasks
# ---------------------------------------------------------------------------


def test_generate_winogrande_tasks_count():
    """generate_winogrande_tasks should return exactly n tasks."""
    tasks = generate_winogrande_tasks(6)
    assert len(tasks) == 6


def test_generate_winogrande_tasks_type():
    """All winogrande tasks should have task_type='winogrande'."""
    tasks = generate_winogrande_tasks(4)
    assert all(t.task_type == "winogrande" for t in tasks)


def test_generate_winogrande_tasks_correct_idx_in_range():
    """correct_idx must be within [0, n_choices)."""
    tasks = generate_winogrande_tasks(10)
    for t in tasks:
        assert 0 <= t.correct_idx < len(t.choices)


# ---------------------------------------------------------------------------
# generate_arc_tasks
# ---------------------------------------------------------------------------


def test_generate_arc_tasks_count():
    """generate_arc_tasks should return exactly n tasks."""
    tasks = generate_arc_tasks(8)
    assert len(tasks) == 8


def test_generate_arc_tasks_type():
    """All arc tasks should have task_type='arc'."""
    tasks = generate_arc_tasks(5)
    assert all(t.task_type == "arc" for t in tasks)


def test_generate_arc_tasks_correct_idx_in_range():
    """correct_idx must be within [0, n_choices) for ARC tasks."""
    tasks = generate_arc_tasks(10)
    for t in tasks:
        assert 0 <= t.correct_idx < len(t.choices)


def test_generate_arc_tasks_four_choices():
    """Each ARC task should have exactly 4 choices."""
    tasks = generate_arc_tasks(5)
    assert all(len(t.choices) == 4 for t in tasks)


# ---------------------------------------------------------------------------
# score_task_by_likelihood
# ---------------------------------------------------------------------------


def test_score_task_by_likelihood_valid_index(small_model):
    """score_task_by_likelihood must return a valid choice index."""
    task = CommonsenseTask(
        task_type="hellaswag",
        context="A person is cooking a meal.",
        choices=["stirs the ingredients", "drives away", "reads a book", "waters plants"],
        correct_idx=0,
    )
    pred = score_task_by_likelihood(small_model, byte_encode, task)
    assert 0 <= pred < len(task.choices)


def test_score_task_by_likelihood_returns_int(small_model):
    """score_task_by_likelihood must return an int."""
    task = CommonsenseTask(
        task_type="arc",
        context="What is the boiling point of water?",
        choices=[
            "100 degrees Celsius",
            "0 degrees Celsius",
            "50 degrees Celsius",
            "200 degrees Celsius",
        ],
        correct_idx=0,
    )
    pred = score_task_by_likelihood(small_model, byte_encode, task)
    assert isinstance(pred, int)


# ---------------------------------------------------------------------------
# compute_accuracy
# ---------------------------------------------------------------------------


def test_compute_accuracy_perfect():
    """Perfect predictions should yield accuracy 1.0."""
    tasks = [
        CommonsenseTask("arc", "Q1", ["a", "b"], 0),
        CommonsenseTask("arc", "Q2", ["c", "d"], 1),
        CommonsenseTask("arc", "Q3", ["e", "f"], 0),
    ]
    predictions = [t.correct_idx for t in tasks]
    assert compute_accuracy(predictions, tasks) == 1.0


def test_compute_accuracy_none_correct():
    """All wrong predictions should yield accuracy 0.0."""
    tasks = [
        CommonsenseTask("arc", "Q1", ["a", "b"], 0),
        CommonsenseTask("arc", "Q2", ["c", "d"], 1),
    ]
    predictions = [1, 0]  # all wrong
    assert compute_accuracy(predictions, tasks) == 0.0


def test_compute_accuracy_partial():
    """Partial correct predictions should yield the right fraction."""
    tasks = [
        CommonsenseTask("arc", "Q1", ["a", "b"], 0),
        CommonsenseTask("arc", "Q2", ["c", "d"], 1),
        CommonsenseTask("arc", "Q3", ["e", "f"], 0),
        CommonsenseTask("arc", "Q4", ["g", "h"], 1),
    ]
    predictions = [0, 0, 0, 1]  # 3/4 correct (Q1, Q3, Q4)
    acc = compute_accuracy(predictions, tasks)
    assert abs(acc - 0.75) < 1e-9


def test_compute_accuracy_empty():
    """Empty task list should return 0.0."""
    assert compute_accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# length_normalize_scores
# ---------------------------------------------------------------------------


def test_length_normalize_scores_divides_by_length():
    """length_normalize_scores must divide each score by its choice length."""
    scores = [-10.0, -20.0, -5.0]
    choices = ["hello", "world!", "hi"]
    normalized = length_normalize_scores(scores, choices)
    assert abs(normalized[0] - (-10.0 / 5)) < 1e-9
    assert abs(normalized[1] - (-20.0 / 6)) < 1e-9
    assert abs(normalized[2] - (-5.0 / 2)) < 1e-9


def test_length_normalize_scores_length():
    """Output length should match input length."""
    scores = [-1.0, -2.0]
    choices = ["a", "bb"]
    result = length_normalize_scores(scores, choices)
    assert len(result) == 2


def test_length_normalize_scores_empty_choice():
    """Empty choice string should not cause division by zero."""
    scores = [-4.0]
    choices = [""]
    result = length_normalize_scores(scores, choices)
    assert result[0] == -4.0  # divided by max(0, 1) = 1


# ---------------------------------------------------------------------------
# CommonsenseEvaluator.evaluate
# ---------------------------------------------------------------------------


def test_evaluator_evaluate_returns_keys(evaluator):
    """evaluate() must return a dict with required keys."""
    tasks = generate_hellaswag_tasks(3)
    result = evaluator.evaluate(tasks)
    assert "accuracy" in result
    assert "n_correct" in result
    assert "n_total" in result
    assert "by_task_type" in result


def test_evaluator_evaluate_accuracy_in_range(evaluator):
    """evaluate() accuracy must be in [0, 1]."""
    tasks = generate_arc_tasks(4)
    result = evaluator.evaluate(tasks)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluator_evaluate_n_total(evaluator):
    """evaluate() n_total should equal number of tasks."""
    tasks = generate_winogrande_tasks(5)
    result = evaluator.evaluate(tasks)
    assert result["n_total"] == 5


def test_evaluator_evaluate_n_correct_consistent(evaluator):
    """n_correct should be consistent with accuracy and n_total."""
    tasks = generate_arc_tasks(6)
    result = evaluator.evaluate(tasks)
    expected_nc = result["n_correct"]
    computed_nc = round(result["accuracy"] * result["n_total"])
    assert abs(expected_nc - computed_nc) <= 1  # allow rounding tolerance


def test_evaluator_evaluate_by_task_type_key(evaluator):
    """by_task_type must contain the evaluated task type."""
    tasks = generate_hellaswag_tasks(3)
    result = evaluator.evaluate(tasks)
    assert "hellaswag" in result["by_task_type"]


# ---------------------------------------------------------------------------
# CommonsenseEvaluator.evaluate_suite
# ---------------------------------------------------------------------------


def test_evaluator_evaluate_suite_returns_all_task_types(evaluator):
    """evaluate_suite() must return results for all 3 task types."""
    result = evaluator.evaluate_suite(n_per_task=3)
    assert "hellaswag" in result
    assert "winogrande" in result
    assert "arc" in result


def test_evaluator_evaluate_suite_overall_key(evaluator):
    """evaluate_suite() must include an 'overall' key."""
    result = evaluator.evaluate_suite(n_per_task=3)
    assert "overall" in result


def test_evaluator_evaluate_suite_per_type_n_total(evaluator):
    """Each per-type result in evaluate_suite should match n_per_task."""
    n = 4
    result = evaluator.evaluate_suite(n_per_task=n)
    assert result["hellaswag"]["n_total"] == n
    assert result["winogrande"]["n_total"] == n
    assert result["arc"]["n_total"] == n
