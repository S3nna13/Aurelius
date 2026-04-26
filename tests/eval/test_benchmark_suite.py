"""Tests for src/eval/benchmark_suite.py — 14 tests."""

from __future__ import annotations

import pytest
import torch

from src.eval.benchmark_suite import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    evaluate_generation_task,
    evaluate_multiple_choice_task,
    evaluate_perplexity_task,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

SEQ_LEN = 8
N_SAMPLES = 3
MAX_NEW_TOKENS = 2


@pytest.fixture(scope="module")
def model():
    m = AureliusTransformer(CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def ppl_samples():
    return [torch.randint(0, 256, (SEQ_LEN,)) for _ in range(N_SAMPLES)]


@pytest.fixture(scope="module")
def mc_questions():
    questions = []
    for i in range(N_SAMPLES):
        questions.append(
            {
                "context_ids": torch.randint(0, 256, (SEQ_LEN,)),
                "choice_ids": [
                    torch.randint(0, 256, (4,)),
                    torch.randint(0, 256, (4,)),
                ],
                "correct_idx": 0,
            }
        )
    return questions


@pytest.fixture(scope="module")
def gen_prompts():
    return [torch.randint(0, 256, (SEQ_LEN,)) for _ in range(N_SAMPLES)]


@pytest.fixture(scope="module")
def gen_references():
    return ["ref_a", "ref_b", "ref_c"]


def byte_decode(ids):
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. BenchmarkTask fields accessible
# ---------------------------------------------------------------------------


def test_benchmark_task_fields():
    task = BenchmarkTask(name="my_task", task_type="perplexity", n_samples=50, weight=2.0)
    assert task.name == "my_task"
    assert task.task_type == "perplexity"
    assert task.n_samples == 50
    assert task.weight == 2.0


# ---------------------------------------------------------------------------
# 2. BenchmarkResult fields accessible
# ---------------------------------------------------------------------------


def test_benchmark_result_fields():
    res = BenchmarkResult(task_name="foo", metric=0.5, n_evaluated=10, details={"x": 1})
    assert res.task_name == "foo"
    assert res.metric == 0.5
    assert res.n_evaluated == 10
    assert res.details == {"x": 1}


# ---------------------------------------------------------------------------
# 3. evaluate_perplexity_task returns BenchmarkResult
# ---------------------------------------------------------------------------


def test_evaluate_perplexity_task_returns_type(model, ppl_samples):
    result = evaluate_perplexity_task(model, ppl_samples, max_len=SEQ_LEN)
    assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# 4. perplexity metric > 0
# ---------------------------------------------------------------------------


def test_evaluate_perplexity_metric_positive(model, ppl_samples):
    result = evaluate_perplexity_task(model, ppl_samples, max_len=SEQ_LEN)
    assert result.metric > 0.0


# ---------------------------------------------------------------------------
# 5. evaluate_multiple_choice_task returns BenchmarkResult
# ---------------------------------------------------------------------------


def test_evaluate_multiple_choice_task_returns_type(model, mc_questions):
    result = evaluate_multiple_choice_task(model, mc_questions, task_name="mc_test")
    assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# 6. multiple_choice accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_evaluate_multiple_choice_accuracy_range(model, mc_questions):
    result = evaluate_multiple_choice_task(model, mc_questions, task_name="mc_test")
    assert 0.0 <= result.metric <= 1.0


# ---------------------------------------------------------------------------
# 7. evaluate_generation_task returns BenchmarkResult
# ---------------------------------------------------------------------------


def test_evaluate_generation_task_returns_type(model, gen_prompts, gen_references):
    result = evaluate_generation_task(
        model, gen_prompts, gen_references, byte_decode, MAX_NEW_TOKENS, "gen_test"
    )
    assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# 8. generation exact_match in [0, 1]
# ---------------------------------------------------------------------------


def test_evaluate_generation_exact_match_range(model, gen_prompts, gen_references):
    result = evaluate_generation_task(
        model, gen_prompts, gen_references, byte_decode, MAX_NEW_TOKENS, "gen_test"
    )
    assert 0.0 <= result.metric <= 1.0


# ---------------------------------------------------------------------------
# 9. BenchmarkSuite.run_all returns required keys
# ---------------------------------------------------------------------------


def test_run_all_returns_required_keys(model, ppl_samples, mc_questions):
    tasks = [
        BenchmarkTask(name="ppl_task", task_type="perplexity"),
        BenchmarkTask(name="mc_task", task_type="multiple_choice"),
    ]
    suite = BenchmarkSuite(tasks, model, byte_decode)
    data = {
        "ppl_task": {"samples": ppl_samples, "max_len": SEQ_LEN},
        "mc_task": {"questions": mc_questions},
    }
    out = suite.run_all(data)
    assert "results" in out
    assert "aggregate_score" in out
    assert "task_scores" in out


# ---------------------------------------------------------------------------
# 10. aggregate_score is float in [0, inf)
# ---------------------------------------------------------------------------


def test_aggregate_score_is_float_nonnegative(model, ppl_samples):
    tasks = [BenchmarkTask(name="ppl_task", task_type="perplexity")]
    suite = BenchmarkSuite(tasks, model)
    results = [BenchmarkResult(task_name="ppl_task", metric=42.5, n_evaluated=3)]
    score = suite.aggregate_score(results)
    assert isinstance(score, float)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# 11. task_scores has all task names as keys
# ---------------------------------------------------------------------------


def test_task_scores_has_all_names(model, ppl_samples, mc_questions):
    tasks = [
        BenchmarkTask(name="ppl_task", task_type="perplexity"),
        BenchmarkTask(name="mc_task", task_type="multiple_choice"),
    ]
    suite = BenchmarkSuite(tasks, model, byte_decode)
    data = {
        "ppl_task": {"samples": ppl_samples, "max_len": SEQ_LEN},
        "mc_task": {"questions": mc_questions},
    }
    out = suite.run_all(data)
    for task in tasks:
        assert task.name in out["task_scores"]


# ---------------------------------------------------------------------------
# 12. BenchmarkSuite with single perplexity task works
# ---------------------------------------------------------------------------


def test_suite_single_perplexity_task(model, ppl_samples):
    tasks = [BenchmarkTask(name="ppl_only", task_type="perplexity", weight=1.0)]
    suite = BenchmarkSuite(tasks, model)
    data = {"ppl_only": {"samples": ppl_samples, "max_len": SEQ_LEN}}
    out = suite.run_all(data)
    assert len(out["results"]) == 1
    assert out["results"][0].metric > 0.0


# ---------------------------------------------------------------------------
# 13. aggregate_score weighted correctly (higher weight = more influence)
# ---------------------------------------------------------------------------


def test_aggregate_score_weighted_correctly(model):
    # Two tasks with metrics 10.0 and 1.0; weight 9:1 should pull score near 10
    tasks = [
        BenchmarkTask(name="heavy", task_type="perplexity", weight=9.0),
        BenchmarkTask(name="light", task_type="perplexity", weight=1.0),
    ]
    suite = BenchmarkSuite(tasks, model)
    results = [
        BenchmarkResult(task_name="heavy", metric=10.0, n_evaluated=3),
        BenchmarkResult(task_name="light", metric=1.0, n_evaluated=3),
    ]
    score = suite.aggregate_score(results)
    # weighted mean = (10*9 + 1*1) / 10 = 9.1
    assert abs(score - 9.1) < 1e-6
    # Score should be closer to 10 than to 1
    assert abs(score - 10.0) < abs(score - 1.0)


# ---------------------------------------------------------------------------
# 14. run_all with 2 tasks returns 2 results
# ---------------------------------------------------------------------------


def test_run_all_two_tasks_two_results(model, ppl_samples, mc_questions):
    tasks = [
        BenchmarkTask(name="ppl_task", task_type="perplexity"),
        BenchmarkTask(name="mc_task", task_type="multiple_choice"),
    ]
    suite = BenchmarkSuite(tasks, model, byte_decode)
    data = {
        "ppl_task": {"samples": ppl_samples, "max_len": SEQ_LEN},
        "mc_task": {"questions": mc_questions},
    }
    out = suite.run_all(data)
    assert len(out["results"]) == 2
