"""Unit tests for AMC memory benchmark scaffold."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.eval.amc_memory_benchmark import AMCMemoryBenchmark, AMCMemoryExample


def test_cross_session_recall_builder_returns_expected_value_in_prompt():
    bench = AMCMemoryBenchmark()
    example = bench.build_cross_session_recall(context_tokens=128, seed=0)

    assert isinstance(example, AMCMemoryExample)
    assert example.task == "cross_session_recall"
    assert example.expected in example.prompt
    assert example.metadata["target_key"] in example.prompt


def test_surprise_gate_builder_selects_highest_importance_observation():
    bench = AMCMemoryBenchmark()
    example = bench.build_surprise_gate_selectivity(context_tokens=128, seed=2)

    assert example.expected == "obs_b"
    assert "importance=9" in example.prompt
    assert "episodic memory" in example.prompt


def test_consolidation_builder_repeats_durable_memory():
    bench = AMCMemoryBenchmark()
    example = bench.build_consolidation_preference(context_tokens=128, seed=1)

    assert example.expected in example.prompt
    assert example.prompt.count(example.expected) == 3
    assert example.metadata["repetitions"] == 3


def test_contradiction_builder_requires_quarantine():
    bench = AMCMemoryBenchmark()
    example = bench.build_contradiction_quarantine(context_tokens=128, seed=3)

    assert example.expected == "quarantine"
    assert "conflict" in example.prompt.lower() or "conflicting" in example.prompt.lower()
    assert "127.0.0.1" in example.prompt
    assert "0.0.0.0" in example.prompt


def test_builders_are_deterministic_by_seed():
    bench = AMCMemoryBenchmark()
    first = bench.build_cross_session_recall(context_tokens=128, seed=42)
    second = bench.build_cross_session_recall(context_tokens=128, seed=42)
    different = bench.build_cross_session_recall(context_tokens=128, seed=43)

    assert first == second
    assert first.prompt != different.prompt or first.expected != different.expected


def test_oracle_evaluation_scores_one():
    bench = AMCMemoryBenchmark()
    lookup = {}
    for task in bench.TASKS:
        for seed in range(2):
            example = bench._build_task(task, context_tokens=128, seed=seed)
            lookup[example.prompt] = example.expected

    results = bench.evaluate(lambda prompt: lookup[prompt], context_tokens=128, samples_per=2)

    assert bench.overall_score(results) == 1.0
    assert all(score == 1.0 for score in bench.score_per_task(results).values())


def test_null_evaluation_scores_zero():
    bench = AMCMemoryBenchmark()
    results = bench.evaluate(lambda _prompt: "", context_tokens=128, samples_per=2)

    assert bench.overall_score(results) == 0.0
    assert all(score == 0.0 for score in bench.score_per_task(results).values())


def test_exact_match_rejects_substring_false_positives():
    bench = AMCMemoryBenchmark()

    assert bench._check("quarantine", "quarantine") is True
    assert bench._check("quarantine", "do not quarantine") is False
    assert bench._check("127.0.0.1", "not 127.0.0.1") is False
    assert bench._check("obs_b", "obs_b is wrong") is False


def test_evaluate_subset_and_cells_shape():
    bench = AMCMemoryBenchmark()
    task = "contradiction_quarantine"
    example = bench._build_task(task, context_tokens=128, seed=0)

    results = bench.evaluate(
        lambda _prompt: example.expected,
        tasks=[task],
        context_tokens=128,
        samples_per=1,
    )

    assert list(results) == [task]
    assert results[task]["pass_rate"] == 1.0
    assert results[task]["n"] == 1
    cells = cast(list[dict[str, Any]], results[task]["cells"])
    assert cells[0]["metadata"]["conflict"] == "host_default"


def test_invalid_inputs_raise():
    bench = AMCMemoryBenchmark()

    with pytest.raises(ValueError):
        AMCMemoryBenchmark(approx_chars_per_token=0)
    with pytest.raises(ValueError):
        bench.build_cross_session_recall(context_tokens=0)
    with pytest.raises(ValueError):
        bench.evaluate(lambda _prompt: "", tasks=[])
    with pytest.raises(ValueError):
        bench.evaluate(lambda _prompt: "", tasks=["unknown"])
    with pytest.raises(ValueError):
        bench.evaluate(lambda _prompt: "", samples_per=0)
    with pytest.raises(ValueError):
        bench.evaluate("not callable")  # type: ignore[arg-type]
