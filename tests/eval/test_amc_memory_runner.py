"""Tests for the tiny AMC-Memory JSON runner."""

from __future__ import annotations

import json

from src.eval.amc_memory_runner import append_jsonl_result, build_engine_generate_fn, main, run_benchmark


def test_run_benchmark_oracle_outputs_summary_and_cells():
    payload = run_benchmark(generator="oracle", context_tokens=64, samples_per=1)

    assert payload["suite"] == "amc_memory"
    assert payload["generator"] == "oracle"
    assert payload["overall_score"] == 1.0
    assert set(payload["scores"]) == {
        "cross_session_recall",
        "surprise_gate_selectivity",
        "consolidation_preference",
        "contradiction_quarantine",
    }
    assert payload["results"]["cross_session_recall"]["n"] == 1


def test_run_benchmark_null_outputs_zero_score_for_subset():
    payload = run_benchmark(
        generator="null",
        tasks=["contradiction_quarantine"],
        context_tokens=64,
        samples_per=2,
    )

    assert payload["generator"] == "null"
    assert payload["overall_score"] == 0.0
    assert payload["scores"] == {"contradiction_quarantine": 0.0}
    assert payload["results"]["contradiction_quarantine"]["n"] == 2


def test_engine_generate_fn_wraps_serving_backend_request():
    captured = {}

    def fake_build_engine(**kwargs):
        captured["build_kwargs"] = kwargs

        def fake_generate(request):
            captured["request"] = request
            return request.messages[-1]["content"].upper()

        return fake_generate, "fake-backend", object()

    generate = build_engine_generate_fn(
        backend="mock",
        model_path="/tmp/aurelius-test-checkpoint",
        model="aurelius-test",
        max_tokens=17,
        temperature=0.25,
        system_prompt="AMC benchmark system prompt",
        engine_builder=fake_build_engine,
    )

    assert generate("remember the host") == "REMEMBER THE HOST"
    assert captured["build_kwargs"]["backend"] == "mock"
    assert captured["build_kwargs"]["model_path"] == "/tmp/aurelius-test-checkpoint"
    request = captured["request"]
    assert request.model == "aurelius-test"
    assert request.messages == [{"role": "user", "content": "remember the host"}]
    assert request.max_tokens == 17
    assert request.temperature == 0.25
    assert request.system == "AMC benchmark system prompt"


def test_jsonl_persistence_appends_compact_records(tmp_path):
    output_path = tmp_path / "benchmark-results" / "amc_memory_runs.jsonl"
    first = {"suite": "amc_memory", "generator": "oracle", "overall_score": 1.0}
    second = {"suite": "amc_memory", "generator": "null", "overall_score": 0.0}

    append_jsonl_result(output_path, first)
    append_jsonl_result(output_path, second)

    lines = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert lines == [first, second]


def test_main_appends_jsonl_output(tmp_path):
    output_path = tmp_path / "benchmark-results" / "amc_memory_runs.jsonl"

    exit_code = main(
        [
            "--generator",
            "oracle",
            "--tasks",
            "contradiction_quarantine",
            "--context-tokens",
            "64",
            "--samples-per",
            "1",
            "--jsonl-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["suite"] == "amc_memory"
    assert records[0]["scores"] == {"contradiction_quarantine": 1.0}


def test_main_writes_json_output(tmp_path):
    output_path = tmp_path / "amc_memory.json"

    exit_code = main(
        [
            "--generator",
            "oracle",
            "--tasks",
            "contradiction_quarantine",
            "--context-tokens",
            "64",
            "--samples-per",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["overall_score"] == 1.0
    assert list(payload["scores"]) == ["contradiction_quarantine"]
