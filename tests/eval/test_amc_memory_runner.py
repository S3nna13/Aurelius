"""Tests for the tiny AMC-Memory JSON runner."""

from __future__ import annotations

import json

from src.eval.amc_memory_runner import main, run_benchmark


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
