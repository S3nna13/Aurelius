"""Unit tests for ``src.agent.dispatch_task``."""

from __future__ import annotations

import time

import pytest

from src.agent.dispatch_task import (
    Dispatcher,
    DispatchOutcome,
    DispatchTask,
    classify_error,
)


class _CountTask(DispatchTask):
    name = "count_task"

    def build_prompt(self, input_item):
        return f"count:{input_item}"

    def process_result(self, input_item, result):
        return {"item": input_item, "echo": result}

    def finalize(self, processed_results):
        return {"total": len(processed_results)}


def _echo_llm(prompt, schema):
    return f"ok[{prompt}]"


def test_concrete_subclass_runs_five_inputs():
    disp = Dispatcher(max_workers=2)
    report = disp.dispatch(_CountTask(), [1, 2, 3, 4, 5], _echo_llm)
    assert report.task_name == "count_task"
    assert len(report.outcomes) == 5
    assert report.status_counts.get("success") == 5
    assert report.finalized == {"total": 5}
    assert sum(report.status_counts.values()) == 5


def test_parallel_execution_actually_parallel():
    """With max_workers=5, five 100ms sleeps should finish in < 2x 100ms."""

    class SleepTask(_CountTask):
        name = "sleep_task"

    def slow_llm(prompt, schema):
        time.sleep(0.1)
        return prompt

    disp = Dispatcher(max_workers=5, per_task_timeout_s=5.0)
    t0 = time.monotonic()
    report = disp.dispatch(SleepTask(), [1, 2, 3, 4, 5], slow_llm)
    elapsed = time.monotonic() - t0
    assert report.status_counts.get("success") == 5
    assert elapsed < 0.2, f"parallel dispatch too slow: {elapsed:.3f}s"


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("403 Forbidden", "auth"),
        ("401 Unauthorized", "auth"),
        ("Rate limit exceeded", "quota"),
        ("quota exhausted", "quota"),
        ("Request timed out", "timeout"),
        ("content filter blocked the request", "blocked"),
        ("refused to answer", "blocked"),
        ("random failure", "error"),
    ],
)
def test_classify_error_categories(msg, expected):
    assert classify_error(msg) == expected


def test_classify_error_exceptions():
    assert classify_error(TimeoutError("boom")) == "timeout"
    assert classify_error(ValueError("unauthorized access")) == "auth"
    assert classify_error(None) == "success"
    assert classify_error("") == "success"


def test_per_task_timeout_enforced():
    class SleepTask(_CountTask):
        pass

    def hang(prompt, schema):
        time.sleep(2.0)
        return "never"

    disp = Dispatcher(max_workers=1, per_task_timeout_s=0.2)
    report = disp.dispatch(SleepTask(), [1, 2], hang)
    # First item may or may not complete depending on scheduling, but
    # at least one should be marked timeout; none should be success.
    assert report.status_counts.get("success", 0) == 0
    assert report.status_counts.get("timeout", 0) >= 1


def test_failure_threshold_aborts_midpoint():
    def always_fail(prompt, schema):
        time.sleep(0.05)
        raise RuntimeError("unauthorized")

    disp = Dispatcher(max_workers=1, per_task_timeout_s=5.0, failure_threshold=0.4)
    report = disp.dispatch(_CountTask(), list(range(20)), always_fail)
    assert report.circuit_open is True
    assert any(o.error_class == "CircuitOpen" for o in report.outcomes)
    assert report.status_counts.get("success", 0) == 0


def test_empty_input_list():
    disp = Dispatcher()
    report = disp.dispatch(_CountTask(), [], _echo_llm)
    assert report.outcomes == []
    assert report.finalized == {"total": 0}
    assert report.status_counts == {}


def test_single_input():
    disp = Dispatcher()
    report = disp.dispatch(_CountTask(), ["only"], _echo_llm)
    assert len(report.outcomes) == 1
    assert report.outcomes[0].status == "success"


def test_malformed_llm_output_surfaces_as_error():
    class StrictTask(_CountTask):
        def process_result(self, input_item, result):
            if "bad" in result:
                raise ValueError("malformed llm output")
            return result

    def bad_llm(prompt, schema):
        return "bad-stuff"

    disp = Dispatcher(failure_threshold=0.99)
    report = disp.dispatch(StrictTask(), [1, 2], bad_llm)
    assert report.status_counts.get("error") == 2
    assert all(o.error_class == "ValueError" for o in report.outcomes)


def test_finalize_called_with_processed_results():
    captured: list = []

    class RecordTask(_CountTask):
        def finalize(self, processed_results):
            captured.append(list(processed_results))
            return {"n": len(processed_results)}

    disp = Dispatcher()
    report = disp.dispatch(RecordTask(), [1, 2, 3], _echo_llm)
    assert report.finalized == {"n": 3}
    assert len(captured) == 1
    assert len(captured[0]) == 3


def test_determinism_same_inputs_same_outcomes():
    disp = Dispatcher(max_workers=2)
    r1 = disp.dispatch(_CountTask(), [1, 2, 3], _echo_llm)
    r2 = disp.dispatch(_CountTask(), [1, 2, 3], _echo_llm)
    assert [o.processed for o in r1.outcomes] == [o.processed for o in r2.outcomes]
    assert [o.status for o in r1.outcomes] == [o.status for o in r2.outcomes]


def test_validate_input_rejects_bad_entries():
    class PickyTask(_CountTask):
        def validate_input(self, item):
            return isinstance(item, int) and item > 0

    disp = Dispatcher()
    report = disp.dispatch(PickyTask(), [1, 0, 2, -1, 3], _echo_llm)
    assert report.status_counts.get("success") == 3
    assert report.status_counts.get("error") == 2
    assert any(o.error_class == "ValidationError" for o in report.outcomes)


def test_status_counts_sum_equals_len_inputs():
    def mixed(prompt, schema):
        if "2" in prompt:
            raise RuntimeError("403 forbidden")
        return prompt

    disp = Dispatcher()
    inputs = [1, 2, 3, 4, 5]
    report = disp.dispatch(_CountTask(), inputs, mixed)
    assert sum(report.status_counts.values()) == len(inputs)


def test_report_serializable_to_dict():
    disp = Dispatcher()
    report = disp.dispatch(_CountTask(), [1, 2], _echo_llm)
    d = report.to_dict()
    assert d["task_name"] == "count_task"
    assert isinstance(d["outcomes"], list)
    assert isinstance(d["outcomes"][0], dict)
    assert "status" in d["outcomes"][0]
    assert "status_counts" in d
    assert d["finalized"] == {"total": 2}


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DispatchTask()  # type: ignore[abstract]


def test_dispatcher_rejects_invalid_construction():
    with pytest.raises(ValueError):
        Dispatcher(max_workers=0)
    with pytest.raises(ValueError):
        Dispatcher(per_task_timeout_s=0)
    with pytest.raises(ValueError):
        Dispatcher(failure_threshold=0)
    with pytest.raises(ValueError):
        Dispatcher(failure_threshold=1.5)


def test_outcome_to_dict_round_trip():
    o = DispatchOutcome(
        input_item=1,
        raw_result="r",
        processed={"k": 1},
        status="success",
        duration_s=0.01,
        error_class=None,
    )
    d = o.to_dict()
    assert d["status"] == "success"
    assert d["processed"] == {"k": 1}


def test_get_schema_default_none():
    class Plain(_CountTask):
        pass

    assert Plain().get_schema() is None

    seen = []

    class Schemaful(_CountTask):
        def get_schema(self):
            return {"type": "object"}

    def spy(prompt, schema):
        seen.append(schema)
        return prompt

    Dispatcher().dispatch(Schemaful(), [1], spy)
    assert seen == [{"type": "object"}]
