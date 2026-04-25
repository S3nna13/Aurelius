import pytest

from src.workflow.step_pipeline import (
    STEP_PIPELINE_REGISTRY,
    PipelineStep,
    StepPipeline,
    StepResult,
)


def test_registry_default():
    assert STEP_PIPELINE_REGISTRY["default"] is StepPipeline


def test_step_result_is_frozen():
    r = StepResult(step_name="s", success=True, output=1, duration_ms=0.0)
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


def test_pipeline_step_defaults():
    s = PipelineStep(name="x", fn=lambda v: v)
    assert s.retry_count == 0
    assert s.timeout_ms is None


def test_add_step_fluent_returns_self():
    p = StepPipeline()
    assert p.add_step("a", lambda v: v) is p


def test_on_step_complete_fluent():
    p = StepPipeline()
    assert p.on_step_complete(lambda n, r: None) is p


def test_on_failure_fluent():
    p = StepPipeline()
    assert p.on_failure(lambda n, r: None) is p


def test_chain_fluent_api():
    p = (
        StepPipeline()
        .add_step("a", lambda v: (v or 0) + 1)
        .add_step("b", lambda v: v * 2)
    )
    results = p.run(0)
    assert [r.output for r in results] == [1, 2]


def test_run_passes_output_to_next():
    p = StepPipeline().add_step("a", lambda v: 10).add_step("b", lambda v: v + 5)
    results = p.run()
    assert results[1].output == 15


def test_run_empty_pipeline():
    p = StepPipeline()
    assert p.run() == []


def test_all_steps_marked_success():
    p = StepPipeline().add_step("a", lambda v: 1).add_step("b", lambda v: 2)
    results = p.run()
    assert all(r.success for r in results)


def test_step_failure_stops_pipeline():
    def boom(v):
        raise RuntimeError("x")

    p = StepPipeline().add_step("a", lambda v: 1).add_step("b", boom).add_step("c", lambda v: 2)
    results = p.run()
    assert len(results) == 2
    assert results[1].success is False


def test_retry_succeeds_after_failure():
    state = {"n": 0}

    def flaky(v):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("first")
        return "ok"

    p = StepPipeline().add_step("a", flaky, retry_count=1)
    results = p.run()
    assert results[0].success is True
    assert results[0].output == "ok"
    assert state["n"] == 2


def test_retry_count_one_tries_twice():
    state = {"n": 0}

    def always_fail(v):
        state["n"] += 1
        raise RuntimeError("no")

    p = StepPipeline().add_step("a", always_fail, retry_count=1)
    results = p.run()
    assert state["n"] == 2
    assert results[0].success is False


def test_retry_count_zero_tries_once():
    state = {"n": 0}

    def always_fail(v):
        state["n"] += 1
        raise RuntimeError("no")

    p = StepPipeline().add_step("a", always_fail)
    p.run()
    assert state["n"] == 1


def test_on_step_complete_hook_called_each_step():
    seen = []
    p = (
        StepPipeline()
        .add_step("a", lambda v: 1)
        .add_step("b", lambda v: 2)
        .on_step_complete(lambda name, r: seen.append(name))
    )
    p.run()
    assert seen == ["a", "b"]


def test_multiple_complete_hooks_all_called():
    a, b = [], []
    p = (
        StepPipeline()
        .add_step("s", lambda v: 1)
        .on_step_complete(lambda n, r: a.append(n))
        .on_step_complete(lambda n, r: b.append(n))
    )
    p.run()
    assert a == ["s"] and b == ["s"]


def test_on_failure_hook_called_on_final_failure():
    fails = []

    def boom(v):
        raise RuntimeError("x")

    p = (
        StepPipeline()
        .add_step("a", boom)
        .on_failure(lambda n, r: fails.append(n))
    )
    p.run()
    assert fails == ["a"]


def test_on_failure_not_called_on_success():
    fails = []
    p = StepPipeline().add_step("a", lambda v: 1).on_failure(lambda n, r: fails.append(n))
    p.run()
    assert fails == []


def test_on_failure_not_called_if_retry_succeeds():
    fails = []
    state = {"n": 0}

    def flaky(v):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("x")
        return 1

    p = (
        StepPipeline()
        .add_step("a", flaky, retry_count=1)
        .on_failure(lambda n, r: fails.append(n))
    )
    p.run()
    assert fails == []


def test_step_result_records_error():
    def boom(v):
        raise RuntimeError("kaboom")

    p = StepPipeline().add_step("a", boom)
    results = p.run()
    assert "kaboom" in results[0].error


def test_summary_counts():
    def boom(v):
        raise RuntimeError("x")

    p = StepPipeline().add_step("a", lambda v: 1).add_step("b", boom)
    results = p.run()
    summary = p.summary(results)
    assert summary["total_steps"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["total_duration_ms"] >= 0.0


def test_summary_all_passed():
    p = StepPipeline().add_step("a", lambda v: 1)
    summary = p.summary(p.run())
    assert summary["passed"] == 1
    assert summary["failed"] == 0


def test_summary_empty():
    p = StepPipeline()
    s = p.summary([])
    assert s["total_steps"] == 0


def test_initial_input_is_passed():
    p = StepPipeline().add_step("a", lambda v: v + 1)
    results = p.run(initial_input=10)
    assert results[0].output == 11


def test_step_records_duration():
    p = StepPipeline().add_step("a", lambda v: 1)
    results = p.run()
    assert results[0].duration_ms >= 0.0


def test_hook_receives_result_object():
    captured = {}

    def hook(name, result):
        captured[name] = result

    p = StepPipeline().add_step("a", lambda v: 42).on_step_complete(hook)
    p.run()
    assert captured["a"].output == 42
    assert captured["a"].success is True


def test_failure_hook_receives_error():
    captured = {}

    def boom(v):
        raise RuntimeError("zzz")

    def hook(name, result):
        captured[name] = result

    p = StepPipeline().add_step("a", boom).on_failure(hook)
    p.run()
    assert captured["a"].success is False
    assert "zzz" in captured["a"].error
