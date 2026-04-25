import pytest

from src.workflow.workflow_scheduler import (
    WORKFLOW_SCHEDULER_REGISTRY,
    JobResult,
    WorkflowJob,
    WorkflowPriority,
    WorkflowScheduler,
)


def test_registry_default():
    assert WORKFLOW_SCHEDULER_REGISTRY["default"] is WorkflowScheduler


def test_priority_values():
    assert WorkflowPriority.CRITICAL == 0
    assert WorkflowPriority.HIGH == 1
    assert WorkflowPriority.NORMAL == 2
    assert WorkflowPriority.LOW == 3
    assert WorkflowPriority.BACKGROUND == 4


def test_job_result_is_frozen():
    r = JobResult(job_id="x", name="n", success=True, result=1, duration_ms=0.0, attempts=1)
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


def test_workflow_job_auto_id():
    j = WorkflowJob(name="n", fn=lambda: 1)
    assert len(j.job_id) == 8


def test_workflow_job_defaults():
    j = WorkflowJob(name="n", fn=lambda: 1)
    assert j.priority == WorkflowPriority.NORMAL
    assert j.max_retries == 0
    assert j.tags == []


def test_submit_returns_job_id():
    s = WorkflowScheduler()
    jid = s.submit("a", lambda: 1)
    assert isinstance(jid, str) and len(jid) == 8


def test_queue_size_reflects_submits():
    s = WorkflowScheduler()
    s.submit("a", lambda: 1)
    s.submit("b", lambda: 2)
    assert s.queue_size() == 2


def test_run_next_empty_returns_none():
    s = WorkflowScheduler()
    assert s.run_next() is None


def test_priority_ordering_critical_before_low():
    order = []
    s = WorkflowScheduler()
    s.submit("low", lambda: order.append("low"), priority=WorkflowPriority.LOW)
    s.submit("crit", lambda: order.append("crit"), priority=WorkflowPriority.CRITICAL)
    s.run_all()
    assert order == ["crit", "low"]


def test_priority_ordering_mixed():
    order = []
    s = WorkflowScheduler()
    s.submit("bg", lambda: order.append("bg"), priority=WorkflowPriority.BACKGROUND)
    s.submit("hi", lambda: order.append("hi"), priority=WorkflowPriority.HIGH)
    s.submit("nm", lambda: order.append("nm"), priority=WorkflowPriority.NORMAL)
    s.submit("cr", lambda: order.append("cr"), priority=WorkflowPriority.CRITICAL)
    s.run_all()
    assert order == ["cr", "hi", "nm", "bg"]


def test_fifo_within_same_priority():
    order = []
    s = WorkflowScheduler()
    s.submit("a", lambda: order.append("a"))
    s.submit("b", lambda: order.append("b"))
    s.submit("c", lambda: order.append("c"))
    s.run_all()
    assert order == ["a", "b", "c"]


def test_run_next_returns_result():
    s = WorkflowScheduler()
    s.submit("a", lambda: 99)
    r = s.run_next()
    assert r is not None
    assert r.success is True
    assert r.result == 99
    assert r.attempts == 1


def test_run_all_drains_queue():
    s = WorkflowScheduler()
    for i in range(5):
        s.submit(f"j{i}", lambda i=i: i)
    results = s.run_all()
    assert len(results) == 5
    assert s.queue_size() == 0


def test_failure_records_error():
    s = WorkflowScheduler()

    def boom():
        raise RuntimeError("nope")

    s.submit("a", boom)
    r = s.run_next()
    assert r.success is False
    assert "nope" in r.error


def test_retries_on_failure():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("x")
        return "ok"

    s = WorkflowScheduler()
    s.submit("a", flaky, max_retries=2)
    r = s.run_next()
    assert r.success is True
    assert r.attempts == 3
    assert r.result == "ok"


def test_retries_exhausted():
    state = {"n": 0}

    def always_fail():
        state["n"] += 1
        raise RuntimeError("x")

    s = WorkflowScheduler()
    s.submit("a", always_fail, max_retries=2)
    r = s.run_next()
    assert r.success is False
    assert r.attempts == 3
    assert state["n"] == 3


def test_no_retries_default():
    state = {"n": 0}

    def always_fail():
        state["n"] += 1
        raise RuntimeError("x")

    s = WorkflowScheduler()
    s.submit("a", always_fail)
    r = s.run_next()
    assert r.attempts == 1


def test_cancel_removes_from_queue():
    s = WorkflowScheduler()
    jid = s.submit("a", lambda: 1)
    assert s.cancel(jid) is True
    assert s.queue_size() == 0


def test_cancel_unknown_returns_false():
    s = WorkflowScheduler()
    assert s.cancel("nonexistent") is False


def test_cancel_twice_second_false():
    s = WorkflowScheduler()
    jid = s.submit("a", lambda: 1)
    assert s.cancel(jid) is True
    assert s.cancel(jid) is False


def test_cancelled_job_not_run():
    ran = []
    s = WorkflowScheduler()
    jid = s.submit("a", lambda: ran.append("a"))
    s.cancel(jid)
    results = s.run_all()
    assert ran == []
    assert results == []


def test_cancel_one_of_many():
    ran = []
    s = WorkflowScheduler()
    s.submit("a", lambda: ran.append("a"))
    jid = s.submit("b", lambda: ran.append("b"))
    s.submit("c", lambda: ran.append("c"))
    s.cancel(jid)
    s.run_all()
    assert ran == ["a", "c"]


def test_pending_by_priority_counts():
    s = WorkflowScheduler()
    s.submit("a", lambda: 1, priority=WorkflowPriority.CRITICAL)
    s.submit("b", lambda: 2, priority=WorkflowPriority.CRITICAL)
    s.submit("c", lambda: 3, priority=WorkflowPriority.LOW)
    counts = s.pending_by_priority()
    assert counts["CRITICAL"] == 2
    assert counts["LOW"] == 1
    assert counts["NORMAL"] == 0


def test_pending_by_priority_after_run():
    s = WorkflowScheduler()
    s.submit("a", lambda: 1)
    s.run_next()
    counts = s.pending_by_priority()
    assert all(v == 0 for v in counts.values())


def test_tags_preserved_on_job():
    s = WorkflowScheduler()
    s.submit("a", lambda: 1, tags=["urgent", "demo"])
    assert s.queue_size() == 1


def test_result_includes_job_id_and_name():
    s = WorkflowScheduler()
    jid = s.submit("my-job", lambda: 1)
    r = s.run_next()
    assert r.job_id == jid
    assert r.name == "my-job"


def test_duration_ms_non_negative():
    s = WorkflowScheduler()
    s.submit("a", lambda: 1)
    r = s.run_next()
    assert r.duration_ms >= 0.0


def test_queue_size_after_cancel_and_submit():
    s = WorkflowScheduler()
    jid = s.submit("a", lambda: 1)
    s.cancel(jid)
    s.submit("b", lambda: 2)
    assert s.queue_size() == 1


def test_run_all_respects_priority():
    order = []
    s = WorkflowScheduler()
    s.submit("n1", lambda: order.append("n1"), priority=WorkflowPriority.NORMAL)
    s.submit("c1", lambda: order.append("c1"), priority=WorkflowPriority.CRITICAL)
    s.submit("n2", lambda: order.append("n2"), priority=WorkflowPriority.NORMAL)
    s.submit("c2", lambda: order.append("c2"), priority=WorkflowPriority.CRITICAL)
    s.run_all()
    assert order[:2] == ["c1", "c2"]
    assert order[2:] == ["n1", "n2"]
