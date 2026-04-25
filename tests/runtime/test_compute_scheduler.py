from __future__ import annotations

import time

import pytest

from src.runtime.compute_scheduler import ComputeJob, ComputeScheduler, Priority


@pytest.fixture()
def sched():
    return ComputeScheduler(max_concurrent=2, flop_budget_per_s=1e12)


def test_submit_returns_job_with_pending_status(sched):
    job = sched.submit(Priority.NORMAL, 1e9)
    assert isinstance(job, ComputeJob)
    assert job.status == "pending"
    assert job.priority == Priority.NORMAL
    assert job.estimated_flops == 1e9


def test_submit_assigns_unique_job_ids(sched):
    j1 = sched.submit(Priority.NORMAL, 1e9)
    j2 = sched.submit(Priority.NORMAL, 1e9)
    assert j1.job_id != j2.job_id


def test_submit_stores_metadata(sched):
    job = sched.submit(Priority.LOW, 1e8, metadata={"user": "alice"})
    assert job.metadata["user"] == "alice"


def test_next_job_returns_none_when_empty(sched):
    assert sched.next_job() is None


def test_next_job_returns_highest_priority(sched):
    sched.submit(Priority.LOW, 1e9)
    sched.submit(Priority.CRITICAL, 1e9)
    sched.submit(Priority.NORMAL, 1e9)
    job = sched.next_job()
    assert job.priority == Priority.CRITICAL


def test_next_job_fifo_within_same_priority(sched):
    j1 = sched.submit(Priority.NORMAL, 1e9)
    time.sleep(0.001)
    j2 = sched.submit(Priority.NORMAL, 1e9)
    assert sched.next_job().job_id == j1.job_id


def test_start_marks_running(sched):
    job = sched.submit(Priority.NORMAL, 1e9)
    started = sched.start(job.job_id)
    assert started.status == "running"
    assert started.started_at is not None


def test_start_raises_when_max_concurrent_reached(sched):
    j1 = sched.submit(Priority.NORMAL, 1e9)
    j2 = sched.submit(Priority.NORMAL, 1e9)
    j3 = sched.submit(Priority.NORMAL, 1e9)
    sched.start(j1.job_id)
    sched.start(j2.job_id)
    with pytest.raises(RuntimeError, match="max_concurrent"):
        sched.start(j3.job_id)


def test_complete_marks_done(sched):
    job = sched.submit(Priority.NORMAL, 1e9)
    sched.start(job.job_id)
    done = sched.complete(job.job_id)
    assert done.status == "done"
    assert done.completed_at is not None


def test_cancel_marks_cancelled(sched):
    job = sched.submit(Priority.NORMAL, 1e9)
    cancelled = sched.cancel(job.job_id)
    assert cancelled.status == "cancelled"


def test_running_jobs_list(sched):
    j1 = sched.submit(Priority.NORMAL, 1e9)
    j2 = sched.submit(Priority.NORMAL, 1e9)
    sched.start(j1.job_id)
    running = sched.running_jobs()
    assert len(running) == 1
    assert running[0].job_id == j1.job_id


def test_pending_jobs_list(sched):
    j1 = sched.submit(Priority.NORMAL, 1e9)
    j2 = sched.submit(Priority.NORMAL, 1e9)
    sched.start(j1.job_id)
    pending = sched.pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == j2.job_id


def test_stats_initial(sched):
    s = sched.stats()
    assert s["pending"] == 0
    assert s["running"] == 0
    assert s["done"] == 0
    assert s["cancelled"] == 0
    assert s["total_flops_scheduled"] == 0.0


def test_stats_counts_all_states(sched):
    j1 = sched.submit(Priority.CRITICAL, 1e10)
    j2 = sched.submit(Priority.HIGH, 2e10)
    j3 = sched.submit(Priority.NORMAL, 3e10)
    j4 = sched.submit(Priority.LOW, 4e10)
    sched.start(j1.job_id)
    sched.complete(j1.job_id)
    sched.start(j2.job_id)
    sched.cancel(j3.job_id)
    s = sched.stats()
    assert s["done"] == 1
    assert s["running"] == 1
    assert s["cancelled"] == 1
    assert s["pending"] == 1
    assert abs(s["total_flops_scheduled"] - 1e10 - 2e10 - 3e10 - 4e10) < 1


def test_stats_total_flops_accumulates(sched):
    sched.submit(Priority.NORMAL, 5e11)
    sched.submit(Priority.HIGH, 5e11)
    assert sched.stats()["total_flops_scheduled"] == 1e12


def test_unknown_job_start_raises(sched):
    with pytest.raises(KeyError):
        sched.start("nonexistent")


def test_unknown_job_complete_raises(sched):
    with pytest.raises(KeyError):
        sched.complete("nonexistent")


def test_unknown_job_cancel_raises(sched):
    with pytest.raises(KeyError):
        sched.cancel("nonexistent")


def test_priority_enum_ordering():
    assert Priority.CRITICAL < Priority.HIGH < Priority.NORMAL < Priority.LOW < Priority.BACKGROUND


def test_next_job_does_not_pop_from_queue(sched):
    sched.submit(Priority.NORMAL, 1e9)
    sched.next_job()
    assert len(sched.pending_jobs()) == 1
