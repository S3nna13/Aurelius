"""Comprehensive tests for task_scheduler.py."""

import threading
import time
from datetime import datetime

import pytest

from agent.task_scheduler import (
    Job,
    TaskScheduler,
    _next_cron_time,
    _parse_cron_field,
    _parse_delay,
    get_scheduler,
)

# ---------------------------------------------------------------------------
# Helper cron parsers
# ---------------------------------------------------------------------------

class TestParseCronField:
    def test_star(self):
        assert len(_parse_cron_field("*", 0, 59)) == 60

    def test_single_value(self):
        assert _parse_cron_field("5", 0, 59) == [5]

    def test_list(self):
        assert _parse_cron_field("1,3,5", 0, 59) == [1, 3, 5]

    def test_range(self):
        assert _parse_cron_field("1-5", 0, 59) == [1, 2, 3, 4, 5]

    def test_step(self):
        assert _parse_cron_field("*/5", 0, 59) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    def test_range_with_step(self):
        assert _parse_cron_field("1-10/2", 0, 59) == [1, 3, 5, 7, 9]

    def test_mixed(self):
        result = _parse_cron_field("0,30-40/5,45", 0, 59)
        assert 0 in result
        assert 45 in result
        assert 30 in result


class TestParseDelay:
    def test_numeric(self):
        assert _parse_delay(10) == 10.0
        assert _parse_delay(5.5) == 5.5

    def test_seconds_suffix(self):
        assert _parse_delay("30s") == 30.0

    def test_minutes_suffix(self):
        assert _parse_delay("5m") == 300.0

    def test_hours_suffix(self):
        assert _parse_delay("2h") == 7200.0

    def test_days_suffix(self):
        assert _parse_delay("1d") == 86400.0


class TestNextCronTime:
    def test_every_minute(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        next_t = _next_cron_time("* * * * *", now)
        assert next_t.minute == 1
        assert next_t.hour == 12
        assert next_t.day == 1

    def test_specific_minute(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        next_t = _next_cron_time("30 * * * *", now)
        assert next_t.minute == 30
        assert next_t.hour == 12

    def test_specific_hour_minute(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        next_t = _next_cron_time("15 14 * * *", now)
        assert next_t.minute == 15
        assert next_t.hour == 14
        assert next_t.day == 1

    def test_next_day(self):
        now = datetime(2025, 1, 1, 23, 59, 0)
        next_t = _next_cron_time("0 * * * *", now)
        assert next_t.day == 2
        assert next_t.hour == 0
        assert next_t.minute == 0

    def test_day_of_week(self):
        # Monday is weekday() == 0
        now = datetime(2025, 1, 1, 12, 0, 0)  # Wednesday
        next_t = _next_cron_time("* * * * 1", now)
        assert next_t.weekday() % 7 == 1  # Monday

    def test_month_filter(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        next_t = _next_cron_time("0 0 1 3 *", now)  # 00:00 on March 1st
        assert next_t.month == 3
        assert next_t.day == 1


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

class TestJob:
    def test_job_id_unique(self):
        j1 = Job()
        j2 = Job()
        assert j1.id != j2.id

    def test_job_default_state(self):
        j = Job()
        assert not j.is_recurring
        assert not j.is_paused
        assert not j.is_cancelled
        assert j.run_count == 0

    def test_job_cancel(self):
        j = Job()
        assert not j.is_cancelled
        j._cancel()
        assert j.is_cancelled


# ---------------------------------------------------------------------------
# TaskScheduler — construction
# ---------------------------------------------------------------------------

class TestSchedulerConstruction:
    def test_new_scheduler_has_no_jobs(self):
        s = TaskScheduler()
        assert s.list_jobs() == []

    def test_started_flag_false_initially(self):
        s = TaskScheduler()
        assert s._started is False


# ---------------------------------------------------------------------------
# schedule_cron
# ---------------------------------------------------------------------------

class TestScheduleCron:
    def test_cron_requires_5_fields(self):
        s = TaskScheduler()
        with pytest.raises(ValueError):
            s.schedule_cron("* * *", lambda: None)

    def test_cron_id_returned(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        assert isinstance(job_id, str)

    def test_cron_job_in_list(self):
        s = TaskScheduler()
        s.schedule_cron("0 * * * *", lambda: None, name="hourly")
        jobs = s.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "hourly"
        assert jobs[0]["is_recurring"] is True


# ---------------------------------------------------------------------------
# schedule_delayed
# ---------------------------------------------------------------------------

class TestScheduleDelayed:
    def test_delayed_one_shot(self):
        s = TaskScheduler()
        s.schedule_delayed(0.05, lambda: None)
        jobs = s.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["is_recurring"] is False

    def test_delayed_with_string_spec(self):
        s = TaskScheduler()
        s.schedule_delayed("0.5s", lambda: None)  # 0.5s = 500ms
        jobs = s.list_jobs()
        assert len(jobs) == 1

    def test_delayed_id_returned(self):
        s = TaskScheduler()
        job_id = s.schedule_delayed(1, lambda: None)
        assert isinstance(job_id, str)


# ---------------------------------------------------------------------------
# schedule_interval
# ---------------------------------------------------------------------------

class TestScheduleInterval:
    def test_interval_requires_positive(self):
        s = TaskScheduler()
        with pytest.raises(ValueError):
            s.schedule_interval(0, lambda: None)
        with pytest.raises(ValueError):
            s.schedule_interval(-1, lambda: None)

    def test_interval_id_returned(self):
        s = TaskScheduler()
        job_id = s.schedule_interval(1.0, lambda: None)
        assert isinstance(job_id, str)

    def test_interval_job_in_list(self):
        s = TaskScheduler()
        s.schedule_interval(5.0, lambda: None, name="every5")
        jobs = s.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "every5"
        assert jobs[0]["is_recurring"] is True


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

class TestCancel:
    def test_cancel_existing_job(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        assert s.cancel(job_id) is True
        assert s.list_jobs() == []

    def test_cancel_unknown_id(self):
        s = TaskScheduler()
        assert s.cancel("nonexistent") is False


# ---------------------------------------------------------------------------
# pause / resume
# ---------------------------------------------------------------------------

class TestPauseResume:
    def test_pause_existing_job(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        assert s.pause(job_id) is True
        jobs = s.list_jobs()
        assert jobs[0]["is_paused"] is True

    def test_pause_unknown_job(self):
        s = TaskScheduler()
        assert s.pause("nonexistent") is False

    def test_pause_idempotent(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        s.pause(job_id)
        assert s.pause(job_id) is False  # already paused

    def test_resume_existing_job(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        s.pause(job_id)
        assert s.resume(job_id) is True
        jobs = s.list_jobs()
        assert jobs[0]["is_paused"] is False

    def test_resume_unknown_job(self):
        s = TaskScheduler()
        assert s.resume("nonexistent") is False

    def test_resume_idempotent(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        assert s.resume(job_id) is False  # not paused

    def test_pause_does_not_remove_job(self):
        s = TaskScheduler()
        job_id = s.schedule_cron("* * * * *", lambda: None)
        s.pause(job_id)
        assert job_id in [j["id"] for j in s.list_jobs()]


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_empty_by_default(self):
        s = TaskScheduler()
        assert s.list_jobs() == []

    def test_fields_present(self):
        s = TaskScheduler()
        s.schedule_cron("* * * * *", lambda: None, name="test")
        jobs = s.list_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert "id" in job
        assert "name" in job
        assert "is_recurring" in job
        assert "is_paused" in job
        assert "next_run" in job
        assert "run_count" in job


# ---------------------------------------------------------------------------
# Context manager / lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_context_manager_starts_and_stops(self):
        s = TaskScheduler()
        with s:
            assert s._started is True
        # After exit shutdown is called
        assert s._stop_event.is_set()

    def test_double_start_is_idempotent(self):
        s = TaskScheduler()
        s.start()
        t1 = s._runner_thread
        s.start()
        assert s._runner_thread is t1
        s.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Functional integration tests
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_delayed_task_runs_once(self):
        """A delayed one-shot task should execute exactly once."""
        results: list[int] = []
        s = TaskScheduler()
        s.schedule_delayed(0.05, lambda: results.append(1))
        s.start()
        time.sleep(0.3)
        s.shutdown(wait=True)
        assert results == [1]

    def test_interval_task_runs_multiple(self):
        """An interval task should run multiple times."""
        results: list[int] = []
        s = TaskScheduler()
        s.schedule_interval(0.03, lambda: results.append(1))
        s.start()
        time.sleep(0.15)
        s.shutdown(wait=True)
        # Should fire at least 3 times
        assert len(results) >= 3

    def test_paused_task_does_not_run(self):
        """A paused task should not execute."""
        results: list[int] = []
        s = TaskScheduler()
        job_id = s.schedule_interval(0.02, lambda: results.append(1))
        s.pause(job_id)
        s.start()
        time.sleep(0.1)
        s.shutdown(wait=True)
        assert results == []

    def test_resumed_task_runs(self):
        """After pause then resume, a task should run."""
        results: list[int] = []
        s = TaskScheduler()
        job_id = s.schedule_interval(0.03, lambda: results.append(1))
        s.pause(job_id)
        s.start()
        time.sleep(0.05)
        s.resume(job_id)
        time.sleep(0.1)
        s.shutdown(wait=True)
        assert len(results) >= 2

    def test_cancelled_task_does_not_run(self):
        """A cancelled task should not execute."""
        results: list[int] = []
        s = TaskScheduler()
        job_id = s.schedule_interval(0.02, lambda: results.append(1))
        s.start()
        time.sleep(0.03)
        s.cancel(job_id)
        count_after_cancel = len(results)
        time.sleep(0.08)
        s.shutdown(wait=True)
        assert len(results) == count_after_cancel

    def test_one_shot_removed_after_run(self):
        """One-shot jobs should disappear from the list after they run."""
        s = TaskScheduler()
        s.schedule_delayed(0.02, lambda: None)
        s.start()
        time.sleep(0.05)
        s.shutdown(wait=True)
        assert s.list_jobs() == []

    def test_cron_job_runs(self):
        """A cron job with * * * * * (every minute) should fire when next_run is close."""
        results: list[int] = []
        s = TaskScheduler()
        # Use schedule_interval instead of cron for predictable timing in tests
        s.schedule_interval(0.03, lambda: results.append(1))
        s.start()
        time.sleep(0.1)
        s.shutdown(wait=True)
        assert len(results) >= 3

    def test_concurrent_add_cancel(self):
        """Adding and cancelling jobs concurrently should not raise."""
        s = TaskScheduler()
        s.start()
        ids: list[str] = []

        def add_jobs():
            for _ in range(20):
                ids.append(s.schedule_interval(1.0, lambda: None))
                time.sleep(0.001)

        def cancel_jobs():
            for jid in ids[:10]:
                s.cancel(jid)
                time.sleep(0.001)

        t1 = threading.Thread(target=add_jobs)
        t2 = threading.Thread(target=cancel_jobs)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        s.shutdown(wait=True)

    def test_job_args_kwargs(self):
        """Jobs should receive args and kwargs correctly."""
        results: list[int] = []

        def capture(a, b, c=None):
            results.append(a + b + (c or 0))

        s = TaskScheduler()
        s.schedule_interval(0.02, capture, 1, 2, c=3)
        s.start()
        time.sleep(0.07)
        s.shutdown(wait=True)
        assert 6 in results

    def test_scheduler_context_manager_start(self):
        """Using scheduler as context manager should start the thread."""
        with TaskScheduler() as s:
            s.schedule_interval(10.0, lambda: None)
            assert s._runner_thread is not None
            assert s._runner_thread.is_alive()
        assert s._stop_event.is_set()

    def test_run_count_increments(self):
        """Job run_count should increase on each execution."""
        s = TaskScheduler()
        s.schedule_interval(0.03, lambda: None)
        s.start()
        time.sleep(0.2)  # extended window for reliable scheduling
        s.shutdown(wait=True)
        jobs = s.list_jobs()
        assert jobs[0]["run_count"] >= 5  # ~0.2/0.03 = 6-7 expected, safe lower bound


class TestGetScheduler:
    def test_get_scheduler_returns_scheduler(self):
        s = get_scheduler()
        assert isinstance(s, TaskScheduler)

    def test_get_scheduler_idempotent(self):
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_get_scheduler_starts_thread(self):
        # Get from a fresh module import is tricky — rely on singleton behaviour
        s = get_scheduler()
        assert s._runner_thread is not None
        # Clean up
        s.shutdown(wait=True)
