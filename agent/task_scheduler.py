"""Kronos-style cron scheduler: recurring jobs, delayed tasks, background thread execution."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


def _make_runner(cmd_list: list[str]) -> Callable[[], None]:
    """Return a zero-arg callable that launches `cmd_list` via subprocess.Popen."""

    def _run() -> None:
        try:
            subprocess.Popen(  # noqa: S603 — intentional shell command execution
                cmd_list,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[scheduler] failed to start {cmd_list}: {exc}", file=sys.stderr)

    return _run


# ---------------------------------------------------------------------------
# Cron expression helpers
# ---------------------------------------------------------------------------

CRON_FIELDS = ["minute", "hour", "day", "month", "day_of_week"]


def _parse_cron_field(value: str, lo: int, hi: int) -> list[int]:
    """Parse a single cron field into a sorted list of integers."""
    result: list[int] = []
    for part in value.split(","):
        if "/" in part:
            base, step = part.split("/", 1)
            step = int(step)
            if base == "*":
                start, stop = lo, hi
            elif "-" in base:
                start, stop = map(int, base.split("-", 1))
            else:
                start = int(base)
                stop = hi
            for v in range(start, stop + 1, step):
                if lo <= v <= hi:
                    result.append(v)
        elif "-" in part:
            start, stop = map(int, part.split("-", 1))
            for v in range(start, stop + 1):
                if lo <= v <= hi:
                    result.append(v)
        elif part == "*":
            for v in range(lo, hi + 1):
                result.append(v)
        else:
            v = int(part)
            if lo <= v <= hi:
                result.append(v)
    return sorted(set(result))


def _next_cron_time(
    expr: str | list[str], after: datetime, tz: datetime.tzinfo | None = None
) -> datetime:
    """Return the next datetime matching *expr* strictly after *after*.

    Args:
        expr: Either a whitespace-separated cron string or a pre-split list
              of the 5 fields [minute, hour, day, month, day_of_week].
    """
    if isinstance(expr, str):
        fields = expr.split()
    else:
        fields = expr
    minute = _parse_cron_field(fields[0], 0, 59)
    hour = _parse_cron_field(fields[1], 0, 23)
    day = _parse_cron_field(fields[2], 1, 31)
    month = _parse_cron_field(fields[3], 1, 12)
    dow = _parse_cron_field(fields[4], 0, 6)

    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    for _ in range(366 * 24 * 60):  # safety bound
        if (
            candidate.month in month
            and candidate.day in day
            and candidate.hour in hour
            and candidate.minute in minute
            and (candidate.weekday() % 7) in dow
        ):
            if tz is not None:
                candidate = candidate.astimezone(tz)
            return candidate
        candidate += timedelta(minutes=1)

    raise RuntimeError("Cannot find next cron time within one year")


def _parse_delay(value: str | int | float) -> float:
    """Convert a delay spec into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value.endswith("s"):
        return float(value[:-1])
    if value.endswith("m"):
        return float(value[:-1]) * 60
    if value.endswith("h"):
        return float(value[:-1]) * 3600
    if value.endswith("d"):
        return float(value[:-1]) * 86400
    return float(value)


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    cron_expr: str | None = None  # None = one-shot delayed task
    interval_secs: float | None = None  # alternative to cron
    func: Callable[[], None] | None = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    next_run: datetime | None = None
    last_run: datetime | None = None
    run_count: int = 0
    is_recurring: bool = False
    is_paused: bool = False
    is_cancelled: bool = False
    shell_cmd: list[str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _cancel(self) -> None:
        with self._lock:
            self.is_cancelled = True


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """
    Kronos-style task scheduler.

    Methods
    -------
    schedule_cron(cron_expr, func, *args, name="", **kwargs)
        Add a recurring job parsed from a 5-field cron expression.
    schedule_delayed(delay, func, *args, name="", **kwargs)
        Add a one-shot task that runs once after *delay* seconds.
    schedule_interval(secs, func, *args, name="", **kwargs)
        Add a recurring job that runs every *secs* seconds.
    cancel(job_id)
        Cancel and remove a job.
    pause(job_id)
        Pause a job.
    resume(job_id)
        Resume a paused job.
    list_jobs()
        Return a list of job descriptors (dict).
    shutdown(wait=True)
        Stop the scheduler and optionally wait for running jobs.
    """

    _instance_lock = threading.Lock()

    def __init__(self, store_path: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._runner_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._idle_event = threading.Event()
        self._wake_event = threading.Event()  # 用于唤醒睡眠循环以响应状态变更
        self._jobs_lock = threading.Lock()
        self._started = False
        self._store_path = (
            Path(store_path) if store_path else Path.home() / ".cache" / "aurelius" / "jobs.json"
        )
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_store()

    # -----------------------------------------------------------------------
    # Job management (thread-safe)
    # -----------------------------------------------------------------------

    def _add_job(self, job: Job) -> str:
        with self._jobs_lock:
            self._jobs[job.id] = job
        return job.id

    def _record_for_job(self, job: Job) -> dict:
        """Serialize a job into a JSON-serializable record."""
        rec: dict = {
            "id": job.id,
            "name": job.name,
            "is_recurring": job.is_recurring,
            "is_paused": job.is_paused,
            "is_cancelled": job.is_cancelled,
            "next_run": job.next_run.isoformat() if job.next_run else None,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "run_count": job.run_count,
        }
        if job.cron_expr:
            rec["cron_expr"] = job.cron_expr
        if job.interval_secs is not None:
            rec["interval_secs"] = job.interval_secs
        if job.shell_cmd:
            rec["shell_cmd"] = job.shell_cmd
        return rec

    def _load_store(self) -> None:
        """Load persisted jobs from the store file."""
        if not self._store_path.exists():
            return
        try:
            with open(self._store_path) as f:
                records = json.load(f)
        except Exception:
            return  # corrupted or unreadable; ignore

        for rec in records:
            try:
                # Parse mandatory fields
                job = Job(
                    id=rec["id"],
                    name=rec.get("name", ""),
                    cron_expr=rec.get("cron_expr"),
                    interval_secs=rec.get("interval_secs"),
                    func=None,  # will set below
                    args=(),
                    kwargs={},
                    next_run=(
                        datetime.fromisoformat(rec["next_run"]) if rec.get("next_run") else None
                    ),
                    last_run=(
                        datetime.fromisoformat(rec["last_run"]) if rec.get("last_run") else None
                    ),
                    run_count=rec.get("run_count", 0),
                    is_recurring=rec.get("is_recurring", False),
                    is_paused=rec.get("is_paused", False),
                    is_cancelled=rec.get("is_cancelled", False),
                    shell_cmd=rec.get("shell_cmd"),
                )
                # Recreate runner if needed
                if job.shell_cmd:
                    job.func = _make_runner(job.shell_cmd)
                # Add directly without triggering another save
                with self._jobs_lock:
                    self._jobs[job.id] = job
            except Exception:  # noqa: S112
                continue

    def _save_store(self) -> None:
        """Write all persisted jobs to the store file."""
        records = [self._record_for_job(job) for job in self._jobs.values() if job.shell_cmd]
        try:
            with open(self._store_path, "w") as f:
                json.dump(records, f, indent=2, default=str)
        except OSError:
            pass  # best-effort

    def clear(self) -> None:
        """Cancel all jobs and clear the persisted store."""
        with self._jobs_lock:
            self._jobs.clear()
        self._save_store()
        self._wake_event.set()

    def schedule_cron(
        self,
        cron_expr: str,
        func: Callable[[], None],
        *args,
        name: str = "",
        shell_cmd: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Add a recurring job from a 5-field cron expression (min hour day month dow)."""
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError("cron_expr must have 5 fields: minute hour day month day_of_week")
        job = Job(
            name=name or f"cron:{cron_expr}",
            cron_expr=cron_expr,
            func=func,
            args=args,
            kwargs=kwargs,
            is_recurring=True,
            next_run=_next_cron_time(parts, datetime.now()),
        )
        job.shell_cmd = shell_cmd
        job_id = self._add_job(job)
        self._save_store()
        self._wake_event.set()
        return job_id

    def schedule_delayed(
        self,
        delay: str | int | float,
        func: Callable[[], None],
        *args,
        name: str = "",
        shell_cmd: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Add a one-shot task that runs once after *delay* (e.g. '30s', '5m', 10)."""
        secs = _parse_delay(delay)
        job = Job(
            name=name or f"delayed:{secs}s",
            interval_secs=secs,
            func=func,
            args=args,
            kwargs=kwargs,
            is_recurring=False,
            next_run=datetime.now() + timedelta(seconds=secs),
        )
        job.shell_cmd = shell_cmd
        job_id = self._add_job(job)
        self._save_store()
        self._wake_event.set()
        return job_id

    def schedule_interval(
        self,
        secs: float,
        func: Callable[[], None],
        *args,
        name: str = "",
        shell_cmd: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Add a recurring job that runs every *secs* seconds."""
        if secs <= 0:
            raise ValueError("interval must be positive")
        job = Job(
            name=name or f"interval:{secs}s",
            interval_secs=secs,
            func=func,
            args=args,
            kwargs=kwargs,
            is_recurring=True,
            next_run=datetime.now() + timedelta(seconds=secs),
        )
        job.shell_cmd = shell_cmd
        job_id = self._add_job(job)
        self._save_store()
        self._wake_event.set()
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Remove a job. Returns True if the job existed."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job._cancel()
            del self._jobs[job_id]
        self._save_store()
        self._wake_event.set()
        return True

    def pause(self, job_id: str) -> bool:
        """Pause a job. Returns True if the job was found and not already paused."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            with job._lock:
                if job.is_paused:
                    return False
                job.is_paused = True
        self._save_store()
        self._wake_event.set()
        return True

    def resume(self, job_id: str) -> bool:
        """Resume a paused job. Returns True if the job was found and was paused."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            with job._lock:
                if not job.is_paused:
                    return False
                job.is_paused = False
        self._save_store()
        self._wake_event.set()
        return True

    def list_jobs(self) -> list[dict]:
        """Return a list of job descriptors including scheduling details."""
        with self._jobs_lock:
            result = []
            for job in list(self._jobs.values()):
                with job._lock:
                    rec: dict = {
                        "id": job.id,
                        "name": job.name,
                        "is_recurring": job.is_recurring,
                        "is_paused": job.is_paused,
                        "is_cancelled": job.is_cancelled,
                        "next_run": job.next_run.isoformat() if job.next_run else None,
                        "last_run": job.last_run.isoformat() if job.last_run else None,
                        "run_count": job.run_count,
                    }
                    if job.cron_expr:
                        rec["cron_expr"] = job.cron_expr
                    if job.interval_secs is not None:
                        rec["interval_secs"] = job.interval_secs
                    # delay/one-shot doesn't need a separate field; is_recurring=False
                    result.append(rec)
            return result

    # -----------------------------------------------------------------------
    # Background runner thread
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler thread (idempotent)."""
        with self._instance_lock:
            if self._started:
                return
            self._started = True
        self._stop_event.clear()
        self._idle_event.set()
        self._runner_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="KronosScheduler"
        )
        self._runner_thread.start()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._idle_event.set()
            # Collect jobs to run now
            now = datetime.now()
            to_run: list[Job] = []

            with self._jobs_lock:
                for job in list(self._jobs.values()):
                    with job._lock:
                        if job.is_cancelled or job.is_paused:
                            continue
                        if job.next_run and job.next_run <= now:
                            to_run.append(job)

            for job in to_run:
                # Mark last_run before execution
                with job._lock:
                    job.last_run = datetime.now()
                    job.run_count += 1
                self._execute_job(job)
                # Decide reschedule
                with job._lock:
                    if job.is_cancelled:
                        continue
                    if job.is_recurring:
                        if job.cron_expr:
                            job.next_run = _next_cron_time(job.cron_expr.split(), datetime.now())
                        elif job.interval_secs:
                            # maintain fixed interval schedule
                            job.next_run += timedelta(seconds=job.interval_secs)
                    else:
                        # One-shot: mark cancelled after first run
                        job.is_cancelled = True

            with self._jobs_lock:
                # Remove cancelled jobs
                self._jobs = {jid: j for jid, j in self._jobs.items() if not j.is_cancelled}
            with self._jobs_lock:
                # Remove cancelled jobs
                self._jobs = {jid: j for jid, j in self._jobs.items() if not j.is_cancelled}

            # Persist store after one-shot jobs are cleaned up
            self._save_store()
            print("[loop] after save_store")

            # Sleep until next job or stop signal
            self._sleep_until_next()

    def _execute_job(self, job: Job) -> None:
        """Execute a job's func in a background thread."""
        if job.func is None:
            return
        t = threading.Thread(
            target=self._safe_run,
            args=(job.func, job.args, job.kwargs, job.id),
            daemon=True,
            name=f"Job-{job.id}",
        )
        t.start()

    def _safe_run(self, func: Callable[[], None], args: tuple, kwargs: dict, job_id: str) -> None:
        try:
            func(*args, **kwargs)
        except Exception:
            # Swallow exceptions to keep the scheduler alive
            import traceback

            traceback.print_exc()

    def _sleep_until_next(self) -> None:
        """Sleep until the next job is due, the stop event fires, or a wake signal arrives."""
        with self._jobs_lock:
            next_times = [
                job.next_run
                for job in self._jobs.values()
                if job.next_run and not job.is_cancelled and not job.is_paused
            ]
        if not next_times:
            # No active jobs; wait for stop or a state change (wake)
            remaining = 1.0
        else:
            next_dt = min(next_times)
            remaining = (next_dt - datetime.now()).total_seconds()
            if remaining <= 0:
                remaining = 0.01
        # Clear any stale wake signal before waiting
        self._wake_event.clear()
        # Wait for either stop event or wake event, whichever comes first
        # Use wait on a combined condition? We'll wait on wake_event with timeout,
        # but also return early if stop_event is set (checked after wake).
        self._wake_event.wait(timeout=max(0.0, min(remaining, 60.0)))

        # On wake (either by timeout, stop, or explicit signal), loop will re-evaluate.

    def shutdown(self, wait: bool = True) -> None:
        """Stop the scheduler. If wait=True, block until the run loop exits."""
        self._stop_event.set()
        self._wake_event.set()  # wake the loop if it's sleeping
        if self._runner_thread and wait:
            self._runner_thread.join(timeout=5.0)

    # For convenience: start immediately when instantiated
    def __enter__(self) -> TaskScheduler:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_scheduler: TaskScheduler | None = None
_default_lock = threading.Lock()


def get_scheduler() -> TaskScheduler:
    """Return the module-level default scheduler (lazily created)."""
    global _default_scheduler
    with _default_lock:
        if _default_scheduler is None:
            _default_scheduler = TaskScheduler()
            _default_scheduler.start()
        elif _default_scheduler._stop_event.is_set():
            # Previous instance was shut down — create a fresh one
            _default_scheduler = TaskScheduler()
            _default_scheduler.start()
        return _default_scheduler
