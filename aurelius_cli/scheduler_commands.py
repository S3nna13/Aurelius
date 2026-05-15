"""aurelius_cli/scheduler_commands.py

CLI for the in-process TaskScheduler — schedule cron/interval/delayed shell
commands directly from the terminal.

Usage:
  aurelius schedule cron "<cron_expr>" -- <command> [args...]
  aurelius schedule interval <seconds> -- <command> [args...]
  aurelius schedule once <delay_seconds> -- <command> [args...]

Examples:
  # Run backup.py every day at 02:00
  aurelius schedule cron "0 2 * * *" -- python backup.py

  # Heartbeat every 30 seconds
  aurelius schedule interval 30 -- curl -X POST https://hc.io/ping

  # Delay a notification by 5 minutes
  aurelius schedule once 300 -- osascript -e 'display notification "Done!"'

The scheduler runs in the foreground until interrupted (Ctrl+C). Jobs execute
concurrently in background threads. Press Ctrl+C to stop the scheduler and
wait for in-flight jobs to finish (best-effort).
"""

from __future__ import annotations

import argparse
import sys

from src.agent.task_scheduler import TaskScheduler, _make_runner


def _parse_delay(delay: str | int | float) -> float:
    """Parse a delay spec like "30s", "5m", or a raw number (seconds)."""
    if isinstance(delay, (int, float)):
        return float(delay)
    s = str(delay).strip().lower()
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    raise ValueError(f"invalid delay format: {delay}")


def build_schedule_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "schedule",
        help="Schedule recurring/delayed shell commands or manage job store",
        description=(
            "Run shell commands on a schedule using Aurelius' in-process "
            "TaskScheduler. Subcommands create new jobs or manage existing ones. "
            "Creation commands block until Ctrl+C."
        ),
        epilog=(
            "examples:\\n"
            '  aurelius schedule cron "0 2 * * *" -- python backup.py\\n'
            "  aurelius schedule interval 60 -- curl -X POST https://hc.io/ping\\n"
            "  aurelius schedule once 300 -- say 'task complete'\\n"
            "  aurelius schedule list\\n"
            "  aurelius schedule cancel <job_id>\\n"
            "  aurelius schedule pause <job_id>\\n"
            "  aurelius schedule resume <job_id>\\n"
            "  aurelius schedule clear --yes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="schedule_cmd", required=True, help="scheduling mode")

    # cron (creation)
    cron = subs.add_parser("cron", help="Schedule a cron-expression job")
    cron.add_argument(
        "cron_expr",
        help='Cron expression (minute hour day month day-of-week). Example: "0 2 * * *"',
    )
    cron.add_argument(
        "shell_cmd",
        nargs=argparse.REMAINDER,
        help="Shell command and arguments to run",
    )

    # interval
    interval = subs.add_parser("interval", help="Schedule a repeating job every N seconds")
    interval.add_argument("seconds", type=float, help="Interval in seconds")
    interval.add_argument(
        "shell_cmd",
        nargs=argparse.REMAINDER,
        help="Shell command and arguments to run",
    )

    # once
    once = subs.add_parser("once", help="Schedule a one-shot job after a delay")
    once.add_argument("delay", type=float, help="Delay in seconds before running")
    once.add_argument(
        "shell_cmd",
        nargs=argparse.REMAINDER,
        help="Shell command and arguments to run",
    )

    # list (management)
    _ = subs.add_parser("list", help="List all scheduled jobs")

    # cancel
    cancel = subs.add_parser("cancel", help="Cancel a job by ID")
    cancel.add_argument("job_id", help="Job identifier to cancel")

    # pause
    pause = subs.add_parser("pause", help="Pause a job by ID")
    pause.add_argument("job_id", help="Job identifier to pause")

    # resume
    resume = subs.add_parser("resume", help="Resume a paused job by ID")
    resume.add_argument("job_id", help="Job identifier to resume")

    # clear
    clear = subs.add_parser("clear", help="Remove all jobs from the store")
    clear.add_argument(
        "--yes", action="store_true", default=False, help="Confirm removal without prompting"
    )


def handle_schedule(args: argparse.Namespace) -> int:
    """Dispatch scheduling mode and block until interrupted for creation,
    or perform management operations (list/cancel/pause/resume/clear)."""
    # Management commands do not require a shell command
    if args.schedule_cmd in ("cron", "interval", "once"):
        # Creation modes require a shell command
        if not args.shell_cmd:
            print("error: no command specified", file=sys.stderr)
            return 2
        runner = _make_runner(args.shell_cmd)
        sched = TaskScheduler()

        if args.schedule_cmd == "cron":
            job_id = sched.schedule_cron(args.cron_expr, runner, shell_cmd=args.shell_cmd)
            print(f" Scheduled cron job {job_id}: {args.shell_cmd}")
        elif args.schedule_cmd == "interval":
            job_id = sched.schedule_interval(args.seconds, runner, shell_cmd=args.shell_cmd)
            print(f" Scheduled interval job {job_id} every {args.seconds}s: {args.shell_cmd}")
        elif args.schedule_cmd == "once":
            job_id = sched.schedule_delayed(args.delay, runner, shell_cmd=args.shell_cmd)
            secs = _parse_delay(args.delay)
            print(f" Scheduled one-shot job {job_id} in {secs}s: {args.shell_cmd}")

        print(" Press Ctrl+C to stop the scheduler.")
        try:
            sched.start()
        except KeyboardInterrupt:
            print("\n Interrupted — stopping scheduler…")
        finally:
            sched.shutdown(wait=True)
        return 0

    # ---- Management commands ----
    sched = TaskScheduler()  # loads persisted jobs

    if args.schedule_cmd == "list":
        jobs = sched.list_jobs()
        if not jobs:
            print("No jobs scheduled.")
            return 0
        # Build a simple table
        header = f"{'ID':<8} {'Name':<20} {'Schedule':<20} {'Next Run':<19} {'Paused'}"
        print(header)
        print("-" * len(header))
        for job in jobs:
            # Determine schedule string
            if job.get("cron_expr"):
                schedule = f"cron {job['cron_expr']}"
            elif job.get("interval_secs") is not None:
                secs = job["interval_secs"]
                if job.get("is_recurring"):
                    schedule = f"every {secs}s"
                else:
                    schedule = f"once after {secs}s"
            else:
                schedule = "unknown"
            next_run = job.get("next_run", "")
            paused = "yes" if job.get("is_paused") else "no"
            job_line = (
                f"{job['id']:<8} {job['name'][:20]:<20} {schedule:<20} "
                f"{str(next_run)[:19]:<19} {paused}"
            )
        print(job_line)
        return 0

    elif args.schedule_cmd == "cancel":
        job_id = args.job_id
        if sched.cancel(job_id):
            print(f" Cancelled job {job_id}")
            return 0
        else:
            print(f"error: job {job_id} not found", file=sys.stderr)
            return 1

    elif args.schedule_cmd == "pause":
        job_id = args.job_id
        if sched.pause(job_id):
            print(f" Paused job {job_id}")
            return 0
        else:
            print(f"error: job {job_id} not found or already paused", file=sys.stderr)
            return 1

    elif args.schedule_cmd == "resume":
        job_id = args.job_id
        if sched.resume(job_id):
            print(f" Resumed job {job_id}")
            return 0
        else:
            print(f"error: job {job_id} not found or not paused", file=sys.stderr)
            return 1

    elif args.schedule_cmd == "clear":
        confirmed = args.yes or False  # we'll add a --yes flag maybe
        # For safety, require --yes to actually clear, else prompt? Simpler: always require --yes
        if confirmed:
            sched.clear()
            print(" All jobs cleared.")
            return 0
        else:
            print("error: --yes confirmation required to clear all jobs", file=sys.stderr)
            return 2

    else:
        print(f"error: unknown subcommand {args.schedule_cmd}", file=sys.stderr)
        return 2
