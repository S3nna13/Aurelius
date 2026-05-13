# Example: scheduling a cron job with TaskScheduler
# Demonstrates how to run a Python function every day at 2am.


"""
Example: schedule a cron job with Aurelius TaskScheduler.

Usage:
    python -m examples.schedule_cron_job

Then press Ctrl+C to stop the scheduler.
"""

import time

from agent.task_scheduler import TaskScheduler


def backup_job():
    """Simulate a backup operation."""
    print("[backup_job] Running daily backup...")
    # Insert real backup logic here

if __name__ == "__main__":
    sched = TaskScheduler()
    # Schedule to run every day at 02:00
    job_id = sched.schedule_cron("0 2 * * *", backup_job)
    print(f"Scheduled backup_job as {job_id}")

    sched.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down scheduler...")
        sched.shutdown(wait=True)