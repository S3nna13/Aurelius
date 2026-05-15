# Example: periodic interval-based background task
# 
"""
Example: schedule an interval job with Aurelius TaskScheduler.

Usage:
    python -m examples.schedule_interval_job
"""

import time

from src.agent.task_scheduler import TaskScheduler


def heartbeat():
    print("[heartbeat] tick")

if __name__ == "__main__":
    sched = TaskScheduler()
    # Run every 10 seconds
    job_id = sched.schedule_interval(10.0, heartbeat)
    print(f"Interval job {job_id} started (every 10s)")

    sched.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping scheduler...")
        sched.shutdown(wait=True)