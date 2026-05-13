# Example: one-shot delayed execution
# 
"""
Example: schedule a one-shot delayed job with Aurelius TaskScheduler.

Usage:
    python -m examples.schedule_delayed_job [delay_seconds]
"""

import sys
import time

from agent.task_scheduler import TaskScheduler


def reminder():
    print("[reminder] Time to stand up and stretch!")

if __name__ == "__main__":
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    sched = TaskScheduler()
    job_id = sched.schedule_delayed(delay, reminder)
    print(f"Reminder scheduled in {delay}s as {job_id}")

    sched.start()
    # Wait for job to fire
    try:
        time.sleep(delay + 2)
    except KeyboardInterrupt:
        pass
    finally:
        sched.shutdown(wait=True)