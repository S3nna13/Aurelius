# Examples

Runnable Python examples for Aurelius developer utilities.

## Requirements

- Python 3.12+ and the Aurelius virtual environment (`.venv/`).
- Run from the repository root so module imports resolve correctly.

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate  # or use your preferred method

# Run any example as a module
python -m examples.schedule_cron_job
python -m examples.pipeline_filter_map
python -m examples.sre_metrics_basic
python -m examples.sre_metrics_demo
```

## Examples

| Script | Feature | Description |
|--------|--------|-------------|
| `schedule_cron_job.py` | Task Scheduler | Schedule a cron-based recurring job (runs daily at 2 AM). |
| `schedule_interval_job.py` | Task Scheduler | Periodic job executing every N seconds. |
| `schedule_delayed_job.py` | Task Scheduler | One-shot delayed execution after a countdown. |
| `pipeline_filter_map.py` | Data Pipeline | ETL-style chaining: filter → map → sort → head/tail. |
| `sre_metrics_basic.py` | SRE Metrics | Library usage: collect latency, error, traffic, saturation. |
| `sre_metrics_demo.py` | SRE Metrics | CLI demo: simulate workload and print formatted report. |

## Notes

- Scheduler examples block indefinitely until `Ctrl+C`. This is intentional: the scheduler runs in the foreground.
- Pipeline examples use pure Python lambdas; replace with real transforms as needed.
- SRE examples simulate synthetic data; integrate with your service by calling `record_request`, `record_error`, etc. around actual request handling.
