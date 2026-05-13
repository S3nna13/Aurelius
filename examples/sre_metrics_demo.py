"""Example: SRE metrics collector — simulate workload and print health score.

Usage:
    python examples/sre_metrics_demo.py [--requests N] [--error-rate F] [--seed S]

The script records request latencies (Gaussian distributed) and error flags
into an SREMetricsCollector, then prints a formatted report identical to the
`aurelius metrics demo` command output.

Run with:
    .venv/bin/python examples/sre_metrics_demo.py --requests 200 --error-rate 0.1
"""

from __future__ import annotations

import argparse
import random
import sys

from src.monitoring.sre_metrics import SREMetricsCollector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Error fraction 0–1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--latency-mean", type=float, default=120.0, help="Mean latency ms")
    parser.add_argument("--latency-std", type=float, default=40.0, help="Std-dev latency ms")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    collector = SREMetricsCollector()

    for _ in range(args.requests):
        err = rng.random() < args.error_rate
        latency = max(0.0, rng.gauss(args.latency_mean, args.latency_std))
        collector.record_request(latency_ms=latency, success=not err)

    total = collector.get_request_count()
    errors = collector.get_error_count()
    if total == 0:
        print("No requests — nothing to report.")
        return 0
    print(f"  Total requests : {total}")
    print(f"  Errors         : {errors} ({errors/total:.2%})")
    p50 = collector.get_percentile(50)
    p90 = collector.get_percentile(90)
    p99 = collector.get_percentile(99)
    print(f"  Latency  p50   : {p50:.1f} ms")
    print(f"  Latency  p90   : {p90:.1f} ms")
    print(f"  Latency  p99   : {p99:.1f} ms")
    print(f"  Health score  : {collector.get_health_score():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

