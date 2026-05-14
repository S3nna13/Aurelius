"""aurelius_cli/metrics_commands.py

CLI for SRE metrics — quick demos and health summaries.
"""

from __future__ import annotations

import argparse
import json
import random

from src.monitoring.sre_metrics import SREMetricsCollector


def build_metrics_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add ``aurelius metrics …`` commands to the top-level CLI."""
    parser = subparsers.add_parser("metrics", help="SRE metrics utilities")
    sub = parser.add_subparsers(dest="metrics_cmd", required=True)

    demo = sub.add_parser("demo", help="Run a synthetic workload and print metrics")
    demo.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of synthetic requests to simulate",
    )
    demo.add_argument(
        "--error-rate",
        type=float,
        default=0.05,
        help="Fraction of requests that error (0-1)",
    )
    demo.add_argument(
        "--latency-mean",
        type=float,
        default=120.0,
        help="Mean latency in milliseconds (default 120)",
    )
    demo.add_argument(
        "--latency-std",
        type=float,
        default=40.0,
        help="Std-dev of latency in milliseconds (default 40)",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    demo.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit raw JSON instead of formatted table",
    )
    demo.set_defaults(func=_run_demo)


def _run_demo(args: argparse.Namespace) -> int:
    """Generate synthetic request data and emit a metrics report."""
    requests = getattr(args, "requests", 100)
    error_rate = getattr(args, "error_rate", 0.05)
    latency_mean = getattr(args, "latency_mean", 120.0)
    latency_std = getattr(args, "latency_std", 40.0)
    seed = getattr(args, "seed", 42)
    json_output = getattr(args, "json", False)

    # Zero-request case – avoid any collector calls
    if requests == 0:
        if json_output:
            print(
                json.dumps(
                    {
                        "total_requests": 0,
                        "errors": 0,
                        "error_rate": None,
                        "p50_latency_ms": None,
                        "p90_latency_ms": None,
                        "p99_latency_ms": None,
                        "health_score": None,
                        "saturation": None,
                    }
                )
            )
            return 0

        print(f"{'Total requests':<15}: 0")
        print(f"{'Error count':<15}: 0")
        print(f"{'Error rate':<15}: N/A")
        print(f"{'Latency p50':<15}: N/A")
        print(f"{'Latency p90':<15}: N/A")
        print(f"{'Latency p99':<15}: N/A")
        print(f"{'Health score':<15}: N/A")
        print(f"{'Saturation':<15}: N/A")
        return 0

    rng = random.Random(seed)
    collector = SREMetricsCollector()

    # Simulate some saturation samples (independent of request count)
    for _ in range(max(1, requests // 10)):
        collector.record_saturation(rng.uniform(0.4, 0.8))

    for _ in range(requests):
        is_error = rng.random() < error_rate
        latency = max(0.0, rng.gauss(latency_mean, latency_std))
        collector.record_request(latency_ms=latency, success=not is_error)

    total = collector.get_request_count()
    errors = collector.get_error_count()

    error_rate_frac = errors / total
    p50 = collector.get_percentile(50)
    p90 = collector.get_percentile(90)
    p99 = collector.get_percentile(99)
    health = collector.get_health_score()
    sat_info = collector.get_saturation_stats()

    if json_output:
        stats = {
            "total_requests": total,
            "errors": errors,
            "error_rate": error_rate_frac,
            "p50_latency_ms": p50,
            "p90_latency_ms": p90,
            "p99_latency_ms": p99,
            "health_score": health,
            "saturation": sat_info,
        }
        print(json.dumps(stats))
        return 0

    print(f"{'Total requests':<15}: {total}")
    print(f"{'Error count':<15}: {errors}")
    print(f"{'Error rate':<15}: {error_rate_frac:.2%}")
    print(f"{'Latency p50':<15}: {p50:.1f} ms")
    print(f"{'Latency p90':<15}: {p90:.1f} ms")
    print(f"{'Latency p99':<15}: {p99:.1f} ms")
    print(f"{'Health score':<15}: {health:.3f}")
    peak_val = sat_info.get("peak", 0.0)
    mean_val = sat_info.get("mean", 0.0)
    print(f"{'Saturation':<15}: peak={peak_val:.2f}, mean={mean_val:.2f}")

    return 0


def handle_metrics(args: argparse.Namespace) -> int:
    """Dispatch subcommands for ``aurelius metrics``."""
    if args.metrics_cmd == "demo":
        return _run_demo(args)
    return 1
