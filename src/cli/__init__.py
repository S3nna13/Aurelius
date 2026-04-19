"""src.cli — Aurelius CLI entry points."""

# Additive exports for the benchmark runner (safe, lazy).
try:
    from .benchmark_runner import (
        BenchmarkRun,
        run_benchmark,
        format_report,
        main as benchmark_runner_main,
    )
except Exception:  # pragma: no cover - keep package importable on partial setups
    pass
