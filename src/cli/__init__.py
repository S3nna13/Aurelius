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

try:
    from .plugin_commands import (
        PluginCommandResult,
        PluginCommands,
        PLUGIN_COMMANDS,
    )
except Exception:  # pragma: no cover
    pass

try:
    from .debug_commands import (
        LogLevel,
        DebugSnapshot,
        DebugCommands,
        DEBUG_COMMANDS,
    )
except Exception:  # pragma: no cover
    pass

try:
    from .output_compressor import (
        CompressionConfig,
        CompressionResult,
        OutputCompressor,
        DEFAULT_COMPRESSOR,
        OUTPUT_COMPRESSOR_REGISTRY,
    )
except Exception:  # pragma: no cover
    pass
