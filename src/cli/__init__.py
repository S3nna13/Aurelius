"""src.cli — Aurelius CLI entry points."""

# Additive exports for the benchmark runner (safe, lazy).
try:
    from .benchmark_runner import (
        BenchmarkRun as BenchmarkRun,
    )
    from .benchmark_runner import (
        format_report as format_report,
    )
    from .benchmark_runner import (
        main as main,
    )
    from .benchmark_runner import (
        run_benchmark as run_benchmark,
    )
except Exception:  # pragma: no cover - keep package importable on partial setups  # noqa: S110
    pass

try:
    from .plugin_commands import (
        PLUGIN_COMMANDS as PLUGIN_COMMANDS,
    )
    from .plugin_commands import (
        PluginCommandResult as PluginCommandResult,
    )
    from .plugin_commands import (
        PluginCommands as PluginCommands,
    )
except Exception:  # pragma: no cover  # noqa: S110
    pass

try:
    from .debug_commands import (
        DEBUG_COMMANDS as DEBUG_COMMANDS,
    )
    from .debug_commands import (
        DebugCommands as DebugCommands,
    )
    from .debug_commands import (
        DebugSnapshot as DebugSnapshot,
    )
    from .debug_commands import (
        LogLevel as LogLevel,
    )
except Exception:  # pragma: no cover  # noqa: S110
    pass

try:
    from .output_compressor import (
        DEFAULT_COMPRESSOR as DEFAULT_COMPRESSOR,
    )
    from .output_compressor import (
        OUTPUT_COMPRESSOR_REGISTRY as OUTPUT_COMPRESSOR_REGISTRY,
    )
    from .output_compressor import (
        CompressionConfig as CompressionConfig,
    )
    from .output_compressor import (
        CompressionResult as CompressionResult,
    )
    from .output_compressor import (
        OutputCompressor as OutputCompressor,
    )
except Exception:  # pragma: no cover  # noqa: S110
    pass
