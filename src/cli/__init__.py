"""src.cli — Aurelius CLI entry points."""

import logging

__all__ = [
    "BenchmarkRun",
    "CompressionConfig",
    "CompressionResult",
    "DEBUG_COMMANDS",
    "DebugCommands",
    "DebugSnapshot",
    "DEFAULT_COMPRESSOR",
    "LogLevel",
    "OUTPUT_COMPRESSOR_REGISTRY",
    "OutputCompressor",
    "PLUGIN_COMMANDS",
    "PluginCommandResult",
    "PluginCommands",
    "format_report",
    "main",
    "run_benchmark",
]

_logger = logging.getLogger(__name__)

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
except ImportError:
    _logger.warning("benchmark_runner not available")

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
except ImportError:
    _logger.warning("plugin_commands not available")

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
except ImportError:
    _logger.warning("debug_commands not available")

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
except ImportError:
    _logger.warning("output_compressor not available")
