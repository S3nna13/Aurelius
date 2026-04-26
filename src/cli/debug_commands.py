"""CLI debug commands: log level control, trace toggling, memory snapshot."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DebugSnapshot:
    timestamp: str
    log_level: str
    traces_enabled: bool
    metrics: dict


class DebugCommands:
    def __init__(self) -> None:
        self._log_level: LogLevel = LogLevel.INFO
        self._traces_enabled: bool = False
        self._metrics: dict = {}

    def set_log_level(self, level: str) -> bool:
        try:
            self._log_level = LogLevel(level.lower())
            return True
        except ValueError:
            return False

    def get_log_level(self) -> str:
        return self._log_level.value

    def enable_traces(self) -> None:
        self._traces_enabled = True

    def disable_traces(self) -> None:
        self._traces_enabled = False

    def traces_enabled(self) -> bool:
        return self._traces_enabled

    def record_metric(self, key: str, value: float) -> None:
        self._metrics[key] = value

    def snapshot(self) -> DebugSnapshot:
        return DebugSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            log_level=self._log_level.value,
            traces_enabled=self._traces_enabled,
            metrics=dict(self._metrics),
        )

    def reset(self) -> None:
        self._log_level = LogLevel.INFO
        self._traces_enabled = False
        self._metrics = {}


DEBUG_COMMANDS = DebugCommands()
