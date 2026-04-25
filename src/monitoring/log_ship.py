"""Log shipping for remote aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LogDestination(Protocol):
    """Protocol for log shipping destinations."""
    def send(self, records: list[dict]) -> None: ...


@dataclass
class LogRecord:
    """A single log record to ship."""
    timestamp: str
    level: str
    message: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LogShipper:
    """Shipper that batches and sends log records."""
    destination: LogDestination
    batch_size: int = 100
    _buffer: list[LogRecord] = field(default_factory=list, repr=False)

    def ship(self, record: LogRecord) -> None:
        """Queue a log record for batching and shipping."""
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Send all buffered records."""
        if not self._buffer:
            return
        records = [{
            "timestamp": r.timestamp,
            "level": r.level,
            "message": r.message,
            "source": r.source,
            "metadata": r.metadata,
        } for r in self._buffer]
        self.destination.send(records)
        self._buffer.clear()


@dataclass
class StdoutLogDestination:
    """Log destination writing to stdout."""
    def send(self, records: list[dict]) -> None:
        import json
        for r in records:
            print(json.dumps(r))


STDOUT_DESTINATION = StdoutLogDestination()
LOG_SHIPPER = LogShipper(destination=STDOUT_DESTINATION)