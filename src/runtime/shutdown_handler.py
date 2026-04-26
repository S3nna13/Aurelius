"""Graceful shutdown handler with drain and timeout."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ShutdownHandler:
    """Handle graceful shutdown with drain period and timeout."""

    drain_seconds: float = 5.0
    timeout_seconds: float = 10.0
    _handlers: list[Callable[[], None]] = field(default_factory=list, repr=False)
    _shutdown_started: bool = False

    def register(self, handler: Callable[[], None]) -> None:
        self._handlers.append(handler)

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        deadline = time.monotonic() + self.timeout_seconds
        for handler in self._handlers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                handler()
            except Exception:  # noqa: S110
                pass

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())
        signal.signal(signal.SIGINT, lambda *_: self.shutdown())


SHUTDOWN_HANDLER = ShutdownHandler()
