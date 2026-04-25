"""Log tailer utility for CLI debug and monitoring workflows."""

from __future__ import annotations

import os
import re
import time
from collections import deque
from collections.abc import Iterator


class LogTailerError(ValueError):
    """Raised when a log tailer operation is invalid or unsafe."""


class LogTailer:
    """Tail, follow, and search a log file using only the stdlib."""

    def __init__(self, log_path: str, max_lines: int = 1000) -> None:
        if ".." in log_path.split(os.sep):
            raise LogTailerError(f"Path traversal rejected: {log_path}")

        resolved = os.path.expanduser(log_path)
        if not os.path.exists(resolved):
            raise LogTailerError(f"File not found: {log_path}")
        if not os.path.isfile(resolved):
            raise LogTailerError(f"Not a file: {log_path}")
        if not os.access(resolved, os.R_OK):
            raise LogTailerError(f"File not readable: {log_path}")

        self.log_path = resolved
        self.max_lines = max_lines
        self._lines: deque[str] = deque(maxlen=max_lines)

    def _refresh(self) -> None:
        """Read the file and update the internal line cache."""
        with open(self.log_path) as f:
            self._lines.clear()
            for line in f:
                self._lines.append(line.rstrip("\n"))

    def tail(self, n: int = 10) -> list[str]:
        """Return the last *n* lines of the log file."""
        self._refresh()
        return list(self._lines)[-n:]

    def follow(self, timeout: float = 5.0) -> Iterator[str]:
        """Yield new lines as they are appended to the log file.

        Starts at the current end-of-file and polls until *timeout* seconds
        have elapsed.
        """
        with open(self.log_path) as f:
            f.seek(0, os.SEEK_END)
            start_time = time.time()

            while time.time() - start_time < timeout:
                current = f.tell()
                f.seek(0, os.SEEK_END)
                new_end = f.tell()

                if new_end < current:
                    # File was truncated; start from the beginning.
                    f.seek(0)
                else:
                    f.seek(current)

                line = f.readline()
                if line:
                    yield line.rstrip("\n")
                    continue

                time.sleep(0.05)

    def search(self, pattern: str) -> list[str]:
        """Return lines matching the given regular expression."""
        self._refresh()
        compiled = re.compile(pattern)
        return [line for line in self._lines if compiled.search(line)]

    def clear_cache(self) -> None:
        """Clear the internal line cache."""
        self._lines.clear()
