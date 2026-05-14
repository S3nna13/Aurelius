"""Server-Sent Events (SSE) wire encoding for streaming completions.

Implements the framing rules from the HTML Living Standard / WHATWG SSE
section (``text/event-stream``): ``event:``, ``data:``, optional ``id:``,
and records terminated by a blank line.

This is a **pure encoder** — no sockets, no asyncio — so inference workers can
share one implementation.
"""

from __future__ import annotations


class SSEStreamEncoder:
    """Encode logical chunks into UTF-8 SSE frames."""

    def __init__(self, *, utf8_errors: str = "replace") -> None:
        if utf8_errors not in ("strict", "replace", "ignore", "surrogateescape"):
            raise ValueError(f"unsupported utf8_errors={utf8_errors!r}")
        self._utf8_errors = utf8_errors

    def encode_event(
        self,
        *,
        data: str,
        event: str | None = None,
        event_id: str | None = None,
    ) -> bytes:
        """Return one complete SSE message (including trailing blank line)."""
        if not isinstance(data, str):
            raise TypeError("data must be str")
        if event is not None and not isinstance(event, str):
            raise TypeError("event must be str or None")
        if event_id is not None and not isinstance(event_id, str):
            raise TypeError("event_id must be str or None")

        lines: list[str] = []
        if event is not None:
            _reject_if_newline("event", event)
            lines.append(f"event: {event}")
        if event_id is not None:
            _reject_if_newline("id", event_id)
            lines.append(f"id: {event_id}")

        for part in data.split("\n"):
            _reject_if_newline("data_line", part)
            lines.append(f"data: {part}")

        lines.append("")  # blank line terminator
        text = "\n".join(lines) + "\n"
        return text.encode("utf-8", errors=self._utf8_errors)

    def encode_comment(self, comment: str) -> bytes:
        """Encode a keep-alive / diagnostic comment line (no blank-line end)."""
        if not isinstance(comment, str):
            raise TypeError("comment must be str")
        for line in comment.split("\n"):
            _reject_if_newline("comment_line", line)
        body = "\n".join(f": {line}" for line in comment.split("\n"))
        return (body + "\n").encode("utf-8", errors=self._utf8_errors)


def _reject_if_newline(field: str, value: str) -> None:
    if "\r" in value or "\n" in value:
        raise ValueError(f"{field} must not contain CR/LF — refusing ambiguous framing")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL bytes")


def split_sse_records(blob: bytes) -> list[bytes]:
    """Split a byte blob on SSE record separators ``\\n\\n`` (test helper)."""
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("blob must be bytes")
    raw = bytes(blob).replace(b"\r\n", b"\n")
    parts = raw.split(b"\n\n")
    return [p for p in parts if p.strip() != b""]


__all__ = ["SSEStreamEncoder", "split_sse_records"]
