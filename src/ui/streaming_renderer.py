"""Rich Live streaming renderer for LLM output in the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, multi-tab session lifecycle), Anthropic Claude Code
(MIT, streaming output), clean-room reimplementation with original Aurelius design.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class StreamingRendererError(Exception):
    """Raised when the streaming renderer encounters malformed state or input."""


class StreamingState(enum.Enum):
    """Lifecycle state of a :class:`StreamingRenderer`."""

    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETE = "complete"
    ERROR = "error"


_STATE_STYLES: dict[StreamingState, str] = {
    StreamingState.IDLE: "dim",
    StreamingState.STREAMING: "bold cyan",
    StreamingState.PAUSED: "yellow",
    StreamingState.COMPLETE: "bold green",
    StreamingState.ERROR: "bold red",
}

_STATE_LABELS: dict[StreamingState, str] = {
    StreamingState.IDLE: "IDLE",
    StreamingState.STREAMING: "STREAMING",
    StreamingState.PAUSED: "PAUSED",
    StreamingState.COMPLETE: "COMPLETE",
    StreamingState.ERROR: "ERROR",
}


@dataclass
class TokenChunk:
    """A single token chunk emitted by the LLM streaming pipeline.

    Attributes:
        text: The decoded text of this token.
        token_id: Optional token id from the vocabulary.
        logprob: Optional log-probability for the token.
        finish_reason: Optional finish reason (e.g. ``"stop"``, ``"length"``).
    """

    text: str
    token_id: int | None = None
    logprob: float | None = None
    finish_reason: str | None = None


class StreamingRenderer:
    """Stateful Rich renderer for incremental LLM token output.

    The renderer accumulates token text in a ``list[str]`` buffer and exposes
    methods to inspect, render, and reset the stream.  It does *not* use
    :class:`rich.live.Live` (which requires a context manager) — it renders
    directly to a :class:`~rich.console.Console` on demand.

    Attributes:
        _buffer: List of token text fragments pushed via :meth:`push_chunk`.
        _state: Current :class:`StreamingState`.
        _error_message: Set when :meth:`error` is called.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._state: StreamingState = StreamingState.IDLE
        self._error_message: str | None = None

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def push_chunk(self, chunk: TokenChunk) -> None:
        """Append a single token chunk to the buffer.

        Transitions state to :attr:`StreamingState.STREAMING`.

        Args:
            chunk: The :class:`TokenChunk` to append.

        Raises:
            StreamingRendererError: If *chunk* is not a :class:`TokenChunk`.
        """
        if not isinstance(chunk, TokenChunk):
            raise StreamingRendererError(f"chunk must be a TokenChunk, got {type(chunk).__name__}")
        if self._state in (StreamingState.COMPLETE, StreamingState.ERROR):
            raise StreamingRendererError(
                f"cannot push chunk in state {self._state.value!r}; call reset() first"
            )
        self._buffer.append(chunk.text)
        self._state = StreamingState.STREAMING

    def push_chunks(self, chunks: list[TokenChunk]) -> None:
        """Batch-append multiple token chunks to the buffer.

        Args:
            chunks: A list of :class:`TokenChunk` objects.

        Raises:
            StreamingRendererError: If *chunks* is not a list.
        """
        if not isinstance(chunks, list):
            raise StreamingRendererError(f"chunks must be a list, got {type(chunks).__name__}")
        for chunk in chunks:
            self.push_chunk(chunk)

    def complete(self, finish_reason: str = "stop") -> None:
        """Transition the renderer to :attr:`StreamingState.COMPLETE`.

        Args:
            finish_reason: The finish reason reported by the model.
        """
        self._state = StreamingState.COMPLETE

    def error(self, message: str) -> None:
        """Transition the renderer to :attr:`StreamingState.ERROR`.

        Args:
            message: A human-readable description of the error.

        Raises:
            StreamingRendererError: If *message* is not a non-empty string.
        """
        if not isinstance(message, str) or not message.strip():
            raise StreamingRendererError("error message must be a non-empty string")
        self._state = StreamingState.ERROR
        self._error_message = message

    def reset(self) -> None:
        """Clear the buffer and return the renderer to :attr:`StreamingState.IDLE`."""
        self._buffer = []
        self._state = StreamingState.IDLE
        self._error_message = None

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_text(self) -> str:
        """Return the full accumulated text from the buffer."""
        return "".join(self._buffer)

    def word_count(self) -> int:
        """Return an approximate word count of the buffer text."""
        text = self.get_text()
        if not text.strip():
            return 0
        return len(text.split())

    def token_count(self) -> int:
        """Return the number of token chunks pushed (one entry per :meth:`push_chunk`)."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Render API
    # ------------------------------------------------------------------

    def render_panel(self, console: Console, title: str = "Aurelius") -> None:
        """Render the buffered text as a styled Rich Panel.

        Args:
            console: A :class:`~rich.console.Console` to print to.
            title: Panel title string.
        """
        state_label = _STATE_LABELS[self._state]
        state_style = _STATE_STYLES[self._state]
        text = self.get_text()
        if self._state == StreamingState.ERROR and self._error_message:
            body = Text(f"[ERROR] {self._error_message}", style="bold red")
        elif text:
            body = Text(text)
        else:
            body = Text("(empty)", style="dim")
        header = Text(f" [{state_label}] ", style=state_style)
        panel_title = Text.assemble(title, " ", header)
        console.print(Panel(body, title=panel_title, border_style=state_style))

    def render_live(self, console: Console) -> None:
        """Render current buffer inline — does NOT use Rich.Live.

        Renders a :class:`~rich.panel.Panel` directly to *console*, suitable
        for repeated calls in a polling loop.

        Args:
            console: A :class:`~rich.console.Console` to print to.
        """
        self.render_panel(console, title="Aurelius (live)")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Named pool of :class:`StreamingRenderer` instances.
STREAMING_RENDERER_REGISTRY: dict[str, StreamingRenderer] = {}
