"""Preserve Thinking — cross-turn reasoning state replay.

Implements K2.6-inspired "preserve thinking": a ring-buffer of thinking
snapshots whose tokens can be prepended to future turns so the orchestrator
does not re-derive long reasoning chains from scratch.

Reference: Kimi K2.6 long-horizon agent, 98,304-token thinking budget,
50+ step tasks.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThinkingSnapshot:
    """A single captured reasoning chain plus its optional compressed form."""

    turn_id: int
    thinking_tokens: list[int]      # token IDs of the thinking chain (possibly truncated)
    summary_tokens: list[int]       # compressed/summary representation (shorter)
    timestamp: float                # time.time() at snapshot creation
    metadata: dict                  # arbitrary key-value tags


@dataclass
class PreserveThinkingConfig:
    """Configuration for :class:`PreserveThinkingBuffer`."""

    max_snapshots: int = 8
    max_tokens_per_snapshot: int = 4096
    use_summary: bool = False
    max_prepend_tokens: int = 2048


class PreserveThinkingBuffer:
    """Ring-buffer of thinking snapshots with token-prepend support.

    When the buffer reaches ``config.max_snapshots`` capacity the oldest
    snapshot is evicted to make room.  The buffer can return either the full
    thinking chain or its summary depending on ``config.use_summary``.
    """

    def __init__(self, config: Optional[PreserveThinkingConfig] = None) -> None:
        self._config = config if config is not None else PreserveThinkingConfig()
        self._buffer: deque[ThinkingSnapshot] = deque()
        self._eviction_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_snapshot(
        self,
        turn_id: int,
        thinking_tokens: list[int],
        summary_tokens: Optional[list[int]] = None,
        metadata: Optional[dict] = None,
    ) -> ThinkingSnapshot:
        """Add a new snapshot to the ring buffer.

        Truncates *thinking_tokens* to ``config.max_tokens_per_snapshot``
        before storing.  Evicts the oldest snapshot if the buffer is at
        capacity.

        Returns the stored :class:`ThinkingSnapshot`.
        """
        truncated = thinking_tokens[: self._config.max_tokens_per_snapshot]
        snap = ThinkingSnapshot(
            turn_id=turn_id,
            thinking_tokens=truncated,
            summary_tokens=list(summary_tokens) if summary_tokens is not None else [],
            timestamp=time.time(),
            metadata=dict(metadata) if metadata is not None else {},
        )

        if len(self._buffer) >= self._config.max_snapshots:
            self._buffer.popleft()
            self._eviction_count += 1

        self._buffer.append(snap)
        return snap

    def get_prepend_tokens(self, max_tokens: Optional[int] = None) -> list[int]:
        """Return tokens to prepend to the next turn's context.

        Uses the most recent snapshot.  If ``config.use_summary`` is True the
        summary tokens are returned; otherwise the full thinking tokens are
        returned.  If ``summary_tokens`` is empty and ``use_summary`` is True
        the method falls back to the full thinking tokens.

        The result is truncated to *max_tokens* (if given) or
        ``config.max_prepend_tokens``.
        """
        if not self._buffer:
            return []

        latest = self._buffer[-1]

        if self._config.use_summary and latest.summary_tokens:
            tokens = latest.summary_tokens
        else:
            tokens = latest.thinking_tokens

        limit = max_tokens if max_tokens is not None else self._config.max_prepend_tokens
        return tokens[:limit]

    def get_all_snapshots(self) -> list[ThinkingSnapshot]:
        """Return all snapshots in insertion order (oldest first)."""
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all snapshots from the buffer (eviction count is unaffected)."""
        self._buffer.clear()

    def eviction_count(self) -> int:
        """Total number of snapshots evicted since the buffer was created."""
        return self._eviction_count

    def __len__(self) -> int:
        return len(self._buffer)
