"""Multi-turn conversation state manager.

Tracks messages, tool calls, tool results, a system prompt, and attachments.
Enforces truncation policies (oldest-first, keep-system, sliding-window) and
integrates with ``src.longcontext.context_compaction.ContextCompactor`` for
automatic summarization when a conversation exceeds its token budget.

Pure stdlib. No silent fallbacks: bad input raises.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_VALID_ROLES = frozenset({"user", "assistant", "system", "tool"})
_VALID_KINDS = frozenset({"message", "tool_call", "tool_result", "system"})


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation.

    ``kind`` partitions turns: ``message`` (normal chat), ``tool_call``
    (assistant-issued tool invocation), ``tool_result`` (tool output), or
    ``system`` (system-prompt slot, never truncated).
    """

    turn_id: int
    role: str
    content: str
    kind: str = "message"
    tool_name: str | None = None
    tool_call_id: str | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.turn_id, int):
            raise TypeError("turn_id must be int")
        if not isinstance(self.role, str) or self.role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("content must be str")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {self.kind!r}")
        if self.tool_name is not None and not isinstance(self.tool_name, str):
            raise TypeError("tool_name must be str or None")
        if self.tool_call_id is not None and not isinstance(self.tool_call_id, str):
            raise TypeError("tool_call_id must be str or None")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be float")


def _default_token_counter(s: str) -> int:
    """Cheap whitespace token counter; replace with a real tokenizer if desired."""
    if not isinstance(s, str):
        raise TypeError("token counter input must be str")
    return len(s.split())


class ConversationState:
    """Policy-driven multi-turn conversation state.

    Parameters
    ----------
    system_prompt:
        Optional string. When non-empty, a single system turn is seeded at
        construction time and is protected from truncation.
    max_turns:
        Sliding-window cap on non-system turns. Enforced by
        ``truncate_if_needed``.
    max_tokens:
        Token budget across all turns (system included). When exceeded, the
        configured ``compactor`` is used if present; otherwise oldest-first
        truncation is applied (system turns preserved).
    token_counter:
        Callable ``str -> int``. Defaults to whitespace splitting.
    compactor:
        Optional object with a ``.compact(turns: list[Turn]) -> list[Turn]``
        method (e.g. ``src.longcontext.context_compaction.ContextCompactor``).
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_turns: int = 100,
        max_tokens: int = 8000,
        token_counter: Callable[[str], int] | None = None,
        compactor: Any = None,
    ) -> None:
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be str")
        if not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive int")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive int")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable or None")
        if compactor is not None and not hasattr(compactor, "compact"):
            raise TypeError("compactor must expose a .compact() method")

        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._token_counter = token_counter or _default_token_counter
        self.compactor = compactor

        self._turns: list[ConversationTurn] = []
        self._next_id: int = 0
        self._last_ts: float = 0.0
        self.attachments: list[dict] = []

        if system_prompt:
            self._append_turn(
                role="system",
                content=system_prompt,
                kind="system",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        # Enforce strictly monotonic timestamps even on coarse clocks.
        t = time.time()
        if t <= self._last_ts:
            t = self._last_ts + 1e-6
        self._last_ts = t
        return t

    def _append_turn(
        self,
        *,
        role: str,
        content: str,
        kind: str,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            turn_id=self._next_id,
            role=role,
            content=content,
            kind=kind,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            timestamp=self._now(),
        )
        self._next_id += 1
        self._turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def append_message(
        self,
        role: str,
        content: str,
        kind: str = "message",
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> ConversationTurn:
        """Append a message turn. Invalid role/kind raises."""
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}")
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}")
        if not isinstance(content, str):
            raise TypeError("content must be str")
        return self._append_turn(
            role=role,
            content=content,
            kind=kind,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    def append_tool_call(self, name: str, arguments: dict, call_id: str) -> ConversationTurn:
        """Record an assistant-issued tool call."""
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty str")
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be dict")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("call_id must be a non-empty str")
        payload = json.dumps(
            {"name": name, "arguments": arguments, "call_id": call_id},
            sort_keys=True,
        )
        return self._append_turn(
            role="assistant",
            content=payload,
            kind="tool_call",
            tool_name=name,
            tool_call_id=call_id,
        )

    def append_tool_result(
        self, call_id: str, result: str, is_error: bool = False
    ) -> ConversationTurn:
        """Record a tool result. ``call_id`` must match a prior tool_call."""
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("call_id must be a non-empty str")
        if not isinstance(result, str):
            raise TypeError("result must be str")
        match: ConversationTurn | None = None
        for t in self._turns:
            if t.kind == "tool_call" and t.tool_call_id == call_id:
                match = t
                break
        if match is None:
            raise ValueError(
                f"no prior tool_call with call_id={call_id!r}; "
                "tool_result must be preceded by its matching tool_call"
            )
        content = json.dumps(
            {"call_id": call_id, "is_error": bool(is_error), "result": result},
            sort_keys=True,
        )
        return self._append_turn(
            role="tool",
            content=content,
            kind="tool_result",
            tool_name=match.tool_name,
            tool_call_id=call_id,
        )

    def to_messages(self) -> list[dict]:
        """Return an ordered list of ``{"role", "content", ...}`` dicts."""
        out: list[dict] = []
        for t in self._turns:
            d: dict[str, Any] = {"role": t.role, "content": t.content}
            if t.tool_name is not None:
                d["tool_name"] = t.tool_name
            if t.tool_call_id is not None:
                d["tool_call_id"] = t.tool_call_id
            d["kind"] = t.kind
            out.append(d)
        return out

    def current_tokens(self) -> int:
        return sum(self._token_counter(t.content) for t in self._turns)

    def summary_stats(self) -> dict:
        counts = {"total": 0, "user": 0, "assistant": 0, "system": 0, "tool": 0}
        kinds = {"message": 0, "tool_call": 0, "tool_result": 0, "system": 0}
        for t in self._turns:
            counts["total"] += 1
            counts[t.role] = counts.get(t.role, 0) + 1
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        return {
            "counts_by_role": counts,
            "counts_by_kind": kinds,
            "tokens": self.current_tokens(),
            "num_turns": len(self._turns),
            "attachments": len(self.attachments),
        }

    def add_attachment(self, attachment: dict) -> None:
        if not isinstance(attachment, dict):
            raise TypeError("attachment must be a dict")
        self.attachments.append(attachment)

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------
    def truncate_if_needed(self) -> None:
        """Apply sliding-window + token-budget policies.

        1. Enforce ``max_turns`` over non-system turns (oldest-first drop).
        2. If still over ``max_tokens``, call ``compactor.compact`` if wired,
           else drop oldest non-system turns until within budget.
        """
        # Sliding window: bound non-system turns.
        non_system = [t for t in self._turns if t.kind != "system"]
        overflow = len(non_system) - self.max_turns
        if overflow > 0:
            to_drop = set()
            dropped = 0
            for t in self._turns:
                if dropped >= overflow:
                    break
                if t.kind != "system":
                    to_drop.add(t.turn_id)
                    dropped += 1
            self._turns = [t for t in self._turns if t.turn_id not in to_drop]

        # Token budget.
        if self.current_tokens() <= self.max_tokens:
            return

        if self.compactor is not None:
            try:
                from src.longcontext.context_compaction import Turn as _CompTurn
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "compactor supplied but context_compaction module not importable"
                ) from e
            comp_turns = [
                _CompTurn(role=t.role, content=t.content, kind=t.kind) for t in self._turns
            ]
            new_turns = self.compactor.compact(comp_turns)
            # Re-seed internal turn list preserving role/kind/content; assign
            # fresh monotonic turn_ids but keep relative order.
            rebuilt: list[ConversationTurn] = []
            for ct in new_turns:
                rebuilt.append(
                    ConversationTurn(
                        turn_id=self._next_id,
                        role=ct.role,
                        content=ct.content,
                        kind=ct.kind,
                        timestamp=self._now(),
                    )
                )
                self._next_id += 1
            self._turns = rebuilt

        # Fallback (or post-compaction safety): drop oldest non-system until
        # within budget.
        while self.current_tokens() > self.max_tokens:
            idx = next(
                (i for i, t in enumerate(self._turns) if t.kind != "system"),
                None,
            )
            if idx is None:
                break
            self._turns.pop(idx)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)


__all__ = ["ConversationTurn", "ConversationState"]
