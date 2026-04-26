"""Context compaction for agent loops and long conversations.

Model-agnostic: callers supply a ``summarize_fn`` (list[Turn] -> str) and a
``token_counter`` (str -> int). This module enforces a token budget by
compacting older turns while preserving:

* the system prompt (optional but default ON),
* the last ``keep_last_n`` turns verbatim,
* a slot-filled "facts" prefix holding numeric / named-entity facts mined
  from the dropped turns (regex-only; no ML).

Policies
--------
- ``oldest_first``: fold ``turns[:-keep_last_n]`` into a single summary turn.
- ``middle_biased``: keep head (system) and tail verbatim, summarize middle.
- ``tool_output_aggregated``: summarize tool_result turns separately from
  message turns so tool citations stay clustered.

A hash cache keyed by ``(role, content)`` prevents re-summarizing turns that
have already been compacted in a previous call.

Pure stdlib. No silent fallbacks: bad input raises.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

_VALID_KINDS = frozenset({"message", "tool_call", "tool_result", "system"})
_VALID_POLICIES = frozenset({"oldest_first", "middle_biased", "tool_output_aggregated"})


@dataclass
class Turn:
    """A single conversation turn.

    ``kind`` partitions turns for policy dispatch. ``system`` turns are
    protected from compaction when the compactor's ``keep_system`` is True.
    """

    role: str
    content: str
    kind: str = "message"

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise TypeError(f"Turn.role must be str, got {type(self.role).__name__}")
        if not isinstance(self.content, str):
            raise TypeError(f"Turn.content must be str, got {type(self.content).__name__}")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Turn.kind must be one of {sorted(_VALID_KINDS)}, got {self.kind!r}")


# Fact-extraction regexes. Deterministic, documented.
_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # key=value or key: value (digits or quoted/bareword)
    ("kv", re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([\"']?[\w\-./]+[\"']?)")),
    # "port 8080", "version 3.14"
    (
        "labeled_num",
        re.compile(r"\b(port|version|id|count|limit|timeout)\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    ),
    # URLs
    ("url", re.compile(r"https?://[^\s)>\]]+")),
    # File paths
    ("path", re.compile(r"(?:^|\s)(/[A-Za-z0-9_./\-]+|[A-Za-z]:\\[\w\\\.\-]+)")),
    # Capitalized multi-word named entities (simple heuristic: 2+ capitalized tokens)
    ("entity", re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")),
)


def _fact_key(kind: str, match: tuple) -> str:
    if kind == "kv":
        return f"{match[0]}={match[1]}"
    if kind == "labeled_num":
        return f"{match[0].lower()} {match[1]}"
    if kind == "entity":
        return match[0] if isinstance(match, tuple) else match
    return match if isinstance(match, str) else match[0]


def _hash_turn(turn: Turn) -> str:
    h = hashlib.sha256()
    h.update(turn.role.encode("utf-8"))
    h.update(b"\x1f")
    h.update(turn.kind.encode("utf-8"))
    h.update(b"\x1f")
    h.update(turn.content.encode("utf-8"))
    return h.hexdigest()


def _hash_group(turns: Iterable[Turn]) -> str:
    h = hashlib.sha256()
    for t in turns:
        h.update(_hash_turn(t).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class ContextCompactor:
    """Budget-aware, policy-driven context compactor.

    Parameters
    ----------
    summarize_fn:
        Callable mapping a list of Turns to a summary string. The callable is
        invoked at most once per unique ``(role, content)`` group (hash cache).
    token_counter:
        Callable mapping a string to its token count. Used to measure budget
        adherence; the compactor does not assume a particular tokenizer.
    target_tokens:
        Soft budget. Compaction runs when ``current_tokens > target_tokens``.
    keep_last_n:
        Tail turns preserved verbatim. Must be >= 0.
    keep_system:
        If True, all ``kind == "system"`` turns are preserved verbatim in
        original order at the front of the returned list.
    policy:
        One of ``"oldest_first"``, ``"middle_biased"``,
        ``"tool_output_aggregated"``.
    """

    summarize_fn: Callable[[list[Turn]], str]
    token_counter: Callable[[str], int]
    target_tokens: int
    keep_last_n: int = 4
    keep_system: bool = True
    policy: str = "oldest_first"
    _cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.summarize_fn):
            raise TypeError("summarize_fn must be callable")
        if not callable(self.token_counter):
            raise TypeError("token_counter must be callable")
        if not isinstance(self.target_tokens, int) or self.target_tokens <= 0:
            raise ValueError("target_tokens must be a positive int")
        if not isinstance(self.keep_last_n, int) or self.keep_last_n < 0:
            raise ValueError("keep_last_n must be a non-negative int")
        if self.policy not in _VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(_VALID_POLICIES)}, got {self.policy!r}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def current_tokens(self, turns: list[Turn]) -> int:
        """Return total tokens across turn contents per ``token_counter``."""
        self._validate_turns(turns)
        return sum(self.token_counter(t.content) for t in turns)

    def extract_facts(self, turns: list[Turn]) -> list[str]:
        """Extract numeric / entity facts from turns. Deterministic regex.

        Returns a de-duplicated, order-preserving list of fact strings such as
        ``"port 8080"`` or ``"user_id=42"``.
        """
        self._validate_turns(turns)
        seen: dict[str, None] = {}
        for t in turns:
            for kind, pat in _FACT_PATTERNS:
                for m in pat.finditer(t.content):
                    if m.groups():
                        match = m.groups()
                        if len(match) == 1:
                            key = _fact_key(kind, match[0] if kind != "kv" else match)
                        else:
                            key = _fact_key(kind, match)
                    else:
                        key = _fact_key(kind, m.group(0))
                    if key and key not in seen:
                        seen[key] = None
        return list(seen.keys())

    def compact(self, turns: list[Turn]) -> list[Turn]:
        """Return a possibly-compacted turn list.

        If ``current_tokens(turns) <= target_tokens`` the input is returned
        unchanged. Otherwise the selected policy is applied. System turns are
        preserved when ``keep_system`` is True. ``keep_last_n`` trailing turns
        are always preserved verbatim.
        """
        self._validate_turns(turns)
        if not turns:
            return []
        if self.current_tokens(turns) <= self.target_tokens:
            return list(turns)

        if self.policy == "oldest_first":
            return self._oldest_first(turns)
        if self.policy == "middle_biased":
            return self._middle_biased(turns)
        if self.policy == "tool_output_aggregated":
            return self._tool_output_aggregated(turns)
        # unreachable — guarded in __post_init__
        raise ValueError(f"unknown policy: {self.policy}")

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------
    def _split_system(self, turns: list[Turn]) -> tuple[list[Turn], list[Turn]]:
        if not self.keep_system:
            return [], list(turns)
        system = [t for t in turns if t.kind == "system"]
        rest = [t for t in turns if t.kind != "system"]
        return system, rest

    def _tail(self, rest: list[Turn]) -> tuple[list[Turn], list[Turn]]:
        n = self.keep_last_n
        if n == 0 or n >= len(rest):
            if n >= len(rest):
                return list(rest), []
            return [], list(rest)
        return rest[-n:], rest[:-n]

    def _summarize_cached(self, group: list[Turn], label: str) -> Turn | None:
        if not group:
            return None
        key = _hash_group(group)
        if key in self._cache:
            summary = self._cache[key]
        else:
            summary = self.summarize_fn(group)
            if not isinstance(summary, str):
                raise TypeError("summarize_fn must return str")
            self._cache[key] = summary
        facts = self.extract_facts(group)
        facts_prefix = ""
        if facts:
            facts_prefix = "FACTS: " + "; ".join(facts) + "\n"
        return Turn(
            role="system",
            content=f"[CONTEXT SUMMARY {label}] {facts_prefix}{summary}",
            kind="system",
        )

    def _oldest_first(self, turns: list[Turn]) -> list[Turn]:
        system, rest = self._split_system(turns)
        tail, head = self._tail(rest)
        summary_turn = self._summarize_cached(head, "oldest_first")
        out: list[Turn] = list(system)
        if summary_turn is not None:
            out.append(summary_turn)
        out.extend(tail)
        return out

    def _middle_biased(self, turns: list[Turn]) -> list[Turn]:
        system, rest = self._split_system(turns)
        tail, head = self._tail(rest)
        # head = everything except the protected tail. Split head in two: keep
        # first 1 turn verbatim (anchor), summarize the middle.
        if len(head) <= 1:
            return list(system) + list(head) + list(tail)
        anchor = head[:1]
        middle = head[1:]
        summary_turn = self._summarize_cached(middle, "middle_biased")
        out: list[Turn] = list(system) + list(anchor)
        if summary_turn is not None:
            out.append(summary_turn)
        out.extend(tail)
        return out

    def _tool_output_aggregated(self, turns: list[Turn]) -> list[Turn]:
        system, rest = self._split_system(turns)
        tail, head = self._tail(rest)
        tool_group = [t for t in head if t.kind == "tool_result"]
        msg_group = [t for t in head if t.kind != "tool_result"]
        out: list[Turn] = list(system)
        msg_summary = self._summarize_cached(msg_group, "messages")
        if msg_summary is not None:
            out.append(msg_summary)
        tool_summary = self._summarize_cached(tool_group, "tool_results")
        if tool_summary is not None:
            out.append(tool_summary)
        out.extend(tail)
        return out

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_turns(turns: list[Turn]) -> None:
        if not isinstance(turns, list):
            raise TypeError("turns must be a list[Turn]")
        for i, t in enumerate(turns):
            if not isinstance(t, Turn):
                raise TypeError(f"turns[{i}] is not a Turn instance")
            if not isinstance(t.content, str):
                raise TypeError(f"turns[{i}].content must be str")


__all__ = ["Turn", "ContextCompactor"]
