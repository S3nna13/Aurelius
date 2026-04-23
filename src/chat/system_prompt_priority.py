"""System-prompt priority encoding.

Enforces a principal hierarchy when multiple system prompts are present.
Lower numeric priority wins. Immutable fragments cannot be truncated.

Pure stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum


class SystemPromptPriority(IntEnum):
    DEVELOPER = 0
    OPERATOR = 1
    USER = 2
    TOOL = 3
    MODEL_DEFAULT = 4


@dataclass(frozen=True)
class SystemPromptFragment:
    priority: SystemPromptPriority
    content: str
    source_id: str
    immutable: bool = False


class PrincipalHierarchyConflict(Exception):
    """Raised when principal hierarchy cannot be reconciled."""


_ALWAYS_RE = re.compile(r"always\s+(\w+)", re.IGNORECASE)
_NEVER_RE = re.compile(r"never\s+(\w+)", re.IGNORECASE)


class SystemPromptPriorityEncoder:
    def __init__(
        self,
        max_total_chars: int = 16000,
        separator: str = "\n---\n",
    ) -> None:
        if max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        self.max_total_chars = max_total_chars
        self.separator = separator

    @staticmethod
    def _format(fragment: SystemPromptFragment) -> str:
        return (
            f"[SOURCE:{fragment.source_id} "
            f"PRIORITY:{fragment.priority.name}]\n{fragment.content}"
        )

    def _sorted(
        self, fragments: list[SystemPromptFragment]
    ) -> list[SystemPromptFragment]:
        # Filter empty-content gracefully, retain insertion order within priority.
        indexed = [
            (i, f) for i, f in enumerate(fragments) if f.content.strip() != ""
        ]
        indexed.sort(key=lambda pair: (int(pair[1].priority), pair[0]))
        return [f for _, f in indexed]

    def merge(self, fragments: list[SystemPromptFragment]) -> str:
        ordered = self._sorted(list(fragments))
        if not ordered:
            return ""

        pieces = [self._format(f) for f in ordered]
        total = sum(len(p) for p in pieces) + len(self.separator) * (
            len(pieces) - 1
        )

        if total <= self.max_total_chars:
            return self.separator.join(pieces)

        # Drop lowest-priority (highest numeric) first; skip immutables.
        # Build a list of (index, fragment) sorted by priority descending
        # (lowest priority first for eviction).
        evict_order = sorted(
            range(len(ordered)),
            key=lambda i: (-int(ordered[i].priority), -i),
        )
        keep = [True] * len(ordered)
        for idx in evict_order:
            if total <= self.max_total_chars:
                break
            if ordered[idx].immutable:
                continue
            total -= len(pieces[idx])
            if sum(keep) > 1:
                total -= len(self.separator)
            keep[idx] = False

        if total > self.max_total_chars:
            # All remaining are immutable and still overflow.
            raise PrincipalHierarchyConflict(
                "immutable fragments exceed max_total_chars"
            )

        kept_pieces = [pieces[i] for i in range(len(ordered)) if keep[i]]
        return self.separator.join(kept_pieces)

    def detect_conflicts(
        self, fragments: list[SystemPromptFragment]
    ) -> list[tuple[SystemPromptFragment, SystemPromptFragment, str]]:
        conflicts: list[
            tuple[SystemPromptFragment, SystemPromptFragment, str]
        ] = []
        parsed: list[tuple[SystemPromptFragment, set[str], set[str]]] = []
        for frag in fragments:
            if not frag.content.strip():
                continue
            always = {m.group(1).lower() for m in _ALWAYS_RE.finditer(frag.content)}
            never = {m.group(1).lower() for m in _NEVER_RE.finditer(frag.content)}
            parsed.append((frag, always, never))

        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                a_frag, a_always, a_never = parsed[i]
                b_frag, b_always, b_never = parsed[j]
                shared = (a_always & b_never) | (a_never & b_always)
                for topic in sorted(shared):
                    conflicts.append((a_frag, b_frag, topic))
        return conflicts

    def resolve(self, fragments: list[SystemPromptFragment]) -> str:
        conflicts = self.detect_conflicts(fragments)
        dropped: set[int] = set()
        for a_frag, b_frag, _topic in conflicts:
            # Higher priority (lower number) wins; drop the other.
            if int(a_frag.priority) < int(b_frag.priority):
                loser = b_frag
            elif int(b_frag.priority) < int(a_frag.priority):
                loser = a_frag
            else:
                # Same priority: drop the later one (stable tiebreak).
                loser = b_frag
            # Drop all matching identity fragments by id()
            for idx, f in enumerate(fragments):
                if f is loser:
                    dropped.add(idx)

        surviving = [f for i, f in enumerate(fragments) if i not in dropped]
        return self.merge(surviving)


__all__ = [
    "PrincipalHierarchyConflict",
    "SystemPromptFragment",
    "SystemPromptPriority",
    "SystemPromptPriorityEncoder",
]
