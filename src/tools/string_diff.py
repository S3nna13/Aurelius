"""String diff utility for comparing tool outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiffResult:
    added: list[str]
    removed: list[str]
    unchanged: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


@dataclass
class StringDiff:
    """Line-based string diff using LCS similarity."""

    def diff(self, old: str, new: str) -> DiffResult:
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        old_set = set(old_lines)
        new_set = set(new_lines)
        return DiffResult(
            added=[line for line in new_lines if line not in old_set],
            removed=[line for line in old_lines if line not in new_set],
            unchanged=[line for line in new_lines if line in old_set],
        )

    def similarity(self, a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        set_a = set(a.split())
        set_b = set(b.split())
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 1.0


STRING_DIFF = StringDiff()
