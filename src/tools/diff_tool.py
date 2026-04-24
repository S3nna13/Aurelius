from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DiffFormat(str, Enum):
    UNIFIED = "unified"
    CONTEXT = "context"
    HTML = "html"
    SIDE_BY_SIDE_TEXT = "side_by_side_text"


@dataclass(frozen=True)
class DiffResult:
    lines_added: int
    lines_removed: int
    hunks: int
    diff_text: str


class DiffTool:
    def __init__(self, context_lines: int = 3) -> None:
        self.context_lines = context_lines

    def diff_strings(
        self,
        a: str,
        b: str,
        fromfile: str = "a",
        tofile: str = "b",
        fmt: DiffFormat = DiffFormat.UNIFIED,
    ) -> DiffResult:
        a_lines = a.splitlines(keepends=True)
        b_lines = b.splitlines(keepends=True)

        if fmt == DiffFormat.UNIFIED:
            diff_lines = list(difflib.unified_diff(
                a_lines, b_lines, fromfile=fromfile, tofile=tofile, n=self.context_lines,
            ))
            diff_text = "".join(diff_lines)
            lines_added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
            lines_removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
            hunks = sum(1 for l in diff_lines if l.startswith("@@"))
        elif fmt == DiffFormat.CONTEXT:
            diff_lines = list(difflib.context_diff(
                a_lines, b_lines, fromfile=fromfile, tofile=tofile, n=self.context_lines,
            ))
            diff_text = "".join(diff_lines)
            lines_added = sum(1 for l in diff_lines if l.startswith("+ "))
            lines_removed = sum(1 for l in diff_lines if l.startswith("- "))
            hunks = sum(1 for l in diff_lines if l.startswith("***************"))
        elif fmt == DiffFormat.HTML:
            diff_text = difflib.HtmlDiff().make_file(a_lines, b_lines, fromfile, tofile)
            diff_lines = diff_text.splitlines()
            lines_added = sum(1 for l in diff_lines if 'class="diff_add"' in l)
            lines_removed = sum(1 for l in diff_lines if 'class="diff_sub"' in l)
            hunks = 0
        else:
            diff_lines = list(difflib.ndiff(a_lines, b_lines))
            diff_text = "".join(diff_lines)
            lines_added = sum(1 for l in diff_lines if l.startswith("+ "))
            lines_removed = sum(1 for l in diff_lines if l.startswith("- "))
            hunks = 0

        return DiffResult(
            lines_added=lines_added,
            lines_removed=lines_removed,
            hunks=hunks,
            diff_text=diff_text,
        )

    def diff_files(self, path_a: str | Path, path_b: str | Path, **kwargs) -> DiffResult:
        a = Path(path_a).read_text()
        b = Path(path_b).read_text()
        return self.diff_strings(a, b, fromfile=str(path_a), tofile=str(path_b), **kwargs)

    def similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def apply_patch(self, original: str, patch_lines: list[str]) -> str:
        return original


DIFF_TOOL_REGISTRY: dict[str, type[DiffTool]] = {"default": DiffTool}
