"""Unified-diff renderer for code review and patch display in Aurelius.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Aider-AI/aider (MIT, diff/edit formats), clean-room reimplementation
with original Aurelius branding.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class DiffViewerError(Exception):
    """Raised when the diff viewer encounters malformed input."""


@dataclass
class DiffLine:
    """A single line in a diff chunk.

    Attributes:
        line_type: One of ``"context"``, ``"add"``, ``"remove"``, or ``"header"``.
        content: The raw line text (without the leading +/-/space sigil).
        line_no_old: 1-based line number in the old file, or ``None``.
        line_no_new: 1-based line number in the new file, or ``None``.
    """

    line_type: Literal["context", "add", "remove", "header"]
    content: str
    line_no_old: int | None = None
    line_no_new: int | None = None


@dataclass
class DiffChunk:
    """A diff hunk — a header line plus zero or more :class:`DiffLine` objects.

    Attributes:
        header: The ``@@`` hunk header string.
        lines: Ordered list of :class:`DiffLine` objects in this chunk.
    """

    header: str
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class ParsedDiff:
    """The result of parsing a unified diff string.

    Attributes:
        filename_old: Path label for the old version (from ``---`` line).
        filename_new: Path label for the new version (from ``+++`` line).
        chunks: Ordered list of :class:`DiffChunk` objects.
    """

    filename_old: str
    filename_new: str
    chunks: list[DiffChunk] = field(default_factory=list)


# Matches: @@ -L,S +L,S @@
_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
)
_OLD_FILE_RE = re.compile(r"^---\s+(.+)$")
_NEW_FILE_RE = re.compile(r"^\+\+\+\s+(.+)$")


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """Parse a unified diff string into a :class:`ParsedDiff`.

    Uses only ``re`` from the stdlib.  A malformed or empty diff returns a
    :class:`ParsedDiff` with empty ``chunks`` rather than raising an exception.

    Args:
        diff_text: A unified-diff formatted string.

    Returns:
        A :class:`ParsedDiff` with ``filename_old``, ``filename_new``, and
        all parsed :class:`DiffChunk` objects.
    """
    filename_old = ""
    filename_new = ""
    chunks: list[DiffChunk] = []
    current_chunk: DiffChunk | None = None
    old_lineno: int = 0
    new_lineno: int = 0

    for raw_line in diff_text.splitlines():
        # --- a/file
        m_old = _OLD_FILE_RE.match(raw_line)
        if m_old:
            filename_old = m_old.group(1).strip()
            continue

        # +++ b/file
        m_new = _NEW_FILE_RE.match(raw_line)
        if m_new:
            filename_new = m_new.group(1).strip()
            continue

        # @@ hunk header
        m_hunk = _HUNK_HEADER_RE.match(raw_line)
        if m_hunk:
            current_chunk = DiffChunk(header=raw_line)
            chunks.append(current_chunk)
            old_lineno = int(m_hunk.group(1))
            new_lineno = int(m_hunk.group(3))
            continue

        if current_chunk is None:
            continue

        if raw_line.startswith("+"):
            current_chunk.lines.append(
                DiffLine(
                    line_type="add",
                    content=raw_line[1:],
                    line_no_old=None,
                    line_no_new=new_lineno,
                )
            )
            new_lineno += 1
        elif raw_line.startswith("-"):
            current_chunk.lines.append(
                DiffLine(
                    line_type="remove",
                    content=raw_line[1:],
                    line_no_old=old_lineno,
                    line_no_new=None,
                )
            )
            old_lineno += 1
        elif raw_line.startswith(" ") or raw_line == "":
            current_chunk.lines.append(
                DiffLine(
                    line_type="context",
                    content=raw_line[1:] if raw_line.startswith(" ") else "",
                    line_no_old=old_lineno,
                    line_no_new=new_lineno,
                )
            )
            old_lineno += 1
            new_lineno += 1
        # skip "diff --git" and other meta lines silently

    return ParsedDiff(
        filename_old=filename_old,
        filename_new=filename_new,
        chunks=chunks,
    )


_LINE_TYPE_STYLES: dict[str, str] = {
    "add": "bold green",
    "remove": "bold red",
    "context": "dim",
    "header": "bold blue",
}

_LINE_TYPE_SIGILS: dict[str, str] = {
    "add": "+",
    "remove": "-",
    "context": " ",
    "header": " ",
}


class DiffViewer:
    """Renders a :class:`ParsedDiff` to a Rich console surface.

    All methods are stateless — create a single instance and call
    :meth:`render_diff` or :meth:`render_inline` as needed.
    """

    def render_diff(
        self,
        console: Console,
        diff: ParsedDiff,
        syntax_lang: str = "python",
    ) -> None:
        """Render *diff* as a Rich Panel with coloured +/- lines.

        Args:
            console: Rich :class:`Console` to print to.
            diff: The :class:`ParsedDiff` to render.
            syntax_lang: Language hint stored for future Syntax integration
                (not used for markup — coloured +/- lines are always used).
        """
        title = f"[bold]Diff[/bold]  {diff.filename_old} → {diff.filename_new}"
        body = Text()

        if not diff.chunks:
            body.append("(empty diff)", style="dim")
        else:
            for chunk_idx, chunk in enumerate(diff.chunks):
                if chunk_idx > 0:
                    body.append("\n")
                body.append(chunk.header + "\n", style="bold blue")
                for diff_line in chunk.lines:
                    style = _LINE_TYPE_STYLES.get(diff_line.line_type, "")
                    sigil = _LINE_TYPE_SIGILS.get(diff_line.line_type, " ")
                    body.append(f"{sigil}{diff_line.content}\n", style=style)

        console.print(Panel(body, title=title, border_style="dim"))

    def render_inline(
        self,
        console: Console,
        old_text: str,
        new_text: str,
    ) -> None:
        """Render a side-by-side comparison of *old_text* and *new_text*.

        Splits both strings into lines and displays them in a two-column
        Rich :class:`Table`.  Lines present only in old are shown in red;
        lines present only in new are shown in green; matching lines are
        shown dim.

        Args:
            console: Rich :class:`Console` to print to.
            old_text: The original text.
            new_text: The updated text.
        """
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        max_len = max(len(old_lines), len(new_lines), 1)

        table = Table(
            title="Inline Diff",
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=False,
        )
        table.add_column("Old", style="", no_wrap=False)
        table.add_column("New", style="", no_wrap=False)

        for i in range(max_len):
            old_cell_text = old_lines[i] if i < len(old_lines) else ""
            new_cell_text = new_lines[i] if i < len(new_lines) else ""

            if old_cell_text == new_cell_text:
                old_cell = Text(old_cell_text, style="dim")
                new_cell = Text(new_cell_text, style="dim")
            else:
                old_cell = Text(old_cell_text, style="red" if old_cell_text else "dim")
                new_cell = Text(new_cell_text, style="green" if new_cell_text else "dim")

            table.add_row(old_cell, new_cell)

        console.print(table)


__all__ = [
    "DiffLine",
    "DiffChunk",
    "ParsedDiff",
    "parse_unified_diff",
    "DiffViewer",
    "DiffViewerError",
]
