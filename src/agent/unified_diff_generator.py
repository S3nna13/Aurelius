"""Unified-diff generation and validation for agent code edits.

Inverse of :func:`src.eval.swebench_lite_scorer.apply_patch_via_python`:
rather than *applying* patches, this module *produces* them from
before/after snapshots of virtual file trees. It is a pure-stdlib
utility backed by :mod:`difflib`; no torch, no foreign deps.

Usage::

    gen = UnifiedDiffGenerator(context_lines=3)
    result = gen.from_single_edit("a.py", "old\\n", "new\\n")
    new_tree = gen.apply_round_trip({"a.py": "old\\n"}, result.diff)

The emitted patches use the subset of the unified-diff dialect that
``apply_patch_via_python`` already understands: ``--- a/path`` /
``+++ b/path`` headers (with ``/dev/null`` on either side for
create/delete), ``@@ -l,s +l,s @@`` hunk banners, and space / plus /
minus / backslash line prefixes.
"""

from __future__ import annotations

import difflib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping


@dataclass
class DiffResult:
    """Structured result of a diff-generation call.

    Attributes:
        diff: Concatenated unified-diff text across all changed files.
            Empty string when the before and after snapshots are
            identical.
        changed_files: File paths (relative, as provided) that appear
            in the diff, in deterministic sorted order.
        line_changes: Number of "+" and "-" lines in the diff (excluding
            the ``+++``/``---`` header markers). Counts insertions and
            deletions together; a pure substitution of one line counts
            as two.
    """

    diff: str
    changed_files: List[str] = field(default_factory=list)
    line_changes: int = 0


class UnifiedDiffGenerator:
    """Produce unified diffs for single-file or multi-file agent edits."""

    def __init__(self, context_lines: int = 3) -> None:
        if not isinstance(context_lines, int) or context_lines < 0:
            raise ValueError("context_lines must be a non-negative int")
        self.context_lines = context_lines

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def from_single_edit(self, path: str, before: str, after: str) -> DiffResult:
        """Generate a unified diff for a single file.

        ``before`` == ``after`` yields an empty :class:`DiffResult`.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        self._check_text(before, "before")
        self._check_text(after, "after")
        return self.from_file_pairs({path: before}, {path: after})

    def from_file_pairs(
        self,
        before: Mapping[str, str],
        after: Mapping[str, str],
    ) -> DiffResult:
        """Generate a combined unified diff from two path->content maps.

        Handles create (path only in ``after``), delete (path only in
        ``before``), and modify (path in both, content differs) cases.
        Files whose content is identical on both sides are skipped.
        """
        if before is None or after is None:
            raise TypeError("before/after must be dict-like, not None")
        for label, mapping in (("before", before), ("after", after)):
            for p, c in mapping.items():
                if not isinstance(p, str) or not p:
                    raise ValueError(f"{label} contains non-string/empty path")
                self._check_text(c, f"{label}[{p!r}]")

        all_paths = sorted(set(before) | set(after))
        hunks: List[str] = []
        changed: List[str] = []
        line_changes = 0

        for path in all_paths:
            b = before.get(path)
            a = after.get(path)
            if b == a:
                continue
            if b is None:
                section = self._render_create(path, a or "")
            elif a is None:
                section = self._render_delete(path, b or "")
            else:
                section = self._render_modify(path, b, a)
                if section is None:
                    # difflib reports no diff (shouldn't happen given b!=a)
                    continue
            hunks.append(section)
            changed.append(path)
            line_changes += self._count_changes(section)

        return DiffResult(
            diff="".join(hunks),
            changed_files=changed,
            line_changes=line_changes,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, diff: str) -> bool:
        """Well-formedness check (not applicability).

        Returns True for empty strings (a valid no-op patch) and for any
        string composed of one or more correctly-shaped file sections.
        Does *not* verify that hunks would apply to any particular base
        content.
        """
        if not isinstance(diff, str):
            return False
        if not diff.strip():
            return True

        lines = diff.splitlines()
        i = 0
        n = len(lines)
        saw_any_section = False
        while i < n:
            # Skip any leading blank lines between sections.
            if not lines[i].strip():
                i += 1
                continue
            if not lines[i].startswith("--- "):
                return False
            if i + 1 >= n or not lines[i + 1].startswith("+++ "):
                return False
            i += 2
            # Expect at least one hunk header.
            if i >= n or not lines[i].startswith("@@"):
                return False
            saw_any_section = True
            # Walk hunks until the next "--- " header or EOF.
            while i < n and not lines[i].startswith("--- "):
                line = lines[i]
                if line.startswith("@@"):
                    if not self._valid_hunk_header(line):
                        return False
                elif line == "" or line[0] in (" ", "+", "-", "\\"):
                    pass
                else:
                    return False
                i += 1
        return saw_any_section

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------
    def apply_round_trip(
        self, before: Mapping[str, str], diff: str
    ) -> Dict[str, str]:
        """Apply ``diff`` to an in-memory ``before`` map and return the new state.

        Materializes ``before`` into a temp directory, delegates to
        :func:`src.eval.swebench_lite_scorer.apply_patch_via_python`, and
        reads the resulting files back. Raises :class:`RuntimeError` if
        the patch fails to apply.
        """
        # Internal import: avoids an import cycle between eval and agent
        # surfaces at module load, and keeps this a deferred dependency.
        from src.eval.swebench_lite_scorer import apply_patch_via_python

        if before is None:
            raise TypeError("before must not be None")
        for p, c in before.items():
            if not isinstance(p, str) or not p:
                raise ValueError("before contains non-string/empty path")
            self._check_text(c, f"before[{p!r}]")

        if not isinstance(diff, str):
            raise TypeError("diff must be a string")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for rel, content in before.items():
                target = (root / rel).resolve()
                # Keep within root (guard against "../" traversal).
                if root not in target.parents and target != root:
                    raise ValueError(f"path escapes root: {rel!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

            ok = apply_patch_via_python(diff, str(root))
            if not ok:
                raise RuntimeError("apply_patch_via_python failed on diff")

            result: Dict[str, str] = {}
            for path in root.rglob("*"):
                if path.is_file():
                    rel = str(path.relative_to(root))
                    result[rel] = path.read_text()
            return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _check_text(value: object, label: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{label} must be str, got {type(value).__name__}")

    @staticmethod
    def _split_keep_newlines(text: str) -> List[str]:
        """Split into lines *without* keepends for unified_diff consumption.

        We deliberately strip newlines so we can control the ``\\ No
        newline at end of file`` marker ourselves and keep output
        deterministic regardless of the exact trailing-newline layout of
        the inputs.
        """
        if text == "":
            return []
        had_trailing = text.endswith("\n")
        lines = text.split("\n")
        if had_trailing:
            lines = lines[:-1]
        return lines

    def _render_modify(self, path: str, before: str, after: str) -> str | None:
        b_lines = self._split_keep_newlines(before)
        a_lines = self._split_keep_newlines(after)
        before_had_nl = before.endswith("\n") or before == ""
        after_had_nl = after.endswith("\n") or after == ""

        diff_iter = difflib.unified_diff(
            b_lines,
            a_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=self.context_lines,
            lineterm="",
        )
        lines = list(diff_iter)
        if not lines:
            return None
        out: List[str] = []
        last_tag: str | None = None
        for raw in lines:
            out.append(raw + "\n")
            if raw.startswith(("---", "+++", "@@")):
                last_tag = None
            elif raw:
                last_tag = raw[0]
        # Emit "\ No newline at end of file" markers when appropriate.
        # We conservatively add it only when the *last* payload line's
        # origin file lacks a trailing newline.
        if not before_had_nl and last_tag in ("-", " "):
            out.append("\\ No newline at end of file\n")
        if not after_had_nl and last_tag in ("+", " "):
            out.append("\\ No newline at end of file\n")
        return "".join(out)

    def _render_create(self, path: str, content: str) -> str:
        lines = self._split_keep_newlines(content)
        had_nl = content.endswith("\n") or content == ""
        header = f"--- /dev/null\n+++ b/{path}\n"
        if not lines:
            # Creating an empty file: emit a zero-length hunk that the
            # apply routine treats as a no-op write of "".
            return header + "@@ -0,0 +0,0 @@\n"
        hunk = f"@@ -0,0 +1,{len(lines)} @@\n"
        body_parts = [f"+{ln}\n" for ln in lines]
        body = "".join(body_parts)
        if not had_nl:
            body += "\\ No newline at end of file\n"
        return header + hunk + body

    def _render_delete(self, path: str, content: str) -> str:
        lines = self._split_keep_newlines(content)
        had_nl = content.endswith("\n") or content == ""
        header = f"--- a/{path}\n+++ /dev/null\n"
        if not lines:
            return header + "@@ -0,0 +0,0 @@\n"
        hunk = f"@@ -1,{len(lines)} +0,0 @@\n"
        body = "".join(f"-{ln}\n" for ln in lines)
        if not had_nl:
            body += "\\ No newline at end of file\n"
        return header + hunk + body

    @staticmethod
    def _count_changes(section: str) -> int:
        count = 0
        for line in section.splitlines():
            if line.startswith("+++ ") or line.startswith("--- "):
                continue
            if line.startswith("+") or line.startswith("-"):
                count += 1
        return count

    @staticmethod
    def _valid_hunk_header(line: str) -> bool:
        # "@@ -l[,s] +l[,s] @@ [optional section heading]"
        if not line.startswith("@@"):
            return False
        try:
            rest = line[2:].lstrip()
            if " @@" not in rest:
                return False
            core, _tail = rest.split(" @@", 1)
            parts = core.split()
            if len(parts) != 2:
                return False
            minus, plus = parts
            if not minus.startswith("-") or not plus.startswith("+"):
                return False
            for spec in (minus[1:], plus[1:]):
                if "," in spec:
                    a, b = spec.split(",", 1)
                    int(a)
                    int(b)
                else:
                    int(spec)
            return True
        except (ValueError, IndexError):
            return False


__all__ = ["DiffResult", "UnifiedDiffGenerator"]
