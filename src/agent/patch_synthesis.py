"""Unified-diff patch synthesis and application engine for Aurelius.

Inspired by StarCoder2 PSM/SPM FIM (bigcode, Apache-2.0),
Kimi-Dev patch synthesis (MoonshotAI, Apache-2.0, 2025),
Aider unified-diff format (MIT), clean-room reimplementation.

Uses stdlib ``difflib`` only; no external patch libraries.
"""

from __future__ import annotations

import difflib
import re
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PatchError(Exception):
    """Raised when a patch cannot be parsed or applied."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PatchHunk:
    """A single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)
    """Raw diff hunk lines: each prefixed with '+', '-', or ' '."""


@dataclass
class Patch:
    """A complete unified-diff patch for a single file."""

    filename: str
    hunks: list[PatchHunk] = field(default_factory=list)
    patch_id: str = ""


# ---------------------------------------------------------------------------
# PatchSynthesizer
# ---------------------------------------------------------------------------


class PatchSynthesizer:
    """Produce and apply unified-diff patches."""

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------

    def make_patch(
        self,
        original: str,
        modified: str,
        filename: str = "file.py",
    ) -> Patch:
        """Produce a minimal unified-diff :class:`Patch` by comparing
        *original* vs *modified* line-by-line via ``difflib.unified_diff``.

        Returns a :class:`Patch` with zero hunks when the texts are identical.
        """
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)

        raw = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                n=3,
            )
        )

        patch = Patch(filename=filename, patch_id=str(uuid.uuid4()))
        if not raw:
            return patch

        # Parse the raw unified diff into PatchHunk objects
        patch.hunks = self._parse_hunks(raw)
        return patch

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def patch_to_str(self, patch: Patch) -> str:
        """Render *patch* as a standard unified diff string."""
        if not patch.hunks:
            return ""

        parts: list[str] = [
            f"--- a/{patch.filename}\n",
            f"+++ b/{patch.filename}\n",
        ]
        for hunk in patch.hunks:
            old_range = (
                f"{hunk.old_start}" if hunk.old_count == 1 else f"{hunk.old_start},{hunk.old_count}"
            )
            new_range = (
                f"{hunk.new_start}" if hunk.new_count == 1 else f"{hunk.new_start},{hunk.new_count}"
            )
            parts.append(f"@@ -{old_range} +{new_range} @@\n")
            for line in hunk.lines:
                # Ensure each line ends with a newline
                if line.endswith("\n"):
                    parts.append(line)
                else:
                    parts.append(line + "\n")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_patch(self, source: str, patch: Patch) -> str:
        """Apply *patch* hunks to *source*.

        Raises :class:`PatchError` if a hunk cannot be applied because the
        expected old lines do not match the source at the indicated position.
        """
        result_lines = source.splitlines(keepends=True)

        # Apply hunks in reverse order so that earlier line numbers remain
        # valid as we mutate the list.
        offset = 0  # cumulative line shift from previously applied hunks

        for hunk in patch.hunks:
            old_start_0 = hunk.old_start - 1 + offset  # convert to 0-based

            # Collect old (removed) lines from the hunk
            old_hunk_lines = [
                ln[1:] for ln in hunk.lines if ln.startswith("-") or ln.startswith(" ")
            ]
            new_hunk_lines = [
                ln[1:] for ln in hunk.lines if ln.startswith("+") or ln.startswith(" ")
            ]

            # Verify old lines match
            actual = result_lines[old_start_0 : old_start_0 + len(old_hunk_lines)]
            # Normalise for comparison (strip trailing newline differences)
            if [line.rstrip("\n") for line in actual] != [line.rstrip("\n") for line in old_hunk_lines]:  # noqa: E501
                raise PatchError(
                    f"Hunk at old_start={hunk.old_start} does not match source. "
                    f"Expected:\n{''.join(old_hunk_lines)}\nGot:\n{''.join(actual)}"
                )

            # Replace old lines with new lines
            result_lines[old_start_0 : old_start_0 + len(old_hunk_lines)] = new_hunk_lines
            offset += len(new_hunk_lines) - len(old_hunk_lines)

        return "".join(result_lines)

    def apply_str_patch(self, source: str, patch_str: str) -> str:
        """Parse *patch_str* (a unified diff) then apply it to *source*.

        Raises :class:`PatchError` on parse failure or application failure.
        """
        patch = self._parse_patch_str(patch_str)
        return self.apply_patch(source, patch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    def _parse_hunks(self, diff_lines: list[str]) -> list[PatchHunk]:
        """Parse raw ``difflib.unified_diff`` output into :class:`PatchHunk` list."""
        hunks: list[PatchHunk] = []
        current: PatchHunk | None = None

        for line in diff_lines:
            if line.startswith("--- ") or line.startswith("+++ "):
                continue
            m = self._HUNK_HEADER_RE.match(line)
            if m:
                if current is not None:
                    hunks.append(current)
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                current = PatchHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
                continue
            if current is not None and line and line[0] in (" ", "+", "-"):
                current.lines.append(line)

        if current is not None:
            hunks.append(current)

        return hunks

    def _parse_patch_str(self, patch_str: str) -> Patch:
        """Parse a unified diff string into a :class:`Patch`.

        Raises :class:`PatchError` on format errors.
        """
        if not isinstance(patch_str, str) or not patch_str.strip():
            raise PatchError("patch_str is empty or not a string")

        lines = patch_str.splitlines(keepends=True)

        # Find the --- / +++ header
        filename = "file.py"
        header_found = False
        body_lines: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("--- "):
                # Next line should be +++
                if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                    # Extract filename from +++ line (strip "b/" prefix)
                    plus_line = lines[i + 1].rstrip("\n")
                    raw_name = plus_line[4:]  # strip "+++ "
                    if raw_name.startswith("b/"):
                        raw_name = raw_name[2:]
                    filename = raw_name or filename
                    header_found = True
                    i += 2
                    body_lines = lines[i:]
                    break
                else:
                    raise PatchError("--- header not followed by +++ header")
            i += 1

        if not header_found:
            raise PatchError("No unified diff header (--- / +++) found in patch_str")

        hunks = self._parse_hunks(body_lines)
        if not hunks:
            raise PatchError("No hunks found in patch_str")

        return Patch(filename=filename, hunks=hunks, patch_id=str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Registry mapping patch style names to the :class:`PatchSynthesizer` class.
PATCH_REGISTRY: dict[str, type] = {
    "unified": PatchSynthesizer,
}


__all__ = [
    "Patch",
    "PatchError",
    "PatchHunk",
    "PATCH_REGISTRY",
    "PatchSynthesizer",
]
