"""SWE-bench-lite scoring harness.

Implements a minimal, pure-stdlib scorer for SWE-bench-style tasks
(Jimenez et al., 2024; arXiv:2310.06770). A problem is described by a
repository snapshot (files in memory), a gold patch, and a test command.
Given a candidate unified-diff patch, the scorer materializes the repo in
a temporary directory, applies the patch with a minimal pure-Python
unified-diff applier (no external `patch` binary), runs the specified
test command under a timeout, and records pass/fail.

Intended for in-process unit testing on synthetic micro-repos. In
production, the same API can be pointed at real SWE-bench task data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SWEProblem:
    """A single SWE-bench-style problem.

    Attributes
    ----------
    task_id: Unique identifier for the task.
    repo_files: Mapping of relative path -> file contents; written to a
        temporary directory before tests are run.
    gold_patch: The reference unified-diff patch (for bookkeeping / oracle
        evaluation).
    test_command: A shell-free argv list, e.g. ``["pytest", "tests/"]``.
    test_should_pass_after_patch: Names of tests that should pass once a
        correct patch is applied (informational; the harness only checks
        overall exit code).
    """

    task_id: str
    repo_files: dict[str, str]
    gold_patch: str
    test_command: list[str]
    test_should_pass_after_patch: list[str] = field(default_factory=list)


@dataclass
class SWEResult:
    task_id: str
    patch_applied: bool
    tests_passed: bool
    stdout: str
    stderr: str
    duration_ms: float


# ---------------------------------------------------------------------------
# Repository materialization
# ---------------------------------------------------------------------------


def materialize_repo(problem: SWEProblem, root_dir: str) -> None:
    """Write every file in ``problem.repo_files`` under ``root_dir``.

    Parent directories are created as needed. Existing files are
    overwritten. Paths must be relative and contained within
    ``root_dir`` (no ``..`` escapes).
    """
    root = Path(root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, contents in problem.repo_files.items():
        if os.path.isabs(rel_path):
            raise ValueError(f"repo_files path must be relative: {rel_path!r}")
        target = (root / rel_path).resolve()
        # Prevent path traversal escape.
        if root not in target.parents and target != root:
            raise ValueError(f"repo_files path escapes root_dir: {rel_path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)


# ---------------------------------------------------------------------------
# Minimal unified-diff applier
# ---------------------------------------------------------------------------


def _parse_file_header(line: str) -> str | None:
    """Extract path from a '--- a/foo' or '+++ b/foo' header."""
    if not (line.startswith("--- ") or line.startswith("+++ ")):
        return None
    path = line[4:].strip()
    # Common conventions: /dev/null, a/..., b/...
    if path == "/dev/null":
        return None
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    """Parse '@@ -l,s +l,s @@ [context]' -> (old_start, old_len, new_start, new_len)."""
    if not line.startswith("@@"):
        return None
    try:
        inner = line.split("@@", 2)[1].strip()
        old_part, new_part = inner.split(" ", 1)
        new_part = new_part.split(" ", 1)[0]
        if not (old_part.startswith("-") and new_part.startswith("+")):
            return None

        def _range(spec: str) -> tuple[int, int]:
            spec = spec[1:]
            if "," in spec:
                a, b = spec.split(",", 1)
                return int(a), int(b)
            return int(spec), 1

        old_start, old_len = _range(old_part)
        new_start, new_len = _range(new_part)
        return old_start, old_len, new_start, new_len
    except (ValueError, IndexError):
        return None


def _split_into_file_sections(diff: str) -> list[list[str]]:
    """Group diff lines into sections, one per file."""
    lines = diff.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git") or line.startswith("--- "):
            if current:
                sections.append(current)
            current = [line]
        else:
            if not current:
                # Stray prelude lines before first header; ignore.
                i += 1
                continue
            current.append(line)
        i += 1
    if current:
        sections.append(current)
    return sections


def _extract_paths(section: list[str]) -> tuple[str | None, str | None]:
    """Return (old_path, new_path) from section headers; None if /dev/null."""
    old_path: str | None = None
    new_path: str | None = None
    saw_minus = False
    saw_plus = False
    for line in section:
        if line.startswith("--- ") and not saw_minus:
            raw = line[4:].strip()
            old_path = None if raw == "/dev/null" else _parse_file_header(line)
            saw_minus = True
        elif line.startswith("+++ ") and not saw_plus:
            raw = line[4:].strip()
            new_path = None if raw == "/dev/null" else _parse_file_header(line)
            saw_plus = True
        if saw_minus and saw_plus:
            break
    return old_path, new_path


def _apply_hunks_to_lines(original: list[str], section: list[str]) -> list[str] | None:
    """Apply all hunks in ``section`` to ``original`` (list of lines without newlines).

    Returns the new list of lines, or None on any context mismatch / bad hunk.
    """
    # Find indices of hunk headers.
    hunk_indices = [i for i, line in enumerate(section) if line.startswith("@@")]
    if not hunk_indices:
        # No hunks: treat as no-op but only if file already exists; caller handles.
        return list(original)

    result: list[str] = []
    cursor = 0  # index into ``original`` (0-based)

    for idx, h_start in enumerate(hunk_indices):
        header = _parse_hunk_header(section[h_start])
        if header is None:
            return None
        old_start, old_len, _new_start, _new_len = header
        # Unified diff line numbers are 1-based; old_start may be 0 for new files.
        target_idx = max(old_start - 1, 0)
        if target_idx < cursor:
            return None
        # Copy lines from cursor up to target_idx unchanged.
        if target_idx > len(original):
            return None
        result.extend(original[cursor:target_idx])
        cursor = target_idx

        h_end = hunk_indices[idx + 1] if idx + 1 < len(hunk_indices) else len(section)
        body = section[h_start + 1 : h_end]

        old_consumed = 0
        for line in body:
            if not line:
                # Blank line inside a hunk: treat as a context blank line.
                tag, payload = " ", ""
            else:
                tag, payload = line[0], line[1:]
            if tag == "\\":
                # "\ No newline at end of file" -- ignore.
                continue
            if tag == " ":
                if cursor >= len(original) or original[cursor] != payload:
                    return None
                result.append(payload)
                cursor += 1
                old_consumed += 1
            elif tag == "-":
                if cursor >= len(original) or original[cursor] != payload:
                    return None
                cursor += 1
                old_consumed += 1
            elif tag == "+":
                result.append(payload)
            else:
                # Unknown tag.
                return None

        if old_len and old_consumed != old_len:
            # Soft check; hunks with trailing context should match declared length.
            # Accept off-by-one when the final "\ No newline" marker is absent.
            if abs(old_consumed - old_len) > 1:
                return None

    # Append any trailing lines after last hunk.
    result.extend(original[cursor:])
    return result


def apply_patch_via_python(diff: str, root_dir: str) -> bool:
    """Apply a unified-diff ``diff`` string under ``root_dir``.

    Supports create, modify, and delete operations. Returns True on full
    success, False on any malformed hunk, context mismatch, or I/O error.
    Partial applications are not rolled back (the tmp directory should be
    considered tainted on failure).
    """
    if not isinstance(diff, str) or not diff.strip():
        # Empty patches are considered a no-op success.
        return True

    root = Path(root_dir).resolve()
    if not root.exists():
        return False

    try:
        sections = _split_into_file_sections(diff)
    except Exception:
        return False
    if not sections:
        return False

    for section in sections:
        # Must contain both ---/+++ markers.
        has_minus = any(line.startswith("--- ") for line in section)
        has_plus = any(line.startswith("+++ ") for line in section)
        if not (has_minus and has_plus):
            return False

        old_path, new_path = _extract_paths(section)
        target_rel = new_path if new_path is not None else old_path
        if target_rel is None:
            # Both /dev/null -- meaningless.
            return False

        target_path = (root / target_rel).resolve()
        if root not in target_path.parents and target_path != root:
            return False

        # Deletion: +++ is /dev/null.
        if new_path is None:
            try:
                if target_path.exists():
                    target_path.unlink()
                return_ok = True
            except OSError:
                return False
            if not return_ok:
                return False
            continue

        # Creation: --- is /dev/null, file should not exist (but be tolerant).
        if old_path is None:
            try:
                hunk_indices = [i for i, line in enumerate(section) if line.startswith("@@")]
                if not hunk_indices:
                    return False
                added: list[str] = []
                for idx, h_start in enumerate(hunk_indices):
                    h_end = hunk_indices[idx + 1] if idx + 1 < len(hunk_indices) else len(section)
                    for line in section[h_start + 1 : h_end]:
                        if not line:
                            continue
                        tag, payload = line[0], line[1:]
                        if tag == "+":
                            added.append(payload)
                        elif tag == "\\":
                            continue
                        elif tag in (" ", "-"):
                            # New files should have no context or deletions.
                            return False
                        else:
                            return False
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Preserve trailing newline convention: write lines joined by \n
                # plus a trailing newline when there were any added lines.
                content = "\n".join(added) + ("\n" if added else "")
                target_path.write_text(content)
                continue
            except OSError:
                return False

        # Modify: read, apply hunks, write.
        try:
            if not target_path.exists():
                return False
            original_text = target_path.read_text()
        except OSError:
            return False

        had_trailing_newline = original_text.endswith("\n")
        original_lines = original_text.split("\n")
        if had_trailing_newline and original_lines and original_lines[-1] == "":
            original_lines = original_lines[:-1]

        new_lines = _apply_hunks_to_lines(original_lines, section)
        if new_lines is None:
            return False

        new_text = "\n".join(new_lines)
        if had_trailing_newline:
            new_text += "\n"

        try:
            target_path.write_text(new_text)
        except OSError:
            return False

    return True


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def run_tests(
    test_command: list[str], root_dir: str, timeout: float = 30.0
) -> tuple[bool, str, str]:
    """Run ``test_command`` in ``root_dir`` under a timeout.

    Returns ``(passed, stdout, stderr)``. ``passed`` is True iff the
    command exits with code 0. On timeout, ``passed`` is False and
    ``stderr`` contains a marker string.
    """
    if not test_command:
        return False, "", "empty test_command"
    try:
        proc = subprocess.run(  # noqa: S603
            list(test_command),
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return False, stdout, stderr + f"\n[TIMEOUT after {timeout}s]"
    except FileNotFoundError as exc:
        return False, "", f"command not found: {exc}"
    except OSError as exc:
        return False, "", f"OSError: {exc}"
    return proc.returncode == 0, proc.stdout or "", proc.stderr or ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_single(
    problem: SWEProblem,
    candidate_patch: str,
    timeout_seconds: float = 30.0,
) -> SWEResult:
    """Apply ``candidate_patch`` to ``problem`` and run its tests."""
    start = time.monotonic()
    tmp = tempfile.mkdtemp(prefix=f"swebench_{problem.task_id}_")
    try:
        materialize_repo(problem, tmp)
        patch_ok = apply_patch_via_python(candidate_patch, tmp)
        if not patch_ok:
            duration_ms = (time.monotonic() - start) * 1000.0
            return SWEResult(
                task_id=problem.task_id,
                patch_applied=False,
                tests_passed=False,
                stdout="",
                stderr="patch application failed",
                duration_ms=duration_ms,
            )
        passed, stdout, stderr = run_tests(problem.test_command, tmp, timeout=timeout_seconds)
        duration_ms = (time.monotonic() - start) * 1000.0
        return SWEResult(
            task_id=problem.task_id,
            patch_applied=True,
            tests_passed=passed,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_problems(
    problems: list[SWEProblem],
    candidate_patches: list[str],
    timeout_seconds: float = 30.0,
) -> dict:
    """Score a list of problems and aggregate pass@1.

    If there are no problems, returns ``{"pass@1": 0.0, "per_task": [],
    "n_problems": 0}``. Raises ``ValueError`` if lengths mismatch.
    """
    if len(problems) != len(candidate_patches):
        raise ValueError(
            f"problems ({len(problems)}) and candidate_patches "
            f"({len(candidate_patches)}) length mismatch"
        )
    if not problems:
        return {"pass@1": 0.0, "per_task": [], "n_problems": 0}

    per_task: list[dict] = []
    passed = 0
    for prob, patch in zip(problems, candidate_patches):
        result = score_single(prob, patch, timeout_seconds=timeout_seconds)
        per_task.append(
            {
                "task_id": result.task_id,
                "patch_applied": result.patch_applied,
                "tests_passed": result.tests_passed,
                "duration_ms": result.duration_ms,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.tests_passed:
            passed += 1

    return {
        "pass@1": passed / len(problems),
        "per_task": per_task,
        "n_problems": len(problems),
    }


__all__ = [
    "SWEProblem",
    "SWEResult",
    "materialize_repo",
    "apply_patch_via_python",
    "run_tests",
    "score_single",
    "score_problems",
]
