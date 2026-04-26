"""Security gate tests — C139 adversarial regression suite.

Verifies the four HIGH bandit findings fixed in sprint C139 remain clean,
and that ShellTool deny-list enforcement is intact.

Findings covered:
  C139-1  B605 os.system removed from src/cli/main.py
  C139-2  B602 shell=True removed from ShellTool; allow-list + shell=False enforced
  C139-3  B324 hashlib.md5 in synthetic_code.py uses usedforsecurity=False
  C139-4  B324 hashlib.sha1 in corpus_indexer.py uses usedforsecurity=False
  C139-5  ShellTool blocks every pattern in SHELL_DENY_PATTERNS
  C139-6  ShellTool allows benign commands (deny-list is not over-broad)
  C139-7  ShellTool uses shell=False and allow-list validation in source
  C139-8  md5 deduplication still functions correctly after the fix
  C139-9  sha1 chunk-id is deterministic after the fix
"""

from __future__ import annotations

import pathlib

from src.data.synthetic_code import CodeExample, SyntheticCodePipeline
from src.retrieval.corpus_indexer import CorpusIndexer
from src.tools.shell_tool import SHELL_DENY_PATTERNS, ShellTool

# Forbidden call fragment — split so this source file doesn't match its own check.
_FORBIDDEN_CALL = "os." + "system("


# ---------------------------------------------------------------------------
# C139-1  os.system must NOT appear in src/cli/main.py
# ---------------------------------------------------------------------------

def test_no_os_system_in_cli_main():
    """C139-1: src/cli/main.py must not call os.system()."""
    src = pathlib.Path("src/cli/main.py").read_text()
    assert _FORBIDDEN_CALL not in src, (
        f"{_FORBIDDEN_CALL} found in src/cli/main.py — B605 regression detected"
    )


# ---------------------------------------------------------------------------
# C139-2  subprocess.run in cli/main.py (the /clear command) must exist
# ---------------------------------------------------------------------------

def test_cli_clear_uses_subprocess_run():
    """C139-2 (cli): /clear must use subprocess.run, not os.system."""
    src = pathlib.Path("src/cli/main.py").read_text()
    assert "subprocess.run" in src, (
        "subprocess.run not found in src/cli/main.py — clear-command fix missing"
    )


# ---------------------------------------------------------------------------
# C139-3  hashlib.md5 in synthetic_code.py must pass usedforsecurity=False
# ---------------------------------------------------------------------------

def test_synthetic_code_md5_uses_usedforsecurity_false():
    """C139-3: hashlib.md5 in synthetic_code.py must pass usedforsecurity=False."""
    src = pathlib.Path("src/data/synthetic_code.py").read_text()
    assert "usedforsecurity=False" in src, (
        "usedforsecurity=False missing from hashlib.md5 call in synthetic_code.py"
    )
    # Confirm no bare md5() call (without the kwarg) exists
    assert "hashlib.md5(ex.code.encode()).hexdigest()" not in src, (
        "Bare hashlib.md5() call (without usedforsecurity=False) still present"
    )


# ---------------------------------------------------------------------------
# C139-4  hashlib.sha1 in corpus_indexer.py must pass usedforsecurity=False
# ---------------------------------------------------------------------------

def test_corpus_indexer_sha1_uses_usedforsecurity_false():
    """C139-4: hashlib.sha1 in corpus_indexer.py must pass usedforsecurity=False."""
    src = pathlib.Path("src/retrieval/corpus_indexer.py").read_text()
    assert "usedforsecurity=False" in src, (
        "usedforsecurity=False missing from hashlib.sha1 call in corpus_indexer.py"
    )
    assert "hashlib.sha1()" not in src, (
        "Bare hashlib.sha1() call (without usedforsecurity=False) still present"
    )


# ---------------------------------------------------------------------------
# C139-5  ShellTool deny-list blocks every listed dangerous pattern
# ---------------------------------------------------------------------------

def test_shell_tool_deny_list_blocks_all_patterns():
    """C139-5: ShellTool.is_denied() must return True for every deny pattern."""
    tool = ShellTool()
    for pattern in SHELL_DENY_PATTERNS:
        cmd = f"echo hello; {pattern}"
        assert tool.is_denied(cmd), (
            f"ShellTool failed to block deny-pattern: {pattern!r}"
        )


# ---------------------------------------------------------------------------
# C139-6  ShellTool allows benign commands (deny-list is not over-broad)
# ---------------------------------------------------------------------------

def test_shell_tool_allows_benign_commands():
    """C139-6: ShellTool.is_denied() must return False for benign commands."""
    tool = ShellTool()
    benign = [
        "echo hello",
        "ls -la /tmp",
        "python3 --version",
        "cat /etc/hostname",
    ]
    for cmd in benign:
        assert not tool.is_denied(cmd), (
            f"ShellTool incorrectly blocked benign command: {cmd!r}"
        )


# ---------------------------------------------------------------------------
# C139-7  shell_tool.py uses shell=False and allow-list validation (B602 eliminated)
# ---------------------------------------------------------------------------

def test_shell_tool_uses_shell_false_and_allowlist():
    """C139-7: shell_tool.py must use shell=False and validate against an allow-list."""
    src = pathlib.Path("src/tools/shell_tool.py").read_text()
    # shell=True must be gone — it was the original B602 finding
    assert "shell=True" not in src, (
        "shell=True still present in src/tools/shell_tool.py — B602 not eliminated"
    )
    # shell=False must be present as the safe replacement
    assert "shell=False" in src, (
        "shell=False missing from src/tools/shell_tool.py — safe execution not enforced"
    )
    # Allow-list validation must exist so we only run known-safe base commands
    assert "ALLOWLIST" in src, (
        "ALLOWLIST missing from src/tools/shell_tool.py — command allow-list not enforced"
    )
    # shlex.split must be used for safe tokenization
    assert "shlex.split" in src, (
        "shlex.split missing from src/tools/shell_tool.py — safe tokenization not enforced"
    )


# ---------------------------------------------------------------------------
# C139-8  md5 deduplication still works correctly after usedforsecurity fix
# ---------------------------------------------------------------------------

def test_synthetic_code_deduplication_functional():
    """C139-8: SyntheticCodeGenerator.deduplicate() must still remove exact dupes."""
    gen = SyntheticCodePipeline.__new__(SyntheticCodePipeline)

    def _make(code: str) -> CodeExample:
        return CodeExample(
            instruction="test",
            code=code,
            language="python",
            difficulty="easy",
        )

    examples = [_make("x = 1"), _make("x = 2"), _make("x = 1")]
    deduped = gen.deduplicate(examples)
    assert len(deduped) == 2, (
        f"Expected 2 unique examples after dedup, got {len(deduped)}"
    )
    codes = [e.code for e in deduped]
    assert "x = 1" in codes
    assert "x = 2" in codes


# ---------------------------------------------------------------------------
# C139-9  sha1 chunk-id is deterministic after usedforsecurity fix
# ---------------------------------------------------------------------------

def test_corpus_indexer_chunk_id_deterministic():
    """C139-9: CorpusIndexer._chunk_id() must return identical hashes on repeated calls."""
    indexer = CorpusIndexer.__new__(CorpusIndexer)
    id1 = indexer._chunk_id("file.py", 0, 50, "def foo(): pass")
    id2 = indexer._chunk_id("file.py", 0, 50, "def foo(): pass")
    assert id1 == id2, (
        f"_chunk_id is not deterministic: {id1!r} != {id2!r}"
    )
    # Must also differ when content changes
    id3 = indexer._chunk_id("file.py", 0, 50, "def bar(): pass")
    assert id1 != id3, "_chunk_id does not distinguish different content"
