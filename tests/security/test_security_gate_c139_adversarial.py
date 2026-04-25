"""Security gate tests — C139 adversarial regression suite.

Verifies the four HIGH bandit findings fixed in sprint C139 remain clean,
and that ShellTool deny-list enforcement is intact.

Findings covered:
  C139-1  B605 os.system removed from src/cli/main.py
  C139-2  B602 shell=True in ShellTool is annotated; deny-list blocks dangerous cmds
  C139-3  B324 hashlib.md5 in synthetic_code.py uses usedforsecurity=False
  C139-4  B324 hashlib.sha1 in corpus_indexer.py uses usedforsecurity=False
  C139-5  ShellTool blocks every pattern in SHELL_DENY_PATTERNS
  C139-6  ShellTool allows benign commands (deny-list is not over-broad)
  C139-7  ShellTool nosec annotation is present in source
  C139-8  md5 deduplication still functions correctly after the fix
  C139-9  sha1 chunk-id is deterministic after the fix
"""

from __future__ import annotations

import pathlib

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
# C139-5  ShellTool allow-list rejects dangerous commands
# ---------------------------------------------------------------------------

def test_shell_tool_allow_list_rejects_dangerous_commands():
    """C139-5: ShellTool must reject dangerous commands not in the allow-list."""
    from src.tools.shell_tool import ShellTool

    tool = ShellTool()
    dangerous = [
        "rm -rf /tmp",
        "mkfs.ext4 /dev/sdb",
        "shutdown now",
        "reboot",
        "chmod 777 /",
    ]
    for cmd in dangerous:
        result = tool.run(cmd)
        assert result.success is False, (
            f"ShellTool failed to block dangerous command: {cmd!r}"
        )


# ---------------------------------------------------------------------------
# C139-6  ShellTool allows benign commands (allow-list is not over-broad)
# ---------------------------------------------------------------------------

def test_shell_tool_allows_benign_commands():
    """C139-6: ShellTool must allow benign commands in the allow-list."""
    from src.tools.shell_tool import ShellTool

    tool = ShellTool()
    benign = [
        "echo hello",
        "ls -la /tmp",
        "python3 --version",
        "cat /etc/hostname",
    ]
    for cmd in benign:
        result = tool.run(cmd)
        assert result.success is True, (
            f"ShellTool incorrectly blocked benign command: {cmd!r}"
        )


# ---------------------------------------------------------------------------
# C139-7  shell_tool.py uses shell=False (hardened post-C139)
# ---------------------------------------------------------------------------

def test_shell_tool_uses_shell_false():
    """C139-7: src/tools/shell_tool.py must use shell=False (not shell=True)."""
    src = pathlib.Path("src/tools/shell_tool.py").read_text()
    assert "shell=False" in src, (
        "shell=False missing from src/tools/shell_tool.py"
    )
    assert "shell=True" not in src, (
        "shell=True still present in src/tools/shell_tool.py"
    )


# ---------------------------------------------------------------------------
# C139-8  md5 deduplication still works correctly after usedforsecurity fix
# ---------------------------------------------------------------------------

def test_synthetic_code_deduplication_functional():
    """C139-8: SyntheticCodeGenerator.deduplicate() must still remove exact dupes."""
    from src.data.synthetic_code import SyntheticCodePipeline, CodeExample

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
    from src.retrieval.corpus_indexer import CorpusIndexer

    indexer = CorpusIndexer.__new__(CorpusIndexer)
    id1 = indexer._chunk_id("file.py", 0, 50, "def foo(): pass")
    id2 = indexer._chunk_id("file.py", 0, 50, "def foo(): pass")
    assert id1 == id2, (
        f"_chunk_id is not deterministic: {id1!r} != {id2!r}"
    )
    # Must also differ when content changes
    id3 = indexer._chunk_id("file.py", 0, 50, "def bar(): pass")
    assert id1 != id3, "_chunk_id does not distinguish different content"
