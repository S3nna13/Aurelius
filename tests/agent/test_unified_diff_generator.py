"""Unit tests for :mod:`src.agent.unified_diff_generator`."""

from __future__ import annotations

import pytest

from src.agent.unified_diff_generator import DiffResult, UnifiedDiffGenerator


def test_modify_single_file_contains_hunk_header():
    gen = UnifiedDiffGenerator()
    before = "a\nb\nc\n"
    after = "a\nB\nc\n"
    res = gen.from_single_edit("f.py", before, after)
    assert isinstance(res, DiffResult)
    assert "@@" in res.diff
    assert "--- a/f.py" in res.diff
    assert "+++ b/f.py" in res.diff
    assert res.changed_files == ["f.py"]
    assert res.line_changes == 2  # one - and one +


def test_create_new_file_uses_devnull_source():
    gen = UnifiedDiffGenerator()
    res = gen.from_file_pairs({}, {"new.py": "hello\nworld\n"})
    assert "--- /dev/null" in res.diff
    assert "+++ b/new.py" in res.diff
    assert res.changed_files == ["new.py"]
    assert res.line_changes == 2


def test_delete_file_uses_devnull_target():
    gen = UnifiedDiffGenerator()
    res = gen.from_file_pairs({"old.py": "x\ny\n"}, {})
    assert "--- a/old.py" in res.diff
    assert "+++ /dev/null" in res.diff
    assert res.changed_files == ["old.py"]
    assert res.line_changes == 2


def test_round_trip_recovers_after_state():
    gen = UnifiedDiffGenerator()
    before = {"f.py": "a\nb\nc\n", "g.py": "keep\n"}
    after = {"f.py": "a\nB\nc\n", "g.py": "keep\n", "new.py": "created\n"}
    res = gen.from_file_pairs(before, after)
    recovered = gen.apply_round_trip(before, res.diff)
    # g.py unchanged; f.py modified; new.py created.
    assert recovered["f.py"] == after["f.py"]
    assert recovered["new.py"] == after["new.py"]
    assert recovered["g.py"] == "keep\n"


def test_multi_file_changes_all_in_diff():
    gen = UnifiedDiffGenerator()
    before = {"a.py": "1\n", "b.py": "2\n", "c.py": "3\n"}
    after = {"a.py": "one\n", "b.py": "2\n", "d.py": "four\n"}
    res = gen.from_file_pairs(before, after)
    # a.py modify, b.py unchanged (skipped), c.py delete, d.py create.
    assert "a.py" in res.changed_files
    assert "c.py" in res.changed_files
    assert "d.py" in res.changed_files
    assert "b.py" not in res.changed_files
    # sorted determinism
    assert res.changed_files == sorted(res.changed_files)


def test_context_lines_zero_works():
    gen = UnifiedDiffGenerator(context_lines=0)
    before = "a\nb\nc\nd\n"
    after = "a\nb\nX\nd\n"
    res = gen.from_single_edit("f.py", before, after)
    assert "@@" in res.diff
    # No surrounding context: the only payload lines should be -c and +X.
    payload_lines = [
        ln
        for ln in res.diff.splitlines()
        if ln and ln[0] in (" ", "+", "-") and not ln.startswith(("+++", "---"))
    ]
    assert "-c" in payload_lines
    assert "+X" in payload_lines
    # No " a"/" b"/" d" context lines.
    assert not any(ln.startswith(" ") for ln in payload_lines)


def test_validate_well_formed_returns_true():
    gen = UnifiedDiffGenerator()
    res = gen.from_single_edit("f.py", "a\n", "b\n")
    assert gen.validate(res.diff) is True


def test_validate_garbage_returns_false():
    gen = UnifiedDiffGenerator()
    assert gen.validate("this is not a diff\njust some text\n") is False
    assert gen.validate("@@ random @@\n+foo\n") is False
    assert gen.validate(123) is False  # type: ignore[arg-type]


def test_empty_inputs_yield_empty_diff():
    gen = UnifiedDiffGenerator()
    res = gen.from_file_pairs({}, {})
    assert res.diff == ""
    assert res.changed_files == []
    assert res.line_changes == 0
    # Empty diff should validate as a trivial no-op.
    assert gen.validate(res.diff) is True


def test_identical_before_after_empty_diff():
    gen = UnifiedDiffGenerator()
    res = gen.from_file_pairs({"a.py": "same\n"}, {"a.py": "same\n"})
    assert res.diff == ""
    assert res.line_changes == 0
    assert res.changed_files == []


def test_determinism():
    gen = UnifiedDiffGenerator()
    before = {"b.py": "2\n", "a.py": "1\n"}
    after = {"a.py": "one\n", "b.py": "2\n", "c.py": "new\n"}
    r1 = gen.from_file_pairs(before, after)
    r2 = gen.from_file_pairs(before, after)
    assert r1.diff == r2.diff
    assert r1.changed_files == r2.changed_files


def test_large_content_10k_lines():
    gen = UnifiedDiffGenerator()
    before = "\n".join(str(i) for i in range(10_000)) + "\n"
    # Change one line in the middle.
    after_lines = [str(i) for i in range(10_000)]
    after_lines[5000] = "CHANGED"
    after = "\n".join(after_lines) + "\n"
    res = gen.from_single_edit("big.txt", before, after)
    assert "@@" in res.diff
    assert res.line_changes == 2
    recovered = gen.apply_round_trip({"big.txt": before}, res.diff)
    assert recovered["big.txt"] == after


def test_line_changes_counts_adds_plus_deletes():
    gen = UnifiedDiffGenerator()
    before = "a\nb\nc\n"
    after = "a\nb\nc\nd\ne\n"  # two additions, zero deletions
    res = gen.from_single_edit("f.py", before, after)
    assert res.line_changes == 2

    before2 = "a\nb\nc\nd\n"
    after2 = "a\nd\n"  # delete b and c
    res2 = gen.from_single_edit("g.py", before2, after2)
    # b and c removed = 2 changes
    assert res2.line_changes == 2


def test_non_string_content_raises():
    """Binary-like / non-str content is rejected up-front.

    Documented behaviour: this generator handles text only. Callers must
    decode bytes (or skip binary files) before feeding them in.
    """
    gen = UnifiedDiffGenerator()
    with pytest.raises(TypeError):
        gen.from_single_edit("f.bin", b"\x00\x01", "text\n")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        gen.from_file_pairs({"a": "ok\n"}, {"a": b"bytes"})  # type: ignore[dict-item]


def test_invalid_context_lines_rejected():
    with pytest.raises(ValueError):
        UnifiedDiffGenerator(context_lines=-1)


def test_round_trip_delete():
    gen = UnifiedDiffGenerator()
    before = {"doomed.py": "bye\n"}
    after: dict[str, str] = {}
    res = gen.from_file_pairs(before, after)
    recovered = gen.apply_round_trip(before, res.diff)
    assert "doomed.py" not in recovered
