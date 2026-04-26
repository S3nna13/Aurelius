import tempfile

from src.tools.diff_tool import DIFF_TOOL_REGISTRY, DiffFormat, DiffResult, DiffTool

A = "line1\nline2\nline3\n"
B = "line1\nlineX\nline3\nline4\n"


def test_unified_diff():
    tool = DiffTool()
    result = tool.diff_strings(A, B)
    assert isinstance(result, DiffResult)
    assert result.lines_added >= 1
    assert result.lines_removed >= 1
    assert "@@" in result.diff_text


def test_unified_hunk_count():
    tool = DiffTool()
    result = tool.diff_strings(A, B, fmt=DiffFormat.UNIFIED)
    assert result.hunks == result.diff_text.count("@@") // 2


def test_context_diff():
    tool = DiffTool()
    result = tool.diff_strings(A, B, fmt=DiffFormat.CONTEXT)
    assert result.lines_added >= 1
    assert result.hunks >= 1
    assert "***" in result.diff_text


def test_html_diff():
    tool = DiffTool()
    result = tool.diff_strings(A, B, fmt=DiffFormat.HTML)
    assert "<html" in result.diff_text.lower() or "<!DOCTYPE" in result.diff_text


def test_side_by_side_text():
    tool = DiffTool()
    result = tool.diff_strings(A, B, fmt=DiffFormat.SIDE_BY_SIDE_TEXT)
    assert isinstance(result.diff_text, str)


def test_identical_strings_no_diff():
    tool = DiffTool()
    result = tool.diff_strings(A, A)
    assert result.lines_added == 0
    assert result.lines_removed == 0
    assert result.hunks == 0


def test_similarity_identical():
    tool = DiffTool()
    assert tool.similarity("hello", "hello") == 1.0


def test_similarity_different():
    tool = DiffTool()
    s = tool.similarity("abc", "xyz")
    assert 0.0 <= s < 1.0


def test_diff_files():
    tool = DiffTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fa:
        fa.write(A)
        pa = fa.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fb:
        fb.write(B)
        pb = fb.name
    result = tool.diff_files(pa, pb)
    assert result.lines_added >= 1


def test_apply_patch_stub():
    tool = DiffTool()
    assert tool.apply_patch("original", []) == "original"


def test_registry_key():
    assert "default" in DIFF_TOOL_REGISTRY
    assert DIFF_TOOL_REGISTRY["default"] is DiffTool
