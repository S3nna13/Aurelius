"""Unit tests for src/cli/output_compressor.py."""

from __future__ import annotations

from src.cli.output_compressor import (
    DEFAULT_COMPRESSOR,
    OUTPUT_COMPRESSOR_REGISTRY,
    CompressionConfig,
    OutputCompressor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> OutputCompressor:
    """Return a fresh compressor to avoid test pollution."""
    return OutputCompressor()


# ---------------------------------------------------------------------------
# 1. test_basic_compression_removes_empty_lines
# ---------------------------------------------------------------------------


def test_basic_compression_removes_empty_lines():
    comp = _fresh()
    text = "alpha\n\nbeta\n\n\ngamma"
    result = comp.compress(text)
    assert result.output == "alpha\nbeta\ngamma"
    assert "remove_empty" in result.strategies_applied
    assert result.original_lines == 6
    assert result.compressed_lines == 3


# ---------------------------------------------------------------------------
# 2. test_deduplication_of_repeated_lines
# ---------------------------------------------------------------------------


def test_deduplication_of_repeated_lines():
    comp = _fresh()
    text = "a\na\na\nb\nb"
    result = comp.compress(text)
    assert result.output == "a (x3)\nb (x2)"
    assert "deduplicate" in result.strategies_applied


# ---------------------------------------------------------------------------
# 3. test_line_length_truncation
# ---------------------------------------------------------------------------


def test_line_length_truncation():
    comp = OutputCompressor(config=CompressionConfig(max_line_length=10, max_lines=100))
    text = "x" * 20
    result = comp.compress(text)
    assert result.output == "xxxxxxx..."
    assert "truncate_lines" in result.strategies_applied


# ---------------------------------------------------------------------------
# 4. test_total_line_truncation_with_summary
# ---------------------------------------------------------------------------


def test_total_line_truncation_with_summary():
    comp = OutputCompressor(config=CompressionConfig(max_lines=3))
    text = "1\n2\n3\n4\n5"
    result = comp.compress(text)
    assert result.output == "1\n2\n3\n... 2 lines omitted"
    assert "truncate_total" in result.strategies_applied
    assert result.compressed_lines == 4


# ---------------------------------------------------------------------------
# 5. test_group_by_prefix
# ---------------------------------------------------------------------------


def test_group_by_prefix():
    comp = _fresh()
    text = "abc one\nabc two\nabc three\ndef four"
    result = comp.compress(text)
    assert "abc one" in result.output
    assert "abc... +2 more" in result.output
    assert "def four" in result.output
    assert "group_by_prefix" in result.strategies_applied


# ---------------------------------------------------------------------------
# 6. test_git_status_specialization
# ---------------------------------------------------------------------------


def test_git_status_specialization():
    comp = _fresh()
    text = (
        "On branch main\n"
        "Your branch is up to date with 'origin/main'.\n"
        "\n"
        "Changes to be committed:\n"
        '  (use "git restore --staged <file>..." to unstage)\n'
        "\tmodified:   a.py\n"
    )
    result = comp.compress_git_status(text)
    assert "On branch main" in result.output
    assert "modified:   a.py" in result.output
    assert 'use "git restore' not in result.output
    assert "remove_hints" in result.strategies_applied
    assert "compress_branch" in result.strategies_applied


# ---------------------------------------------------------------------------
# 7. test_ls_specialization
# ---------------------------------------------------------------------------


def test_ls_specialization():
    comp = _fresh()
    text = "foo.py\nbar.py\nbaz.txt\nqux/"
    result = comp.compress_ls(text)
    assert ".py: 2" in result.output
    assert ".txt: 1" in result.output
    assert "(dir): 1" in result.output
    assert "group_by_extension" in result.strategies_applied


# ---------------------------------------------------------------------------
# 8. test_test_output_failures_only
# ---------------------------------------------------------------------------


def test_test_output_failures_only():
    comp = _fresh()
    text = "test_a PASSED\ntest_b FAILED\ntest_c PASSED\n"
    result = comp.compress_test_output(text)
    assert "test_b FAILED" in result.output
    assert "test_a PASSED" not in result.output
    assert "2 passed" in result.output
    assert "failures_only" in result.strategies_applied


# ---------------------------------------------------------------------------
# 9. test_grep_output_specialization
# ---------------------------------------------------------------------------


def test_grep_output_specialization():
    comp = _fresh()
    text = "file1.txt:10:match\nfile1.txt:20:another\nfile2.txt:5:match\n"
    result = comp.compress_grep(text)
    assert "file1.txt: 2 matches" in result.output
    assert "file2.txt: 1 match" in result.output
    assert "group_by_file" in result.strategies_applied


# ---------------------------------------------------------------------------
# 10. test_compression_ratio_calculation
# ---------------------------------------------------------------------------


def test_compression_ratio_calculation():
    comp = _fresh()
    text = "a\n\nb\n\nc"
    result = comp.compress(text)
    assert result.original_lines == 5
    assert result.compressed_lines == 3
    assert result.compression_ratio == 3 / 5


# ---------------------------------------------------------------------------
# 11. test_config_customization
# ---------------------------------------------------------------------------


def test_config_customization():
    config = CompressionConfig(
        max_lines=10,
        max_line_length=4,
        deduplicate=False,
        group_by_prefix=False,
        remove_empty=False,
        show_summary=False,
    )
    comp = OutputCompressor(config=config)
    text = "\n\nabc\nabc\n12345\n67890\n"
    result = comp.compress(text)
    assert "1..." in result.output
    assert "6..." in result.output
    assert "truncate_lines" in result.strategies_applied


# ---------------------------------------------------------------------------
# 12. test_no_compression_needed_short_input
# ---------------------------------------------------------------------------


def test_no_compression_needed_short_input():
    comp = _fresh()
    text = "hello world"
    result = comp.compress(text)
    assert result.output == "hello world"
    assert result.strategies_applied == []
    assert result.compression_ratio == 1.0


# ---------------------------------------------------------------------------
# 13. test_registry_singleton_exists
# ---------------------------------------------------------------------------


def test_registry_singleton_exists():
    assert "default" in OUTPUT_COMPRESSOR_REGISTRY
    assert OUTPUT_COMPRESSOR_REGISTRY["default"] is DEFAULT_COMPRESSOR


# ---------------------------------------------------------------------------
# 14. test_custom_compressor_in_registry
# ---------------------------------------------------------------------------


def test_custom_compressor_in_registry():
    custom = OutputCompressor(config=CompressionConfig(max_lines=1))
    OUTPUT_COMPRESSOR_REGISTRY["custom"] = custom
    assert OUTPUT_COMPRESSOR_REGISTRY["custom"] is custom
    del OUTPUT_COMPRESSOR_REGISTRY["custom"]


# ---------------------------------------------------------------------------
# 15. test_large_input_handled_safely
# ---------------------------------------------------------------------------


def test_large_input_handled_safely():
    comp = _fresh()
    text = "\n".join([f"line {i}" for i in range(10_000)])
    result = comp.compress(text)
    assert result.compressed_lines <= comp.config.max_lines + 1
    assert result.original_lines == 10_000


# ---------------------------------------------------------------------------
# 16. test_empty_string_compression
# ---------------------------------------------------------------------------


def test_empty_string_compression():
    comp = _fresh()
    result = comp.compress("")
    assert result.output == ""
    assert result.original_lines == 0
    assert result.compressed_lines == 0
    assert result.compression_ratio == 0.0


# ---------------------------------------------------------------------------
# 17. test_deduplication_without_consecutive_repeat
# ---------------------------------------------------------------------------


def test_deduplication_without_consecutive_repeat():
    comp = _fresh()
    text = "a\nb\na\nb"
    result = comp.compress(text)
    assert " (x" not in result.output
    assert "deduplicate" not in result.strategies_applied


# ---------------------------------------------------------------------------
# 18. test_group_by_prefix_short_lines_ignored
# ---------------------------------------------------------------------------


def test_group_by_prefix_short_lines_ignored():
    comp = _fresh()
    text = "ab\nab\nac"
    result = comp.compress(text)
    # Lines shorter than 3 chars should not be grouped
    assert "... +" not in result.output
    assert "group_by_prefix" not in result.strategies_applied


# ---------------------------------------------------------------------------
# 19. test_show_summary_false_omits_summary_line
# ---------------------------------------------------------------------------


def test_show_summary_false_omits_summary_line():
    comp = OutputCompressor(config=CompressionConfig(max_lines=2, show_summary=False))
    text = "1\n2\n3\n4"
    result = comp.compress(text)
    assert "omitted" not in result.output
    assert result.compressed_lines == 2


# ---------------------------------------------------------------------------
# 20. test_git_status_keeps_untracked_files
# ---------------------------------------------------------------------------


def test_git_status_keeps_untracked_files():
    comp = _fresh()
    text = (
        "On branch dev\n"
        "Untracked files:\n"
        '  (use "git add <file>..." to include in what will be committed)\n'
        "\tnew_file.py\n"
    )
    result = comp.compress_git_status(text)
    assert "new_file.py" in result.output
    assert "Untracked files:" in result.output
    assert 'use "git add' not in result.output


# ---------------------------------------------------------------------------
# 21. test_ls_skips_total_line
# ---------------------------------------------------------------------------


def test_ls_skips_total_line():
    comp = _fresh()
    text = "total 42\nfoo.py\nbar.py"
    result = comp.compress_ls(text)
    assert "total" not in result.output
    assert ".py: 2" in result.output


# ---------------------------------------------------------------------------
# 22. test_grep_single_match_grammar
# ---------------------------------------------------------------------------


def test_grep_single_match_grammar():
    comp = _fresh()
    text = "single.txt:1:hello"
    result = comp.compress_grep(text)
    assert "single.txt: 1 match" in result.output
    assert "matches" not in result.output
