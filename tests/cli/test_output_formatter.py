"""Tests for src/cli/output_formatter.py."""

import json

from src.cli.output_formatter import FormatterConfig, OutputFormat, OutputFormatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(fmt: OutputFormat, **kwargs) -> FormatterConfig:
    return FormatterConfig(format=fmt, **kwargs)


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------


def test_output_format_values():
    assert OutputFormat.PLAIN == "plain"
    assert OutputFormat.JSON == "json"
    assert len(OutputFormat) == 5


# ---------------------------------------------------------------------------
# PLAIN
# ---------------------------------------------------------------------------


def test_plain_returns_text_as_is():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.PLAIN)
    text = "Hello, world!"
    assert formatter.format(text, cfg) == text


def test_plain_preserves_whitespace():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.PLAIN)
    text = "line1\n\nline2\n"
    assert formatter.format(text, cfg) == text


# ---------------------------------------------------------------------------
# STREAM
# ---------------------------------------------------------------------------


def test_stream_returns_text_unchanged():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.STREAM)
    text = "streaming token"
    assert formatter.format(text, cfg) == text


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_json_format_without_tokens():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.JSON, show_tokens=False)
    result = json.loads(formatter.format("hello world", cfg))
    assert result["response"] == "hello world"
    assert "tokens" not in result


def test_json_format_with_tokens():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.JSON, show_tokens=True)
    result = json.loads(formatter.format("hello world", cfg))
    assert result["tokens"] == 2


def test_json_format_is_valid_json():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.JSON)
    raw = formatter.format('{"nested": true}', cfg)
    parsed = json.loads(raw)
    assert "response" in parsed


# ---------------------------------------------------------------------------
# COMPACT
# ---------------------------------------------------------------------------


def test_compact_strips_blank_lines():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.COMPACT)
    text = "line1\n\n\nline2\n\nline3"
    result = formatter.format(text, cfg)
    assert "\n\n" not in result
    assert "line1" in result
    assert "line2" in result


def test_compact_wraps_long_lines():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.COMPACT, max_width=10)
    text = "A" * 25
    result = formatter.format(text, cfg)
    for line in result.splitlines():
        assert len(line) <= 10


# ---------------------------------------------------------------------------
# RICH_MARKDOWN
# ---------------------------------------------------------------------------


def test_rich_markdown_does_not_double_wrap_fenced_blocks():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.RICH_MARKDOWN)
    text = "```python\nprint('hi')\n```"
    result = formatter.format(text, cfg)
    # Should not add extra fences
    assert result.count("```") == 2


def test_rich_markdown_passes_through_plain_text():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.RICH_MARKDOWN)
    text = "Just plain text."
    result = formatter.format(text, cfg)
    assert "Just plain text." in result


# ---------------------------------------------------------------------------
# format_error
# ---------------------------------------------------------------------------


def test_format_error_plain():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.PLAIN)
    err = ValueError("something went wrong")
    result = formatter.format_error(err, cfg)
    assert "ValueError" in result
    assert "something went wrong" in result


def test_format_error_json():
    formatter = OutputFormatter()
    cfg = make_config(OutputFormat.JSON)
    err = RuntimeError("boom")
    result = json.loads(formatter.format_error(err, cfg))
    assert result["type"] == "RuntimeError"
    assert result["error"] == "boom"


# ---------------------------------------------------------------------------
# format_tokens_used
# ---------------------------------------------------------------------------


def test_format_tokens_used():
    formatter = OutputFormatter()
    result = formatter.format_tokens_used(100, 50)
    assert "100" in result
    assert "50" in result
    assert "150" in result
