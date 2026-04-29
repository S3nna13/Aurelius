"""Adversarial / security tests for edit_tool, grep_tool, web_tool.

Control-ID: C-139. All assertions verify graceful failure — no exceptions raised.
"""

from __future__ import annotations

from src.tools.edit_tool import _MAX_CONTENT_LEN as EDIT_MAX
from src.tools.edit_tool import EditTool
from src.tools.grep_tool import GrepTool
from src.tools.web_tool import WebTool

# ── edit_tool adversarial ─────────────────────────────────────────────────────


def test_edit_search_not_found_no_exception():
    tool = EditTool()
    result = tool.apply_edit("hello world", "NOTPRESENT", "replacement")
    assert not result.success
    assert result.error != ""


def test_edit_ambiguous_search_no_exception():
    tool = EditTool()
    # "foo" appears twice — ambiguous
    result = tool.apply_edit("foo and foo again", "foo", "bar")
    assert not result.success
    assert "ambiguous" in result.error or result.error != ""


def test_edit_content_over_size_blocked():
    tool = EditTool()
    big = "A" * (EDIT_MAX + 1)
    result = tool.apply_edit(big, "A", "B")
    assert not result.success
    assert "exceeds" in result.error


def test_edit_empty_search_blocked():
    tool = EditTool()
    result = tool.apply_edit("some text", "", "anything")
    assert not result.success
    assert result.error != ""


# ── grep_tool adversarial ─────────────────────────────────────────────────────


def test_grep_invalid_regex_no_exception():
    tool = GrepTool()
    result = tool.search("[[[invalid", "some content here")
    assert not result.success
    assert "invalid pattern" in result.error


def test_grep_catastrophic_regex_small_input():
    tool = GrepTool()
    # Known backtracking pattern; small input so it terminates quickly
    result = tool.search(r"a+b", "a" * 15 + "c")
    # Must return (success or not) without hanging/excepting
    assert isinstance(result.success, bool)


# ── web_tool adversarial ──────────────────────────────────────────────────────


def test_web_ssrf_imds_blocked():
    tool = WebTool()
    result = tool.fetch("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
    assert not result.success
    assert result.error != ""


def test_web_ssrf_localhost_blocked():
    tool = WebTool()
    result = tool.fetch("http://localhost:8080/internal")
    assert not result.success


def test_web_file_scheme_blocked():
    tool = WebTool()
    result = tool.fetch("file:///etc/passwd")
    assert not result.success
    assert "not allowed" in result.error or result.error != ""


def test_web_url_length_bomb_blocked():
    tool = WebTool()
    long_url = "https://example.com/" + "x" * 3000
    result = tool.fetch(long_url)
    assert not result.success
    assert "exceeds" in result.error


def test_web_rfc1918_10_blocked():
    tool = WebTool()
    result = tool.fetch("http://10.10.10.10/secret")
    assert not result.success


def test_web_rfc1918_192_168_blocked():
    tool = WebTool()
    result = tool.fetch("http://192.168.0.1/router")
    assert not result.success
