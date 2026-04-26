"""Unit tests for `src/agent/web_browse_tool.py`.

These tests exercise **pure logic only** — no real HTTP requests.
"""

from __future__ import annotations

import pytest

from src.agent.web_browse_tool import (
    DEFAULT_TOOL_DESCRIPTOR,
    PrivateHostBlocked,
    UrlValidationError,
    WebBrowseTool,
    WebFetchResult,
    WebRequestSpec,
)

# ---------------------------------------------------------------------------
# build_request shape & defaults
# ---------------------------------------------------------------------------


def test_build_request_shape_defaults():
    tool = WebBrowseTool()
    spec = tool.build_request("https://example.com/")
    assert isinstance(spec, WebRequestSpec)
    assert spec.method == "GET"
    assert spec.url == "https://example.com/"
    assert spec.timeout_s == 10.0
    assert spec.max_bytes == 2_000_000
    assert spec.follow_redirects is True
    # default user agent surfaces into headers
    assert spec.headers.get("User-Agent") == "Aurelius-Agent/1.0"


def test_build_request_method_uppercased():
    tool = WebBrowseTool()
    spec = tool.build_request("https://example.com/", method="get")
    assert spec.method == "GET"


def test_build_request_headers_default_empty():
    tool = WebBrowseTool(default_user_agent=None)
    spec = tool.build_request("https://example.com/")
    assert spec.headers == {}


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="CONNECT", url="https://example.com/")


def test_non_http_scheme_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="GET", url="ftp://example.com/")


def test_localhost_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://localhost/")


def test_ipv6_localhost_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://[::1]/")


def test_10_net_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://10.0.0.1/")


def test_192_168_net_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://192.168.1.1/")


def test_172_16_net_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://172.16.3.4/")


def test_127_loopback_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://127.0.0.1:8080/")


def test_link_local_blocked():
    with pytest.raises(PrivateHostBlocked):
        WebRequestSpec(method="GET", url="http://169.254.169.254/latest/meta-data/")


def test_public_ip_allowed():
    # 8.8.8.8 is Google public DNS — outside any private range.
    spec = WebRequestSpec(method="GET", url="http://8.8.8.8/")
    assert spec.url == "http://8.8.8.8/"


def test_timeout_zero_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="GET", url="https://example.com/", timeout_s=0)


def test_timeout_negative_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="GET", url="https://example.com/", timeout_s=-1.0)


def test_max_bytes_zero_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="GET", url="https://example.com/", max_bytes=0)


def test_max_bytes_negative_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="GET", url="https://example.com/", max_bytes=-5)


def test_empty_url_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(method="GET", url="")


def test_headers_non_string_raises():
    with pytest.raises(UrlValidationError):
        WebRequestSpec(
            method="GET",
            url="https://example.com/",
            headers={"X-Foo": 123},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# validate_request & summarize_result
# ---------------------------------------------------------------------------


def test_validate_request_accepts_valid():
    tool = WebBrowseTool()
    spec = tool.build_request("https://example.com/")
    tool.validate_request(spec)  # does not raise


def test_validate_request_rejects_wrong_type():
    tool = WebBrowseTool()
    with pytest.raises(UrlValidationError):
        tool.validate_request("not a spec")  # type: ignore[arg-type]


def test_summarize_result_truncates_at_max_chars():
    result = WebFetchResult(
        status=200,
        url="https://ex.test/",
        body_sample="q" * 5000,
        bytes_read=5000,
        elapsed_s=0.123,
    )
    summary = WebBrowseTool.summarize_result(result, max_chars=100)
    assert "HTTP 200" in summary
    assert "truncated" in summary
    assert "q" * 100 in summary
    # body portion truncated to exactly 100 q's
    assert summary.count("q") == 100


def test_summarize_result_no_truncation_when_small():
    result = WebFetchResult(
        status=200,
        url="https://example.com/",
        body_sample="hello",
        bytes_read=5,
        elapsed_s=0.001,
    )
    summary = WebBrowseTool.summarize_result(result)
    assert "hello" in summary
    assert "truncated" not in summary


def test_summarize_result_rejects_bad_types():
    with pytest.raises(UrlValidationError):
        WebBrowseTool.summarize_result("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Descriptor schema
# ---------------------------------------------------------------------------


def test_descriptor_required_keys():
    for key in ("name", "description", "parameters"):
        assert key in DEFAULT_TOOL_DESCRIPTOR
    params = DEFAULT_TOOL_DESCRIPTOR["parameters"]
    assert params["type"] == "object"
    assert "url" in params["properties"]
    assert "url" in params["required"]


def test_descriptor_enumerates_methods():
    methods = DEFAULT_TOOL_DESCRIPTOR["parameters"]["properties"]["method"]["enum"]
    assert set(methods) == {"GET", "HEAD", "POST", "PUT", "DELETE"}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_request_deterministic():
    tool = WebBrowseTool()
    a = tool.build_request("https://example.com/a", headers={"X": "1"})
    b = tool.build_request("https://example.com/a", headers={"X": "1"})
    assert a == b


# ---------------------------------------------------------------------------
# IDN / unicode URLs
# ---------------------------------------------------------------------------


def test_unicode_url_accepts_or_rejects_cleanly():
    # IDN host: bücher.example should either validate cleanly or raise
    # UrlValidationError — it must never propagate a raw UnicodeError.
    tool = WebBrowseTool()
    url = "https://bücher.example/"
    try:
        spec = tool.build_request(url)
        assert spec.url == url
    except UrlValidationError:
        pass


def test_malformed_idn_rejected_cleanly():
    # Pure-ASCII path can still contain a bogus label; exercise the
    # idna-encode path with a deliberately invalid host.
    with pytest.raises(UrlValidationError):
        # A label longer than 63 chars fails idna encoding.
        WebRequestSpec(
            method="GET",
            url="https://" + ("a" * 64) + ".example/",
        )


# ---------------------------------------------------------------------------
# Redirect handling is still structural — target URLs must validate too.
# ---------------------------------------------------------------------------


def test_redirect_target_validated():
    # If the runtime later follows a redirect to a private host, the
    # same validator must refuse the resulting URL.
    tool = WebBrowseTool()
    with pytest.raises(PrivateHostBlocked):
        tool.build_request("http://10.0.0.5/redirected")


# ---------------------------------------------------------------------------
# Unknown overrides are rejected (typo safety).
# ---------------------------------------------------------------------------


def test_unknown_override_rejected():
    tool = WebBrowseTool()
    with pytest.raises(UrlValidationError):
        tool.build_request("https://example.com/", verify_ssl=False)
