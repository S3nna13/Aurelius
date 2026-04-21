"""Integration: web_browse_tool wiring into agent package + config flag."""

from __future__ import annotations

from src import agent as agent_pkg
from src.agent.web_browse_tool import DEFAULT_TOOL_DESCRIPTOR, WebBrowseTool
from src.model.config import AureliusConfig


def test_tool_registered_in_agent_package():
    assert hasattr(agent_pkg, "TOOL_REGISTRY")
    assert "web_browse" in agent_pkg.TOOL_REGISTRY
    assert agent_pkg.TOOL_REGISTRY["web_browse"] is DEFAULT_TOOL_DESCRIPTOR


def test_public_symbols_exported():
    for name in (
        "WebBrowseTool",
        "WebRequestSpec",
        "WebFetchResult",
        "UrlValidationError",
        "PrivateHostBlocked",
        "WEB_BROWSE_TOOL_DESCRIPTOR",
        "TOOL_REGISTRY",
    ):
        assert name in agent_pkg.__all__, name
        assert hasattr(agent_pkg, name), name


def test_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert cfg.agent_web_browse_tool_enabled is False


def test_end_to_end_build_validate_summarize():
    # Exercise the full tool surface without any I/O.
    tool = WebBrowseTool()
    spec = tool.build_request("https://example.com/path", headers={"Accept": "text/html"})
    tool.validate_request(spec)

    from src.agent.web_browse_tool import WebFetchResult

    fake = WebFetchResult(
        status=200,
        url=spec.url,
        body_sample="hello world",
        bytes_read=11,
        elapsed_s=0.05,
    )
    summary = tool.summarize_result(fake)
    assert "HTTP 200" in summary
    assert "hello world" in summary
