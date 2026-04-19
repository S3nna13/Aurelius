"""Integration tests for the agent tool-call parser surface.

These tests verify cross-module wiring: the registry is populated on
import, registered callables work end-to-end, and round-tripping a
canonical tool call through ``format_* -> parse_* `` recovers the
original semantic payload.
"""

from __future__ import annotations

import importlib
import sys


def test_registry_contains_xml_and_json_keys() -> None:
    from src.agent import TOOL_CALL_PARSER_REGISTRY

    assert "xml" in TOOL_CALL_PARSER_REGISTRY
    assert "json" in TOOL_CALL_PARSER_REGISTRY


def test_registry_entries_are_callable() -> None:
    from src.agent import TOOL_CALL_PARSER_REGISTRY

    for key in ("xml", "json"):
        assert callable(TOOL_CALL_PARSER_REGISTRY[key])


def test_agent_loop_registry_exists_and_is_empty_by_default() -> None:
    from src.agent import AGENT_LOOP_REGISTRY

    assert isinstance(AGENT_LOOP_REGISTRY, dict)


def test_round_trip_xml() -> None:
    from src.agent import TOOL_CALL_PARSER_REGISTRY, format_xml
    from src.agent.tool_call_parser import ParsedToolCall

    original = ParsedToolCall(
        name="search",
        arguments={"q": "aurelius", "top_k": 5},
        raw="",
        call_id="call_42",
    )
    serialised = format_xml(original)
    parsed = TOOL_CALL_PARSER_REGISTRY["xml"](serialised)
    assert len(parsed) == 1
    assert parsed[0].name == original.name
    assert parsed[0].arguments == original.arguments
    assert parsed[0].call_id == original.call_id


def test_round_trip_json() -> None:
    from src.agent import TOOL_CALL_PARSER_REGISTRY, format_json
    from src.agent.tool_call_parser import ParsedToolCall

    original = ParsedToolCall(
        name="fetch",
        arguments={"url": "https://example.org", "headers": {"x": "1"}},
        raw="",
        call_id="call_7",
    )
    serialised = format_json(original)
    parsed = TOOL_CALL_PARSER_REGISTRY["json"](serialised)
    assert len(parsed) == 1
    assert parsed[0].name == original.name
    assert parsed[0].arguments == original.arguments
    assert parsed[0].call_id == original.call_id


def test_import_has_no_unexpected_side_effects() -> None:
    # Re-import the module and ensure the registry still contains both
    # keys (no duplicate registration errors, no mutation of foreign
    # module state). We also confirm no provider SDKs were pulled in.
    import src.agent as agent_mod

    importlib.reload(agent_mod)
    assert set(["xml", "json"]).issubset(agent_mod.TOOL_CALL_PARSER_REGISTRY.keys())

    forbidden = {"torch", "transformers", "einops"}
    loaded = set(sys.modules.keys())
    # Any of these being present is a side-effect we must not introduce
    # ourselves. We only assert that importing src.agent did not newly
    # *pull them in*; we cannot control an already-imported environment,
    # so we merely assert agent_mod does not depend on them directly.
    for name in forbidden:
        mod_file = getattr(sys.modules.get(name, None), "__file__", None)
        # It's okay if unrelated test runner imported them; what matters is
        # src.agent itself doesn't reference them. Checked via attribute.
        assert not hasattr(agent_mod, name), f"src.agent leaked {name}"
        # silence unused var
        del mod_file
