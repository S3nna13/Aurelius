"""Integration tests: FIM formatter is exposed on src.chat surface."""

from __future__ import annotations

import src.chat as chat
from src.chat import (
    CHAT_TEMPLATE_REGISTRY,
    MESSAGE_FORMAT_REGISTRY,
    FIM_MIDDLE,
    FIM_PAD,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIMExample,
    FIMFormatter,
)


def test_constants_exposed() -> None:
    assert FIM_PREFIX == "<fim_prefix>"
    assert FIM_SUFFIX == "<fim_suffix>"
    assert FIM_MIDDLE == "<fim_middle>"
    assert FIM_PAD == "<fim_pad>"


def test_formatter_class_exposed() -> None:
    assert hasattr(chat, "FIMFormatter")
    assert hasattr(chat, "FIMExample")


def test_prior_registry_entries_intact() -> None:
    # Existing registry entries from chatml / llama3 / harmony must still
    # be resolvable after the additive FIM import.
    for key in ("chatml", "llama3", "harmony"):
        assert key in CHAT_TEMPLATE_REGISTRY
    for key in ("chatml", "tool_result", "harmony"):
        assert key in MESSAGE_FORMAT_REGISTRY


def test_format_parse_roundtrip_via_surface() -> None:
    f = FIMFormatter(mode="psm")
    ex = FIMExample(
        prefix="def add(a, b):\n    return ",
        middle="a + b",
        suffix="\n\nadd(1, 2)\n",
    )
    text = f.format(ex)
    assert FIM_PREFIX in text and FIM_SUFFIX in text and FIM_MIDDLE in text
    assert f.parse(text) == ex


def test_spm_roundtrip_via_surface() -> None:
    f = FIMFormatter(mode="spm")
    ex = FIMExample(prefix="p", middle="m", suffix="s")
    assert f.parse(f.format(ex)) == ex


def test_inference_format_via_surface() -> None:
    f = FIMFormatter(mode="psm")
    out = f.format_for_inference("pre", "suf")
    assert out.endswith(FIM_MIDDLE)
    assert "pre" in out and "suf" in out
