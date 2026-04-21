"""Integration tests — Harmony template in the CHAT_TEMPLATE_REGISTRY.

Verifies that:
  1. "harmony" is present in CHAT_TEMPLATE_REGISTRY.
  2. The registered object can be constructed from the registry and renders
     user+assistant messages with the expected GPT-OSS-120B delimiters.
  3. Existing keys in the registry (regression guard) are unaffected.
"""

from __future__ import annotations

import pytest

from src.chat import CHAT_TEMPLATE_REGISTRY
from src.chat.harmony_template import HarmonyTemplate


# ---------------------------------------------------------------------------
# 1. Registry contains "harmony"
# ---------------------------------------------------------------------------

def test_harmony_key_in_registry() -> None:
    assert "harmony" in CHAT_TEMPLATE_REGISTRY, (
        "'harmony' must be registered in CHAT_TEMPLATE_REGISTRY"
    )


# ---------------------------------------------------------------------------
# 2. Registry value is a HarmonyTemplate instance
# ---------------------------------------------------------------------------

def test_harmony_registry_value_is_instance() -> None:
    assert isinstance(CHAT_TEMPLATE_REGISTRY["harmony"], HarmonyTemplate)


# ---------------------------------------------------------------------------
# 3. Construct from registry, render user+assistant, check delimiters
# ---------------------------------------------------------------------------

def test_render_user_assistant_from_registry() -> None:
    tpl = CHAT_TEMPLATE_REGISTRY["harmony"]
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    out = tpl.render(msgs)

    # BOS
    assert out.startswith("<|begin_of_text|>")
    # User delimiters
    assert "<|user|>" in out
    assert "<|end_user|>" in out
    # Assistant delimiters
    assert "<|assistant|>" in out
    assert "<|end_assistant|>" in out
    # Content present
    assert "What is 2+2?" in out
    assert "4" in out


# ---------------------------------------------------------------------------
# 4. Existing registry key "chatml" is unaffected (regression guard)
# ---------------------------------------------------------------------------

def test_chatml_key_still_present_after_harmony_registration() -> None:
    assert "chatml" in CHAT_TEMPLATE_REGISTRY, (
        "Registering 'harmony' must not remove 'chatml'"
    )


# ---------------------------------------------------------------------------
# 5. Existing registry key "llama3" is unaffected (regression guard)
# ---------------------------------------------------------------------------

def test_llama3_key_still_present_after_harmony_registration() -> None:
    assert "llama3" in CHAT_TEMPLATE_REGISTRY, (
        "Registering 'harmony' must not remove 'llama3'"
    )


# ---------------------------------------------------------------------------
# Additional integration scenarios
# ---------------------------------------------------------------------------

def test_system_user_assistant_tool_pipeline_from_registry() -> None:
    """Full pipeline: system → user → assistant (with think) → tool → assistant."""
    tpl = CHAT_TEMPLATE_REGISTRY["harmony"]
    msgs = [
        {"role": "system", "content": "You are Aurelius, a helpful AI."},
        {"role": "user", "content": "Search for the answer."},
        {
            "role": "assistant",
            "content": "",
            "thinking": "I should call the search tool.",
            "tool_calls": ['search(query="answer")'],
        },
        {"role": "tool", "content": '{"result": "42"}'},
        {"role": "assistant", "content": "The answer is 42."},
    ]
    out = tpl.render(msgs)

    assert "<|begin_of_text|>" in out
    assert "<|system|>" in out and "<|end_system|>" in out
    assert "<|user|>" in out and "<|end_user|>" in out
    assert "<think>I should call the search tool.</think>" in out
    assert '<tool_call>search(query="answer")</tool_call>' in out
    assert "<tool_result>" in out and "</tool_result>" in out
    assert "The answer is 42." in out


def test_add_eos_via_registry_instance() -> None:
    """Confirm add_eos=False on the singleton (default); custom instance works."""
    tpl_default = CHAT_TEMPLATE_REGISTRY["harmony"]
    out_no_eos = tpl_default.render([{"role": "user", "content": "hi"}])
    assert not out_no_eos.endswith("<|end_of_text|>")

    tpl_with_eos = HarmonyTemplate(add_eos=True)
    out_eos = tpl_with_eos.render([{"role": "user", "content": "hi"}])
    assert out_eos.endswith("<|end_of_text|>")


def test_parse_roles_integration() -> None:
    """parse_roles should return the correct sequence from a rendered string."""
    tpl = CHAT_TEMPLATE_REGISTRY["harmony"]
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
    ]
    rendered = tpl.render(msgs)
    roles = tpl.parse_roles(rendered)
    assert roles == ["system", "user", "assistant", "tool"]
