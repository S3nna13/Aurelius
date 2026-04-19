"""Integration tests for the Harmony template in the chat registry."""

from __future__ import annotations

from src.chat import (
    CHAT_TEMPLATE_REGISTRY,
    MESSAGE_FORMAT_REGISTRY,
    HarmonyMessage,
    HarmonyTemplate,
    Message,
)


def test_registry_exposes_all_templates() -> None:
    for name in ("chatml", "llama3", "harmony"):
        assert name in CHAT_TEMPLATE_REGISTRY, name
    assert isinstance(CHAT_TEMPLATE_REGISTRY["harmony"], HarmonyTemplate)


def test_message_format_registry_has_harmony() -> None:
    assert MESSAGE_FORMAT_REGISTRY["harmony"] is HarmonyMessage


def test_harmony_round_trip_via_registry() -> None:
    tpl = CHAT_TEMPLATE_REGISTRY["harmony"]
    msgs = [
        HarmonyMessage(role="system", content="be precise"),
        HarmonyMessage(role="developer", content="no tools"),
        HarmonyMessage(role="user", content="compute 2+2"),
        HarmonyMessage(
            role="assistant", content="2+2=4", channel="analysis"
        ),
        HarmonyMessage(role="assistant", content="4", channel="final"),
    ]
    text = tpl.encode(msgs, add_generation_prompt=False)
    decoded = tpl.decode(text)
    assert len(decoded) == len(msgs)
    for original, got in zip(msgs, decoded):
        assert got.role == original.role
        assert got.content == original.content
        assert got.channel == original.channel


def test_chatml_unchanged_after_harmony_registration() -> None:
    chatml = CHAT_TEMPLATE_REGISTRY["chatml"]
    msgs = [Message(role="user", content="ping")]
    out = chatml.encode(msgs)
    assert out == "<|im_start|>user\nping<|im_end|>\n"
    assert chatml.decode(out) == msgs


def test_llama3_unchanged_after_harmony_registration() -> None:
    llama = CHAT_TEMPLATE_REGISTRY["llama3"]
    msgs = [Message(role="user", content="ping")]
    out = llama.encode(msgs)
    assert out.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>user<|end_header_id|>\n\nping<|eot_id|>" in out
    assert llama.decode(out) == msgs


def test_harmony_generation_prompt_integration() -> None:
    tpl = CHAT_TEMPLATE_REGISTRY["harmony"]
    out = tpl.encode(
        [HarmonyMessage(role="user", content="hi")],
        add_generation_prompt=True,
    )
    assert out.endswith("<|start|>assistant")
