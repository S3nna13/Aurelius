"""Integration tests: registry wiring and cross-template regression."""

from __future__ import annotations

from src.chat import CHAT_TEMPLATE_REGISTRY, Message
from src.chat.chatml_template import ChatMLTemplate
from src.chat.llama3_template import Llama3Template


def test_registry_has_both_templates() -> None:
    assert "chatml" in CHAT_TEMPLATE_REGISTRY
    assert "llama3" in CHAT_TEMPLATE_REGISTRY
    assert isinstance(CHAT_TEMPLATE_REGISTRY["chatml"], ChatMLTemplate)
    assert isinstance(CHAT_TEMPLATE_REGISTRY["llama3"], Llama3Template)


def test_llama3_round_trip_via_registry() -> None:
    tpl = CHAT_TEMPLATE_REGISTRY["llama3"]
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="tool", content="{\"ok\": 1}"),
    ]
    wire = tpl.encode(msgs)
    assert tpl.decode(wire) == msgs


def test_chatml_still_works_regression_guard() -> None:
    tpl = CHAT_TEMPLATE_REGISTRY["chatml"]
    msgs = [
        Message(role="user", content="ping"),
        Message(role="assistant", content="pong"),
    ]
    wire = tpl.encode(msgs, add_generation_prompt=False)
    assert tpl.decode(wire) == msgs
    # Generation prompt still tolerated.
    wire2 = tpl.encode(msgs, add_generation_prompt=True)
    assert tpl.decode(wire2) == msgs


def test_templates_produce_distinct_wire_formats() -> None:
    msgs = [Message(role="user", content="hi")]
    chatml_out = CHAT_TEMPLATE_REGISTRY["chatml"].encode(msgs)
    llama3_out = CHAT_TEMPLATE_REGISTRY["llama3"].encode(msgs)
    assert chatml_out != llama3_out
    assert "<|im_start|>" in chatml_out
    assert "<|start_header_id|>" in llama3_out
