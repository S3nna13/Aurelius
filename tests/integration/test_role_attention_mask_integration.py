"""Integration: role_attention_mask works with ChatML-encoded convos
and is exposed through ``src.chat`` without breaking existing registry
entries.
"""

from __future__ import annotations

import torch

import src.chat as chat_pkg
from src.chat import (
    CHAT_TEMPLATE_REGISTRY,
    MASK_VALUE,
    MESSAGE_FORMAT_REGISTRY,
    ChatMLTemplate,
    Message,
    RoleSpan,
    build_loss_mask,
    build_role_mask,
)


def test_exposed_from_chat_package():
    assert hasattr(chat_pkg, "RoleSpan")
    assert hasattr(chat_pkg, "build_role_mask")
    assert hasattr(chat_pkg, "build_loss_mask")
    assert hasattr(chat_pkg, "validate_spans")


def test_existing_registry_entries_intact():
    assert "chatml" in CHAT_TEMPLATE_REGISTRY
    assert "llama3" in CHAT_TEMPLATE_REGISTRY
    assert "harmony" in CHAT_TEMPLATE_REGISTRY
    assert "chatml" in MESSAGE_FORMAT_REGISTRY
    assert "harmony" in MESSAGE_FORMAT_REGISTRY
    assert "tool_result" in MESSAGE_FORMAT_REGISTRY


def test_build_mask_for_chatml_conversation():
    tmpl = ChatMLTemplate()
    msgs = [
        Message("system", "You are helpful."),
        Message("user", "Hi"),
        Message("assistant", "Hello."),
    ]
    text = tmpl.encode(msgs)
    assert isinstance(text, str) and len(text) > 0

    # Simulate a token-level span tiling. For the integration test we
    # treat each message's encoded string length as its "token count".
    # What matters is that build_role_mask + build_loss_mask accept the
    # span tiling and return correctly-shaped outputs.
    per_msg_lens = [len(tmpl.encode([m])) for m in msgs]
    spans = []
    cursor = 0
    for m, n in zip(msgs, per_msg_lens):
        spans.append(RoleSpan(m.role, cursor, cursor + n))
        cursor += n
    seq_len = cursor

    mask = build_role_mask(spans, seq_len, system_priority=True, causal=True)
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.float32

    # System columns attendable from every row.
    sys_span = spans[0]
    for i in range(seq_len):
        for j in range(sys_span.start, sys_span.end):
            assert mask[i, j].item() == 0.0

    # Strict upper triangle (outside system cols) masked.
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if sys_span.start <= j < sys_span.end:
                continue
            assert mask[i, j].item() == MASK_VALUE

    loss = build_loss_mask(spans, seq_len, loss_roles=("assistant",))
    assert loss.dtype == torch.bool
    assert loss.shape == (seq_len,)
    # Only assistant span is True.
    assistant = spans[2]
    assert loss[assistant.start : assistant.end].all().item()
    # Non-assistant positions are False.
    for i in range(seq_len):
        if not (assistant.start <= i < assistant.end):
            assert not loss[i].item()
