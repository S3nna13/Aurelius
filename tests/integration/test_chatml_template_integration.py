"""Integration tests for the chat surface registry.

Verifies:
  * `src.chat` can be imported without pulling torch / model.config.
  * CHAT_TEMPLATE_REGISTRY["chatml"] is a working template.
  * encode -> decode is lossless for a representative conversation.
"""

from __future__ import annotations

import sys


def test_importing_src_chat_does_not_import_torch_or_model():
    # Ensure we observe the post-import state cleanly.
    for mod in list(sys.modules):
        if mod.startswith("src.chat"):
            del sys.modules[mod]
    torch_before = "torch" in sys.modules
    model_before = "src.model" in sys.modules

    import src.chat  # noqa: F401

    # We only care that *we* did not pull these in. If a prior test
    # already imported torch, that is out of our control; but src.chat
    # itself must not cause a new import.
    if not torch_before:
        assert "torch" not in sys.modules, "src.chat must not import torch"
    if not model_before:
        assert "src.model" not in sys.modules, (
            "src.chat must not import src.model"
        )


def test_registry_exposes_chatml_template():
    from src.chat import CHAT_TEMPLATE_REGISTRY, MESSAGE_FORMAT_REGISTRY

    assert "chatml" in CHAT_TEMPLATE_REGISTRY
    tpl = CHAT_TEMPLATE_REGISTRY["chatml"]
    assert hasattr(tpl, "encode") and callable(tpl.encode)
    assert hasattr(tpl, "decode") and callable(tpl.decode)
    assert "chatml" in MESSAGE_FORMAT_REGISTRY


def test_registry_roundtrip():
    from src.chat import CHAT_TEMPLATE_REGISTRY, Message

    tpl = CHAT_TEMPLATE_REGISTRY["chatml"]
    msgs = [
        Message("system", "You are Aurelius."),
        Message("user", "What is 2+2?"),
        Message("assistant", "4"),
        Message("tool", '{"ok": true}'),
    ]
    wire = tpl.encode(msgs)
    decoded = tpl.decode(wire)
    assert decoded == msgs


def test_registry_generation_prompt_path():
    from src.chat import CHAT_TEMPLATE_REGISTRY, Message

    tpl = CHAT_TEMPLATE_REGISTRY["chatml"]
    wire = tpl.encode(
        [Message("user", "hello")], add_generation_prompt=True
    )
    assert wire.endswith("<|im_start|>assistant\n")
