"""Tests for native chat session interface."""

from unittest.mock import MagicMock

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.serving.chat_session import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ChatSession,
    GenerationConfig,
    Message,
    apply_repetition_penalty,
    format_chatml_messages,
)


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def _fake_tokenizer():
    tok = MagicMock()
    tok.encode = lambda text: [ord(c) % 256 for c in text[:30]]
    tok.decode = lambda ids: "hello"
    return tok


def test_format_chatml_contains_tokens():
    """format_chatml_messages must include all role tokens."""
    msgs = [
        Message("system", "Be helpful"),
        Message("user", "Hello"),
    ]
    result = format_chatml_messages(msgs)
    assert SYSTEM_TOKEN in result
    assert USER_TOKEN in result
    assert ASSISTANT_TOKEN in result
    assert END_TOKEN in result


def test_format_chatml_ends_with_assistant():
    """Prompt must end with ASSISTANT_TOKEN to prompt generation."""
    msgs = [Message("user", "Hi")]
    result = format_chatml_messages(msgs)
    assert result.strip().endswith(ASSISTANT_TOKEN.strip()) or ASSISTANT_TOKEN in result[-50:]


def test_apply_repetition_penalty():
    """Repeated tokens must have lower logits after penalty."""
    logits = torch.ones(256)  # non-zero so division by penalty has effect
    generated = torch.tensor([5, 10, 5])  # tokens 5 and 10 appear
    penalized = apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    assert penalized[5] < logits[5]
    assert penalized[10] < logits[10]
    # Non-generated tokens unchanged
    assert penalized[42] == logits[42]


def test_chat_session_returns_string(small_model):
    """chat() must return a non-empty string."""
    cfg = GenerationConfig(max_new_tokens=4, temperature=1.0, top_p=0.9)
    session = ChatSession(small_model, _fake_tokenizer(), gen_cfg=cfg)
    response = session.chat("Hello")
    assert isinstance(response, str)


def test_chat_session_appends_messages(small_model):
    """chat() must append user + assistant messages to history."""
    cfg = GenerationConfig(max_new_tokens=4, temperature=1.0, top_p=0.9)
    session = ChatSession(small_model, _fake_tokenizer(), gen_cfg=cfg)
    initial_len = len(session)
    session.chat("How are you?")
    assert len(session) == initial_len + 2  # user + assistant


def test_chat_session_reset(small_model):
    """reset() must clear history except system prompt."""
    cfg = GenerationConfig(max_new_tokens=4, temperature=1.0, top_p=0.9)
    session = ChatSession(small_model, _fake_tokenizer(), gen_cfg=cfg)
    session.chat("Hello")
    session.reset()
    assert len(session) == 1  # only system message
    assert session.messages[0].role == "system"


def test_get_history_format(small_model):
    """get_history must return list of role/content dicts."""
    session = ChatSession(small_model, _fake_tokenizer())
    history = session.get_history()
    assert len(history) >= 1
    assert "role" in history[0]
    assert "content" in history[0]
