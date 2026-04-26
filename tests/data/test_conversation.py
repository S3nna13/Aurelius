"""Tests for src/data/conversation.py — multi-turn conversation dataset builder."""

from __future__ import annotations

import torch

from src.data.conversation import (
    ChatTemplate,
    Conversation,
    ConversationDataset,
    Turn,
    build_sft_labels,
    concatenate_conversations,
    conversations_from_pairs,
)


# Simple deterministic tokenizer: maps each char to ord(c) % 128
def _tokenize(s):
    return [ord(c) % 128 for c in s]


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------


def test_turn_dataclass():
    t = Turn(role="user", content="hello")
    assert t.role == "user"
    assert t.content == "hello"


# ---------------------------------------------------------------------------
# Conversation basics
# ---------------------------------------------------------------------------


def test_conversation_len():
    conv = Conversation(
        turns=[
            Turn("user", "hi"),
            Turn("assistant", "hello"),
            Turn("user", "bye"),
            Turn("assistant", "goodbye"),
        ]
    )
    assert len(conv) == 4


def test_conversation_user_turns():
    conv = Conversation(
        turns=[
            Turn("system", "sys"),
            Turn("user", "q1"),
            Turn("assistant", "a1"),
            Turn("user", "q2"),
            Turn("assistant", "a2"),
        ]
    )
    user = conv.user_turns()
    assert len(user) == 2
    assert all(t.role == "user" for t in user)


# ---------------------------------------------------------------------------
# Conversation.is_valid
# ---------------------------------------------------------------------------


def test_conversation_is_valid_alternating():
    conv = Conversation(
        turns=[
            Turn("user", "q1"),
            Turn("assistant", "a1"),
            Turn("user", "q2"),
            Turn("assistant", "a2"),
        ]
    )
    assert conv.is_valid() is True


def test_conversation_is_valid_with_system():
    conv = Conversation(
        turns=[
            Turn("system", "You are helpful."),
            Turn("user", "question"),
            Turn("assistant", "answer"),
        ]
    )
    assert conv.is_valid() is True


def test_conversation_is_valid_double_user():
    conv = Conversation(
        turns=[
            Turn("user", "q1"),
            Turn("user", "q2"),
            Turn("assistant", "a1"),
        ]
    )
    assert conv.is_valid() is False


# ---------------------------------------------------------------------------
# ChatTemplate
# ---------------------------------------------------------------------------


def test_chat_template_format_turn():
    template = ChatTemplate()
    turn = Turn(role="user", content="hello world")
    result = template.format_turn(turn)
    assert result.startswith(template.user_prefix)
    assert "hello world" in result
    assert template.user_suffix in result


def test_chat_template_format_conversation():
    template = ChatTemplate()
    conv = Conversation(
        turns=[
            Turn("system", "sys prompt"),
            Turn("user", "my question"),
            Turn("assistant", "my answer"),
        ]
    )
    result = template.format_conversation(conv)
    assert "sys prompt" in result
    assert "my question" in result
    assert "my answer" in result
    assert template.system_prefix in result
    assert template.user_prefix in result
    assert template.assistant_prefix in result


# ---------------------------------------------------------------------------
# build_sft_labels
# ---------------------------------------------------------------------------


def _make_simple_conv():
    return Conversation(
        turns=[
            Turn("user", "Hello"),
            Turn("assistant", "Hi there"),
        ]
    )


def test_build_sft_labels_shape():
    conv = _make_simple_conv()
    template = ChatTemplate()
    input_ids, labels = build_sft_labels(conv, template, _tokenize, max_seq_len=512)
    assert input_ids.shape == labels.shape
    assert input_ids.ndim == 1
    assert len(input_ids) > 0


def test_build_sft_labels_user_masked():
    conv = _make_simple_conv()
    template = ChatTemplate()
    input_ids, labels = build_sft_labels(conv, template, _tokenize, max_seq_len=512)

    # Tokenize just the user turn to know its length
    user_turn_str = template.format_turn(conv.turns[0])
    user_len = len(_tokenize(user_turn_str))

    # All user-turn positions should be -100
    assert (labels[:user_len] == -100).all(), "User turn labels should be -100"


def test_build_sft_labels_assistant_not_masked():
    conv = _make_simple_conv()
    template = ChatTemplate()
    input_ids, labels = build_sft_labels(conv, template, _tokenize, max_seq_len=512)

    # Tokenize both turns to find where assistant begins
    user_turn_str = template.format_turn(conv.turns[0])
    user_len = len(_tokenize(user_turn_str))

    # Assistant labels should NOT be -100
    assistant_labels = labels[user_len:]
    assert len(assistant_labels) > 0
    assert (assistant_labels != -100).all(), "Assistant turn labels should not be masked"


# ---------------------------------------------------------------------------
# conversations_from_pairs
# ---------------------------------------------------------------------------


def test_conversations_from_pairs():
    pairs = [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]
    convs = conversations_from_pairs(pairs, system_prompt="Be helpful.")
    assert len(convs) == 3
    for conv in convs:
        assert isinstance(conv, Conversation)
        roles = [t.role for t in conv.turns]
        assert roles == ["system", "user", "assistant"]
    # Without system prompt
    convs_no_sys = conversations_from_pairs(pairs)
    for conv in convs_no_sys:
        roles = [t.role for t in conv.turns]
        assert roles == ["user", "assistant"]


# ---------------------------------------------------------------------------
# concatenate_conversations
# ---------------------------------------------------------------------------


def test_concatenate_conversations():
    pairs = [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]
    single_turn_convs = conversations_from_pairs(pairs)
    merged = concatenate_conversations(single_turn_convs, max_turns=10)
    assert isinstance(merged, Conversation)
    # Should have all turns: 3 user + 3 assistant = 6
    assert len(merged) == 6
    assert merged.is_valid()


def test_concatenate_conversations_max_turns():
    pairs = [("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]
    single_turn_convs = conversations_from_pairs(pairs)
    merged = concatenate_conversations(single_turn_convs, max_turns=4)
    assert len(merged) <= 4


# ---------------------------------------------------------------------------
# ConversationDataset
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 3, max_seq_len: int = 512) -> ConversationDataset:
    pairs = [(f"question {i}", f"answer {i}") for i in range(n)]
    convs = conversations_from_pairs(pairs, system_prompt="You are helpful.")
    template = ChatTemplate()
    return ConversationDataset(convs, template, _tokenize, max_seq_len=max_seq_len)


def test_conversation_dataset_len():
    ds = _make_dataset(5)
    assert len(ds) == 5


def test_conversation_dataset_getitem():
    ds = _make_dataset(3)
    item = ds[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["input_ids"].shape == item["labels"].shape
    assert item["input_ids"].ndim == 1


# ---------------------------------------------------------------------------
# filter_by_length
# ---------------------------------------------------------------------------


def test_filter_by_length():
    # Create conversations of varying content lengths
    pairs_short = [("hi", "ok")]  # very short
    pairs_long = [
        (
            f"tell me about topic {i} in great detail",
            f"certainly, topic {i} is very interesting because...",
        )
        for i in range(4)
    ]

    all_pairs = pairs_short + pairs_long
    convs = conversations_from_pairs(all_pairs)
    template = ChatTemplate()
    ds = ConversationDataset(convs, template, _tokenize, max_seq_len=512)

    # Find length of the short conversation
    short_len = len(ds[0]["input_ids"])

    # Filter out conversations shorter than short_len + 1
    filtered = ds.filter_by_length(min_len=short_len + 1)
    assert len(filtered) < len(ds)
    for i in range(len(filtered)):
        assert len(filtered[i]["input_ids"]) >= short_len + 1
