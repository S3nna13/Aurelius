"""Integration tests for message-truncation policies.

Checks that the policy is exposed via the ``src.chat`` package surface
and operates correctly on a real ChatML-produced message list.
"""

from __future__ import annotations

import src.chat as chat
from src.chat.chatml_template import ChatMLTemplate, Message


def _render_and_parse(messages):
    """Round-trip a list of messages through the ChatML encoder to get
    a canonical list-of-dicts back out."""
    tpl = ChatMLTemplate()
    wire = tpl.encode([Message(role=m["role"], content=m["content"]) for m in messages])
    parsed = tpl.decode(wire)
    # ChatMLTemplate.decode returns list[Message]; convert to dicts.
    return [{"role": p.role, "content": p.content} for p in parsed]


def test_policy_exposed_on_src_chat_package():
    assert hasattr(chat, "MessageTruncationPolicy")
    assert hasattr(chat, "TruncatedResult")


def test_apply_to_chatml_message_list_keep_last_n():
    raw = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    msgs = _render_and_parse(raw)
    assert msgs == raw  # sanity: ChatML preserved our dicts

    pol = chat.MessageTruncationPolicy("keep_last_n", max_turns=4)
    out = pol.apply(msgs)
    assert len(out.messages) == 4
    assert out.messages == raw[-4:]
    assert out.dropped_count == 3


def test_prior_chat_entries_remain_intact_and_input_not_mutated():
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    msgs = _render_and_parse(raw)
    snapshot = [dict(m) for m in msgs]

    pol = chat.MessageTruncationPolicy("keep_system_and_last_n", max_turns=2)
    out = pol.apply(msgs)

    # Input list and its contents unchanged.
    assert msgs == snapshot
    # Kept messages are drawn from the original (equality, order).
    assert out.messages[0] == raw[0]
    assert out.messages[-2:] == raw[-2:]


def test_summarize_strategy_integration_with_chatml_list():
    raw = [{"role": "user", "content": "x" * 40} for _ in range(6)]
    msgs = _render_and_parse(raw)

    def summarize(batch):
        return f"summary-of-{len(batch)}"

    pol = chat.MessageTruncationPolicy(
        "token_budget_summarize_oldest",
        max_tokens=25,
        summarize_fn=summarize,
    )
    out = pol.apply(msgs)
    assert out.summary_added is True
    assert any(m["content"].startswith("summary-of-") for m in out.messages)
    # Remaining non-summary messages are a suffix of the original.
    tail = [m for m in out.messages if not m["content"].startswith("summary-of-")]
    assert tail == raw[-len(tail) :]


def test_truncated_output_still_encodable_by_chatml():
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "again"},
    ]
    msgs = _render_and_parse(raw)
    pol = chat.MessageTruncationPolicy("keep_system_and_last_n", max_turns=1)
    out = pol.apply(msgs)

    # The truncated list must itself round-trip through ChatML.
    tpl = ChatMLTemplate()
    wire = tpl.encode([Message(role=m["role"], content=m["content"]) for m in out.messages])
    decoded = tpl.decode(wire)
    assert [(d.role, d.content) for d in decoded] == [
        (m["role"], m["content"]) for m in out.messages
    ]
