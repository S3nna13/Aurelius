"""Unit tests for src.chat.multi_turn_state."""

from __future__ import annotations

import pytest

from src.chat.multi_turn_state import ConversationState, ConversationTurn


def test_append_message_adds_turn_with_correct_role():
    s = ConversationState()
    s.append_message("user", "hello")
    msgs = s.to_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"


def test_tool_call_and_result_pair_correctly():
    s = ConversationState()
    s.append_message("user", "run thing")
    s.append_tool_call("search", {"q": "foo"}, call_id="c1")
    s.append_tool_result("c1", "found", is_error=False)
    msgs = s.to_messages()
    assert msgs[-2]["kind"] == "tool_call"
    assert msgs[-2]["tool_call_id"] == "c1"
    assert msgs[-1]["kind"] == "tool_result"
    assert msgs[-1]["tool_call_id"] == "c1"
    assert msgs[-1]["tool_name"] == "search"


def test_to_messages_returns_ordered_dicts():
    s = ConversationState(system_prompt="sys")
    s.append_message("user", "a")
    s.append_message("assistant", "b")
    s.append_message("user", "c")
    roles = [m["role"] for m in s.to_messages()]
    assert roles == ["system", "user", "assistant", "user"]


def test_truncate_removes_oldest_when_over_max_turns():
    s = ConversationState(max_turns=3, max_tokens=10_000)
    for i in range(5):
        s.append_message("user", f"msg{i}")
    s.truncate_if_needed()
    contents = [m["content"] for m in s.to_messages()]
    assert contents == ["msg2", "msg3", "msg4"]


def test_system_message_preserved_during_truncation():
    s = ConversationState(system_prompt="SYS", max_turns=2, max_tokens=10_000)
    for i in range(5):
        s.append_message("user", f"m{i}")
    s.truncate_if_needed()
    msgs = s.to_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"
    assert len(msgs) == 3  # system + 2 most recent user


def test_token_budget_truncation_triggers_at_max_tokens():
    # Each message is 3 whitespace tokens.
    s = ConversationState(max_turns=100, max_tokens=6)
    for i in range(5):
        s.append_message("user", f"one two {i}")
    s.truncate_if_needed()
    assert s.current_tokens() <= 6


def test_compactor_wired_when_over_budget():
    calls = {"n": 0}

    class FakeCompactor:
        def compact(self, turns):
            calls["n"] += 1
            # Collapse to a single short summary turn.
            from src.longcontext.context_compaction import Turn

            return [Turn(role="system", content="summary", kind="system")]

    s = ConversationState(
        system_prompt="sys",
        max_turns=100,
        max_tokens=3,
        compactor=FakeCompactor(),
    )
    for i in range(5):
        s.append_message("user", f"aaa bbb ccc ddd {i}")
    s.truncate_if_needed()
    assert calls["n"] == 1
    assert any(m["content"] == "summary" for m in s.to_messages())


def test_turn_id_monotonically_increasing():
    s = ConversationState()
    s.append_message("user", "a")
    s.append_message("assistant", "b")
    s.append_message("user", "c")
    ids = [t.turn_id for t in s.turns]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_timestamp_monotonically_increasing():
    s = ConversationState()
    for i in range(5):
        s.append_message("user", str(i))
    ts = [t.timestamp for t in s.turns]
    assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))


def test_summary_stats_counts():
    s = ConversationState(system_prompt="s")
    s.append_message("user", "u1")
    s.append_message("assistant", "a1")
    s.append_tool_call("t", {}, "c1")
    s.append_tool_result("c1", "ok")
    stats = s.summary_stats()
    assert stats["counts_by_role"]["user"] == 1
    assert stats["counts_by_role"]["assistant"] == 2  # message + tool_call
    assert stats["counts_by_role"]["system"] == 1
    assert stats["counts_by_role"]["tool"] == 1
    assert stats["counts_by_role"]["total"] == 5
    assert stats["counts_by_kind"]["tool_call"] == 1
    assert stats["counts_by_kind"]["tool_result"] == 1


def test_empty_conversation_returns_empty_message_list():
    s = ConversationState()
    assert s.to_messages() == []
    s2 = ConversationState(system_prompt="sys")
    msgs = s2.to_messages()
    assert len(msgs) == 1 and msgs[0]["role"] == "system"


def test_determinism_of_ordering():
    def build():
        s = ConversationState(system_prompt="x")
        s.append_message("user", "1")
        s.append_message("assistant", "2")
        s.append_tool_call("search", {"q": "k"}, "c1")
        s.append_tool_result("c1", "r")
        return [(m["role"], m["content"], m["kind"]) for m in s.to_messages()]

    a = build()
    b = build()
    assert a == b


def test_invalid_role_raises():
    s = ConversationState()
    with pytest.raises(ValueError):
        s.append_message("wizard", "hello")


def test_tool_result_without_matching_call_id_raises():
    s = ConversationState()
    with pytest.raises(ValueError):
        s.append_tool_result("bogus", "x")


def test_invalid_kind_raises():
    s = ConversationState()
    with pytest.raises(ValueError):
        s.append_message("user", "hi", kind="nope")


def test_conversation_turn_invalid_role():
    with pytest.raises(ValueError):
        ConversationTurn(turn_id=0, role="alien", content="x")
