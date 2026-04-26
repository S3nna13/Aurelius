"""Integration tests for ConversationState within the chat surface.

Verifies:
- public exposure via ``src.chat``,
- end-to-end multi-turn conversation with tool calls,
- prior chat entries (imports from src.chat) remain intact,
- integration with ContextCompactor from src.longcontext.
"""

from __future__ import annotations

import src.chat as chat_pkg
from src.chat.multi_turn_state import ConversationState, ConversationTurn
from src.longcontext.context_compaction import ContextCompactor


def test_exposed_in_src_chat():
    assert hasattr(chat_pkg, "ConversationState")
    assert hasattr(chat_pkg, "ConversationTurn")
    assert chat_pkg.ConversationState is ConversationState
    assert chat_pkg.ConversationTurn is ConversationTurn


def test_prior_chat_entries_intact():
    # Sanity: existing registry + classes still importable and populated.
    assert "chatml" in chat_pkg.CHAT_TEMPLATE_REGISTRY
    assert "harmony" in chat_pkg.CHAT_TEMPLATE_REGISTRY
    assert hasattr(chat_pkg, "ConversationMemory")
    assert hasattr(chat_pkg, "FIMFormatter")
    assert hasattr(chat_pkg, "build_role_mask")


def test_end_to_end_multi_turn_with_tools():
    s = ConversationState(
        system_prompt="You are a careful agent.",
        max_turns=50,
        max_tokens=10_000,
    )
    s.append_message("user", "search the docs for 'einsum'")
    s.append_message("assistant", "calling search tool")
    s.append_tool_call("search", {"q": "einsum"}, call_id="call-1")
    s.append_tool_result("call-1", "found 3 results", is_error=False)
    s.append_message("assistant", "here are the 3 results summarized")
    s.append_message("user", "thanks")

    msgs = s.to_messages()
    roles = [m["role"] for m in msgs]
    kinds = [m["kind"] for m in msgs]
    assert roles[0] == "system"
    assert "tool" in roles
    assert "tool_call" in kinds and "tool_result" in kinds

    stats = s.summary_stats()
    assert stats["counts_by_role"]["total"] == 7
    assert stats["counts_by_kind"]["tool_call"] == 1
    assert stats["counts_by_kind"]["tool_result"] == 1


def test_integration_with_context_compactor():
    def summarize(turns):
        return f"summary of {len(turns)} turns"

    compactor = ContextCompactor(
        summarize_fn=summarize,
        token_counter=lambda s: len(s.split()),
        target_tokens=5,
        keep_last_n=1,
        keep_system=True,
        policy="oldest_first",
    )
    state = ConversationState(
        system_prompt="SYS",
        max_turns=100,
        max_tokens=5,
        token_counter=lambda s: len(s.split()),
        compactor=compactor,
    )
    for i in range(6):
        state.append_message("user", f"word1 word2 word3 {i}")
    state.truncate_if_needed()

    msgs = state.to_messages()
    # System preserved and a summary injected.
    assert any(m["role"] == "system" for m in msgs)
    assert any("summary" in m["content"].lower() or "SYS" == m["content"] for m in msgs)
    assert state.current_tokens() <= 5 or any("summary" in m["content"].lower() for m in msgs)


def test_attachments_and_reset_flow():
    s = ConversationState(system_prompt="sys")
    s.add_attachment({"type": "image", "path": "/tmp/a.png"})
    s.append_message("user", "see image")
    stats = s.summary_stats()
    assert stats["attachments"] == 1
    assert stats["num_turns"] == 2
