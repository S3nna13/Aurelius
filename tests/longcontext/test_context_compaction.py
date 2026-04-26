"""Unit tests for src.longcontext.context_compaction."""

from __future__ import annotations

import time

import pytest

from src.longcontext.context_compaction import ContextCompactor, Turn


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------
def word_tokens(s: str) -> int:
    """Simple deterministic token counter: whitespace-split word count."""
    return len(s.split())


class CountingSummarizer:
    """Fake deterministic summarizer. Returns a fixed short string and
    records how many times it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, turns: list[Turn]) -> str:
        self.calls += 1
        return f"SUMMARY[{len(turns)} turns]"


def make_turns(n: int, words: int = 10, kind: str = "message") -> list[Turn]:
    turns: list[Turn] = []
    for i in range(n):
        content = " ".join(f"w{i}_{j}" for j in range(words))
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(role=role, content=content, kind=kind))
    return turns


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_under_budget_returns_unchanged():
    turns = make_turns(3, words=5)
    s = CountingSummarizer()
    c = ContextCompactor(
        summarize_fn=s, token_counter=word_tokens, target_tokens=10_000, keep_last_n=2
    )
    out = c.compact(turns)
    assert out == turns
    assert s.calls == 0


def test_over_budget_triggers_compaction_within_slack():
    turns = make_turns(20, words=10)
    s = CountingSummarizer()
    target = 50
    c = ContextCompactor(
        summarize_fn=s, token_counter=word_tokens, target_tokens=target, keep_last_n=2
    )
    out = c.compact(turns)
    assert s.calls >= 1
    # Within 10% slack of target.
    assert c.current_tokens(out) <= int(target * 1.10 + 5)


def test_keep_last_n_preserved_verbatim():
    turns = make_turns(20, words=10)
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=30,
        keep_last_n=3,
    )
    out = c.compact(turns)
    # Tail should be the last 3 of input, same objects / values.
    tail = [t for t in out if t.kind != "system"]
    assert tail == turns[-3:]


def test_system_preserved_when_keep_system_true():
    sys_turn = Turn(role="system", content="You are helpful.", kind="system")
    turns = [sys_turn] + make_turns(15, words=10)
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=20,
        keep_last_n=2,
        keep_system=True,
    )
    out = c.compact(turns)
    assert out[0] is sys_turn or out[0] == sys_turn


def test_system_dropped_when_keep_system_false():
    sys_turn = Turn(role="system", content="You are helpful.", kind="system")
    turns = [sys_turn] + make_turns(15, words=10)
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=20,
        keep_last_n=2,
        keep_system=False,
    )
    out = c.compact(turns)
    # The original system turn must not be preserved verbatim.
    assert sys_turn not in out


def test_policy_oldest_first_summarizes_oldest():
    turns = make_turns(10, words=10)
    s = CountingSummarizer()
    c = ContextCompactor(
        summarize_fn=s,
        token_counter=word_tokens,
        target_tokens=30,
        keep_last_n=2,
        policy="oldest_first",
    )
    out = c.compact(turns)
    # First non-system should be summary
    assert any(t.content.startswith("[CONTEXT SUMMARY") for t in out)
    # Tail preserved.
    assert turns[-2:] == [t for t in out if not t.content.startswith("[CONTEXT SUMMARY")]


def test_policy_tool_output_aggregated():
    msgs = make_turns(6, words=8, kind="message")
    tools = [
        Turn(role="tool", content=f"result_{i} port 80{i}0", kind="tool_result") for i in range(5)
    ]
    turns = msgs + tools + make_turns(2, words=5, kind="message")
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=20,
        keep_last_n=2,
        policy="tool_output_aggregated",
    )
    out = c.compact(turns)
    labels = [t.content for t in out if t.kind == "system"]
    assert any("tool_results" in c for c in labels)
    assert any("messages" in c for c in labels)


def test_extract_facts_numbers_and_entities():
    turns = [
        Turn(role="user", content="Connect to port 8080 at version 3.14"),
        Turn(role="assistant", content="ok user_id=42 using https://example.com/api"),
        Turn(role="user", content="See Golden Gate Bridge docs at /etc/conf.d"),
    ]
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=1000,
    )
    facts = c.extract_facts(turns)
    joined = " | ".join(facts).lower()
    assert "port 8080" in joined
    assert "version 3.14" in joined
    assert "user_id=42" in joined
    assert "https://example.com/api" in joined
    assert "golden gate bridge" in joined


def test_hash_cache_prevents_resummarization():
    turns = make_turns(20, words=10)
    s = CountingSummarizer()
    c = ContextCompactor(summarize_fn=s, token_counter=word_tokens, target_tokens=30, keep_last_n=2)
    out1 = c.compact(turns)
    calls_after_first = s.calls
    out2 = c.compact(turns)
    # Same input -> no additional summarizer invocations.
    assert s.calls == calls_after_first
    assert len(out1) == len(out2)


def test_empty_turns_returns_empty():
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=100,
    )
    assert c.compact([]) == []


def test_single_turn_under_budget_passthrough():
    t = Turn(role="user", content="hello world")
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=100,
    )
    assert c.compact([t]) == [t]


def test_malformed_turn_raises():
    with pytest.raises(TypeError):
        Turn(role="user", content=12345)  # type: ignore[arg-type]

    # Also: compact rejects non-Turn list entries.
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=100,
    )
    with pytest.raises(TypeError):
        c.compact(["not a turn"])  # type: ignore[list-item]


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        Turn(role="user", content="x", kind="bogus")


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        ContextCompactor(
            summarize_fn=CountingSummarizer(),
            token_counter=word_tokens,
            target_tokens=100,
            policy="nope",
        )


def test_large_history_fast():
    turns = make_turns(500, words=8)
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=50,
        keep_last_n=4,
    )
    start = time.perf_counter()
    out = c.compact(turns)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"compact took {elapsed:.3f}s"
    assert len(out) < len(turns)


def test_middle_biased_keeps_anchor_and_tail():
    turns = make_turns(10, words=10)
    c = ContextCompactor(
        summarize_fn=CountingSummarizer(),
        token_counter=word_tokens,
        target_tokens=20,
        keep_last_n=2,
        policy="middle_biased",
    )
    out = c.compact(turns)
    # Should contain the first message-turn as anchor and last 2 as tail.
    non_summary = [t for t in out if not t.content.startswith("[CONTEXT SUMMARY")]
    assert turns[0] in non_summary
    assert turns[-1] in non_summary
    assert turns[-2] in non_summary
