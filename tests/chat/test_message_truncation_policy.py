"""Unit tests for :mod:`src.chat.message_truncation_policy`."""

from __future__ import annotations

import pytest

from src.chat.message_truncation_policy import (
    MessageTruncationPolicy,
    TruncatedResult,
    VALID_STRATEGIES,
)


def _mk(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _convo() -> list:
    return [
        _mk("system", "you are helpful"),
        _mk("user", "hi"),
        _mk("assistant", "hello"),
        _mk("user", "question 1"),
        _mk("assistant", "answer 1"),
        _mk("user", "question 2"),
        _mk("assistant", "answer 2"),
    ]


# --- keep_last_n -------------------------------------------------------


def test_keep_last_n_drops_oldest_and_preserves_last_n():
    pol = MessageTruncationPolicy("keep_last_n", max_turns=3)
    out = pol.apply(_convo())
    assert len(out.messages) == 3
    assert out.messages == _convo()[-3:]
    assert out.dropped_count == 4
    assert out.summary_added is False


# --- keep_system_and_last_n -------------------------------------------


def test_keep_system_and_last_n_keeps_system_plus_last_n():
    pol = MessageTruncationPolicy("keep_system_and_last_n", max_turns=2)
    out = pol.apply(_convo())
    assert out.messages[0]["role"] == "system"
    assert out.messages[0]["content"] == "you are helpful"
    assert out.messages[1:] == _convo()[-2:]
    assert out.dropped_count == len(_convo()) - 3


# --- token_budget_oldest_first ----------------------------------------


def test_token_budget_oldest_first_drops_until_under_budget():
    msgs = [_mk("user", "x" * 40)] * 5  # 10 tokens each @ //4 counter
    pol = MessageTruncationPolicy(
        "token_budget_oldest_first", max_tokens=25
    )
    out = pol.apply(msgs)
    total = sum(len(m["content"]) // 4 for m in out.messages)
    assert total <= 25
    # 5 * 10 = 50 tokens, dropping 3 leaves 20.
    assert len(out.messages) == 2
    assert out.dropped_count == 3
    assert out.summary_added is False


def test_token_budget_oldest_first_preserves_leading_system():
    msgs = [_mk("system", "s" * 40)] + [_mk("user", "u" * 40) for _ in range(3)]
    pol = MessageTruncationPolicy(
        "token_budget_oldest_first", max_tokens=20
    )
    out = pol.apply(msgs)
    assert out.messages[0]["role"] == "system"
    # 1 system (10) + 1 user (10) = 20, fits.
    assert len(out.messages) == 2


# --- token_budget_summarize_oldest -----------------------------------


def test_token_budget_summarize_invokes_summarize_fn_and_flags():
    calls = []

    def summarize(msgs):
        calls.append(list(msgs))
        return "SUMMARY"

    msgs = [_mk("user", "x" * 40) for _ in range(5)]
    pol = MessageTruncationPolicy(
        "token_budget_summarize_oldest",
        max_tokens=30,  # summary (1 token @ //4 = 1) + tail
        summarize_fn=summarize,
    )
    out = pol.apply(msgs)
    assert out.summary_added is True
    assert any(m["content"] == "SUMMARY" for m in out.messages)
    # summarize_fn is invoked repeatedly during the search and once for
    # the final text; at minimum it was called with a non-empty batch.
    assert calls, "summarize_fn must be invoked"
    assert out.dropped_count > 0


def test_summarize_strategy_without_summarize_fn_raises():
    pol = MessageTruncationPolicy(
        "token_budget_summarize_oldest", max_tokens=10
    )
    with pytest.raises(ValueError):
        pol.apply([_mk("user", "x" * 40)])


# --- priority_score ---------------------------------------------------


def test_priority_score_ranks_and_keeps_top_k():
    msgs = [
        _mk("system", "S"),
        _mk("user", "low"),
        _mk("assistant", "med"),
        _mk("user", "high"),
        _mk("assistant", "highest"),
    ]
    scores = {"low": 0.1, "med": 0.5, "high": 0.9, "highest": 1.0}
    pol = MessageTruncationPolicy(
        "priority_score",
        max_turns=2,
        priority_fn=lambda m: scores[m["content"]],
    )
    out = pol.apply(msgs)
    # system is preserved + top-2 from the rest: "high" and "highest".
    contents = [m["content"] for m in out.messages]
    assert contents[0] == "S"
    assert set(contents[1:]) == {"high", "highest"}
    # Order within kept matches original conversation order.
    assert contents[1:] == ["high", "highest"]
    assert out.dropped_count == 2


# --- error / edge cases ------------------------------------------------


def test_unknown_strategy_raises_at_construction():
    with pytest.raises(ValueError):
        MessageTruncationPolicy("does_not_exist", max_turns=1)


def test_empty_messages_returns_empty():
    pol = MessageTruncationPolicy("keep_last_n", max_turns=3)
    out = pol.apply([])
    assert out.messages == []
    assert out.dropped_count == 0
    assert out.summary_added is False


def test_under_budget_input_returns_unchanged():
    msgs = [_mk("user", "x" * 8)]  # 2 tokens
    pol = MessageTruncationPolicy(
        "token_budget_oldest_first", max_tokens=100
    )
    out = pol.apply(msgs)
    assert out.messages == msgs
    assert out.dropped_count == 0


def test_determinism_same_inputs_same_outputs():
    msgs = _convo()
    pol = MessageTruncationPolicy("keep_last_n", max_turns=3)
    a = pol.apply(list(msgs))
    b = pol.apply(list(msgs))
    assert a == b


def test_max_tokens_zero_returns_empty_or_system_only():
    msgs = [_mk("system", "sys"), _mk("user", "hello")]
    pol = MessageTruncationPolicy(
        "token_budget_oldest_first", max_tokens=0
    )
    out = pol.apply(msgs)
    total = sum(len(m["content"]) // 4 for m in out.messages)
    assert total <= 0
    # The "sys" and "hello" both count as 0 tokens under //4, so both
    # fit already; ensure output is a subset of original.
    for m in out.messages:
        assert m in msgs


def test_dropped_count_is_correct_across_strategies():
    msgs = _convo()  # 7 entries
    pol = MessageTruncationPolicy("keep_last_n", max_turns=2)
    out = pol.apply(msgs)
    assert out.dropped_count == 5
    assert len(out.messages) + out.dropped_count == len(msgs)


def test_summary_added_flag_only_for_summarize_strategy():
    pol = MessageTruncationPolicy("keep_last_n", max_turns=1)
    out = pol.apply(_convo())
    assert out.summary_added is False

    pol2 = MessageTruncationPolicy(
        "token_budget_summarize_oldest",
        max_tokens=5,
        summarize_fn=lambda msgs: "S",
    )
    out2 = pol2.apply([_mk("user", "a" * 40) for _ in range(4)])
    assert out2.summary_added is True


def test_valid_strategies_listed():
    expected = {
        "keep_last_n",
        "keep_system_and_last_n",
        "token_budget_oldest_first",
        "token_budget_summarize_oldest",
        "priority_score",
    }
    assert set(VALID_STRATEGIES) == expected


def test_apply_returns_truncated_result_type():
    pol = MessageTruncationPolicy("keep_last_n", max_turns=1)
    out = pol.apply(_convo())
    assert isinstance(out, TruncatedResult)


def test_invalid_message_shape_raises():
    pol = MessageTruncationPolicy("keep_last_n", max_turns=1)
    with pytest.raises(ValueError):
        pol.apply([{"role": "user"}])


def test_keep_last_n_requires_max_turns():
    pol = MessageTruncationPolicy("keep_last_n")
    with pytest.raises(ValueError):
        pol.apply(_convo())


def test_negative_construction_args_rejected():
    with pytest.raises(ValueError):
        MessageTruncationPolicy("keep_last_n", max_turns=-1)
    with pytest.raises(ValueError):
        MessageTruncationPolicy("token_budget_oldest_first", max_tokens=-1)
