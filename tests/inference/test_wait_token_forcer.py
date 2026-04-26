"""Unit tests for src/inference/wait_token_forcer.py.

Covers all 15 required test cases for WaitTokenForcer and WaitTokenForcerConfig.
"""

from __future__ import annotations

from src.inference.wait_token_forcer import WaitTokenForcer, WaitTokenForcerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

END_THINK = 151645  # default end_think_token_id
WAIT_TOK = 151649  # default wait_token_id
MIN_TOKS = 64  # default min_thinking_tokens


def _seq(n: int, end_with_end_think: bool = False) -> list[int]:
    """Build a dummy token sequence of length n."""
    base = list(range(n))
    if end_with_end_think:
        base[-1] = END_THINK
    return base


def _long_seq_ending_end_think(length: int) -> list[int]:
    """Sequence of *length* tokens ending with END_THINK."""
    return [1] * (length - 1) + [END_THINK]


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = WaitTokenForcerConfig()
    assert cfg.end_think_token_id == 151645
    assert cfg.wait_token_id == 151649
    assert cfg.max_thinking_tokens == 16384
    assert cfg.max_wait_injections == 8
    assert cfg.min_thinking_tokens == 64


# ---------------------------------------------------------------------------
# 2. test_should_inject_true
# ---------------------------------------------------------------------------


def test_should_inject_true():
    """should_inject_wait returns True when all conditions are met."""
    forcer = WaitTokenForcer()
    seq = _long_seq_ending_end_think(MIN_TOKS)  # length == min_thinking_tokens
    assert forcer.should_inject_wait(seq) is True


# ---------------------------------------------------------------------------
# 3. test_should_inject_false_not_end_think
# ---------------------------------------------------------------------------


def test_should_inject_false_not_end_think():
    """Returns False when last token is not end_think_token_id."""
    forcer = WaitTokenForcer()
    seq = [1] * MIN_TOKS  # last token is 1, not END_THINK
    assert forcer.should_inject_wait(seq) is False


# ---------------------------------------------------------------------------
# 4. test_should_inject_false_max_tokens
# ---------------------------------------------------------------------------


def test_should_inject_false_max_tokens():
    """Returns False when sequence length == max_thinking_tokens."""
    cfg = WaitTokenForcerConfig(max_thinking_tokens=100)
    forcer = WaitTokenForcer(cfg)
    seq = _long_seq_ending_end_think(100)  # exactly at the limit
    assert forcer.should_inject_wait(seq) is False


# ---------------------------------------------------------------------------
# 5. test_should_inject_false_max_injections
# ---------------------------------------------------------------------------


def test_should_inject_false_max_injections():
    """Returns False when injection counter has hit max_wait_injections."""
    cfg = WaitTokenForcerConfig(max_wait_injections=2, min_thinking_tokens=1)
    forcer = WaitTokenForcer(cfg)
    seq = [END_THINK]
    # exhaust the budget manually
    forcer._injections_used = 2
    assert forcer.should_inject_wait(seq) is False


# ---------------------------------------------------------------------------
# 6. test_should_inject_false_min_tokens
# ---------------------------------------------------------------------------


def test_should_inject_false_min_tokens():
    """Returns False when sequence is shorter than min_thinking_tokens."""
    cfg = WaitTokenForcerConfig(min_thinking_tokens=10)
    forcer = WaitTokenForcer(cfg)
    seq = _long_seq_ending_end_think(5)  # below minimum
    assert forcer.should_inject_wait(seq) is False


# ---------------------------------------------------------------------------
# 7. test_inject_wait_replaces_end_think
# ---------------------------------------------------------------------------


def test_inject_wait_replaces_end_think():
    """inject_wait removes end_think, appends wait_token_id."""
    forcer = WaitTokenForcer()
    seq = _long_seq_ending_end_think(MIN_TOKS)
    result = forcer.inject_wait(seq)
    assert result[-1] == WAIT_TOK
    assert len(result) == len(seq)
    assert END_THINK not in result


# ---------------------------------------------------------------------------
# 8. test_inject_wait_no_change_when_not_needed
# ---------------------------------------------------------------------------


def test_inject_wait_no_change_when_not_needed():
    """inject_wait returns the list unchanged when conditions aren't met."""
    forcer = WaitTokenForcer()
    seq = [1] * MIN_TOKS  # last token is NOT end_think
    result = forcer.inject_wait(seq)
    assert result == seq


# ---------------------------------------------------------------------------
# 9. test_inject_increments_counter
# ---------------------------------------------------------------------------


def test_inject_increments_counter():
    """injections_used increments by 1 after each successful inject_wait."""
    forcer = WaitTokenForcer()
    seq = _long_seq_ending_end_think(MIN_TOKS)
    assert forcer.injections_used == 0
    forcer.inject_wait(seq)
    assert forcer.injections_used == 1
    # Rebuild the sequence so the last token is end_think again
    seq2 = _long_seq_ending_end_think(MIN_TOKS + 1)
    forcer.inject_wait(seq2)
    assert forcer.injections_used == 2


# ---------------------------------------------------------------------------
# 10. test_process_sequence_no_early_stop
# ---------------------------------------------------------------------------


def test_process_sequence_no_early_stop():
    """process_sequence with no end_think token → sequence unchanged, 0 injections."""
    forcer = WaitTokenForcer()
    seq = [10, 20, 30, 40]
    processed, n = forcer.process_sequence(seq)
    assert processed == seq
    assert n == 0


# ---------------------------------------------------------------------------
# 11. test_process_sequence_one_injection
# ---------------------------------------------------------------------------


def test_process_sequence_one_injection():
    """process_sequence replaces a mid-sequence end_think with wait_token_id."""
    cfg = WaitTokenForcerConfig(min_thinking_tokens=4)
    forcer = WaitTokenForcer(cfg)
    # Build: 3 normal tokens + END_THINK at position 3 (len=4 meets min) + more tokens
    seq = [1, 2, 3, END_THINK, 5, 6]
    processed, n = forcer.process_sequence(seq)
    assert n == 1
    assert processed[3] == WAIT_TOK
    assert processed[4] == 5
    assert processed[5] == 6


# ---------------------------------------------------------------------------
# 12. test_process_sequence_max_injections
# ---------------------------------------------------------------------------


def test_process_sequence_max_injections():
    """process_sequence stops injecting once max_wait_injections is reached."""
    cfg = WaitTokenForcerConfig(max_wait_injections=2, min_thinking_tokens=1)
    forcer = WaitTokenForcer(cfg)
    # 5 end-think tokens interspersed with normal tokens
    seq = [END_THINK, 1, END_THINK, 2, END_THINK, 3, END_THINK, 4, END_THINK]
    processed, n = forcer.process_sequence(seq)
    assert n == 2
    assert forcer.injections_used == 2
    # The 3rd, 4th, 5th end-think tokens must remain as END_THINK
    remaining_end_think = processed.count(END_THINK)
    assert remaining_end_think == 3


# ---------------------------------------------------------------------------
# 13. test_reset_clears_counter
# ---------------------------------------------------------------------------


def test_reset_clears_counter():
    """reset() brings injections_used back to 0."""
    forcer = WaitTokenForcer()
    seq = _long_seq_ending_end_think(MIN_TOKS)
    forcer.inject_wait(seq)
    assert forcer.injections_used == 1
    forcer.reset()
    assert forcer.injections_used == 0


# ---------------------------------------------------------------------------
# 14. test_injection_stats_keys
# ---------------------------------------------------------------------------


def test_injection_stats_keys():
    """injection_stats() returns the correct keys and consistent values."""
    forcer = WaitTokenForcer()
    stats = forcer.injection_stats()
    assert set(stats.keys()) == {"injections_used", "max_allowed", "budget_remaining"}
    assert stats["injections_used"] == 0
    assert stats["max_allowed"] == 8
    assert stats["budget_remaining"] == 8

    seq = _long_seq_ending_end_think(MIN_TOKS)
    forcer.inject_wait(seq)
    stats2 = forcer.injection_stats()
    assert stats2["injections_used"] == 1
    assert stats2["budget_remaining"] == 7


# ---------------------------------------------------------------------------
# 15. test_determinism
# ---------------------------------------------------------------------------


def test_determinism():
    """Same input always produces same output (no randomness)."""
    cfg = WaitTokenForcerConfig(min_thinking_tokens=4)
    seq = [1, 2, 3, END_THINK, 5, 6]

    forcer_a = WaitTokenForcer(cfg)
    result_a, n_a = forcer_a.process_sequence(seq)

    forcer_b = WaitTokenForcer(cfg)
    result_b, n_b = forcer_b.process_sequence(seq)

    assert result_a == result_b
    assert n_a == n_b
