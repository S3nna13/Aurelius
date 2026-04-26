"""Tests for src/longcontext/context_window_manager.py (8+ tests)."""

from __future__ import annotations

from src.longcontext.context_window_manager import (
    ContextWindowManager,
    WindowPolicy,
)

MGR = ContextWindowManager(max_tokens=100)
TOKENS = list(range(200))  # 200 tokens, exceeds max_tokens=100


# ---------------------------------------------------------------------------
# 1. fits: True when within limit
# ---------------------------------------------------------------------------
def test_fits_true():
    assert MGR.fits(list(range(50))) is True


# ---------------------------------------------------------------------------
# 2. fits: False when over limit
# ---------------------------------------------------------------------------
def test_fits_false():
    assert MGR.fits(list(range(200))) is False


# ---------------------------------------------------------------------------
# 3. TRUNCATE_LEFT: keeps last max_tokens
# ---------------------------------------------------------------------------
def test_truncate_left():
    result = MGR.apply_policy(TOKENS, WindowPolicy.TRUNCATE_LEFT)
    assert result == TOKENS[-100:]
    assert len(result) == 100


# ---------------------------------------------------------------------------
# 4. TRUNCATE_RIGHT: keeps first max_tokens
# ---------------------------------------------------------------------------
def test_truncate_right():
    result = MGR.apply_policy(TOKENS, WindowPolicy.TRUNCATE_RIGHT)
    assert result == TOKENS[:100]
    assert len(result) == 100


# ---------------------------------------------------------------------------
# 5. SUMMARIZE_MIDDLE: structure is head + placeholder + tail
# ---------------------------------------------------------------------------
def test_summarize_middle_structure():
    result = MGR.apply_policy(TOKENS, WindowPolicy.SUMMARIZE_MIDDLE, summary_size=20)
    head_size = 100 // 4  # 25
    tail_size = 100 // 4  # 25
    # head comes first
    assert result[:head_size] == TOKENS[:head_size]
    # placeholder of -1 follows
    placeholder = result[head_size : head_size + 20]
    assert all(x == -1 for x in placeholder)
    # tail comes last
    assert result[head_size + 20 :] == TOKENS[-tail_size:]


# ---------------------------------------------------------------------------
# 6. SLIDING: length is max_tokens when seq > max_tokens
# ---------------------------------------------------------------------------
def test_sliding_length():
    result = MGR.apply_policy(TOKENS, WindowPolicy.SLIDING)
    assert len(result) == 100


# ---------------------------------------------------------------------------
# 7. SLIDING: returns unchanged when fits
# ---------------------------------------------------------------------------
def test_sliding_no_op_when_fits():
    short = list(range(50))
    result = MGR.apply_policy(short, WindowPolicy.SLIDING)
    assert result == short


# ---------------------------------------------------------------------------
# 8. TRUNCATE_LEFT: no-op when input fits
# ---------------------------------------------------------------------------
def test_truncate_left_noop():
    short = list(range(30))
    result = MGR.apply_policy(short, WindowPolicy.TRUNCATE_LEFT)
    assert result == short


# ---------------------------------------------------------------------------
# 9. token_count_estimate: proportional to words
# ---------------------------------------------------------------------------
def test_token_count_estimate_proportional():
    text_short = "hello world"
    text_long = "hello world " * 10
    est_short = MGR.token_count_estimate(text_short)
    est_long = MGR.token_count_estimate(text_long)
    assert est_long > est_short


# ---------------------------------------------------------------------------
# 10. token_count_estimate: matches formula
# ---------------------------------------------------------------------------
def test_token_count_estimate_formula():
    text = "one two three four five"  # 5 words
    expected = round(5 * 1.3)
    assert MGR.token_count_estimate(text) == expected


# ---------------------------------------------------------------------------
# 11. default max_tokens is 4096
# ---------------------------------------------------------------------------
def test_default_max_tokens():
    mgr = ContextWindowManager()
    assert mgr.max_tokens == 4096
