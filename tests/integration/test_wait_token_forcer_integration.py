"""Integration tests for WaitTokenForcer.

Verifies:
* End-to-end injection budget enforcement across multiple calls.
* reset() correctly restores state.
* DECODER_REGISTRY["wait_token_forcer"] is wired correctly.
"""
from __future__ import annotations

import pytest

from src.inference.wait_token_forcer import WaitTokenForcer, WaitTokenForcerConfig
import src.inference as inference_pkg


END_THINK = 151645
WAIT_TOK  = 151649
MIN_TOKS  = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq_ending_end_think(n: int) -> list[int]:
    """Return a list of n tokens where the last one is END_THINK."""
    return [1] * (n - 1) + [END_THINK]


# ---------------------------------------------------------------------------
# Integration: injection limit across repeated process_sequence calls
# ---------------------------------------------------------------------------

def test_injection_limit_across_calls():
    """Only 3 injections happen when process_sequence is called 5 times."""
    cfg = WaitTokenForcerConfig(max_wait_injections=3, min_thinking_tokens=MIN_TOKS)
    forcer = WaitTokenForcer(cfg)

    total_injections = 0
    for _ in range(5):
        seq = _seq_ending_end_think(MIN_TOKS)
        _, n = forcer.process_sequence(seq)
        total_injections += n

    assert total_injections == 3
    assert forcer.injections_used == 3


# ---------------------------------------------------------------------------
# Integration: reset() restores state for reuse
# ---------------------------------------------------------------------------

def test_reset_restores_full_budget():
    """After reset(), a fresh round of injections is possible."""
    cfg = WaitTokenForcerConfig(max_wait_injections=3, min_thinking_tokens=MIN_TOKS)
    forcer = WaitTokenForcer(cfg)

    # Exhaust the budget
    for _ in range(5):
        seq = _seq_ending_end_think(MIN_TOKS)
        forcer.process_sequence(seq)

    assert forcer.injections_used == 3

    # Reset and verify the counter is gone
    forcer.reset()
    assert forcer.injections_used == 0

    # A fresh pass should allow injections again
    total_after_reset = 0
    for _ in range(5):
        seq = _seq_ending_end_think(MIN_TOKS)
        _, n = forcer.process_sequence(seq)
        total_after_reset += n

    assert total_after_reset == 3
    assert forcer.injections_used == 3


# ---------------------------------------------------------------------------
# Integration: DECODER_REGISTRY wired correctly
# ---------------------------------------------------------------------------

def test_registry_wired():
    """DECODER_REGISTRY['wait_token_forcer'] resolves to WaitTokenForcer."""
    registry = inference_pkg.DECODER_REGISTRY
    assert "wait_token_forcer" in registry
    assert registry["wait_token_forcer"] is WaitTokenForcer


def test_instantiation_from_registry():
    """Class obtained from registry can be instantiated and used."""
    cls = inference_pkg.DECODER_REGISTRY["wait_token_forcer"]
    forcer = cls()
    seq = _seq_ending_end_think(MIN_TOKS)
    result = forcer.inject_wait(seq)
    assert result[-1] == WAIT_TOK
    assert forcer.injections_used == 1


# ---------------------------------------------------------------------------
# Integration: stats are consistent throughout lifecycle
# ---------------------------------------------------------------------------

def test_stats_lifecycle():
    """injection_stats values stay internally consistent throughout a lifecycle."""
    cfg = WaitTokenForcerConfig(max_wait_injections=3, min_thinking_tokens=MIN_TOKS)
    forcer = WaitTokenForcer(cfg)

    for i in range(5):
        seq = _seq_ending_end_think(MIN_TOKS)
        forcer.inject_wait(seq)
        stats = forcer.injection_stats()
        expected_used = min(i + 1, 3)
        assert stats["injections_used"] == expected_used
        assert stats["max_allowed"] == 3
        assert stats["budget_remaining"] == max(0, 3 - (i + 1))

    forcer.reset()
    stats = forcer.injection_stats()
    assert stats["injections_used"] == 0
    assert stats["budget_remaining"] == 3
