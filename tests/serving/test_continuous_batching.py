"""Tests for continuous batching inference manager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.serving.continuous_batching import ContinuousBatcher, Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logits(vocab_size: int, next_token: int, batch_size: int, seq_len: int):
    """Return logits tensor where the max token is always `next_token`."""
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    logits[:, :, next_token] = 100.0  # large value → always chosen by argmax/softmax
    return logits


def _mock_model(next_token: int, vocab_size: int = 100):
    """Return a callable mock that mimics model(input_ids) -> (loss, logits, pkv)."""
    def model(input_ids: torch.Tensor):
        B, T = input_ids.shape
        logits = _make_logits(vocab_size, next_token, B, T)
        return None, logits, None
    return model


def _dummy_encode(text: str) -> list[int]:
    return [ord(c) % 100 for c in text]


def _make_request(rid: str, prompt_len: int = 3, max_new_tokens: int = 5,
                  temperature: float = 0.0) -> Request:
    return Request(
        request_id=rid,
        input_ids=list(range(1, prompt_len + 1)),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Test 1 – add_request puts request in waiting_queue
# ---------------------------------------------------------------------------

def test_add_request_to_queue():
    batcher = ContinuousBatcher(_mock_model(1), _dummy_encode)
    req = _make_request("r1")
    batcher.add_request(req)
    assert len(batcher.waiting_queue) == 1
    assert batcher.waiting_queue[0].request_id == "r1"


# ---------------------------------------------------------------------------
# Test 2 – _fill_batch respects max_batch_size
# ---------------------------------------------------------------------------

def test_fill_batch_respects_max_batch_size():
    batcher = ContinuousBatcher(_mock_model(1), _dummy_encode, max_batch_size=4)
    for i in range(10):
        batcher.add_request(_make_request(f"r{i}"))
    batcher._fill_batch()
    assert len(batcher.active_requests) == 4
    assert len(batcher.waiting_queue) == 6


# ---------------------------------------------------------------------------
# Test 3 – step advances generation by one token
# ---------------------------------------------------------------------------

def test_step_advances_generation():
    next_tok = 7
    batcher = ContinuousBatcher(_mock_model(next_tok), _dummy_encode, max_batch_size=4)
    req = _make_request("r1", max_new_tokens=10, temperature=0.0)
    batcher.add_request(req)
    batcher.step()
    # Request should have exactly one generated token
    assert len(req.generated_ids) == 1
    assert req.generated_ids[0] == next_tok


# ---------------------------------------------------------------------------
# Test 4 – EOS token terminates request in 1 step
# ---------------------------------------------------------------------------

def test_eos_terminates_request():
    eos_id = 2
    batcher = ContinuousBatcher(
        _mock_model(eos_id), _dummy_encode, eos_token_id=eos_id, max_batch_size=4
    )
    req = _make_request("r1", max_new_tokens=50, temperature=0.0)
    batcher.add_request(req)
    completed = batcher.step()
    assert len(completed) == 1
    assert completed[0].request_id == "r1"
    assert completed[0].finish_reason == "eos"
    assert completed[0].is_finished is True


# ---------------------------------------------------------------------------
# Test 5 – max_new_tokens terminates request after exactly N steps
# ---------------------------------------------------------------------------

def test_max_new_tokens_terminates():
    # Use token 5, which is NOT the eos_token_id (2)
    batcher = ContinuousBatcher(
        _mock_model(5), _dummy_encode, eos_token_id=2, max_batch_size=4
    )
    req = _make_request("r1", max_new_tokens=3, temperature=0.0)
    batcher.add_request(req)

    completed_step1 = batcher.step()
    assert len(completed_step1) == 0

    completed_step2 = batcher.step()
    assert len(completed_step2) == 0

    completed_step3 = batcher.step()
    assert len(completed_step3) == 1
    assert completed_step3[0].finish_reason == "length"
    assert len(req.generated_ids) == 3


# ---------------------------------------------------------------------------
# Test 6 – run_until_complete returns all requests
# ---------------------------------------------------------------------------

def test_run_until_complete_returns_all():
    batcher = ContinuousBatcher(
        _mock_model(5), _dummy_encode, eos_token_id=2, max_batch_size=4
    )
    requests = [_make_request(f"r{i}", max_new_tokens=2, temperature=0.0) for i in range(6)]
    results = batcher.run_until_complete(requests)
    assert len(results) == 6
    for r in results:
        assert r.is_finished


# ---------------------------------------------------------------------------
# Test 7 – stats dict has correct counts
# ---------------------------------------------------------------------------

def test_stats_tracking():
    batcher = ContinuousBatcher(
        _mock_model(5), _dummy_encode, eos_token_id=2, max_batch_size=2
    )
    # Add 5 requests: 2 go active, 3 wait
    for i in range(5):
        batcher.add_request(_make_request(f"r{i}", max_new_tokens=10, temperature=0.0))
    batcher._fill_batch()

    stats = batcher.stats
    assert stats["active"] == 2
    assert stats["waiting"] == 3
    assert stats["completed"] == 0


# ---------------------------------------------------------------------------
# Test 8 – slots fill as requests complete
# ---------------------------------------------------------------------------

def test_batch_fills_as_slots_free():
    """With max_batch=2 and 4 requests, first 2 are active;
    once they finish (eos immediately), the next 2 fill in."""
    eos_id = 2
    batcher = ContinuousBatcher(
        _mock_model(eos_id), _dummy_encode, eos_token_id=eos_id, max_batch_size=2
    )
    reqs = [_make_request(f"r{i}", max_new_tokens=50, temperature=0.0) for i in range(4)]
    for r in reqs:
        batcher.add_request(r)

    # After fill, first 2 should be active
    batcher._fill_batch()
    assert len(batcher.active_requests) == 2
    assert len(batcher.waiting_queue) == 2

    # Step: first 2 finish (eos), then fill pulls next 2 in next step
    completed_s1 = batcher.step()
    # step() calls _fill_batch() first internally, but active was already 2 (max),
    # so no new requests added yet; after step the 2 finish → 2 completed
    assert len(completed_s1) == 2
    assert len(batcher.active_requests) == 0
    assert len(batcher.waiting_queue) == 2

    # Next step: _fill_batch promotes remaining 2, then they finish (eos)
    completed_s2 = batcher.step()
    assert len(completed_s2) == 2
    assert len(batcher.active_requests) == 0
    assert len(batcher.waiting_queue) == 0
    assert len(batcher.completed) == 4
