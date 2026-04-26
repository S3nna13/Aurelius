"""Tests for cascade speculative decoding."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.cascade_speculative import (
    CascadeConfig,
    CascadeLevel,
    CascadeSpeculativeDecoder,
    build_cascade,
    compute_cascade_acceptance_rate,
)

# ---------------------------------------------------------------------------
# Mock model for testing
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    def __init__(self, vocab_size=256, always_token=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.always_token = always_token  # if set, always predicts this token
        self.embed = nn.Embedding(vocab_size, 32)
        self.proj = nn.Linear(32, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids).mean(dim=1, keepdim=True)
        logits = self.proj(x).expand(-1, input_ids.shape[1], -1)
        if self.always_token is not None:
            logits = torch.zeros_like(logits)
            logits[:, :, self.always_token] = 100.0
        return (None, logits, None)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_input(batch=1, seq_len=5, vocab_size=256):
    return torch.randint(0, vocab_size, (batch, seq_len))


# ---------------------------------------------------------------------------
# Test 1: CascadeConfig defaults correct
# ---------------------------------------------------------------------------


def test_cascade_config_defaults():
    cfg = CascadeConfig()
    assert cfg.draft_lengths == [4, 2]
    assert cfg.acceptance_thresholds == [0.0, 0.0]
    assert cfg.max_new_tokens == 512


# ---------------------------------------------------------------------------
# Test 2: CascadeLevel.draft returns correct shapes
# ---------------------------------------------------------------------------


def test_cascade_level_draft_shapes():
    vocab_size = 256
    draft_len = 4
    model = MockModel(vocab_size=vocab_size)
    level = CascadeLevel(model=model, draft_len=draft_len)
    input_ids = make_input(batch=1, seq_len=5, vocab_size=vocab_size)

    draft_ids, draft_logprobs = level.draft(input_ids)

    assert draft_ids.shape == (1, draft_len), f"Expected (1, {draft_len}), got {draft_ids.shape}"
    assert draft_logprobs.shape == (1, draft_len, vocab_size), (
        f"Expected (1, {draft_len}, {vocab_size}), got {draft_logprobs.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: CascadeLevel.verify accepts all when model agrees with draft
# ---------------------------------------------------------------------------


def test_cascade_level_verify_all_accepted():
    """When model always predicts token 42, draft of all-42 tokens is fully accepted."""
    always_token = 42
    draft_len = 4
    model = MockModel(vocab_size=256, always_token=always_token)
    level = CascadeLevel(model=model, draft_len=draft_len)

    input_ids = make_input(batch=1, seq_len=5)
    draft_ids = torch.full((1, draft_len), always_token, dtype=torch.long)

    accepted_tokens, n_accepted = level.verify(input_ids, draft_ids)

    assert n_accepted == draft_len, f"Expected all {draft_len} accepted, got {n_accepted}"
    assert accepted_tokens.shape == (1, draft_len)


# ---------------------------------------------------------------------------
# Test 4: CascadeLevel.verify stops at first mismatch
# ---------------------------------------------------------------------------


def test_cascade_level_verify_stops_at_mismatch():
    """Model always predicts token 42; draft has mismatch at position 2."""
    always_token = 42
    model = MockModel(vocab_size=256, always_token=always_token)
    level = CascadeLevel(model=model, draft_len=4)

    input_ids = make_input(batch=1, seq_len=5)
    # First two tokens match, third does not
    draft_ids = torch.tensor([[42, 42, 99, 42]], dtype=torch.long)

    accepted_tokens, n_accepted = level.verify(input_ids, draft_ids)

    assert n_accepted == 2, f"Expected 2 accepted (mismatch at index 2), got {n_accepted}"
    assert accepted_tokens.shape == (1, 2)


# ---------------------------------------------------------------------------
# Test 5: CascadeSpeculativeDecoder with 2 levels instantiates
# ---------------------------------------------------------------------------


def test_cascade_decoder_instantiates():
    level0 = CascadeLevel(MockModel(vocab_size=256), draft_len=4)
    level1 = CascadeLevel(MockModel(vocab_size=256), draft_len=2)
    decoder = CascadeSpeculativeDecoder(levels=[level0, level1])
    assert decoder is not None
    assert len(decoder.levels) == 2


# ---------------------------------------------------------------------------
# Test 6: decode_step returns new tokens and stats dict
# ---------------------------------------------------------------------------


def test_decode_step_returns_tokens_and_stats():
    always_token = 7
    level0 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=4)
    level1 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=2)
    decoder = CascadeSpeculativeDecoder(levels=[level0, level1])

    input_ids = make_input(batch=1, seq_len=5)
    new_tokens, stats = decoder.decode_step(input_ids)

    assert isinstance(new_tokens, torch.Tensor)
    assert new_tokens.ndim == 2
    assert new_tokens.shape[0] == 1
    assert new_tokens.shape[1] >= 0
    assert isinstance(stats, dict)
    assert "n_accepted" in stats
    assert "n_proposed" in stats


# ---------------------------------------------------------------------------
# Test 7: generate() returns tensor longer than input
# ---------------------------------------------------------------------------


def test_generate_longer_than_input():
    always_token = 5
    level0 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=4)
    level1 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=2)
    decoder = CascadeSpeculativeDecoder(levels=[level0, level1], max_new_tokens=8)

    input_ids = make_input(batch=1, seq_len=4)
    output_ids = decoder.generate(input_ids)

    assert output_ids.shape[1] > input_ids.shape[1], (
        f"Output length {output_ids.shape[1]} should exceed input length {input_ids.shape[1]}"
    )
    assert output_ids.shape[1] == input_ids.shape[1] + 8


# ---------------------------------------------------------------------------
# Test 8: get_stats returns dict with acceptance_rates key
# ---------------------------------------------------------------------------


def test_get_stats_has_acceptance_rates():
    always_token = 3
    level0 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=4)
    level1 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=2)
    decoder = CascadeSpeculativeDecoder(levels=[level0, level1], max_new_tokens=4)

    input_ids = make_input(batch=1, seq_len=4)
    decoder.generate(input_ids)

    stats = decoder.get_stats()
    assert isinstance(stats, dict)
    assert "acceptance_rates" in stats
    assert "total_tokens" in stats
    assert "cascade_speedup_estimate" in stats
    assert isinstance(stats["acceptance_rates"], list)


# ---------------------------------------------------------------------------
# Test 9: compute_cascade_acceptance_rate returns correct rates
# ---------------------------------------------------------------------------


def test_compute_cascade_acceptance_rate():
    # Level 0: steps accepted [4, 4, 2] → mean = (4+4+2)/3 = 10/3
    # Level 1: steps accepted [2, 0, 2] → mean = (2+0+2)/3 = 4/3
    level_acceptances = [[4, 4, 2], [2, 0, 2]]
    rates = compute_cascade_acceptance_rate(level_acceptances)

    assert len(rates) == 2
    assert abs(rates[0] - (10 / 3)) < 1e-6, f"Level 0 rate wrong: {rates[0]}"
    assert abs(rates[1] - (4 / 3)) < 1e-6, f"Level 1 rate wrong: {rates[1]}"


def test_compute_cascade_acceptance_rate_empty():
    rates = compute_cascade_acceptance_rate([[], []])
    assert rates == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Test 10: build_cascade factory creates correct structure
# ---------------------------------------------------------------------------


def test_build_cascade_factory():
    models = [MockModel(vocab_size=256), MockModel(vocab_size=256), MockModel(vocab_size=256)]
    draft_lengths = [4, 2, 1]
    decoder = build_cascade(models, draft_lengths)

    assert isinstance(decoder, CascadeSpeculativeDecoder)
    assert len(decoder.levels) == 3
    for i, level in enumerate(decoder.levels):
        assert level.draft_len == draft_lengths[i]
        assert level.model is models[i]


def test_build_cascade_mismatch_raises():
    models = [MockModel(), MockModel()]
    with pytest.raises(ValueError):
        build_cascade(models, [4])  # mismatched lengths


def test_build_cascade_single_model_raises():
    with pytest.raises(ValueError):
        build_cascade([MockModel()], [4])


# ---------------------------------------------------------------------------
# Test 11: Two levels with always-same-token model → 100% acceptance
# ---------------------------------------------------------------------------


def test_two_levels_always_same_token_full_acceptance():
    """When both levels always predict token 7, draft matches perfectly → 100% at final level."""
    always_token = 7
    draft_len = 4
    level0 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=draft_len)
    level1 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=2)
    decoder = CascadeSpeculativeDecoder(levels=[level0, level1], max_new_tokens=draft_len)

    input_ids = make_input(batch=1, seq_len=5)
    _, stats = decoder.decode_step(input_ids)

    # Final level verifies draft produced by level0 (all always_token)
    # Since level1 also always predicts always_token, all draft_len tokens accepted
    assert stats["n_accepted"] == draft_len, (
        f"Expected {draft_len} accepted tokens, got {stats['n_accepted']}"
    )


# ---------------------------------------------------------------------------
# Test 12: Stats track tokens correctly across multiple steps
# ---------------------------------------------------------------------------


def test_stats_track_tokens_across_steps():
    """Verify that total_tokens accumulates correctly over multiple decode_steps."""
    always_token = 9
    draft_len = 4
    level0 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=draft_len)
    level1 = CascadeLevel(MockModel(vocab_size=256, always_token=always_token), draft_len=2)
    decoder = CascadeSpeculativeDecoder(levels=[level0, level1])

    input_ids = make_input(batch=1, seq_len=5)

    n_steps = 3
    total_new = 0
    for _ in range(n_steps):
        new_tokens, _ = decoder.decode_step(input_ids)
        total_new += new_tokens.shape[1]

    stats = decoder.get_stats()
    assert stats["total_tokens"] == total_new, (
        f"total_tokens {stats['total_tokens']} != sum of new_tokens {total_new}"
    )
    # acceptance_rates list has one entry per level
    assert len(stats["acceptance_rates"]) == 2
