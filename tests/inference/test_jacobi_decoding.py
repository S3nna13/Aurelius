"""Tests for src/inference/jacobi_decoding.py.

Tests JacobiDecoder and jacobi_decode convenience function.
Uses a small MockModel (nn.Embedding + nn.Linear) — pure PyTorch.

Configuration: vocab_size=64, d_model=16, prompt_len=4, max_new_tokens=8, batch=1.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.jacobi_decoding import (
    JacobiDecoder,
    jacobi_decode,
)

# ---------------------------------------------------------------------------
# Mock Model
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Tiny mock LM: nn.Embedding + nn.Linear -> (None, logits, None).

    forward(input_ids: (B, T)) -> (None, logits: (B, T, vocab_size), None)
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 16, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)  # (B, T, d_model)
        logits = self.proj(x)  # (B, T, vocab_size)
        return (None, logits, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
D_MODEL = 16
PROMPT_LEN = 4
MAX_NEW_TOKENS = 8
BATCH = 1


@pytest.fixture
def model() -> MockModel:
    return MockModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, seed=42)


@pytest.fixture
def prompt() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, VOCAB_SIZE, (BATCH, PROMPT_LEN))


@pytest.fixture
def decoder(model: MockModel) -> JacobiDecoder:
    return JacobiDecoder(model=model, max_iterations=10, temperature=1.0)


# ---------------------------------------------------------------------------
# Test 1: JacobiDecoder instantiates with defaults
# ---------------------------------------------------------------------------


def test_jacobi_decoder_instantiates():
    model = MockModel()
    dec = JacobiDecoder(model=model)
    assert dec.max_iterations == 10
    assert dec.temperature == 1.0
    assert dec.top_k == 0
    assert dec.eos_token_id is None


# ---------------------------------------------------------------------------
# Test 2: decode() returns tensor of shape (B, prompt_len + max_new_tokens)
# ---------------------------------------------------------------------------


def test_decode_output_shape(decoder, prompt):
    output_ids, n_iter = decoder.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    expected_len = PROMPT_LEN + MAX_NEW_TOKENS
    assert output_ids.shape == (BATCH, expected_len), (
        f"Expected shape ({BATCH}, {expected_len}), got {output_ids.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: Prompt tokens are preserved in output (first T_prompt unchanged)
# ---------------------------------------------------------------------------


def test_prompt_tokens_preserved(decoder, prompt):
    output_ids, _ = decoder.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert torch.equal(output_ids[:, :PROMPT_LEN], prompt), (
        "Prompt tokens should be unchanged in the output."
    )


# ---------------------------------------------------------------------------
# Test 4: n_iterations <= max_iterations
# ---------------------------------------------------------------------------


def test_n_iterations_bounded(decoder, prompt):
    _, n_iter = decoder.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert n_iter <= decoder.max_iterations, (
        f"n_iterations={n_iter} exceeds max_iterations={decoder.max_iterations}"
    )


# ---------------------------------------------------------------------------
# Test 5: decode_with_stats() returns dict with all required keys
# ---------------------------------------------------------------------------


def test_decode_with_stats_keys(decoder, prompt):
    stats = decoder.decode_with_stats(prompt, max_new_tokens=MAX_NEW_TOKENS)
    required_keys = {"output_ids", "n_iterations", "tokens_per_iteration", "convergence_mask"}
    assert required_keys <= set(stats.keys()), f"Missing keys: {required_keys - set(stats.keys())}"


# ---------------------------------------------------------------------------
# Test 6: tokens_per_iteration sums to <= max_new_tokens * max_iterations
# ---------------------------------------------------------------------------


def test_tokens_per_iteration_bounded(decoder, prompt):
    stats = decoder.decode_with_stats(prompt, max_new_tokens=MAX_NEW_TOKENS)
    total = sum(stats["tokens_per_iteration"])
    upper_bound = MAX_NEW_TOKENS * decoder.max_iterations
    assert total <= upper_bound, (
        f"tokens_per_iteration sum {total} exceeds upper bound {upper_bound}"
    )


# ---------------------------------------------------------------------------
# Test 7: convergence_mask is boolean tensor of shape (max_new_tokens,)
# ---------------------------------------------------------------------------


def test_convergence_mask_shape_and_dtype(decoder, prompt):
    stats = decoder.decode_with_stats(prompt, max_new_tokens=MAX_NEW_TOKENS)
    mask = stats["convergence_mask"]
    assert isinstance(mask, torch.Tensor), "convergence_mask must be a Tensor"
    assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"
    assert mask.shape == (MAX_NEW_TOKENS,), f"Expected shape ({MAX_NEW_TOKENS},), got {mask.shape}"


# ---------------------------------------------------------------------------
# Test 8: temperature=0.0 (greedy) is deterministic — same call twice gives same result
# ---------------------------------------------------------------------------


def test_greedy_is_deterministic(model, prompt):
    dec = JacobiDecoder(model=model, max_iterations=5, temperature=0.0)
    out1, _ = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    out2, _ = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert torch.equal(out1, out2), "Greedy decoding should be deterministic."


# ---------------------------------------------------------------------------
# Test 9: temperature=1.0 with seeded torch gives reproducible output
# ---------------------------------------------------------------------------


def test_seeded_decode_reproducible(model, prompt):
    dec = JacobiDecoder(model=model, max_iterations=5, temperature=1.0)
    torch.manual_seed(123)
    out1, _ = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    torch.manual_seed(123)
    out2, _ = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert torch.equal(out1, out2), "Seeded decoding should be reproducible."


# ---------------------------------------------------------------------------
# Test 10: eos_token_id param accepted; decode runs without error
# ---------------------------------------------------------------------------


def test_eos_token_id_accepted(model, prompt):
    dec = JacobiDecoder(model=model, max_iterations=5, temperature=0.0, eos_token_id=1)
    output_ids, n_iter = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert output_ids.shape == (BATCH, PROMPT_LEN + MAX_NEW_TOKENS)
    assert n_iter >= 1


# ---------------------------------------------------------------------------
# Test 11: max_iterations=1 runs exactly 1 forward pass
# ---------------------------------------------------------------------------


def test_max_iterations_one(model, prompt):
    dec = JacobiDecoder(model=model, max_iterations=1, temperature=1.0)
    _, n_iter = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert n_iter == 1, f"Expected exactly 1 iteration, got {n_iter}"


# ---------------------------------------------------------------------------
# Test 12: jacobi_decode() convenience function works
# ---------------------------------------------------------------------------


def test_jacobi_decode_convenience(model, prompt):
    output_ids = jacobi_decode(
        model=model,
        input_ids=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        max_iterations=5,
        temperature=1.0,
    )
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.shape == (BATCH, PROMPT_LEN + MAX_NEW_TOKENS)


# ---------------------------------------------------------------------------
# Test 13: Batch size 2 works correctly (both sequences decoded)
# ---------------------------------------------------------------------------


def test_batch_size_two(model):
    batch = 2
    torch.manual_seed(99)
    prompt = torch.randint(0, VOCAB_SIZE, (batch, PROMPT_LEN))
    dec = JacobiDecoder(model=model, max_iterations=5, temperature=1.0)
    output_ids, n_iter = dec.decode(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert output_ids.shape == (batch, PROMPT_LEN + MAX_NEW_TOKENS), (
        f"Expected shape ({batch}, {PROMPT_LEN + MAX_NEW_TOKENS}), got {output_ids.shape}"
    )
    # Ensure prompt is preserved for both sequences
    assert torch.equal(output_ids[:, :PROMPT_LEN], prompt)
    assert n_iter >= 1


# ---------------------------------------------------------------------------
# Test 14: decode_with_stats with max_iterations=5 returns n_iterations <= 5
# ---------------------------------------------------------------------------


def test_decode_with_stats_iteration_bound(model, prompt):
    dec = JacobiDecoder(model=model, max_iterations=5, temperature=1.0)
    stats = dec.decode_with_stats(prompt, max_new_tokens=MAX_NEW_TOKENS)
    assert stats["n_iterations"] <= 5, (
        f"n_iterations={stats['n_iterations']} exceeds max_iterations=5"
    )
    assert len(stats["tokens_per_iteration"]) == stats["n_iterations"], (
        "tokens_per_iteration length should match n_iterations"
    )
