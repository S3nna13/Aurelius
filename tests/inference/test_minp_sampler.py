"""Tests for src/inference/minp_sampler.py

Coverage
--------
Unit tests  (14)
  1.  test_config_defaults
  2.  test_temperature_scaling
  3.  test_temperature_one_noop
  4.  test_top_k_limits
  5.  test_top_k_zero_noop
  6.  test_min_p_adaptive
  7.  test_min_p_threshold
  8.  test_min_p_all_survive_uniform
  9.  test_top_p_nucleus
  10. test_sample_output_range
  11. test_sample_shape_batched
  12. test_sample_with_probs_sums_to_one
  13. test_effective_vocab_size
  14. test_combined_filters
Integration test (1)
  15. test_integration_vocab1024_batch4
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.inference.minp_sampler import MinPConfig, MinPSampler
from src.inference import DECODER_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_logits(vocab: int) -> torch.Tensor:
    """Return perfectly flat logits of shape (vocab,)."""
    return torch.zeros(vocab)


def _peaked_logits(vocab: int, peak_idx: int = 0, peak_val: float = 10.0) -> torch.Tensor:
    """Return logits with one dominant token."""
    logits = torch.zeros(vocab)
    logits[peak_idx] = peak_val
    return logits


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Test 1: MinPConfig has correct default values."""

    def test_config_defaults(self) -> None:
        cfg = MinPConfig()
        assert cfg.min_p == 0.05
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 1.0


class TestTemperatureScaling:
    """Tests 2–3: temperature primitive."""

    def test_temperature_scaling(self) -> None:
        """T=2.0 should halve every logit value."""
        sampler = MinPSampler(MinPConfig(temperature=2.0))
        logits = torch.tensor([2.0, 4.0, 6.0])
        out = sampler.apply_temperature(logits)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(out, expected), f"Expected {expected}, got {out}"

    def test_temperature_one_noop(self) -> None:
        """T=1.0 must return the identical tensor (same data, no-op)."""
        sampler = MinPSampler(MinPConfig(temperature=1.0))
        logits = torch.tensor([1.0, 2.0, 3.0])
        out = sampler.apply_temperature(logits)
        assert torch.equal(out, logits), "T=1.0 should be a no-op"


class TestTopK:
    """Tests 4–5: top-k primitive."""

    def test_top_k_limits(self) -> None:
        """Only the top-k tokens should remain finite; the rest become -inf."""
        k = 3
        vocab = 10
        sampler = MinPSampler(MinPConfig(top_k=k))
        logits = torch.arange(float(vocab))        # 0..9, ascending
        out = sampler.apply_top_k(logits)

        finite_mask = out > float("-inf")
        assert finite_mask.sum().item() == k, (
            f"Expected {k} finite logits, got {finite_mask.sum().item()}"
        )
        # The k surviving indices should be the k largest original logits.
        surviving_idx = finite_mask.nonzero(as_tuple=True)[0].tolist()
        expected_idx = list(range(vocab - k, vocab))
        assert sorted(surviving_idx) == expected_idx

    def test_top_k_zero_noop(self) -> None:
        """top_k=0 must leave logits unchanged."""
        sampler = MinPSampler(MinPConfig(top_k=0))
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = sampler.apply_top_k(logits)
        assert torch.equal(out, logits)


class TestMinP:
    """Tests 6–8: min-p primitive."""

    def test_min_p_adaptive(self) -> None:
        """Sharp distribution → fewer survivors than a flat distribution."""
        vocab = 50
        sampler = MinPSampler(MinPConfig(min_p=0.1))

        sharp_logits = _peaked_logits(vocab, peak_val=10.0)
        flat_logits = _uniform_logits(vocab)

        sharp_survivors = (sampler.apply_min_p(sharp_logits) > float("-inf")).sum()
        flat_survivors = (sampler.apply_min_p(flat_logits) > float("-inf")).sum()

        assert sharp_survivors < flat_survivors, (
            f"Sharp dist should yield fewer survivors ({sharp_survivors}) "
            f"than flat ({flat_survivors})"
        )

    def test_min_p_threshold(self) -> None:
        """Tokens with probability < min_p * p_max are explicitly masked."""
        vocab = 5
        # Construct a distribution where one token strongly dominates.
        logits = torch.tensor([5.0, 0.0, -5.0, -10.0, -10.0])
        sampler = MinPSampler(MinPConfig(min_p=0.2))

        probs = F.softmax(logits, dim=-1)
        p_max = probs.max().item()
        threshold = 0.2 * p_max

        out = sampler.apply_min_p(logits)

        for i in range(vocab):
            if probs[i].item() < threshold:
                assert out[i].item() == float("-inf"), (
                    f"Token {i} (prob={probs[i]:.4f}) should be masked"
                )
            else:
                assert out[i].item() != float("-inf"), (
                    f"Token {i} (prob={probs[i]:.4f}) should survive"
                )

    def test_min_p_all_survive_uniform(self) -> None:
        """Perfectly uniform logits → every token survives min_p filter."""
        vocab = 20
        sampler = MinPSampler(MinPConfig(min_p=0.05))
        logits = _uniform_logits(vocab)
        out = sampler.apply_min_p(logits)
        # With uniform probs all equal to 1/vocab, threshold = min_p/vocab,
        # which is less than 1/vocab, so all survive.
        survivors = (out > float("-inf")).sum().item()
        assert survivors == vocab, f"All {vocab} tokens should survive; got {survivors}"


class TestTopP:
    """Test 9: top-p (nucleus) primitive."""

    def test_top_p_nucleus(self) -> None:
        """Cumsum threshold should filter the long tail."""
        vocab = 10
        # Create a distribution where the first 3 tokens hold ~90 % of mass.
        logits = torch.tensor([5.0, 4.0, 3.0] + [0.0] * 7)
        sampler = MinPSampler(MinPConfig(top_p=0.90))
        out = sampler.apply_top_p(logits)

        surviving = (out > float("-inf")).sum().item()
        # The first 3 tokens dominate; at least those should survive.
        assert surviving >= 1
        # Full vocab should not survive.
        assert surviving < vocab, "Top-p should prune the tail"


class TestSample:
    """Tests 10–11: sample method."""

    def test_sample_output_range(self) -> None:
        """Sampled token id must be in [0, vocab_size)."""
        vocab = 100
        sampler = MinPSampler()
        logits = torch.randn(vocab)
        token = sampler.sample(logits)
        assert token.shape == torch.Size([]), "Unbatched input → scalar output"
        assert 0 <= token.item() < vocab

    def test_sample_shape_batched(self) -> None:
        """Batched input (B, V) → output shape (B,)."""
        B, V = 8, 200
        sampler = MinPSampler()
        logits = torch.randn(B, V)
        tokens = sampler.sample(logits)
        assert tokens.shape == (B,), f"Expected shape ({B},), got {tokens.shape}"
        assert tokens.min().item() >= 0
        assert tokens.max().item() < V


class TestSampleWithProbs:
    """Test 12: sample_with_probs method."""

    def test_sample_with_probs_sums_to_one(self) -> None:
        """Returned filtered_probs must sum to ≈1 for each row."""
        B, V = 4, 50
        sampler = MinPSampler(MinPConfig(min_p=0.05))
        logits = torch.randn(B, V)
        token_ids, filtered_probs = sampler.sample_with_probs(logits)

        assert token_ids.shape == (B,)
        assert filtered_probs.shape == (B, V)

        row_sums = filtered_probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), (
            f"Row sums: {row_sums.tolist()}"
        )


class TestEffectiveVocabSize:
    """Test 13: effective_vocab_size method."""

    def test_effective_vocab_size(self) -> None:
        """Effective vocab size should decrease with a higher min_p threshold."""
        vocab = 100
        logits = _peaked_logits(vocab, peak_idx=0, peak_val=8.0)

        sampler_loose = MinPSampler(MinPConfig(min_p=0.01))
        sampler_strict = MinPSampler(MinPConfig(min_p=0.5))

        size_loose = sampler_loose.effective_vocab_size(logits).item()
        size_strict = sampler_strict.effective_vocab_size(logits).item()

        assert size_strict <= size_loose, (
            f"Strict min_p ({size_strict}) should yield <= loose ({size_loose})"
        )
        assert size_strict >= 1, "At least the top token should always survive"


class TestCombinedFilters:
    """Test 14: full filter pipeline applied in the correct order."""

    def test_combined_filters(self) -> None:
        """top_k=10, min_p=0.05, top_p=0.9 — survivors ≤ top_k and ≥ 1."""
        vocab = 200
        k = 10
        sampler = MinPSampler(MinPConfig(top_k=k, min_p=0.05, top_p=0.9))
        logits = torch.randn(vocab)

        out = sampler.sample(logits)
        assert 0 <= out.item() < vocab

        # Verify that the effective vocab after top_k alone is exactly k.
        sampler_k_only = MinPSampler(MinPConfig(top_k=k))
        filtered_topk = sampler_k_only.apply_top_k(logits)
        assert (filtered_topk > float("-inf")).sum().item() == k

        # After full pipeline the surviving set should be ≤ k.
        full_filtered = sampler.apply_top_p(
            sampler.apply_min_p(
                sampler.apply_top_k(
                    sampler.apply_temperature(logits)
                )
            )
        )
        survivors = (full_filtered > float("-inf")).sum().item()
        assert 1 <= survivors <= k


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    """Test 15: Integration — vocab=1024, B=4."""

    def test_integration_vocab1024_batch4(self) -> None:
        """End-to-end sample run with realistic size.

        Verifies:
        - Token ids are in [0, 1024).
        - filtered_probs contain no -inf entries (distribution is valid).
        - Registry lookup yields the correct class.
        """
        B, V = 4, 1024
        cfg = MinPConfig(min_p=0.05, temperature=0.9, top_k=50, top_p=0.95)
        sampler = MinPSampler(cfg)

        torch.manual_seed(42)
        logits = torch.randn(B, V)

        token_ids, filtered_probs = sampler.sample_with_probs(logits)

        # Shape checks.
        assert token_ids.shape == (B,)
        assert filtered_probs.shape == (B, V)

        # All token ids in valid range.
        assert (token_ids >= 0).all() and (token_ids < V).all(), (
            f"Token ids out of range: {token_ids.tolist()}"
        )

        # filtered_probs must not contain -inf (softmax turns -inf logit → 0,
        # but the tensor itself should be a valid probability distribution).
        assert not torch.any(torch.isinf(filtered_probs)), (
            "filtered_probs should not contain -inf values"
        )
        assert not torch.any(torch.isnan(filtered_probs)), (
            "filtered_probs should not contain NaN values"
        )

        # Row sums ≈ 1.
        row_sums = filtered_probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), (
            f"Row sums: {row_sums.tolist()}"
        )

        # Registry check.
        assert DECODER_REGISTRY["minp_sampler"] is MinPSampler, (
            "DECODER_REGISTRY['minp_sampler'] should point to MinPSampler"
        )
