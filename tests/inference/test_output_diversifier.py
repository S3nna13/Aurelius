"""Tests for output diversification (src/inference/output_diversifier.py)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.output_diversifier import (
    DiversifierConfig,
    DiversityPenalty,
    NgramBlocker,
    OutputDiversifier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 32
PROMPT_LEN = 4
MAX_NEW_TOKENS = 6


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Returns deterministic fixed logits for any input."""

    def __init__(self, vocab_size: int = VOCAB, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        # Fixed logit vector — register as buffer so it participates in no_grad.
        self.register_buffer("fixed_logits", torch.randn(vocab_size))
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return (1, seq_len, vocab_size) logits (last position varies by step)."""
        batch, seq_len = input_ids.shape
        # Tile fixed logits across the sequence so generate_diverse_batch can
        # index position -1.
        logits = self.fixed_logits.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, self.vocab_size)
        return logits.contiguous()


# ---------------------------------------------------------------------------
# Test 1: DiversifierConfig defaults correct
# ---------------------------------------------------------------------------


def test_diversifier_config_defaults():
    cfg = DiversifierConfig()
    assert cfg.strategy == "nucleus"
    assert cfg.diversity_penalty == 0.5
    assert cfg.ngram_size == 3
    assert cfg.min_diversity == 0.3
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# Test 2: NgramBlocker.update adds ngrams to block set
# ---------------------------------------------------------------------------


def test_ngram_blocker_update_adds_ngrams():
    blocker = NgramBlocker(ngram_size=2)
    seq = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    blocker.update(seq)
    # Expected bigrams: (1,2), (2,3), (3,4)
    assert (1, 2) in blocker._blocked
    assert (2, 3) in blocker._blocked
    assert (3, 4) in blocker._blocked
    # Should not contain non-consecutive pairs.
    assert (1, 3) not in blocker._blocked


# ---------------------------------------------------------------------------
# Test 3: NgramBlocker blocks tokens that would repeat ngrams
# ---------------------------------------------------------------------------


def test_ngram_blocker_blocks_repeated_ngrams():
    blocker = NgramBlocker(ngram_size=2)
    # Sequence already seen: tokens [5, 7]
    blocker.update(torch.tensor([5, 7], dtype=torch.long))

    logits = torch.zeros(VOCAB)
    # Context ends with token 5; token 7 would form bigram (5, 7) → blocked.
    context = torch.tensor([5], dtype=torch.long)
    masked = blocker.apply_blocking_mask(logits, context)

    assert masked[7].item() == float("-inf"), "Token 7 should be blocked"
    # Other tokens should be unaffected.
    assert masked[0].item() == 0.0
    assert masked[6].item() == 0.0


# ---------------------------------------------------------------------------
# Test 4: NgramBlocker.reset clears all blocked ngrams
# ---------------------------------------------------------------------------


def test_ngram_blocker_reset():
    blocker = NgramBlocker(ngram_size=2)
    blocker.update(torch.tensor([1, 2, 3], dtype=torch.long))
    assert len(blocker._blocked) > 0

    blocker.reset()
    assert len(blocker._blocked) == 0

    # After reset, nothing should be blocked.
    logits = torch.zeros(VOCAB)
    context = torch.tensor([1], dtype=torch.long)
    masked = blocker.apply_blocking_mask(logits, context)
    assert not torch.any(masked == float("-inf")).item()


# ---------------------------------------------------------------------------
# Test 5: DiversityPenalty.compute_diversity_bonus returns (vocab_size,) tensor
# ---------------------------------------------------------------------------


def test_diversity_penalty_bonus_shape():
    dp = DiversityPenalty(penalty_type="hamming", alpha=0.5)
    candidate_ids = torch.arange(VOCAB)
    bonus = dp.compute_diversity_bonus(candidate_ids, previous_ids=[])
    assert bonus.shape == (VOCAB,), f"Expected ({VOCAB},), got {bonus.shape}"


# ---------------------------------------------------------------------------
# Test 6: Tokens not in previous sequences get higher bonus
# ---------------------------------------------------------------------------


def test_diversity_penalty_novel_tokens_get_higher_bonus():
    dp = DiversityPenalty(penalty_type="hamming", alpha=1.0)
    candidate_ids = torch.arange(VOCAB)
    # Previous sequence uses tokens 0–9.
    prev = [torch.arange(10, dtype=torch.long)]
    bonus = dp.compute_diversity_bonus(candidate_ids, previous_ids=prev)

    # Tokens 0–9 (seen) should have bonus 0; tokens 10+ (unseen) should have bonus 1.
    for tok in range(10):
        assert bonus[tok].item() == 0.0, f"Seen token {tok} should have 0 bonus"
    for tok in range(10, VOCAB):
        assert bonus[tok].item() == 1.0, f"Unseen token {tok} should have 1.0 bonus"


# ---------------------------------------------------------------------------
# Test 7: OutputDiversifier.generate_diverse_batch returns correct shape
# ---------------------------------------------------------------------------


def test_generate_diverse_batch_shape():
    model = MockModel(vocab_size=VOCAB)
    diversifier = OutputDiversifier(model, strategy="nucleus", diversity_penalty=0.5)
    prompt = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    n_samples = 4

    result = diversifier.generate_diverse_batch(
        prompt, n_samples=n_samples, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0
    )
    assert result.shape == (n_samples, MAX_NEW_TOKENS), (
        f"Expected ({n_samples}, {MAX_NEW_TOKENS}), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# Test 8: compute_pairwise_diversity returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_pairwise_diversity_range():
    model = MockModel(vocab_size=VOCAB)
    diversifier = OutputDiversifier(model)
    torch.manual_seed(7)
    seqs = torch.randint(0, VOCAB, (5, 10))
    diversity = diversifier.compute_pairwise_diversity(seqs)
    assert isinstance(diversity, float), "Expected a Python float"
    assert 0.0 <= diversity <= 1.0, f"Diversity {diversity} not in [0, 1]"


# ---------------------------------------------------------------------------
# Test 9: Identical sequences have diversity=0
# ---------------------------------------------------------------------------


def test_pairwise_diversity_identical_sequences():
    model = MockModel(vocab_size=VOCAB)
    diversifier = OutputDiversifier(model)
    seq = torch.tensor([[1, 2, 3, 4, 5]] * 4, dtype=torch.long)
    diversity = diversifier.compute_pairwise_diversity(seq)
    assert diversity == 0.0, f"Identical sequences should have diversity 0, got {diversity}"


# ---------------------------------------------------------------------------
# Test 10: Completely different sequences have diversity=1
# ---------------------------------------------------------------------------


def test_pairwise_diversity_completely_different():
    model = MockModel(vocab_size=VOCAB)
    diversifier = OutputDiversifier(model)
    # Two sequences with no overlapping token ids.
    seq_a = torch.zeros(1, 5, dtype=torch.long)  # all token 0
    seq_b = torch.ones(1, 5, dtype=torch.long)  # all token 1
    seqs = torch.cat([seq_a, seq_b], dim=0)
    diversity = diversifier.compute_pairwise_diversity(seqs)
    assert diversity == 1.0, (
        f"Completely different sequences should have diversity 1, got {diversity}"
    )


# ---------------------------------------------------------------------------
# Test 11: filter_diverse removes near-duplicates
# ---------------------------------------------------------------------------


def test_filter_diverse_removes_near_duplicates():
    model = MockModel(vocab_size=VOCAB)
    diversifier = OutputDiversifier(model)

    # Create a batch: 2 nearly identical sequences + 1 very different one.
    base = torch.zeros(8, dtype=torch.long)  # all 0s
    near_dup = torch.zeros(8, dtype=torch.long)  # also all 0s (duplicate)
    different = torch.ones(8, dtype=torch.long) * 15  # all token 15

    seqs = torch.stack([base, near_dup, different])  # (3, 8)
    filtered = diversifier.filter_diverse(seqs, min_diversity=0.3)

    # The near-duplicate should be removed; we should have at most 2 sequences.
    assert filtered.shape[0] <= 2, (
        f"Expected at most 2 sequences after filtering, got {filtered.shape[0]}"
    )
    assert filtered.shape[0] >= 1, "At least one sequence must survive filtering"


# ---------------------------------------------------------------------------
# Test 12: generate_diverse_batch with n_samples=1 works correctly
# ---------------------------------------------------------------------------


def test_generate_diverse_batch_single_sample():
    model = MockModel(vocab_size=VOCAB)
    diversifier = OutputDiversifier(model, strategy="nucleus", diversity_penalty=0.5)
    prompt = torch.tensor([1, 2], dtype=torch.long)

    result = diversifier.generate_diverse_batch(
        prompt, n_samples=1, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0
    )
    assert result.shape == (1, MAX_NEW_TOKENS), (
        f"Expected (1, {MAX_NEW_TOKENS}), got {result.shape}"
    )
    assert result.dtype == torch.long
