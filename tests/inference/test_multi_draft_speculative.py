"""Tests for multi-draft speculative decoding."""

import torch

from src.inference.multi_draft_speculative import (
    MultiDraftConfig,
    MultiDraftDecoder,
    batch_verify_drafts,
    compute_draft_diversity,
    levenshtein_distance,
    typical_acceptance,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# typical_acceptance tests
# ---------------------------------------------------------------------------


def test_typical_acceptance_returns_valid_index():
    """typical_acceptance must return a value in [-1, n_drafts-1]."""
    n_drafts = 4
    draft_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    target_probs = torch.tensor([0.1, 0.3, 0.4, 0.2])

    results = set()
    torch.manual_seed(42)
    for _ in range(50):
        result = typical_acceptance(draft_probs, target_probs)
        results.add(result)
        assert -1 <= result < n_drafts, f"Out of range: {result}"


def test_typical_acceptance_high_ratio_accepts():
    """When target_prob >> draft_prob, the candidate should be accepted with high probability."""
    # draft probability is tiny; target probability is large → ratio >> 1 → accept almost always
    draft_probs = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
    target_probs = torch.tensor([0.9, 0.03, 0.04, 0.03])

    accepted_count = 0
    torch.manual_seed(0)
    for _ in range(100):
        result = typical_acceptance(draft_probs, target_probs)
        if result != -1:
            accepted_count += 1

    # With such a high ratio, the acceptance rate should be very high
    assert accepted_count >= 90, f"Expected high acceptance rate, got {accepted_count}/100"


# ---------------------------------------------------------------------------
# levenshtein_distance tests
# ---------------------------------------------------------------------------


def test_levenshtein_distance_identity():
    """Levenshtein distance between identical sequences must be 0."""
    assert levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0


def test_levenshtein_distance_single_edit():
    """Levenshtein distance with one substitution must be 1."""
    assert levenshtein_distance([1, 2, 3], [1, 4, 3]) == 1


# ---------------------------------------------------------------------------
# compute_draft_diversity tests
# ---------------------------------------------------------------------------


def test_compute_draft_diversity_unique_tokens():
    """Three candidates with the same first token → unique_first_tokens == 1."""
    candidates = [
        [5, 1, 2],
        [5, 3, 4],
        [5, 7, 8],
    ]
    result = compute_draft_diversity(candidates)
    assert result["unique_first_tokens"] == 1


def test_compute_draft_diversity_entropy_range():
    """Entropy must be >= 0 for any input."""
    candidates = [
        [1, 2, 3],
        [4, 5, 6],
        [1, 7, 8],
    ]
    result = compute_draft_diversity(candidates)
    assert result["entropy"] >= 0.0


# ---------------------------------------------------------------------------
# batch_verify_drafts tests
# ---------------------------------------------------------------------------


def test_batch_verify_returns_tensor():
    """batch_verify_drafts must return a (Tensor, list) tuple."""
    model = _make_model()
    prefix_ids = list(range(1, 6))  # [1, 2, 3, 4, 5]
    draft_candidates = [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
    ]

    target_probs, accepted_tokens = batch_verify_drafts(
        target_model=model,
        prefix_ids=prefix_ids,
        draft_candidates=draft_candidates,
        max_seq_len=64,
    )

    assert isinstance(target_probs, torch.Tensor), "target_probs must be a Tensor"
    assert target_probs.shape == (3,), f"Expected shape (3,), got {target_probs.shape}"
    assert isinstance(accepted_tokens, list), "accepted_tokens must be a list"


# ---------------------------------------------------------------------------
# MultiDraftDecoder tests
# ---------------------------------------------------------------------------


def test_multi_draft_decoder_generate_returns_tokens():
    """MultiDraftDecoder.generate must return a non-empty list of token IDs."""
    torch.manual_seed(0)
    target = _make_model()
    draft = _make_model()

    cfg = MultiDraftConfig(n_drafts=2, draft_steps=2, temperature=1.0)
    decoder = MultiDraftDecoder(
        target_model=target,
        draft_model=draft,
        config=cfg,
        eos_token_id=2,
    )

    prompt_ids = [1, 5, 10, 20]
    generated, stats = decoder.generate(prompt_ids, max_new_tokens=8)

    assert isinstance(generated, list), "generate must return a list"
    assert len(generated) > 0, "generate must produce at least one token"


def test_multi_draft_decoder_stats_keys():
    """stats dict must contain all required keys."""
    torch.manual_seed(1)
    target = _make_model()
    draft = _make_model()

    cfg = MultiDraftConfig(n_drafts=2, draft_steps=2, temperature=1.0)
    decoder = MultiDraftDecoder(
        target_model=target,
        draft_model=draft,
        config=cfg,
        eos_token_id=2,
    )

    prompt_ids = [1, 5, 10]
    _, stats = decoder.generate(prompt_ids, max_new_tokens=4)

    required_keys = {"n_accepted", "n_rejected", "acceptance_rate", "tokens_per_step"}
    assert required_keys.issubset(stats.keys()), (
        f"Missing keys: {required_keys - set(stats.keys())}"
    )


# ---------------------------------------------------------------------------
# Additional diversity test
# ---------------------------------------------------------------------------


def test_draft_diversity_all_different():
    """Four candidates with completely different first tokens → unique_first_tokens == 4."""
    candidates = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    result = compute_draft_diversity(candidates)
    assert result["unique_first_tokens"] == 4
