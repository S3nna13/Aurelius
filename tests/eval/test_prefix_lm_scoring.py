"""Tests for src/eval/prefix_lm_scoring.py

All tests use tiny tensors and a trivial mock model so they run fast
with no GPU and no external dependencies beyond PyTorch.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.eval.prefix_lm_scoring import (
    PrefixLMScorer,
    ScoringConfig,
    compute_token_log_probs,
    perplexity_from_log_probs,
    rank_completions,
    score_completion,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _uniform_logits(T: int, vocab: int, seed: int = 0) -> Tensor:
    """Return deterministic random logits of shape (T, vocab)."""
    torch.manual_seed(seed)
    return torch.randn(T, vocab)


def _make_ids(T: int, vocab: int, seed: int = 1) -> Tensor:
    """Return deterministic random token ids of shape (T,)."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (T,))


def _dummy_model(vocab: int = 32, seed: int = 42):
    """Return a callable that ignores input and returns fixed logits."""

    def model_fn(input_ids: Tensor) -> Tensor:
        T = input_ids.shape[-1]
        torch.manual_seed(seed)
        return torch.randn(T, vocab)

    return model_fn


# ---------------------------------------------------------------------------
# ScoringConfig tests
# ---------------------------------------------------------------------------


def test_scoring_config_defaults():
    """ScoringConfig should have correct default values."""
    cfg = ScoringConfig()
    assert cfg.reduction == "mean"
    assert cfg.log_base == 2.0
    assert cfg.normalize_by_length is True


def test_scoring_config_custom():
    """ScoringConfig should accept custom values without error."""
    cfg = ScoringConfig(reduction="sum", log_base=10.0, normalize_by_length=False)
    assert cfg.reduction == "sum"
    assert cfg.log_base == 10.0
    assert cfg.normalize_by_length is False


def test_scoring_config_invalid_reduction():
    """ScoringConfig should raise ValueError for unknown reduction."""
    with pytest.raises(ValueError, match="reduction"):
        ScoringConfig(reduction="invalid")


def test_scoring_config_invalid_log_base():
    """ScoringConfig should raise ValueError for non-positive log_base."""
    with pytest.raises(ValueError, match="log_base"):
        ScoringConfig(log_base=-1.0)


# ---------------------------------------------------------------------------
# compute_token_log_probs tests
# ---------------------------------------------------------------------------


def test_compute_token_log_probs_shape():
    """Output must be 1-D with length T."""
    T, vocab = 8, 32
    logits = _uniform_logits(T, vocab)
    ids = _make_ids(T, vocab)
    lp = compute_token_log_probs(logits, ids)
    assert lp.shape == (T,), f"Expected ({T},), got {lp.shape}"


def test_compute_token_log_probs_non_positive():
    """All log-probs must be <= 0 (they are log-softmax values)."""
    T, vocab = 10, 64
    logits = _uniform_logits(T, vocab)
    ids = _make_ids(T, vocab)
    lp = compute_token_log_probs(logits, ids)
    assert (lp <= 0).all(), "Some log-probs are positive, which is impossible"


def test_compute_token_log_probs_sequence_log_prob_non_positive():
    """Sum of per-token log-probs (sequence log-prob) must be <= 0."""
    T, vocab = 12, 16
    logits = _uniform_logits(T, vocab)
    ids = _make_ids(T, vocab)
    lp = compute_token_log_probs(logits, ids)
    assert lp.sum().item() <= 0.0


def test_compute_token_log_probs_known_value():
    """For a 2-token, 2-vocab case with known logits the result is exact."""
    # logits = [[0, 0], [100, -100]]
    # position 0: log_softmax([0,0]) = [-ln2, -ln2]; token 0 → -ln2
    # position 1: log_softmax([100,-100]) ≈ [0, -200]; token 1 → ≈ -200
    logits = torch.tensor([[0.0, 0.0], [100.0, -100.0]])
    ids = torch.tensor([0, 1])
    lp = compute_token_log_probs(logits, ids)
    assert abs(lp[0].item() - (-math.log(2))) < 1e-5
    assert lp[1].item() < -100.0  # very negative


# ---------------------------------------------------------------------------
# score_completion tests
# ---------------------------------------------------------------------------


def test_score_completion_mean_returns_scalar():
    """reduction='mean' must return a 0-D tensor."""
    T, vocab = 10, 32
    lp = compute_token_log_probs(_uniform_logits(T, vocab), _make_ids(T, vocab))
    mask = torch.zeros(T, dtype=torch.bool)
    mask[5:] = True
    cfg = ScoringConfig(reduction="mean")
    score = score_completion(lp, mask, cfg)
    assert score.ndim == 0, f"Expected scalar (0-D), got shape {score.shape}"


def test_score_completion_sum_returns_scalar():
    """reduction='sum' must return a 0-D tensor."""
    T, vocab = 8, 16
    lp = compute_token_log_probs(_uniform_logits(T, vocab), _make_ids(T, vocab))
    mask = torch.zeros(T, dtype=torch.bool)
    mask[4:] = True
    cfg = ScoringConfig(reduction="sum")
    score = score_completion(lp, mask, cfg)
    assert score.ndim == 0


def test_score_completion_none_returns_completion_length():
    """reduction='none' must return a 1-D tensor of length n_completion_tokens."""
    T, vocab = 10, 32
    n_completion = 4
    lp = compute_token_log_probs(_uniform_logits(T, vocab), _make_ids(T, vocab))
    mask = torch.zeros(T, dtype=torch.bool)
    mask[-n_completion:] = True
    cfg = ScoringConfig(reduction="none")
    out = score_completion(lp, mask, cfg)
    assert out.ndim == 1
    assert out.shape[0] == n_completion


def test_score_completion_log_base_conversion():
    """Score in base 2 should equal score in nats divided by ln(2)."""
    T, vocab = 8, 16
    lp = compute_token_log_probs(_uniform_logits(T, vocab), _make_ids(T, vocab))
    mask = torch.zeros(T, dtype=torch.bool)
    mask[4:] = True

    cfg_nats = ScoringConfig(reduction="sum", log_base=math.e)
    cfg_bits = ScoringConfig(reduction="sum", log_base=2.0)
    score_nats = score_completion(lp, mask, cfg_nats).item()
    score_bits = score_completion(lp, mask, cfg_bits).item()
    # bits = nats / ln(2)
    assert abs(score_bits - score_nats / math.log(2)) < 1e-4


# ---------------------------------------------------------------------------
# rank_completions tests
# ---------------------------------------------------------------------------


def test_rank_completions_highest_score_gets_rank_0():
    """The completion with the highest score must have rank 0."""
    scores = torch.tensor([-3.0, -1.0, -5.0, -2.0])
    ranks = rank_completions(scores)
    best_idx = scores.argmax().item()
    assert ranks[best_idx].item() == 0


def test_rank_completions_length():
    """rank_completions output length must equal n_completions."""
    n = 7
    scores = torch.randn(n)
    ranks = rank_completions(scores)
    assert ranks.shape == (n,)


def test_rank_completions_is_permutation():
    """Ranks should be a permutation of 0..n-1."""
    n = 6
    torch.manual_seed(5)
    scores = torch.randn(n)
    ranks = rank_completions(scores)
    assert sorted(ranks.tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# PrefixLMScorer tests
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    vocab = 32
    cfg = ScoringConfig(reduction="mean")
    model = _dummy_model(vocab=vocab)
    return PrefixLMScorer(model_fn=model, config=cfg)


def test_prefix_lm_scorer_score_returns_scalar(scorer):
    """PrefixLMScorer.score must return a 0-D tensor."""
    ids = torch.randint(0, 32, (10,))
    result = scorer.score(ids, completion_start=6)
    assert result.ndim == 0


def test_prefix_lm_scorer_score_batch_shape(scorer):
    """PrefixLMScorer.score_batch must return a 1-D tensor of length n."""
    n = 4
    inputs = [torch.randint(0, 32, (10,)) for _ in range(n)]
    starts = [5] * n
    out = scorer.score_batch(inputs, starts)
    assert out.shape == (n,)


def test_prefix_lm_scorer_rank_best_is_rank_0():
    """The completion with the highest score must have rank 0."""
    vocab = 16

    # Build a model whose output logits we control: for input starting with
    # token 0 we return logits that heavily favour the continuation.
    def controlled_model(input_ids: Tensor) -> Tensor:
        T = input_ids.shape[-1]
        # Put high mass on token 0 for the first input, uniform for others
        logits = torch.zeros(T, vocab)
        if input_ids[0].item() == 0:
            logits[:, 0] = 100.0  # near-zero loss → high score
        return logits

    cfg = ScoringConfig(reduction="mean")
    scorer = PrefixLMScorer(model_fn=controlled_model, config=cfg)

    # First input: starts with token 0, completion continues with token 0
    inp0 = torch.zeros(6, dtype=torch.long)  # all token-0
    inp1 = torch.ones(6, dtype=torch.long)  # all token-1 (worse score)
    inputs = [inp0, inp1]
    starts = [3, 3]

    ranks = scorer.rank(inputs, starts)
    assert ranks[0].item() == 0, "Input 0 should have rank 0 (best score)"


# ---------------------------------------------------------------------------
# perplexity_from_log_probs tests
# ---------------------------------------------------------------------------


def test_perplexity_from_log_probs_finite_positive():
    """Perplexity must be a finite, positive float."""
    log_probs = torch.tensor([-1.5, -2.0, -0.8, -3.1])
    ppl = perplexity_from_log_probs(log_probs, base=2.0)
    assert math.isfinite(ppl)
    assert ppl > 0.0


def test_perplexity_from_log_probs_all_zeros_gives_one():
    """If all log-probs are 0, perplexity = base^0 = 1."""
    log_probs = torch.zeros(5)
    ppl = perplexity_from_log_probs(log_probs, base=2.0)
    assert abs(ppl - 1.0) < 1e-6, f"Expected 1.0, got {ppl}"


def test_perplexity_from_log_probs_known_value():
    """Verify perplexity formula: base^(-mean(log_probs))."""
    base = 2.0
    log_probs = torch.tensor([-2.0, -4.0])  # mean = -3
    expected = base**3  # = 8.0
    ppl = perplexity_from_log_probs(log_probs, base=base)
    assert abs(ppl - expected) < 1e-5, f"Expected {expected}, got {ppl}"
