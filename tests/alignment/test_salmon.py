"""Tests for SALMON (Self-Alignment with Instructable Reward Models).

Reference: arXiv:2310.05910

Covers:
 1. SALMONScorer returns tensor of shape (n_principles,)
 2. aggregate_score returns scalar
 3. Scores are finite for normal inputs
 4. Determinism under torch.manual_seed
 5. Higher overlap → higher score (monotonicity proxy)
 6. SALMONFilter selects highest-scoring candidate
 7. SALMONFilter with 1 candidate returns index 0
 8. SALMONLoss is scalar
 9. SALMONLoss gradients are finite
10. SALMONLoss decreases when winner log_prob increases
11. No NaN/Inf on zero-length (empty) sequences
12. No NaN/Inf on long sequences
13. n_principles=1 edge case
14. scores shape consistent with n_principles
15. batch dimension handled correctly (multi-principle batch)
"""

from __future__ import annotations

import pytest
import torch

from src.alignment.salmon import (
    _VOCAB_SIZE,
    PrincipleConditionedRM,
    SALMONFilter,
    SALMONLoss,
    SALMONScorer,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

PRINCIPLES_3 = ["Be helpful", "Be harmless", "Be honest"]
PRINCIPLES_1 = ["Be honest"]

VOCAB = _VOCAB_SIZE


def _tok(length: int, seed: int = 0) -> torch.Tensor:
    """Create a random 1-D LongTensor of *length* token ids."""
    torch.manual_seed(seed)
    if length == 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.randint(0, VOCAB, (length,))


@pytest.fixture(scope="module")
def scorer_3() -> SALMONScorer:
    torch.manual_seed(42)
    return SALMONScorer(PRINCIPLES_3)


@pytest.fixture(scope="module")
def scorer_1() -> SALMONScorer:
    torch.manual_seed(7)
    return SALMONScorer(PRINCIPLES_1)


# ---------------------------------------------------------------------------
# Test 1 — SALMONScorer returns shape (n_principles,)
# ---------------------------------------------------------------------------


def test_scorer_output_shape(scorer_3: SALMONScorer) -> None:
    prompt = _tok(8, seed=1)
    response = _tok(16, seed=2)
    scores = scorer_3.score(prompt, response)
    assert scores.shape == (len(PRINCIPLES_3),), (
        f"Expected shape ({len(PRINCIPLES_3)},), got {scores.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 — aggregate_score returns a scalar tensor
# ---------------------------------------------------------------------------


def test_aggregate_score_is_scalar(scorer_3: SALMONScorer) -> None:
    prompt = _tok(8, seed=3)
    response = _tok(16, seed=4)
    agg = scorer_3.aggregate_score(prompt, response)
    assert agg.dim() == 0, f"Expected scalar (dim=0), got dim={agg.dim()}"


# ---------------------------------------------------------------------------
# Test 3 — Scores are finite for normal inputs
# ---------------------------------------------------------------------------


def test_scores_finite(scorer_3: SALMONScorer) -> None:
    prompt = _tok(10, seed=5)
    response = _tok(20, seed=6)
    scores = scorer_3.score(prompt, response)
    assert torch.all(torch.isfinite(scores)), "Non-finite values in score tensor"
    agg = scorer_3.aggregate_score(prompt, response)
    assert torch.isfinite(agg), f"Non-finite aggregate score: {agg.item()}"


# ---------------------------------------------------------------------------
# Test 4 — Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    torch.manual_seed(99)
    scorer_a = SALMONScorer(PRINCIPLES_3)
    torch.manual_seed(99)
    scorer_b = SALMONScorer(PRINCIPLES_3)

    prompt = _tok(8, seed=10)
    response = _tok(12, seed=11)

    scores_a = scorer_a.score(prompt, response)
    scores_b = scorer_b.score(prompt, response)
    assert torch.allclose(scores_a, scores_b), "Scores differ between identical seeds"


# ---------------------------------------------------------------------------
# Test 5 — Higher overlap → higher score (monotonicity proxy)
# ---------------------------------------------------------------------------


def test_monotonicity_higher_overlap() -> None:
    """Responses whose mean embedding aligns with the principle embedding score
    higher than those that anti-align.

    We directly manipulate the embedding weights so that:
      - principle character 'A' maps to the unit vector e_0 = [1, 0, 0, ...].
      - response_high tokens map to the same direction → cosine ~ +1.
      - response_low tokens map to the opposite direction → cosine ~ -1.
    This guarantees score_high > score_low regardless of random seed.
    """
    torch.manual_seed(0)
    scorer = SALMONScorer(["A"])  # single-char principle
    pcrm = scorer.pcrm
    D = pcrm.embed_dim

    a_id = ord("A") % pcrm.principle_vocab_size

    # Craft a direction vector
    direction = torch.zeros(D)
    direction[0] = 1.0  # unit vector along dim 0

    with torch.no_grad():
        # Principle embedding for 'A' → direction
        pcrm.principle_embed.weight[a_id].copy_(direction)
        # Token id 1 → aligned with direction
        pcrm.token_embed.weight[1].copy_(direction)
        # Token id 2 → anti-aligned with direction
        pcrm.token_embed.weight[2].copy_(-direction)

    response_high = torch.ones(8, dtype=torch.long)  # all token id 1
    response_low = torch.full((8,), 2, dtype=torch.long)  # all token id 2
    prompt = _tok(4, seed=20)

    score_high = scorer.aggregate_score(prompt, response_high).item()
    score_low = scorer.aggregate_score(prompt, response_low).item()
    assert score_high > score_low, (
        f"Expected score_high ({score_high:.4f}) > score_low ({score_low:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 6 — SALMONFilter selects highest-scoring candidate
# ---------------------------------------------------------------------------


def test_filter_selects_best(scorer_3: SALMONScorer) -> None:
    torch.manual_seed(55)
    filt = SALMONFilter(scorer_3)
    prompt = _tok(8, seed=30)
    candidates = [_tok(12, seed=31 + i) for i in range(4)]

    best_idx, scores = filt.select_best(prompt, candidates)
    expected_idx = int(scores.argmax().item())
    assert best_idx == expected_idx, (
        f"SALMONFilter returned idx {best_idx}, expected {expected_idx}"
    )
    assert scores.shape == (len(candidates),)


# ---------------------------------------------------------------------------
# Test 7 — SALMONFilter with 1 candidate returns index 0
# ---------------------------------------------------------------------------


def test_filter_single_candidate(scorer_3: SALMONScorer) -> None:
    filt = SALMONFilter(scorer_3)
    prompt = _tok(8, seed=40)
    candidates = [_tok(12, seed=41)]
    best_idx, scores = filt.select_best(prompt, candidates)
    assert best_idx == 0
    assert scores.shape == (1,)


# ---------------------------------------------------------------------------
# Test 8 — SALMONLoss is a scalar tensor
# ---------------------------------------------------------------------------


def test_loss_is_scalar() -> None:
    loss_fn = SALMONLoss(alpha=0.1)
    winner_lp = torch.tensor(-2.5, requires_grad=True)
    loser_lps = [torch.tensor(-3.0), torch.tensor(-4.0)]
    loss = loss_fn(winner_lp, loser_lps)
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 9 — SALMONLoss gradients are finite
# ---------------------------------------------------------------------------


def test_loss_gradients_finite() -> None:
    loss_fn = SALMONLoss(alpha=0.1)
    winner_lp = torch.tensor(-2.5, requires_grad=True)
    loser_lps = [torch.tensor(-3.0, requires_grad=True)]
    loss = loss_fn(winner_lp, loser_lps)
    loss.backward()
    assert winner_lp.grad is not None
    assert torch.isfinite(winner_lp.grad), f"Non-finite grad: {winner_lp.grad}"


# ---------------------------------------------------------------------------
# Test 10 — SALMONLoss decreases when winner log_prob increases
# ---------------------------------------------------------------------------


def test_loss_decreases_with_higher_winner_logprob() -> None:
    loss_fn = SALMONLoss(alpha=0.0)  # pure SFT, no contrastive
    loss_low = loss_fn(torch.tensor(-5.0), []).item()
    loss_high = loss_fn(torch.tensor(-1.0), []).item()
    assert loss_high < loss_low, (
        f"Expected loss to decrease when winner log-prob rises: "
        f"loss_low={loss_low:.4f}, loss_high={loss_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11 — No NaN/Inf on zero-length (empty) sequences
# ---------------------------------------------------------------------------


def test_no_nan_on_empty_sequences(scorer_3: SALMONScorer) -> None:
    prompt = _tok(0)
    response = _tok(0)
    scores = scorer_3.score(prompt, response)
    assert torch.all(torch.isfinite(scores)), f"Non-finite scores on empty seqs: {scores}"
    agg = scorer_3.aggregate_score(prompt, response)
    assert torch.isfinite(agg), f"Non-finite agg on empty seqs: {agg}"


# ---------------------------------------------------------------------------
# Test 12 — No NaN/Inf on long sequences
# ---------------------------------------------------------------------------


def test_no_nan_on_long_sequences(scorer_3: SALMONScorer) -> None:
    prompt = _tok(512, seed=100)
    response = _tok(512, seed=101)
    scores = scorer_3.score(prompt, response)
    assert torch.all(torch.isfinite(scores)), f"Non-finite scores on long seqs: {scores}"


# ---------------------------------------------------------------------------
# Test 13 — n_principles=1 edge case
# ---------------------------------------------------------------------------


def test_single_principle_scorer(scorer_1: SALMONScorer) -> None:
    prompt = _tok(8, seed=50)
    response = _tok(12, seed=51)
    scores = scorer_1.score(prompt, response)
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"
    agg = scorer_1.aggregate_score(prompt, response)
    assert agg.dim() == 0
    assert torch.isfinite(agg)


# ---------------------------------------------------------------------------
# Test 14 — scores shape consistent with n_principles for varying K
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_scores_shape_with_k_principles(k: int) -> None:
    torch.manual_seed(k)
    principles = [f"Principle number {i}" for i in range(k)]
    scorer = SALMONScorer(principles)
    prompt = _tok(8, seed=200)
    response = _tok(16, seed=201)
    scores = scorer.score(prompt, response)
    assert scores.shape == (k,), f"Expected ({k},), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 15 — "batch" dimension: multiple (prompt, response) pairs
# ---------------------------------------------------------------------------


def test_multiple_pairs_independent(scorer_3: SALMONScorer) -> None:
    """Scoring several pairs should give independent, finite results."""
    results = []
    for i in range(5):
        prompt = _tok(8, seed=300 + i)
        response = _tok(12, seed=400 + i)
        agg = scorer_3.aggregate_score(prompt, response)
        assert torch.isfinite(agg), f"Non-finite score for pair {i}"
        results.append(agg.item())
    # Scores should not all be identical (distinct inputs → distinct scores)
    assert len(set(f"{v:.6f}" for v in results)) > 1, (
        "All aggregate scores are identical across different input pairs"
    )


# ---------------------------------------------------------------------------
# Bonus — PrincipleConditionedRM direct API
# ---------------------------------------------------------------------------


def test_pcrm_score_is_scalar_finite() -> None:
    torch.manual_seed(0)
    pcrm = PrincipleConditionedRM()
    prompt = _tok(8, seed=500)
    response = _tok(8, seed=501)
    r = pcrm.score("Be helpful", prompt, response)
    assert r.dim() == 0, f"Expected scalar, got shape {r.shape}"
    assert torch.isfinite(r), f"Non-finite PCRM score: {r.item()}"


def test_pcrm_raises_on_non_1d() -> None:
    torch.manual_seed(0)
    pcrm = PrincipleConditionedRM()
    with pytest.raises(ValueError, match="prompt_tokens must be 1-D"):
        pcrm.score("Be helpful", torch.zeros(2, 4, dtype=torch.long), _tok(4))
    with pytest.raises(ValueError, match="response_tokens must be 1-D"):
        pcrm.score("Be helpful", _tok(4), torch.zeros(2, 4, dtype=torch.long))
