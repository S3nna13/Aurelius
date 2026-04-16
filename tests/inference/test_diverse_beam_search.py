"""Tests for diverse beam search (src/inference/diverse_beam_search.py)."""

from __future__ import annotations

import math
from typing import List

import pytest
import torch

from src.inference.diverse_beam_search import (
    DiverseBeamConfig,
    DiverseBeamSearch,
    compute_diversity_penalty,
    compute_sequence_diversity,
    group_beam_search_step,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

VOCAB = 32
BEAM = 4
N_GROUPS = 2
PROMPT_LEN = 3


# ---------------------------------------------------------------------------
# Simple mock model
# ---------------------------------------------------------------------------


def make_mock_model(vocab_size: int = VOCAB, seed: int = 0):
    """Return a model_fn that produces uniform-ish logits (deterministic)."""

    def model_fn(token_ids: torch.Tensor) -> torch.Tensor:
        # Deterministic: logits are a small perturbation around 0, based on
        # the last token so the output varies slightly per input.
        torch.manual_seed(seed + int(token_ids[-1].item()))
        return torch.randn(vocab_size) * 0.1

    return model_fn


# ---------------------------------------------------------------------------
# Test 1: DiverseBeamConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = DiverseBeamConfig()
    assert cfg.beam_width == 4
    assert cfg.n_groups == 2
    assert cfg.diversity_penalty == 0.5
    assert cfg.max_new_tokens == 50
    assert cfg.length_penalty == 1.0
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# Test 2: compute_diversity_penalty — shape is (vocab_size,)
# ---------------------------------------------------------------------------


def test_diversity_penalty_shape():
    pen = compute_diversity_penalty([1, 3, 5], vocab_size=VOCAB, penalty=0.5)
    assert pen.shape == (VOCAB,)


# ---------------------------------------------------------------------------
# Test 3: penalty applied to previous tokens only
# ---------------------------------------------------------------------------


def test_diversity_penalty_values():
    prev_tokens = [0, 4, 10]
    penalty = 0.75
    pen = compute_diversity_penalty(prev_tokens, vocab_size=VOCAB, penalty=penalty)

    for t in prev_tokens:
        assert math.isclose(pen[t].item(), -penalty, rel_tol=1e-6), (
            f"Token {t} should have penalty {-penalty}, got {pen[t].item()}"
        )

    # All other tokens must be 0.
    other = [i for i in range(VOCAB) if i not in prev_tokens]
    for t in other:
        assert pen[t].item() == 0.0, f"Token {t} should be 0, got {pen[t].item()}"


# ---------------------------------------------------------------------------
# Test 4: compute_diversity_penalty — empty prev tokens → all zeros
# ---------------------------------------------------------------------------


def test_diversity_penalty_empty_prev_tokens():
    pen = compute_diversity_penalty([], vocab_size=VOCAB, penalty=1.0)
    assert torch.all(pen == 0.0).item()


# ---------------------------------------------------------------------------
# Test 5: initialize_groups returns n_groups groups
# ---------------------------------------------------------------------------


def test_initialize_groups_count():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=1)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([10, 20, 30], dtype=torch.long)
    groups = dbs.initialize_groups(prompt)
    assert len(groups) == N_GROUPS


# ---------------------------------------------------------------------------
# Test 6: each group has beam_width // n_groups beams
# ---------------------------------------------------------------------------


def test_initialize_groups_beam_count():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=1)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([5, 6, 7], dtype=torch.long)
    groups = dbs.initialize_groups(prompt)
    expected_per_group = BEAM // N_GROUPS
    for g, beams in enumerate(groups):
        assert len(beams) == expected_per_group, (
            f"Group {g} has {len(beams)} beams, expected {expected_per_group}"
        )


# ---------------------------------------------------------------------------
# Test 7: search returns n_groups sequences
# ---------------------------------------------------------------------------


def test_search_returns_n_groups_sequences():
    cfg = DiverseBeamConfig(
        beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=3,
        diversity_penalty=0.5
    )
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    results = dbs.search(prompt)
    assert len(results) == N_GROUPS


# ---------------------------------------------------------------------------
# Test 8: search output is a list of 1-D tensors
# ---------------------------------------------------------------------------


def test_search_output_is_list_of_1d_tensors():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=3)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    results = dbs.search(prompt)
    for seq in results:
        assert isinstance(seq, torch.Tensor), "Each result must be a torch.Tensor"
        assert seq.dim() == 1, f"Expected 1-D tensor, got {seq.dim()}-D"


# ---------------------------------------------------------------------------
# Test 9: each sequence length > PROMPT_LEN (tokens were actually generated)
# ---------------------------------------------------------------------------


def test_search_sequences_longer_than_prompt():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=5)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    results = dbs.search(prompt)
    for seq in results:
        assert seq.shape[0] > PROMPT_LEN, (
            f"Sequence length {seq.shape[0]} not > prompt length {PROMPT_LEN}"
        )


# ---------------------------------------------------------------------------
# Test 10: search_with_scores returns n_groups tuples
# ---------------------------------------------------------------------------


def test_search_with_scores_returns_n_groups_tuples():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=3)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([7, 8, 9], dtype=torch.long)
    results = dbs.search_with_scores(prompt)
    assert len(results) == N_GROUPS


# ---------------------------------------------------------------------------
# Test 11: scores from search_with_scores are finite floats
# ---------------------------------------------------------------------------


def test_search_with_scores_finite():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=3)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    results = dbs.search_with_scores(prompt)
    for seq, score in results:
        assert math.isfinite(score), f"Score {score} is not finite"


# ---------------------------------------------------------------------------
# Test 12: compute_sequence_diversity >= 0
# ---------------------------------------------------------------------------


def test_compute_sequence_diversity_non_negative():
    seqs = [
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([5, 6, 7, 8]),
        torch.tensor([1, 6, 3, 9]),
    ]
    d = compute_sequence_diversity(seqs)
    assert d >= 0.0


# ---------------------------------------------------------------------------
# Test 13: identical sequences have diversity 0 (or trivially low)
# ---------------------------------------------------------------------------


def test_compute_sequence_diversity_identical_sequences():
    seq = torch.tensor([1, 2, 3])
    d = compute_sequence_diversity([seq, seq.clone(), seq.clone()])
    assert d == 0.0


# ---------------------------------------------------------------------------
# Test 14: diversity with penalty > diversity without penalty (qualitative)
# ---------------------------------------------------------------------------


def test_diversity_penalty_increases_diversity():
    """Groups with diversity_penalty > 0 should produce more diverse output
    than groups run with diversity_penalty = 0."""
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)

    # With strong diversity penalty.
    cfg_diverse = DiverseBeamConfig(
        beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=10,
        diversity_penalty=5.0, temperature=1.0,
    )
    dbs_diverse = DiverseBeamSearch(make_mock_model(seed=1), cfg_diverse)
    seqs_diverse = dbs_diverse.search(prompt)
    diversity_score = compute_sequence_diversity(seqs_diverse)

    # With zero diversity penalty.
    cfg_nodiv = DiverseBeamConfig(
        beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=10,
        diversity_penalty=0.0, temperature=1.0,
    )
    dbs_nodiv = DiverseBeamSearch(make_mock_model(seed=1), cfg_nodiv)
    seqs_nodiv = dbs_nodiv.search(prompt)
    nodiversity_score = compute_sequence_diversity(seqs_nodiv)

    # The test is qualitative: diversity_score should be >= nodiversity_score.
    # We accept equality in the degenerate case where both produce identical output.
    assert diversity_score >= nodiversity_score or math.isclose(
        diversity_score, nodiversity_score, abs_tol=1e-3
    ), (
        f"Expected diversity_score ({diversity_score:.4f}) >= "
        f"nodiversity_score ({nodiversity_score:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 15: group_beam_search_step returns at most beam_width candidates
# ---------------------------------------------------------------------------


def test_group_beam_search_step_count():
    logits = torch.randn(VOCAB)
    beams = [([1, 2], 0.0), ([1, 3], -0.5)]
    div_pen = torch.zeros(VOCAB)
    result = group_beam_search_step(logits, beams, div_pen, beam_width=BEAM)
    assert len(result) <= BEAM


# ---------------------------------------------------------------------------
# Test 16: search_with_scores sorted descending
# ---------------------------------------------------------------------------


def test_search_with_scores_sorted_descending():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=5)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([3, 4, 5], dtype=torch.long)
    results = dbs.search_with_scores(prompt)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), (
        "search_with_scores must return results sorted by score descending"
    )


# ---------------------------------------------------------------------------
# Test 17: sequences include prompt prefix
# ---------------------------------------------------------------------------


def test_search_sequences_start_with_prompt():
    cfg = DiverseBeamConfig(beam_width=BEAM, n_groups=N_GROUPS, max_new_tokens=4)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([10, 20, 30], dtype=torch.long)
    results = dbs.search(prompt)
    prompt_list = prompt.tolist()
    for seq in results:
        assert seq.tolist()[:PROMPT_LEN] == prompt_list, (
            f"Sequence {seq.tolist()} does not start with prompt {prompt_list}"
        )


# ---------------------------------------------------------------------------
# Test 18: invalid beam_width / n_groups raises ValueError
# ---------------------------------------------------------------------------


def test_mismatched_beam_and_groups_raises():
    cfg = DiverseBeamConfig(beam_width=5, n_groups=2)  # 5 not divisible by 2
    with pytest.raises(ValueError, match="divisible"):
        DiverseBeamSearch(make_mock_model(), cfg)


# ---------------------------------------------------------------------------
# Test 19: compute_sequence_diversity single sequence returns 0
# ---------------------------------------------------------------------------


def test_compute_sequence_diversity_single_sequence():
    d = compute_sequence_diversity([torch.tensor([1, 2, 3])])
    assert d == 0.0


# ---------------------------------------------------------------------------
# Test 20: n_groups=1 search returns exactly 1 sequence
# ---------------------------------------------------------------------------


def test_search_single_group():
    cfg = DiverseBeamConfig(beam_width=4, n_groups=1, max_new_tokens=3)
    dbs = DiverseBeamSearch(make_mock_model(), cfg)
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    results = dbs.search(prompt)
    assert len(results) == 1
    assert isinstance(results[0], torch.Tensor)
    assert results[0].dim() == 1
