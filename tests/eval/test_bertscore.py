"""Tests for pure PyTorch BERTScore utilities."""

import pytest
import torch

from src.eval.bertscore import bertscore, bertscore_from_token_ids, pairwise_cosine_similarity


def test_pairwise_cosine_similarity_returns_expected_shape():
    x = torch.randn(3, 5)
    y = torch.randn(4, 5)
    sims = pairwise_cosine_similarity(x, y)
    assert sims.shape == (1, 3, 4)


def test_bertscore_identical_embeddings_score_one():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = bertscore(embeddings, embeddings)
    assert result.precision.item() == pytest.approx(1.0)
    assert result.recall.item() == pytest.approx(1.0)
    assert result.f1.item() == pytest.approx(1.0)


def test_bertscore_orthogonal_embeddings_have_low_score():
    candidate = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    reference = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    result = bertscore(candidate, reference)
    assert result.f1.item() == pytest.approx(0.0, abs=1e-6)


def test_bertscore_masks_ignore_padding_tokens():
    candidate = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-5.0, -5.0]]])
    candidate_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    reference_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    result = bertscore(candidate, reference, candidate_mask=candidate_mask, reference_mask=reference_mask)
    assert result.f1.item() == pytest.approx(1.0)


def test_bertscore_batched_inputs_return_batch_scores():
    candidate = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    ])
    reference = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 1.0]],
    ])
    result = bertscore(candidate, reference)
    assert result.f1.shape == (2,)
    assert result.f1[0].item() > result.f1[1].item()


def test_bertscore_from_token_ids_matches_direct_call():
    table = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    candidate_ids = torch.tensor([[1, 2, 0]])
    reference_ids = torch.tensor([[1, 2, 0]])
    by_ids = bertscore_from_token_ids(candidate_ids, reference_ids, table, pad_id=0)
    direct = bertscore(table[candidate_ids], table[reference_ids], candidate_ids != 0, reference_ids != 0)
    assert by_ids.f1.item() == pytest.approx(direct.f1.item())


def test_bertscore_rejects_mismatched_batch_sizes():
    with pytest.raises(ValueError):
        bertscore(torch.randn(2, 3, 4), torch.randn(1, 3, 4))
