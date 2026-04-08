"""Pure PyTorch BERTScore-style matching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise cosine similarity for batched token embeddings."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"Expected 2D or 3D inputs, got {x.dim()}D and {y.dim()}D")
    if x.size(0) != y.size(0):
        raise ValueError(f"Batch dimensions must match, got {x.size(0)} and {y.size(0)}")

    x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
    y_norm = y / y.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.matmul(x_norm, y_norm.transpose(-1, -2))


@dataclass(frozen=True)
class BERTScoreResult:
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


def bertscore(
    candidate_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> BERTScoreResult:
    """Compute BERTScore-style precision/recall/F1 from token embeddings."""
    if candidate_embeddings.dim() == 2:
        candidate_embeddings = candidate_embeddings.unsqueeze(0)
    if reference_embeddings.dim() == 2:
        reference_embeddings = reference_embeddings.unsqueeze(0)
    if candidate_embeddings.size(0) != reference_embeddings.size(0):
        raise ValueError("Candidate and reference batch sizes must match")

    batch_size, candidate_len = candidate_embeddings.shape[:2]
    reference_len = reference_embeddings.size(1)

    if candidate_mask is None:
        candidate_mask = torch.ones(
            batch_size, candidate_len, dtype=torch.bool, device=candidate_embeddings.device
        )
    if reference_mask is None:
        reference_mask = torch.ones(
            batch_size, reference_len, dtype=torch.bool, device=reference_embeddings.device
        )

    similarities = pairwise_cosine_similarity(candidate_embeddings, reference_embeddings, eps=eps)
    pair_mask = candidate_mask.unsqueeze(-1) & reference_mask.unsqueeze(-2)
    similarities = similarities.masked_fill(~pair_mask, float("-inf"))

    candidate_best = similarities.max(dim=-1).values.masked_fill(~candidate_mask, 0.0)
    reference_best = similarities.max(dim=-2).values.masked_fill(~reference_mask, 0.0)

    candidate_count = candidate_mask.sum(dim=-1).clamp(min=1)
    reference_count = reference_mask.sum(dim=-1).clamp(min=1)
    precision = candidate_best.sum(dim=-1) / candidate_count
    recall = reference_best.sum(dim=-1) / reference_count
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=eps)

    return BERTScoreResult(precision=precision, recall=recall, f1=f1)


def bertscore_from_token_ids(
    candidate_ids: torch.Tensor,
    reference_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    pad_id: int = 0,
    eps: float = 1e-8,
) -> BERTScoreResult:
    """Compute BERTScore from token ids and an embedding table."""
    candidate_embeddings = embedding_table[candidate_ids]
    reference_embeddings = embedding_table[reference_ids]
    candidate_mask = candidate_ids != pad_id
    reference_mask = reference_ids != pad_id
    return bertscore(
        candidate_embeddings,
        reference_embeddings,
        candidate_mask=candidate_mask,
        reference_mask=reference_mask,
        eps=eps,
    )
