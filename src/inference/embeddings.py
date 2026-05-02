"""Semantic embedding extraction and similarity utilities for AureliusTransformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""

    pooling: str = "mean"  # "mean", "last", "max"
    normalize: bool = True  # L2-normalize embeddings
    layer: str = "last"  # "last" = after final norm (before lm_head)


class EmbeddingExtractor:
    """Extract embeddings from AureliusTransformer via forward hook.

    Usage:
        extractor = EmbeddingExtractor(model, cfg)
        embeddings = extractor.encode(input_ids)  # (B, d_model)
    """

    def __init__(self, model: nn.Module, cfg: EmbeddingConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or EmbeddingConfig()
        self._hidden: torch.Tensor | None = None
        self._hook = None

    def _register_hook(self) -> None:
        """Hook into model.norm to capture its output."""

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self._hidden = output

        self._hook = self.model.norm.register_forward_hook(hook_fn)

    def _remove_hook(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract embeddings for a batch of token sequences.

        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) bool, True=valid. If None, all tokens valid.

        Returns:
            (B, d_model) embedding tensor.
        """
        self._register_hook()
        try:
            with torch.no_grad():
                self.model(input_ids)
            hidden = self._hidden  # (B, S, D)
        finally:
            self._remove_hook()

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        if self.cfg.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.cfg.pooling == "last":
            # Last valid token per sequence
            lengths = attention_mask.sum(dim=1) - 1
            emb = hidden[torch.arange(hidden.size(0)), lengths]
        elif self.cfg.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            hidden_masked = hidden * mask + (-1e9) * (1 - mask)
            emb = hidden_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling strategy: {self.cfg.pooling!r}")

        if self.cfg.normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    def __enter__(self) -> EmbeddingExtractor:
        self._register_hook()
        return self

    def __exit__(self, *args: object) -> None:
        self._remove_hook()


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, D) L2-normalized embeddings

    Returns:
        (N, N) similarity matrix where entry [i,j] = cos_sim(e_i, e_j)
    """
    return embeddings @ embeddings.T


def find_nearest_neighbors(
    query: torch.Tensor,
    corpus: torch.Tensor,
    top_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find top-k nearest neighbors in corpus for each query.

    Args:
        query: (Q, D) query embeddings (L2-normalized)
        corpus: (C, D) corpus embeddings (L2-normalized)
        top_k: number of neighbors to return

    Returns:
        (scores, indices) each of shape (Q, top_k)
        scores: cosine similarity values
        indices: indices into corpus
    """
    sim = query @ corpus.T  # (Q, C)
    scores, indices = torch.topk(sim, min(top_k, corpus.shape[0]), dim=1)
    return scores, indices


def deduplicate_by_similarity(
    embeddings: torch.Tensor,
    threshold: float = 0.95,
) -> list[int]:
    """Return indices of a deduplicated subset.

    Greedily keeps embeddings that are not too similar to already-kept ones.

    Args:
        embeddings: (N, D) embeddings (will be L2-normalized internally)
        threshold: cosine similarity above which two items are considered duplicates

    Returns:
        List of indices to keep (ordered, subset of 0..N-1).
    """
    normed = F.normalize(embeddings, dim=-1)
    kept: list[int] = []
    for i in range(len(normed)):
        if not kept:
            kept.append(i)
            continue
        kept_embs = normed[kept]
        sim = (kept_embs @ normed[i]).max().item()
        if sim < threshold:
            kept.append(i)
    return kept
