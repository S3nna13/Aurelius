"""Semantic Embedding & Similarity utilities for AureliusTransformer.

Provides pooling strategies, embedding extraction, similarity metrics,
and a high-level SemanticSimilarityModel class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimilarityConfig:
    """Configuration for semantic similarity computation.

    Attributes:
        pooling: Hidden-state pooling strategy — "mean" | "last" | "cls".
        normalize: Whether to L2-normalize embeddings before similarity.
        similarity_metric: Distance/similarity metric — "cosine" | "dot" | "euclidean".
    """

    pooling: str = "mean"  # "mean" | "last" | "cls"
    normalize: bool = True
    similarity_metric: str = "cosine"  # "cosine" | "dot" | "euclidean"


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def pool_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pooling: str,
) -> torch.Tensor:
    """Pool (B, T, d) hidden states down to (B, d).

    Args:
        hidden: Tensor of shape (B, T, d).
        attention_mask: Bool/float tensor of shape (B, T). True / 1 = valid token.
            If None, all positions are treated as valid.
        pooling: One of "mean", "last", "cls".

    Returns:
        Tensor of shape (B, d).
    """
    B, T, d = hidden.shape

    if attention_mask is None:
        attention_mask = torch.ones(B, T, dtype=torch.bool, device=hidden.device)

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif pooling == "last":
        # Index of the last valid token per sequence
        lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
        lengths = lengths.clamp(min=0)
        emb = hidden[torch.arange(B, device=hidden.device), lengths]  # (B, d)
    elif pooling == "cls":
        # Use the first token (position 0) as the [CLS]-style representation
        emb = hidden[:, 0, :]  # (B, d)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling!r}. Choose 'mean', 'last', or 'cls'.")

    return emb


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def get_embeddings(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: SimilarityConfig | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the model backbone and extract pooled (optionally normalized) embeddings.

    Uses a forward hook on ``model.norm`` to capture the final hidden states
    before the language-model head, then pools them according to ``config.pooling``.

    Args:
        model: AureliusTransformer (or any model with a ``norm`` submodule).
        input_ids: (B, S) integer token ids.
        config: SimilarityConfig controlling pooling and normalization.
        attention_mask: (B, S) bool mask; True = valid. Defaults to all-ones.

    Returns:
        (B, d_model) embedding tensor.
    """
    if config is None:
        config = SimilarityConfig()

    hidden_states: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        hidden_states.append(output)

    hook = model.norm.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    hidden = hidden_states[0]  # (B, S, d_model)
    emb = pool_hidden_states(hidden, attention_mask, config.pooling)

    if config.normalize:
        emb = F.normalize(emb, dim=-1)

    return emb


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between corresponding rows of a and b.

    Args:
        a: (B, d) tensor.
        b: (B, d) tensor.

    Returns:
        (B,) tensor of cosine similarities in [-1, 1].
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(dim=-1)


def dot_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise dot product between corresponding rows of a and b.

    Args:
        a: (B, d) tensor.
        b: (B, d) tensor.

    Returns:
        (B,) tensor of dot products.
    """
    return (a * b).sum(dim=-1)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L2 (Euclidean) distance between corresponding rows of a and b.

    Args:
        a: (B, d) tensor.
        b: (B, d) tensor.

    Returns:
        (B,) tensor of non-negative distances.
    """
    return (a - b).norm(dim=-1)


def compute_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Dispatch to the appropriate similarity / distance function.

    Args:
        a: (B, d) tensor.
        b: (B, d) tensor.
        metric: One of "cosine", "dot", "euclidean".

    Returns:
        (B,) tensor.
    """
    if metric == "cosine":
        return cosine_similarity(a, b)
    elif metric == "dot":
        return dot_similarity(a, b)
    elif metric == "euclidean":
        return euclidean_distance(a, b)
    else:
        raise ValueError(
            f"Unknown similarity metric: {metric!r}. Choose 'cosine', 'dot', or 'euclidean'."
        )


# ---------------------------------------------------------------------------
# High-level class
# ---------------------------------------------------------------------------


class SemanticSimilarityModel:
    """High-level semantic similarity interface built on AureliusTransformer.

    Usage::

        sim_model = SemanticSimilarityModel(backbone, SimilarityConfig())
        embs = sim_model.embed(input_ids)              # (B, d_model)
        scores = sim_model.similarity(ids_a, ids_b)    # (B,)

    Args:
        backbone: AureliusTransformer (must have a ``norm`` attribute).
        config: SimilarityConfig controlling pooling, normalization, and metric.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: SimilarityConfig | None = None,
    ) -> None:
        self.backbone = backbone
        self.config = config or SimilarityConfig()

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed a batch of token sequences.

        Always returns L2-normalized embeddings (ignores ``config.normalize``
        setting to guarantee unit-norm output from this method).

        Args:
            input_ids: (B, S) token ids.
            attention_mask: Optional (B, S) bool mask.

        Returns:
            (B, d_model) unit-norm embedding tensor.
        """
        # Force normalize=True for this method's output contract
        embed_cfg = SimilarityConfig(
            pooling=self.config.pooling,
            normalize=True,
            similarity_metric=self.config.similarity_metric,
        )
        return get_embeddings(self.backbone, input_ids, embed_cfg, attention_mask)

    def similarity(
        self,
        ids_a: torch.Tensor,
        ids_b: torch.Tensor,
        attention_mask_a: torch.Tensor | None = None,
        attention_mask_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pairwise similarity between two batches of sequences.

        Args:
            ids_a: (B, S) token ids for the first batch.
            ids_b: (B, S) token ids for the second batch.
            attention_mask_a: Optional mask for ids_a.
            attention_mask_b: Optional mask for ids_b.

        Returns:
            (B,) scalar similarity per pair.
        """
        emb_a = get_embeddings(self.backbone, ids_a, self.config, attention_mask_a)
        emb_b = get_embeddings(self.backbone, ids_b, self.config, attention_mask_b)
        return compute_similarity(emb_a, emb_b, self.config.similarity_metric)
