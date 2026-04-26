"""Weight tying utilities: share parameters between embedding and output projection.

Weight tying (e.g., Press & Wolf 2016, arXiv:1608.05859) shares the token
embedding matrix with the final output projection, reducing parameter count
and often improving perplexity.

Utilities provided:
- tie_embedding_weights: point linear.weight at embedding.weight (true sharing)
- TiedEmbedding: single weight for both embed() and project()
- copy_shared_weights / cross_layer_weight_sharing: copy-based initialisation
- SharedLinear: wraps an externally supplied weight tensor
- LanguageModelWithTying: minimal demonstration model
- count_unique_parameters / count_parameter_bytes: accounting helpers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Primitive: tie two existing modules
# ---------------------------------------------------------------------------


def tie_embedding_weights(embedding: nn.Embedding, linear: nn.Linear) -> None:
    """Make *linear* share its weight tensor with *embedding*.

    After this call ``linear.weight is embedding.weight`` is True.
    Any gradient update through either module will be reflected in both.

    Args:
        embedding: Source embedding module; shape (vocab_size, d_model).
        linear: Target linear module; its weight is replaced with the
                embedding weight. Shape must be (vocab_size, d_model) before
                transposing for ``F.linear``.
    """
    linear.weight = embedding.weight


# ---------------------------------------------------------------------------
# TiedEmbedding: single parameter for embed + project
# ---------------------------------------------------------------------------


class TiedEmbedding(nn.Module):
    """Embedding + LM-head projection sharing a single weight matrix.

    The weight has shape ``(vocab_size, d_model)`` (same layout as
    ``nn.Embedding``).  Projection uses ``weight.T``, equivalent to the
    standard un-tied ``nn.Linear(d_model, vocab_size, bias=False)``.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self._weight, std=d_model**-0.5)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the shared parameter."""
        return self._weight

    def embed(self, token_ids: Tensor) -> Tensor:
        """Token lookup.

        Args:
            token_ids: LongTensor of shape (B, T).

        Returns:
            FloatTensor of shape (B, T, d_model).
        """
        return F.embedding(token_ids, self._weight)

    def project(self, hidden: Tensor) -> Tensor:
        """Output projection: ``hidden @ weight.T``.

        Args:
            hidden: FloatTensor of shape (B, T, d_model).

        Returns:
            FloatTensor of shape (B, T, vocab_size).
        """
        return hidden @ self._weight.T

    def forward(self, token_ids: Tensor) -> Tensor:
        """Convenience: embed token ids then project back to vocab.

        Not typically used directly; prefer embed() and project() separately.
        """
        return self.project(self.embed(token_ids))


# ---------------------------------------------------------------------------
# Copy-based weight sharing (not true tying)
# ---------------------------------------------------------------------------


def copy_shared_weights(source: nn.Module, target: nn.Module) -> None:
    """Copy parameter values from *source* to *target* (not tied).

    After this call the two modules have identical weights but **do not**
    share storage — modifying one will not affect the other.

    Args:
        source: Module whose parameters are copied.
        target: Module whose parameters are overwritten in-place.
    """
    source_sd = source.state_dict()
    target.load_state_dict(source_sd, strict=True)


def cross_layer_weight_sharing(
    layers: list[nn.Module],
    share_every_n: int = 2,
) -> None:
    """Copy weights from representative layers into subsequent layers.

    For every index ``i >= share_every_n`` the layer is initialised with
    the weights of ``layers[i % share_every_n]``.  This is a *copy*, not
    true parameter sharing; both sets of parameters continue to be updated
    independently during training.

    Example with ``share_every_n=2`` and 6 layers:
        - layers[0] → representative for group 0
        - layers[1] → representative for group 1
        - layers[2] ← copy of layers[0]
        - layers[3] ← copy of layers[1]
        - layers[4] ← copy of layers[0]
        - layers[5] ← copy of layers[1]

    Args:
        layers: List of ``nn.Module``s to update in-place.
        share_every_n: Group size; layers 0…n-1 are the representatives.
    """
    if share_every_n < 1:
        raise ValueError(f"share_every_n must be >= 1, got {share_every_n}")
    for i in range(share_every_n, len(layers)):
        copy_shared_weights(layers[i % share_every_n], layers[i])


# ---------------------------------------------------------------------------
# SharedLinear: linear layer that borrows its weight from outside
# ---------------------------------------------------------------------------


class SharedLinear(nn.Module):
    """Linear layer whose weight is supplied externally (not owned here).

    Useful when you want a second module to *use* a weight that is
    registered as a parameter of another module.  The weight is stored as a
    plain attribute (not re-wrapped as ``nn.Parameter``) so that it appears
    in ``named_parameters()`` of the *owning* module only.

    Args:
        weight: Tensor of shape (out_features, in_features) — the same
                layout as ``nn.Linear.weight``.
        bias: If ``True`` a learnable bias is added; default ``False``.
    """

    def __init__(self, weight: Tensor, bias: bool = False) -> None:
        super().__init__()
        # Store as a plain attribute so grad flows through but the param
        # is not re-registered here.
        self.weight = weight
        if bias:
            out_features = weight.shape[0]
            self.bias: Tensor | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``F.linear(x, self.weight, self.bias)``."""
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Parameter accounting helpers
# ---------------------------------------------------------------------------


def count_unique_parameters(model: nn.Module) -> int:
    """Count parameters by storage identity to avoid double-counting tied weights.

    Returns:
        Number of **unique** ``nn.Parameter`` tensors (by ``data_ptr``).
    """
    seen: set[int] = set()
    count = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            count += 1
    return count


def count_parameter_bytes(model: nn.Module, deduplicate: bool = True) -> int:
    """Total bytes consumed by model parameters.

    Args:
        model: The module to inspect.
        deduplicate: If ``True`` (default), count shared tensors only once.
                     If ``False``, sum all parameter tensors (may double-count).

    Returns:
        Total bytes as an ``int``.
    """
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if deduplicate:
            if ptr in seen:
                continue
            seen.add(ptr)
        total += p.numel() * p.element_size()
    return total


# ---------------------------------------------------------------------------
# Demonstration model
# ---------------------------------------------------------------------------


class LanguageModelWithTying(nn.Module):
    """Minimal language model demonstrating input/output weight tying.

    Architecture:
        token_ids → TiedEmbedding.embed → mean-pool over T → expand back → project → logits

    The embed and project operations share the same ``TiedEmbedding`` weight.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Hidden (embedding) dimension.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.tied = TiedEmbedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed → mean-pool → broadcast back → project to vocab.

        Args:
            token_ids: LongTensor of shape (B, T).

        Returns:
            FloatTensor of shape (B, T, vocab_size).
        """
        # (B, T, d_model)
        x = self.tied.embed(token_ids)
        # Simple mean pooling over sequence dimension, broadcast back
        pooled = x.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        pooled = pooled.expand_as(x)  # (B, T, d_model)
        # (B, T, vocab_size)
        logits = self.tied.project(pooled)
        return logits
