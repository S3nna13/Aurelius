"""ALBERT-style weight reduction: factorized embeddings + cross-layer sharing.

Two independent techniques:
1. FactorizedEmbedding: vocab → embed_dim → d_model (saves params when embed_dim << d_model)
2. SharedLayerTransformer: n_layers blocks sharing one set of weights

Reference: Lan et al. 2019, arXiv:1909.11942
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from .config import AureliusConfig


@dataclass
class WeightSharingConfig:
    """Configuration for weight reduction techniques."""
    factorized_embed_dim: int = 128    # embedding bottleneck dimension
    share_layers: bool = False         # if True, all layers share weights
    n_shared_groups: int = 1           # 1=all layers share, 2=split in half, etc.
    # n_shared_groups=1: layers 0-23 all same weights
    # n_shared_groups=2: layers 0-11 share one set, 12-23 another
    # n_shared_groups=4: 4 groups of 6 layers each


class FactorizedEmbedding(nn.Module):
    """Two-stage embedding: vocab → embed_dim → d_model.

    Saves parameters when embed_dim << d_model.
    Parameter savings: vocab_size * (d_model - embed_dim) - embed_dim * d_model
    For vocab=128K, d_model=2048, embed_dim=128:
    Standard: 128000 * 2048 = 262M params
    Factorized: 128000 * 128 + 128 * 2048 = 16.6M params (15.7x reduction)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,     # bottleneck
        d_model: int,       # output dimension
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.projection = nn.Linear(embed_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S) token ids → (B, S, d_model)"""
        return self.projection(self.embedding(x))

    def parameter_savings(self, vocab_size: int, embed_dim: int, d_model: int) -> int:
        """Compute parameter savings vs standard embedding."""
        standard = vocab_size * d_model
        factorized = vocab_size * embed_dim + embed_dim * d_model
        return standard - factorized


def apply_factorized_embedding(
    model: nn.Module,
    embed_dim: int,
    config: AureliusConfig,
) -> nn.Module:
    """Replace model.embed with FactorizedEmbedding in-place.

    Also updates model.lm_head if it shares weights with embed.
    Note: After factorization, weight tying is broken (embed is now 2-stage).

    Returns modified model.
    """
    # Determine padding_idx from existing embedding if present
    old_embed = model.embed
    padding_idx = None
    if isinstance(old_embed, nn.Embedding):
        padding_idx = old_embed.padding_idx

    # Replace with factorized embedding
    model.embed = FactorizedEmbedding(
        vocab_size=config.vocab_size,
        embed_dim=embed_dim,
        d_model=config.d_model,
        padding_idx=padding_idx,
    )

    # Weight tying is now broken — lm_head keeps its own independent weights.
    # If lm_head was tied to the old embed, it now holds the old embedding weight
    # (a reference that still exists on the lm_head Linear). We leave it as-is
    # so the lm_head remains functional with independent weights.

    return model


def apply_cross_layer_sharing(
    model: nn.Module,
    n_shared_groups: int = 1,
) -> nn.Module:
    """Apply cross-layer weight sharing in-place.

    n_shared_groups=1: All layers share layer 0's weights.
    n_shared_groups=k: Divide n_layers into k groups, each group shares first layer's weights.

    The actual Python objects in model.layers are replaced with references to
    the "representative" layer for that group.

    Returns modified model.
    """
    layers = model.layers
    n_layers = len(layers)

    if n_shared_groups < 1 or n_shared_groups > n_layers:
        raise ValueError(
            f"n_shared_groups must be between 1 and n_layers ({n_layers}), "
            f"got {n_shared_groups}"
        )

    # Assign layers using interleaved (round-robin) grouping:
    # layer i belongs to group (i % n_shared_groups).
    # The representative for each group is the first layer assigned to it.
    # e.g., n_layers=4, n_groups=2: groups are {0,2} and {1,3}
    #   → layers[0] is rep for group 0, layers[1] is rep for group 1
    #   → new_layers = [layers[0], layers[1], layers[0], layers[1]]
    representatives = [layers[g] for g in range(n_shared_groups)]

    new_layers = []
    for i in range(n_layers):
        group_idx = i % n_shared_groups
        new_layers.append(representatives[group_idx])

    # Replace the ModuleList entries
    # nn.ModuleList doesn't support direct index assignment in a loop cleanly,
    # so we rebuild it.
    model.layers = nn.ModuleList(new_layers)

    return model


def _iter_all_params_with_duplicates(model: nn.Module):
    """Yield all parameters including duplicates from shared layers.

    PyTorch's model.modules() and model.parameters() deduplicate modules,
    so shared layers are only visited once. This helper iterates the layers
    list (including duplicates) explicitly to get the true conceptual total.
    """
    # Parameters from non-layer parts (unique, visited once)
    layer_ids: set[int] = set()
    if hasattr(model, "layers"):
        for layer in model.layers:
            layer_ids.add(id(layer))

    # Non-layer module params (deduplicated by PyTorch — that's fine here)
    for module in model.modules():
        if id(module) not in layer_ids:
            for p in module.parameters(recurse=False):
                yield p

    # Layer params: iterate the list directly (duplicates included)
    if hasattr(model, "layers"):
        for layer in model.layers:
            yield from layer.parameters()


def count_unique_parameters(model: nn.Module) -> dict[str, int]:
    """Count unique (non-shared) parameters.

    With cross-layer sharing, many layers point to the same Parameter objects.
    The "total" count treats each logical layer slot as contributing its own
    parameters (even if they share weights), giving the conceptual parameter
    budget. "unique" counts only distinct tensors by data pointer.

    Returns {"total_params": int, "unique_params": int, "shared_ratio": float}
    """
    total_params = sum(p.numel() for p in _iter_all_params_with_duplicates(model))

    seen_ptrs: set[int] = set()
    unique_params = 0
    for p in _iter_all_params_with_duplicates(model):
        ptr = p.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            unique_params += p.numel()

    shared_ratio = 1.0 - (unique_params / total_params) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "unique_params": unique_params,
        "shared_ratio": shared_ratio,
    }
