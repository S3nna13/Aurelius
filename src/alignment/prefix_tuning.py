"""Prefix Tuning: Layer-wise soft prefix tuning (Li & Liang 2021).

Prepends learned continuous prefix vectors to keys and values at each
attention layer. Supports MLP reparameterization for training stability.

Reference: Li & Liang 2021, arXiv:2101.00190
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PrefixConfig:
    prefix_length: int = 10        # number of prefix tokens per layer
    n_layers: int = 24
    n_kv_heads: int = 8
    head_dim: int = 128
    dropout: float = 0.1
    use_mlp_reparameterization: bool = True  # use MLP for stability during training


class PrefixTuning(nn.Module):
    """Learned prefix K/V vectors prepended to each attention layer's keys and values.

    Architecture (with MLP reparameterization):
    - prefix_embedding: (prefix_length, embed_dim) where embed_dim = n_kv_heads * head_dim
    - MLP: embed_dim -> embed_dim * 2 -> n_layers * 2 * n_kv_heads * head_dim
    - At each layer: extract slice, reshape to (prefix_length, n_kv_heads, head_dim)

    Without reparameterization:
    - Direct parameters: (n_layers, 2, prefix_length, n_kv_heads, head_dim)
      (the 2 is for K and V)
    """

    def __init__(self, cfg: PrefixConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout(p=cfg.dropout)

        if cfg.use_mlp_reparameterization:
            # embed_dim is the per-head-group size
            embed_dim = cfg.n_kv_heads * cfg.head_dim
            intermediate_dim = embed_dim * 2
            final_dim = cfg.n_layers * 2 * cfg.n_kv_heads * cfg.head_dim

            # Embedding table: (prefix_length, embed_dim)
            self.prefix_embedding = nn.Embedding(cfg.prefix_length, embed_dim)

            # MLP reparameterization: embed_dim -> intermediate_dim -> final_dim
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, intermediate_dim),
                nn.Tanh(),
                nn.Linear(intermediate_dim, final_dim),
            )
        else:
            # Direct parameters: (n_layers, 2, prefix_length, n_kv_heads, head_dim)
            self.prefix_params = nn.Parameter(
                torch.randn(cfg.n_layers, 2, cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
                * 0.02
            )

    def _get_prefix_output(self) -> Tensor:
        """Run embedding + optional MLP to get all prefix K/V vectors.

        Returns:
            Tensor of shape (prefix_length, n_layers * 2 * n_kv_heads * head_dim)
            when using MLP, or raw params reshaped accordingly when not.
        """
        if self.cfg.use_mlp_reparameterization:
            # prefix_embedding.weight: (prefix_length, embed_dim)
            embedded = self.prefix_embedding.weight  # (P, embed_dim)
            embedded = self.dropout(embedded)
            # MLP output: (P, n_layers * 2 * n_kv_heads * head_dim)
            return self.mlp(embedded)
        else:
            # Direct params: (n_layers, 2, P, n_kv_heads, head_dim)
            # Apply dropout and reshape to (P, n_layers * 2 * n_kv_heads * head_dim)
            params = self.dropout(self.prefix_params)
            # Permute to (P, n_layers, 2, n_kv_heads, head_dim) then flatten last dims
            params = params.permute(2, 0, 1, 3, 4).contiguous()
            P = self.cfg.prefix_length
            return params.view(P, -1)

    def get_prefix_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Get prefix K and V vectors for a specific layer.

        Args:
            layer_idx: Which transformer layer to get prefixes for.

        Returns:
            (prefix_k, prefix_v) each of shape (prefix_length, n_kv_heads, head_dim)
        """
        cfg = self.cfg
        kv_size = cfg.n_kv_heads * cfg.head_dim  # size per K or V slice

        # Get full output: (prefix_length, n_layers * 2 * n_kv_heads * head_dim)
        full = self._get_prefix_output()  # (P, n_layers * 2 * kv_size)

        # Each layer occupies a contiguous block of size 2 * kv_size
        layer_offset = layer_idx * 2 * kv_size
        k_start = layer_offset
        k_end = layer_offset + kv_size
        v_start = k_end
        v_end = v_start + kv_size

        # Slice and reshape to (prefix_length, n_kv_heads, head_dim)
        prefix_k = full[:, k_start:k_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
        prefix_v = full[:, v_start:v_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)

        return prefix_k, prefix_v

    def get_all_prefix_kvs(self) -> list[tuple[Tensor, Tensor]]:
        """Get prefix K/V vectors for all layers.

        Returns:
            List of (prefix_k, prefix_v) tuples, one per layer.
            Each tensor has shape (prefix_length, n_kv_heads, head_dim).
        """
        cfg = self.cfg
        kv_size = cfg.n_kv_heads * cfg.head_dim

        # Compute once and slice per layer
        full = self._get_prefix_output()  # (P, n_layers * 2 * kv_size)

        result = []
        for layer_idx in range(cfg.n_layers):
            layer_offset = layer_idx * 2 * kv_size
            k_start = layer_offset
            k_end = layer_offset + kv_size
            v_start = k_end
            v_end = v_start + kv_size

            prefix_k = full[:, k_start:k_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
            prefix_v = full[:, v_start:v_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
            result.append((prefix_k, prefix_v))

        return result


def apply_prefix_to_attention(
    prefix_tuning: PrefixTuning,
    k: Tensor,   # (B, S, n_kv_heads, head_dim) — original keys
    v: Tensor,   # (B, S, n_kv_heads, head_dim) — original values
    layer_idx: int,
) -> tuple[Tensor, Tensor]:
    """Prepend prefix K/V vectors to the full k, v tensors.

    Args:
        prefix_tuning: PrefixTuning module with learned prefix parameters.
        k: Original keys of shape (B, S, n_kv_heads, head_dim).
        v: Original values of shape (B, S, n_kv_heads, head_dim).
        layer_idx: Which layer's prefix to use.

    Returns:
        Tuple of (new_k, new_v) each of shape (B, prefix_length + S, n_kv_heads, head_dim).
    """
    B = k.shape[0]

    # Get prefix: (prefix_length, n_kv_heads, head_dim)
    prefix_k, prefix_v = prefix_tuning.get_prefix_kv(layer_idx)

    # Expand to batch dim: (B, prefix_length, n_kv_heads, head_dim)
    prefix_k = prefix_k.unsqueeze(0).expand(B, -1, -1, -1)
    prefix_v = prefix_v.unsqueeze(0).expand(B, -1, -1, -1)

    # Concatenate along sequence dimension
    new_k = torch.cat([prefix_k, k], dim=1)
    new_v = torch.cat([prefix_v, v], dim=1)

    return new_k, new_v


class PrefixTuningTrainer:
    """Freezes backbone model, trains only PrefixTuning parameters."""

    def __init__(
        self,
        model: nn.Module,
        prefix_tuning: PrefixTuning,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.prefix_tuning = prefix_tuning
        self.optimizer = optimizer

    def freeze_backbone(self) -> int:
        """Freeze all model (backbone) parameters.

        Returns:
            Count of frozen parameters (number of tensors, not elements).
        """
        count = 0
        for p in self.model.parameters():
            p.requires_grad = False
            count += 1
        return count

    def unfreeze_backbone(self) -> None:
        """Restore requires_grad=True to all backbone model parameters."""
        for p in self.model.parameters():
            p.requires_grad = True

    def trainable_params(self) -> list[nn.Parameter]:
        """Return only the prefix_tuning parameters (those with requires_grad=True)."""
        return list(self.prefix_tuning.parameters())

    def param_count(self) -> dict[str, int]:
        """Count parameters across backbone + prefix.

        Returns:
            {"total": int, "trainable": int, "frozen": int}
        """
        all_params = list(self.model.parameters()) + list(self.prefix_tuning.parameters())
        total = sum(p.numel() for p in all_params)
        trainable = sum(p.numel() for p in all_params if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
