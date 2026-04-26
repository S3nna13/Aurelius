"""
src/model/megabyte_model.py

MEGABYTE hierarchical model: global model over patches + local model over bytes.
Two transformers cooperate: the global model predicts patch-level context,
the local model autoregressively generates bytes within each patch conditioned
on the global context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MegabyteConfig:
    patch_size: int = 8
    global_d_model: int = 256
    global_n_layers: int = 4
    global_n_heads: int = 4
    local_d_model: int = 128
    local_n_layers: int = 2
    local_n_heads: int = 4
    vocab_size: int = 256
    max_seq_len: int = 2048


def _make_causal_mask(S: int, device: torch.device) -> Tensor:
    """Build a causal boolean mask of shape (S, S).

    True means the position is visible (attend to), False means masked.
    """
    return torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))


class PatchEmbedding(nn.Module):
    """Embeds a byte sequence into patch-level vectors.

    Steps:
        1. Embed each byte token via nn.Embedding -> (B, T, local_d_model).
        2. Reshape into patches -> (B, n_patches, patch_size, local_d_model).
        3. Mean-pool over patch_size -> (B, n_patches, local_d_model).
        4. Linear projection -> (B, n_patches, global_d_model).

    Args:
        config: MegabyteConfig with vocab_size, patch_size, local/global dims.
    """

    def __init__(self, config: MegabyteConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.byte_embed = nn.Embedding(config.vocab_size, config.local_d_model)
        self.proj = nn.Linear(config.local_d_model, config.global_d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Embed byte tokens into patch vectors.

        Args:
            x: Integer tensor of shape (B, T).
               T must be divisible by patch_size.

        Returns:
            Float tensor of shape (B, n_patches, global_d_model).
        """
        B, T = x.shape
        assert T % self.patch_size == 0, (  # noqa: S101
            f"Sequence length {T} must be divisible by patch_size {self.patch_size}"
        )
        n_patches = T // self.patch_size

        h = self.byte_embed(x)  # (B, T, local_d_model)
        h = h.reshape(B, n_patches, self.patch_size, -1)  # (B, n_patches, P, local_d_model)
        h = h.mean(dim=2)  # (B, n_patches, local_d_model)
        return self.proj(h)  # (B, n_patches, global_d_model)


class _TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, MHA, and FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0  # noqa: S101
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        ff_dim = d_model * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward with pre-norm residual.

        Args:
            x: (B, S, d_model).
            mask: Optional (S, S) bool causal mask.

        Returns:
            (B, S, d_model).
        """
        B, S, _ = x.shape

        # Self-attention
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, nh, S, S)

        if mask is not None:
            attn_bias = torch.zeros(S, S, device=x.device, dtype=attn.dtype)
            attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
            attn = attn + attn_bias

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, nh, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        out = self.out_proj(out)
        x = x + out

        # FFN
        x = x + self.ff(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """Small transformer for use as global or local model.

    Args:
        n_layers: Number of transformer blocks.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        ff_mult: Feed-forward expansion factor (default 4).
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with causal masking.

        Args:
            x: (B, S, d_model).

        Returns:
            (B, S, d_model).
        """
        S = x.shape[1]
        device = x.device
        mask = _make_causal_mask(S, device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class MegabyteModel(nn.Module):
    """MEGABYTE-style hierarchical byte-level language model.

    Architecture:
        - Patch embedding: bytes -> patch vectors.
        - Global model (MiniTransformer): processes patch sequence.
        - Local model (MiniTransformer): generates bytes within each patch,
          conditioned on global context via additive bias.
        - Output head: projects local hidden states to byte logits.

    Args:
        config: MegabyteConfig.
    """

    def __init__(self, config: MegabyteConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        # Patch embedding: bytes -> global_d_model
        self.patch_embed = PatchEmbedding(config)

        # Global model operates on patch embeddings
        self.global_model = MiniTransformer(
            n_layers=config.global_n_layers,
            d_model=config.global_d_model,
            n_heads=config.global_n_heads,
        )

        # Project global context down to local_d_model for injection
        self.global_to_local = nn.Linear(config.global_d_model, config.local_d_model, bias=False)

        # Local byte embedding
        self.byte_embed = nn.Embedding(config.vocab_size, config.local_d_model)

        # Local model operates on byte sequence within each patch
        self.local_model = MiniTransformer(
            n_layers=config.local_n_layers,
            d_model=config.local_d_model,
            n_heads=config.local_n_heads,
        )

        # Output projection to vocabulary
        self.output_head = nn.Linear(config.local_d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Compute loss and logits for a byte sequence.

        The model predicts each byte given its predecessors and global
        patch context from previous patches.

        Args:
            input_ids: Integer tensor of shape (B, T).
                       T must be divisible by patch_size.

        Returns:
            (loss, logits) where:
                - loss is a scalar cross-entropy loss.
                - logits has shape (B, T, vocab_size).
        """
        B, T = input_ids.shape
        P = self.patch_size
        assert T % P == 0, f"T={T} must be divisible by patch_size={P}"  # noqa: S101
        n_patches = T // P

        # --- Global model ---
        patch_emb = self.patch_embed(input_ids)  # (B, n_patches, global_d_model)
        global_ctx = self.global_model(patch_emb)  # (B, n_patches, global_d_model)

        # Project global context to local dimension
        # (B, n_patches, local_d_model)
        local_ctx = self.global_to_local(global_ctx)

        # --- Local model (per-patch byte generation) ---
        # Embed all bytes
        byte_emb = self.byte_embed(input_ids)  # (B, T, local_d_model)

        # Reshape into patches: (B*n_patches, P, local_d_model)
        byte_emb_patches = byte_emb.reshape(B * n_patches, P, -1)

        # Expand global context as bias: (B*n_patches, 1, local_d_model)
        local_ctx_flat = local_ctx.reshape(B * n_patches, 1, -1)

        # Inject global context as additive bias to first position / all positions
        byte_emb_conditioned = byte_emb_patches + local_ctx_flat

        # Run local transformer
        local_out = self.local_model(byte_emb_conditioned)  # (B*n_patches, P, local_d_model)

        # Reshape back: (B, T, local_d_model)
        local_out = local_out.reshape(B, T, -1)

        # Project to logits
        logits = self.output_head(local_out)  # (B, T, vocab_size)

        # Compute next-token prediction loss (shift by 1)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab_size)
        shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)

        loss = F.cross_entropy(
            shift_logits.reshape(-1, self.config.vocab_size),
            shift_labels.reshape(-1),
        )

        return loss, logits
