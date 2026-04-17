"""MegaByte: Predicting Million-byte Sequences with Multiscale Transformers.

Reference: Yu et al., Meta AI 2023 — arXiv:2305.07185
Section 3, Figure 2.

Architecture:
    Global model (G): operates on patches of P bytes, predicts next patch repr.
    Local model (L): within each patch, autoregressively predicts each byte
                     conditioned on (global context, previous bytes in patch).

Variable names follow paper notation where possible:
    P  — patch size (bytes per patch)
    T  — total sequence length  (must be divisible by P)
    n  — number of patches  (T // P)
    B  — batch size
    d_g — global model dimension
    d_l — local model dimension
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MegaByteConfig:
    """Hyper-parameters for MegaByte (Section 3 / Table 1 notation)."""

    vocab_size: int = 256          # byte vocabulary {0 .. 255}
    patch_size: int = 4            # P — bytes per patch
    d_local: int = 128             # d_l — local model hidden dim
    d_global: int = 256            # d_g — global model hidden dim
    n_local_layers: int = 2        # depth of local transformer L
    n_global_layers: int = 4       # depth of global transformer G
    n_heads_local: int = 4         # attention heads in L
    n_heads_global: int = 4        # attention heads in G
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Shared building block: a single causal Transformer
# ---------------------------------------------------------------------------

class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no KV-cache; training-only path)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                               # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)                  # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        attn_bias = torch.zeros(T, T, dtype=x.dtype, device=x.device)
        attn_bias = attn_bias.masked_fill(~mask, float("-inf"))

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = attn_weights + attn_bias            # broadcast over B, heads
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)                # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    """Pre-norm block: LayerNorm → Attention → residual → LayerNorm → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _CausalTransformer(nn.Module):
    """Stack of causal transformer blocks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# Named sub-modules (exported for testing convenience)
# ---------------------------------------------------------------------------

class GlobalTransformer(_CausalTransformer):
    """Global model G: operates on patch-level representations (Section 3.2)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, n_layers, dropout)


class LocalTransformer(_CausalTransformer):
    """Local model L: per-patch byte-level autoregressive model (Section 3.3)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, n_layers, dropout)


# ---------------------------------------------------------------------------
# MegaByte main model
# ---------------------------------------------------------------------------

class MegaByteModel(nn.Module):
    """MegaByte model (Section 3, Figure 2).

    Forward pass summary
    --------------------
    Given byte ids x ∈ {0..255}^(B, T)  (T must be divisible by P):

    1.  e  = byte_embed(x)                         # (B, T, d_l)
    2.  For each patch p:
            patch_flat_p = e[:, p*P:(p+1)*P, :].reshape(B, P*d_l)
            x_g[:, p, :] = patch_proj(patch_flat_p)  # (B, n, d_g)
    3.  h_g = GlobalTransformer(x_g)               # (B, n, d_g)
    4.  For each patch p:
            prefix_p  = global_to_local(h_g[:, p, :])   # (B, 1, d_l)
            tok_inp_p = e[:, p*P : (p+1)*P - 1, :]      # (B, P-1, d_l)
            local_in  = cat([prefix_p, tok_inp_p], dim=1) # (B, P, d_l)
            local_out[:, p*P:(p+1)*P, :] = LocalTransformer(local_in)
    5.  logits = output_proj(local_out)             # (B, T, 256)
    """

    def __init__(self, config: MegaByteConfig | None = None) -> None:
        super().__init__()
        self.config = config or MegaByteConfig()
        cfg = self.config
        P, d_l, d_g = cfg.patch_size, cfg.d_local, cfg.d_global

        # Byte embedding table (vocab = 256)
        self.byte_embed = nn.Embedding(cfg.vocab_size, d_l)

        # Patch projection: flattened P bytes × d_l → d_g  (global input)
        self.patch_proj = nn.Linear(P * d_l, d_g, bias=False)

        # Global transformer G
        self.global_transformer = GlobalTransformer(
            d_model=d_g,
            n_heads=cfg.n_heads_global,
            n_layers=cfg.n_global_layers,
            dropout=cfg.dropout,
        )

        # Project global context h_g[:,p,:] → d_l for local prefix token
        self.global_to_local = nn.Linear(d_g, d_l, bias=False)

        # Local transformer L
        self.local_transformer = LocalTransformer(
            d_model=d_l,
            n_heads=cfg.n_heads_local,
            n_layers=cfg.n_local_layers,
            dropout=cfg.dropout,
        )

        # Output projection → byte logits
        self.output_proj = nn.Linear(d_l, cfg.vocab_size, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        byte_ids: torch.LongTensor,
        targets: torch.LongTensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            byte_ids: LongTensor of shape (B, T).  T must be divisible by P.
            targets:  optional LongTensor (B, T) for teacher-forced CE loss.

        Returns:
            logits of shape (B, T, 256)  — or  (loss, logits) when targets given.
        """
        cfg = self.config
        P = cfg.patch_size
        B, T = byte_ids.shape

        if T % P != 0:
            raise ValueError(
                f"Sequence length T={T} must be divisible by patch_size P={P}."
            )

        n = T // P  # number of patches

        # ---- Step 1: byte embeddings ----
        e = self.byte_embed(byte_ids)                  # (B, T, d_l)

        # ---- Step 2: build global input (one vector per patch) ----
        # Reshape: (B, n, P, d_l) → (B, n, P*d_l)
        e_patches = e.view(B, n, P, cfg.d_local)       # (B, n, P, d_l)
        e_flat = e_patches.reshape(B, n, P * cfg.d_local)  # (B, n, P*d_l)
        x_g = self.patch_proj(e_flat)                  # (B, n, d_g)

        # ---- Step 3: global transformer ----
        h_g = self.global_transformer(x_g)             # (B, n, d_g)

        # ---- Step 4: local transformer (vectorised over all patches) ----
        # prefix tokens: project global output for each patch → d_l
        prefix = self.global_to_local(h_g)             # (B, n, d_l)
        prefix = prefix.unsqueeze(2)                   # (B, n, 1, d_l)

        # teacher-forced byte inputs: for patch p take bytes [p*P .. (p+1)*P-1)
        # i.e. the first P-1 bytes of each patch (shift left by 1 for local AR)
        tok_inp = e_patches[:, :, :-1, :]              # (B, n, P-1, d_l)

        # local input: [prefix | first P-1 bytes] → length P per patch
        local_in = torch.cat([prefix, tok_inp], dim=2) # (B, n, P, d_l)

        # Merge batch and patch dims so LocalTransformer sees (B*n, P, d_l)
        local_in_flat = local_in.reshape(B * n, P, cfg.d_local)
        local_out_flat = self.local_transformer(local_in_flat)  # (B*n, P, d_l)

        # Restore shape: (B, T, d_l)
        local_out = local_out_flat.view(B, T, cfg.d_local)

        # ---- Step 5: output projection ----
        logits = self.output_proj(local_out)           # (B, T, 256)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, cfg.vocab_size),
                targets.reshape(B * T),
            )
            return loss, logits

        return logits
