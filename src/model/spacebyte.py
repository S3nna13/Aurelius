"""SpaceByte: Towards Deleting Tokenization from Large Language Models.

Reference: Slagle, 2024 — arXiv:2404.14408
Section 2, Figure 1.

SpaceByte extends MegaByte with *dynamic* patching: patch boundaries are placed
at natural word boundaries (bytes preceding a space character 0x20), rather than
fixed-size blocks.

Variable names follow paper notation where possible:
    T          — total sequence length in bytes
    B          — batch size
    n_patches  — number of patches (variable; depends on content)
    d_l        — local model hidden dimension
    d_g        — global model hidden dimension
    patch_byte — the special byte that signals a patch boundary (default: 0x20 = space)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpaceByteConfig:
    """Hyper-parameters for SpaceByte (Section 2 notation)."""

    vocab_size: int = 256       # byte vocabulary {0 .. 255}
    patch_byte: int = 0x20      # byte value that triggers a new patch boundary
    d_local: int = 64           # d_l — local model hidden dim
    d_global: int = 128         # d_g — global model hidden dim
    n_local_layers: int = 1     # depth of local transformer L
    n_global_layers: int = 2    # depth of global transformer G
    n_heads_local: int = 2      # attention heads in L
    n_heads_global: int = 2     # attention heads in G
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Building blocks: causal Transformer (independent of megabyte.py)
# ---------------------------------------------------------------------------

class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (training path; no KV-cache)."""

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

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        attn_bias = torch.zeros(T, T, dtype=x.dtype, device=x.device)
        attn_bias = attn_bias.masked_fill(~mask, float("-inf"))

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = attn_weights + attn_bias
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)              # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    """Pre-norm block: LN → Attn → residual → LN → FFN → residual."""

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


# Named sub-modules (exported for testing)
class GlobalTransformer(_CausalTransformer):
    """Global model G: operates on patch-level representations (Section 2)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, n_layers, dropout)


class LocalTransformer(_CausalTransformer):
    """Local model L: per-patch byte-level autoregressive model (Section 2)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, n_layers, dropout)


# ---------------------------------------------------------------------------
# SpaceByte main model
# ---------------------------------------------------------------------------

class SpaceByteModel(nn.Module):
    """SpaceByte model (Section 2, arXiv:2404.14408).

    Key difference from MegaByte: patch boundaries are placed dynamically at
    bytes that equal `patch_byte` (default 0x20 = space), giving variable-length
    patches rather than fixed-size P-byte blocks.

    Forward pass summary (Section 2)
    ---------------------------------
    Given byte ids x ∈ {0..255}^(B, T):

    1.  Identify patch boundaries:
            boundary before byte t  iff  x[t] == patch_byte  (or t == 0)
        Produces patch start indices s_0=0, s_1, ..., s_{n-1}  (n_patches total).

    2.  Build global input (one vector per patch):
            For patch p spanning bytes [s_p .. s_{p+1}):
                patch_embed[:, p, :] = mean(byte_embed[:, s_p:s_{p+1}, :])
            → shape (B, n_patches, d_g)   [after projection d_l → d_g]

    3.  Global transformer:
            h_g = GlobalTransformer(patch_embed)   → (B, n_patches, d_g)

    4.  Local transformer for each patch p:
            global_ctx  = global_to_local(h_g[:, p, :])  → (B, 1, d_l)
            local_input = [global_ctx ; byte_embed[:, s_p : s_{p+1}-1, :]]
            local_out   = LocalTransformer(local_input)   → (B, patch_len_p, d_l)
            logits_p    = output_proj(local_out)          → (B, patch_len_p, 256)

    5.  Concatenate patch logits: (B, T, 256)
    """

    def __init__(self, config: SpaceByteConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpaceByteConfig()
        cfg = self.config

        d_l, d_g = cfg.d_local, cfg.d_global

        # Byte embedding table (vocab = 256)
        self.byte_embed = nn.Embedding(cfg.vocab_size, d_l)

        # Project mean-pooled patch bytes (d_l) to global dim (d_g)
        self.patch_proj = nn.Linear(d_l, d_g, bias=False)

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

        # Output projection → byte logits (256-way)
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
    # Patch boundary detection (Section 2)
    # ------------------------------------------------------------------

    def find_patch_boundaries(self, byte_ids: torch.LongTensor) -> List[int]:
        """Return list of patch-start indices for a single sequence.

        A new patch begins at position t when byte_ids[t] == patch_byte.
        Position 0 is always a patch start.

        Args:
            byte_ids: 1-D LongTensor of shape (T,) — a single byte sequence.

        Returns:
            Sorted list of patch start indices (always includes 0).

        Example:
            "hello world foo" (patch_byte=0x20=' ')
            space at positions 5 and 11 → patches start at [0, 5, 11].
        """
        if byte_ids.dim() != 1:
            raise ValueError(
                f"find_patch_boundaries expects a 1-D tensor, got shape {byte_ids.shape}"
            )
        patch_byte = self.config.patch_byte
        boundaries: List[int] = [0]
        ids = byte_ids.tolist()
        for t in range(1, len(ids)):
            if ids[t] == patch_byte:
                boundaries.append(t)
        return boundaries

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
            byte_ids: LongTensor of shape (B, T).
            targets:  optional LongTensor (B, T) for teacher-forced cross-entropy.

        Returns:
            logits  of shape (B, T, 256)   — or —
            (loss, logits) when targets is provided.

        Raises:
            NotImplementedError: if B > 1 sequences have differing patch structures.
        """
        cfg = self.config
        B, T = byte_ids.shape

        if T == 0:
            raise ValueError("byte_ids sequence length T must be > 0.")

        # ---- Determine patch boundaries (from first batch row) ----
        boundaries = self.find_patch_boundaries(byte_ids[0])  # List[int], sorted

        # For B > 1 ensure every row has identical patch structure
        if B > 1:
            for b in range(1, B):
                b_boundaries = self.find_patch_boundaries(byte_ids[b])
                if b_boundaries != boundaries:
                    raise NotImplementedError(
                        "SpaceByteModel: B>1 with differing patch boundaries is not "
                        "supported. Ensure all sequences in a batch have spaces at the "
                        "same positions, or use B=1."
                    )

        # n_patches and patch spans: patch p covers bytes [boundaries[p], boundaries[p+1])
        # Append sentinel T so the last patch is well-defined.
        patch_starts = boundaries                       # list of n_patches start indices
        patch_ends = boundaries[1:] + [T]              # list of n_patches end indices
        n_patches = len(patch_starts)

        # ---- Step 1: byte embeddings ----
        e = self.byte_embed(byte_ids)                  # (B, T, d_l)

        # ---- Step 2: build global input (one vector per patch via mean-pool) ----
        # patch_embed: (B, n_patches, d_l) → project → (B, n_patches, d_g)
        patch_vecs = torch.stack(
            [e[:, patch_starts[p]:patch_ends[p], :].mean(dim=1) for p in range(n_patches)],
            dim=1,
        )                                               # (B, n_patches, d_l)
        x_g = self.patch_proj(patch_vecs)              # (B, n_patches, d_g)

        # ---- Step 3: global transformer ----
        h_g = self.global_transformer(x_g)             # (B, n_patches, d_g)

        # ---- Step 4: local transformer per patch ----
        # For efficiency we run each patch sequentially (B=1 is the simple case;
        # B>1 with uniform boundaries is handled by the same batched ops).
        all_logits: List[torch.Tensor] = []

        for p in range(n_patches):
            s, end = patch_starts[p], patch_ends[p]
            patch_len = end - s                         # number of bytes in patch p

            # Global context for this patch: (B, 1, d_l)
            global_ctx = self.global_to_local(h_g[:, p, :]).unsqueeze(1)  # (B, 1, d_l)

            if patch_len == 1:
                # Single byte: local input = just the global context prefix
                local_in = global_ctx                   # (B, 1, d_l)
            else:
                # Teacher-forced: prepend global_ctx, drop the last byte of the patch
                tok_inp = e[:, s:end - 1, :]           # (B, patch_len-1, d_l)
                local_in = torch.cat([global_ctx, tok_inp], dim=1)  # (B, patch_len, d_l)

            local_out = self.local_transformer(local_in)    # (B, patch_len, d_l)
            logits_p = self.output_proj(local_out)          # (B, patch_len, 256)
            all_logits.append(logits_p)

        # ---- Step 5: concatenate ----
        logits = torch.cat(all_logits, dim=1)              # (B, T, 256)

        if logits.shape[1] != T:
            raise RuntimeError(
                f"Internal error: logits T-dim={logits.shape[1]} != expected T={T}."
            )

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, cfg.vocab_size),
                targets.reshape(B * T),
            )
            return loss, logits

        return logits
