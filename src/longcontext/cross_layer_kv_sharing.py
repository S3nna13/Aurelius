"""Cross-Layer KV Sharing (MLKV / LCKV style).

Halves the KV-cache footprint by having every N-th layer own KV projections
while the intervening layers borrow them. With share_every_n=2 (default),
odd layers reuse the KV from the preceding even layer; only the query
projection differs per layer.

This mirrors the observation in MLKV / GQA-across-layers / LCKV that
adjacent transformer layers produce very similar key-value representations,
so computing separate KV projections wastes memory without much benefit.

Cache-reduction formula: 1 - 1/share_every_n
    share_every_n=2 -> 50 % reduction
    share_every_n=3 -> 33 % reduction

NOTE: this module is self-contained and MUST NOT import from ``src.model``
(see the hermetic-import test in the test suite).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CrossLayerKVConfig:
    """Hyper-parameters for a CrossLayerKVStack.

    Attributes:
        n_layers:       Total number of attention layers in the stack.
        d_model:        Residual stream width.
        n_heads:        Number of query heads per layer.
        n_kv_heads:     Number of key/value heads on owner layers (supports
                        grouped-query attention; must divide n_heads).
        head_dim:       Per-head dimension; n_heads * head_dim must equal
                        d_model.
        share_every_n:  Every share_every_n-th layer (0, n, 2n, …) owns KV
                        projections; the remaining layers borrow from the most
                        recent owner.
    """

    n_layers: int = 24
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    share_every_n: int = 2

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.n_kv_heads < 1:
            raise ValueError(f"n_kv_heads must be >= 1, got {self.n_kv_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.share_every_n < 1:
            raise ValueError(f"share_every_n must be >= 1, got {self.share_every_n}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim must equal d_model: "
                f"{self.n_heads} * {self.head_dim} = {self.n_heads * self.head_dim} "
                f"!= {self.d_model}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads must be divisible by n_kv_heads: {self.n_heads} % {self.n_kv_heads} != 0"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Single attention layer
# ──────────────────────────────────────────────────────────────────────────────


class CrossLayerKVAttention(nn.Module):
    """Single attention layer that either owns or borrows KV projections.

    KV owners (is_kv_owner=True) have ``k_proj`` and ``v_proj`` and produce
    a fresh ``kv_state`` on each forward pass.

    KV borrowers (is_kv_owner=False) have *no* k_proj / v_proj; they receive
    the owner's ``kv_cache`` and return it unchanged.

    Both variants have a ``q_proj`` and an ``out_proj``.

    Args:
        config:      Stack-wide hyper-parameters.
        layer_idx:   Index within the stack (informational; stored as attribute).
        is_kv_owner: Whether this layer owns KV projections.
    """

    def __init__(
        self,
        config: CrossLayerKVConfig,
        layer_idx: int,
        is_kv_owner: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_kv_owner = is_kv_owner

        d = config.d_model
        nh = config.n_heads
        nkv = config.n_kv_heads
        hd = config.head_dim
        kv_dim = nkv * hd

        # Every layer has its own query projection and output projection.
        self.q_proj = nn.Linear(d, nh * hd, bias=False)
        self.out_proj = nn.Linear(nh * hd, d, bias=False)

        # Only owner layers hold KV projections.
        if is_kv_owner:
            self.k_proj: nn.Linear | None = nn.Linear(d, kv_dim, bias=False)
            self.v_proj: nn.Linear | None = nn.Linear(d, kv_dim, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None

        self._scale = 1.0 / math.sqrt(hd)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: Tensor,
        kv_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one cross-layer-KV attention layer.

        Args:
            x:        Input tensor of shape [B, T, d_model].
            kv_cache: For borrower layers: KV state from the owner, shape
                      [B, n_kv_heads, T, 2*head_dim] (k and v packed along
                      last dim).  Ignored / not used by owner layers.

        Returns:
            (output, kv_state) where
                output   : [B, T, d_model]
                kv_state : [B, n_kv_heads, T, 2*head_dim]
                           Owners produce a freshly computed kv_state.
                           Borrowers return the provided kv_cache unchanged.
        """
        B, T, _ = x.shape
        cfg = self.config
        nh = cfg.n_heads
        nkv = cfg.n_kv_heads
        hd = cfg.head_dim
        groups = nh // nkv  # query groups per KV head

        # --- Query projection ------------------------------------------
        # [B, T, nh*hd] -> [B, nh, T, hd]
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)

        # --- Key / Value -----------------------------------------------
        if self.is_kv_owner:
            # Compute fresh k, v and pack into kv_state.
            k = self.k_proj(x).view(B, T, nkv, hd).transpose(1, 2)  # [B, nkv, T, hd]
            v = self.v_proj(x).view(B, T, nkv, hd).transpose(1, 2)  # [B, nkv, T, hd]
            kv_state = torch.cat([k, v], dim=-1)  # [B, nkv, T, 2*hd]
        else:
            # Borrow the owner's KV state.
            if kv_cache is None:
                raise ValueError(
                    f"CrossLayerKVAttention (layer {self.layer_idx}, borrower): "
                    "kv_cache must be provided for non-owner layers"
                )
            kv_state = kv_cache
            k, v = kv_state.split(hd, dim=-1)  # each [B, nkv, T_past, hd]

        # --- Expand KV heads to match query heads (GQA) ----------------
        if groups > 1:
            # [B, nkv, T, hd] -> [B, nh, T, hd]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        T_kv = k.shape[2]

        # --- Scaled dot-product attention (causal) ----------------------
        # [B, nh, T, T_kv]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        # Causal mask: query position i can only attend to key positions <= i.
        mask = torch.ones(T, T_kv, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)  # [B, nh, T, hd]

        # --- Output projection ------------------------------------------
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, nh * hd)
        output = self.out_proj(attn_out)  # [B, T, d_model]

        return output, kv_state


# ──────────────────────────────────────────────────────────────────────────────
# Stack
# ──────────────────────────────────────────────────────────────────────────────


class CrossLayerKVStack(nn.Module):
    """Stack of N layers with cross-layer KV sharing.

    Layers at indices 0, share_every_n, 2*share_every_n, … own their KV
    projections; all other layers borrow from the most recently computed
    owner layer.

    Args:
        config: Stack-wide hyper-parameters.
    """

    def __init__(self, config: CrossLayerKVConfig) -> None:
        super().__init__()
        self.config = config

        layers = []
        for i in range(config.n_layers):
            is_owner = i % config.share_every_n == 0
            layers.append(CrossLayerKVAttention(config, layer_idx=i, is_kv_owner=is_owner))
        self.layers = nn.ModuleList(layers)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: Tensor) -> Tensor:
        """Pass x through the entire stack.

        Owner layers compute fresh KV and store them; each borrower in the
        same group immediately receives the owner's kv_state.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        current_kv: Tensor | None = None

        for layer in self.layers:
            if layer.is_kv_owner:
                x, current_kv = layer(x, kv_cache=None)
            else:
                x, current_kv = layer(x, kv_cache=current_kv)

        return x

    # ------------------------------------------------------------------ #
    # Introspection helpers                                                #
    # ------------------------------------------------------------------ #

    def kv_cache_size_ratio(self) -> float:
        """Fraction of layers that own KV projections.

        Returns:
            1 / share_every_n  (e.g. 0.5 for share_every_n=2).
        """
        return 1.0 / self.config.share_every_n

    def parameter_count(self) -> dict[str, int]:
        """Break parameter counts into semantic groups.

        Returns:
            dict with keys "total", "kv_params", "q_params", "other_params".
        """
        total = 0
        kv_params = 0
        q_params = 0
        other_params = 0

        for layer in self.layers:
            for name, p in layer.named_parameters():
                # name is e.g. "q_proj.weight", "k_proj.weight", "out_proj.weight"
                n = p.numel()
                total += n
                if name in ("k_proj.weight", "v_proj.weight"):
                    kv_params += n
                elif name == "q_proj.weight":
                    q_params += n
                else:
                    other_params += n

        return {
            "total": total,
            "kv_params": kv_params,
            "q_params": q_params,
            "other_params": other_params,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Registry hook
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "CrossLayerKVConfig",
    "CrossLayerKVAttention",
    "CrossLayerKVStack",
]
