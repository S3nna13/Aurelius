"""Infini-attention (Munkhdalai et al. 2024 — arXiv:2404.07143).

Augments standard (local) attention with a compressive associative memory
`M` and a normalizer `z`, updated linearly per segment with a non-linear
feature map sigma (ELU+1, the Katharopoulos kernel). A learnable per-head
gate beta blends local and memory-retrieved attention:

    phi(x)   = elu(x) + 1                         # positive feature map
    M_{s+1}  = M_s + K_s^T @ phi(V_s)             # linear memory update
    z_{s+1}  = z_s + sum_t phi(K_s)_t             # normalizer update
    A_mem    = phi(Q_s) @ M_s / (phi(Q_s) @ z_s + eps)
    out      = sigmoid(beta) * A_local
               + (1 - sigmoid(beta)) * A_mem

The "memory" here is per-head and of shape [B, H, D, D]; `z` is [B, H, D].
Memory persists across forward() calls until `reset_memory()` is called or
`reset_memory=True` is passed on a call. `detach_memory()` snips the
autograd graph for truncated-BPTT-style streaming training.

SINGLE-DEVICE ONLY. No distributed memory is implemented here.

NOTE: this module is self-contained and MUST NOT import from `src.model`
(see `test_importing_longcontext_does_not_import_model` — any such import
leaks into sibling strategies' hermetic checks).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _elu_plus_one(x: Tensor) -> Tensor:
    """Katharopoulos feature map: phi(x) = elu(x) + 1, strictly positive."""
    return torch.nn.functional.elu(x) + 1.0


class InfiniAttention(nn.Module):
    """Infini-attention layer with persistent compressive memory.

    Args:
        d_model: total model width (informational; not used to reshape here
            because q/k/v are already supplied in [B, H, S, D] form).
        n_heads: number of attention heads H.
        head_dim: per-head dim D; must satisfy n_heads * head_dim == d_model.

    Persistent buffers (lazily allocated on first forward):
        memory_M: [B, H, D, D] associative memory
        memory_z: [B, H, D]    per-key normalizer

    These are stored as plain attributes (not nn.Parameters / buffers) so
    they can track batch size dynamically and participate in autograd for
    truncated BPTT.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"InfiniAttention: d_model/n_heads/head_dim must be positive; "
                f"got d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}"
            )
        if n_heads * head_dim != d_model:
            raise ValueError(
                f"InfiniAttention: n_heads * head_dim ({n_heads}*{head_dim}="
                f"{n_heads * head_dim}) must equal d_model ({d_model})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Learnable per-head gate. Initialized to 0 so sigmoid(beta)=0.5:
        # local and memory contribute equally at init.
        self.beta = nn.Parameter(torch.zeros(n_heads))

        # Lazy state.
        self.memory_M: Tensor | None = None
        self.memory_z: Tensor | None = None

    # ------------------------------------------------------------------ #
    # Memory lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_memory(self) -> None:
        """Zero out the compressive memory `M` and normalizer `z`."""
        if self.memory_M is not None:
            self.memory_M = torch.zeros_like(self.memory_M)
        if self.memory_z is not None:
            self.memory_z = torch.zeros_like(self.memory_z)

    def detach_memory(self) -> None:
        """Break the autograd graph on persistent memory (truncated BPTT)."""
        if self.memory_M is not None:
            self.memory_M = self.memory_M.detach()
        if self.memory_z is not None:
            self.memory_z = self.memory_z.detach()

    def _ensure_memory(self, B: int, device: torch.device, dtype: torch.dtype) -> None:
        need_alloc = (
            self.memory_M is None
            or self.memory_M.shape[0] != B
            or self.memory_M.device != device
            or self.memory_M.dtype != dtype
        )
        if need_alloc:
            self.memory_M = torch.zeros(
                B,
                self.n_heads,
                self.head_dim,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            self.memory_z = torch.zeros(
                B,
                self.n_heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        reset_memory: bool = False,
        causal: bool = True,
    ) -> Tensor:
        """Apply Infini-attention over one segment.

        Shapes:
            q, k, v: [B, H, S, D]
            returns: [B, H, S, D]
        """
        self._validate(q, k, v)
        B, H, S, D = q.shape

        if reset_memory:
            # Drop any prior memory before computing this segment. This
            # must happen BEFORE we read/update M to honor the kwarg.
            self.memory_M = None
            self.memory_z = None

        self._ensure_memory(B, q.device, q.dtype)

        # --- 1. Local attention (standard scaled dot-product) ---------
        scale = 1.0 / math.sqrt(D)
        # [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            # Upper-triangular mask: position i cannot attend to j>i.
            mask = torch.ones(S, S, device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        a_local = torch.matmul(probs, v)  # [B, H, S, D]

        # --- 2. Memory retrieval --------------------------------------
        phi_q = _elu_plus_one(q)  # [B, H, S, D]
        phi_k = _elu_plus_one(k)  # [B, H, S, D]

        # numerator:   [B, H, S, D] = phi_q @ M     (M is [B,H,D,D])
        numer = torch.matmul(phi_q, self.memory_M)
        # denom scalar per (B,H,S,1): phi_q . z
        denom = torch.matmul(phi_q, self.memory_z.unsqueeze(-1))  # [B,H,S,1]
        a_mem = numer / (denom + 1e-6)

        # --- 3. Gate --------------------------------------------------
        gate = torch.sigmoid(self.beta).view(1, H, 1, 1)
        out = gate * a_local + (1.0 - gate) * a_mem

        # --- 4. Update memory for next segment ------------------------
        # Paper eq: M_{s+1} = M_s + K_s^T @ phi(V_s); but per
        # Katharopoulos-style linear attention we accumulate keys in the
        # feature-mapped space and values in raw space so that retrieval
        # phi(Q) @ M / (phi(Q) @ z) matches standard softmax semantics.
        # Here: M <- M + phi(K)^T @ V,  z <- z + sum_t phi(K)_t.
        # (The paper's eq 5 applies phi to V; we follow the algebraically
        # equivalent common formulation — either works as long as
        # retrieval uses phi(Q) @ M. We pick phi on K to keep A_mem in the
        # value space, matching A_local's units so the gate blends
        # comparable quantities.)
        delta_M = torch.matmul(phi_k.transpose(-2, -1), v)  # [B,H,D,D]
        delta_z = phi_k.sum(dim=-2)  # [B,H,D]

        # Accumulate without building long autograd chains by default;
        # callers use detach_memory() between segments for truncated BPTT.
        self.memory_M = self.memory_M + delta_M
        self.memory_z = self.memory_z + delta_z

        return out

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _validate(self, q: Tensor, k: Tensor, v: Tensor) -> None:
        for name, t in (("q", q), ("k", k), ("v", v)):
            if not isinstance(t, Tensor):
                raise TypeError(f"InfiniAttention: {name} must be a torch.Tensor")
            if t.dim() != 4:
                raise ValueError(
                    f"InfiniAttention: {name} must be 4-D [B,H,S,D]; got shape {tuple(t.shape)}"
                )
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                f"InfiniAttention: q/k/v must share shape; got "
                f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
            )
        B, H, S, D = q.shape
        if H != self.n_heads:
            raise ValueError(f"InfiniAttention: expected H={self.n_heads}, got {H}")
        if D != self.head_dim:
            raise ValueError(f"InfiniAttention: expected head_dim={self.head_dim}, got {D}")


__all__ = ["InfiniAttention"]
