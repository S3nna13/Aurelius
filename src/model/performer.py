"""Performer attention via FAVOR+ random feature approximation.

Implements "Rethinking Attention with Performers" (Choromanski et al., 2020,
arXiv:2009.14794).

FAVOR+ (Fast Attention Via positive Orthogonal Random features) replaces the
O(T²) softmax attention kernel with an O(T·m) unbiased approximation built
from positive orthogonal random features:

    phi(x) = exp(x @ ω^T - ||x||²/2) / sqrt(m)    ω ∈ R^{m×d}

The random feature matrix ω uses orthogonally-initialised rows (via QR
decomposition) scaled by the chi distribution (Frobenius norms of iid Gaussian
blocks), giving lower variance than ordinary i.i.d. Gaussian features.

Non-causal complexity : O(T · m · d)
Causal (prefix-sum)  : O(T · m · d)   — O(T) memory

Classes
-------
PerformerOrthogonalRF   — Draws and caches orthogonal random features ω.
PerformerAttention      — Single-head FAVOR+ attention (causal or non-causal).
PerformerLayer          — Multi-head Performer replacing standard MHA.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper: positive FAVOR+ feature map
# ---------------------------------------------------------------------------

def _positive_map(x: Tensor, omega: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute positive FAVOR+ random feature map.

    phi(x) = exp(x @ omega^T - ||x||^2 / 2) / sqrt(m)

    For numerical stability we shift by the per-sample maximum before
    exponentiating, then undo the shift so that the resulting values remain
    positive and the ratio phi_Q @ kv / phi_Q @ z is unchanged.

    Args:
        x:      (..., d_head)   input vectors (already scaled by caller).
        omega:  (m, d_head)     orthogonal random features matrix.
        eps:    small constant added to denominator norms.

    Returns:
        Tensor of shape (..., m)  — positive feature vectors.
    """
    # (..., d_head) @ (d_head, m) → (..., m)
    proj = x @ omega.T                              # (..., m)
    # ||x||^2 / 2  — shape (..., 1)
    norm_sq_half = 0.5 * (x * x).sum(dim=-1, keepdim=True)   # (..., 1)
    log_phi = proj - norm_sq_half                   # (..., m)
    # Subtract row-wise max for numerical stability (cancels in ratios)
    log_phi = log_phi - log_phi.max(dim=-1, keepdim=True).values
    m = omega.shape[0]
    phi = torch.exp(log_phi) / math.sqrt(m)         # (..., m)
    return phi


# ---------------------------------------------------------------------------
# PerformerOrthogonalRF
# ---------------------------------------------------------------------------

class PerformerOrthogonalRF(nn.Module):
    """Orthogonal random features for FAVOR+.

    Draws m random feature vectors arranged in orthogonal blocks of size
    d_head × d_head (last block truncated if necessary).  Each block is formed
    by QR-decomposing a random Gaussian matrix and then scaling rows by the
    Frobenius norms of the original blocks (chi-distribution scaling).

    Args:
        d_head:       Dimension of each attention head.
        num_features: Number of random features m.
        seed:         Optional integer seed for reproducibility.
    """

    def __init__(self, d_head: int, num_features: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self.d_head = d_head
        self.num_features = num_features
        self.seed = seed

        # Cached omega; None triggers draw on first call to get_omegas()
        self._omega: Optional[Tensor] = None
        self._needs_redraw: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_omegas(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return the cached (num_features, d_head) omega matrix.

        The matrix is drawn (or redrawn) lazily.  Subsequent calls return the
        same tensor unless ``redraw()`` has been called.

        Args:
            device: Target device.
            dtype:  Target floating-point dtype.

        Returns:
            Tensor of shape (num_features, d_head).
        """
        if self._needs_redraw or self._omega is None:
            self._omega = self._draw_omega(device, dtype)
            self._needs_redraw = False
        elif self._omega.device != device or self._omega.dtype != dtype:
            self._omega = self._omega.to(device=device, dtype=dtype)
        return self._omega

    def redraw(self) -> None:
        """Signal that omegas should be redrawn on the next ``get_omegas`` call."""
        self._needs_redraw = True
        self._omega = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_omega(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Draw a fresh orthogonal random feature matrix."""
        m = self.num_features
        d = self.d_head

        # Number of blocks of size (d, d) we need
        n_blocks = math.ceil(m / d)

        # Use a local RNG so we don't perturb global state
        if self.seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
        else:
            gen = None

        blocks: list[Tensor] = []
        for _ in range(n_blocks):
            # Sample d × d Gaussian matrix
            if gen is not None:
                g = torch.randn(d, d, generator=gen, device=device, dtype=torch.float32)
            else:
                g = torch.randn(d, d, device=device, dtype=torch.float32)

            # QR decomposition — Q has orthonormal rows/cols
            q, _ = torch.linalg.qr(g)          # (d, d)

            # Scale each row by the Frobenius norm of the corresponding
            # Gaussian row (chi distribution scaling, reduces variance)
            norms = torch.linalg.norm(g, dim=1)  # (d,)
            q = q * norms.unsqueeze(1)            # (d, d)

            blocks.append(q)

        # Stack all blocks and truncate to m features
        omega = torch.cat(blocks, dim=0)[:m]     # (m, d)
        return omega.to(dtype=dtype)


# ---------------------------------------------------------------------------
# PerformerAttention
# ---------------------------------------------------------------------------

class PerformerAttention(nn.Module):
    """Single-head FAVOR+ performer attention.

    For *non-causal* mode:
        kv = phi_K^T @ V          (m, d_head)
        z  = phi_K.sum(dim=-2)    (m,)
        out[t] = (phi_Q[t] @ kv) / (phi_Q[t] @ z + eps)

    For *causal* mode (prefix-sum recurrence, O(T) memory):
        S_t  = sum_{s<=t} phi_K[s]^T ⊗ V[s]   accumulated
        z_t  = sum_{s<=t} phi_K[s]             accumulated
        out[t] = (phi_Q[t] @ S_t) / (phi_Q[t] @ z_t + eps)

    Args:
        d_head:       Head dimension.
        num_features: Number of random features m.
        causal:       If True, use causal (prefix-sum) recurrence.
        seed:         Optional seed for the random feature generator.
    """

    def __init__(
        self,
        d_head: int,
        num_features: int = 256,
        causal: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.num_features = num_features
        self.causal = causal
        self.eps = 1e-6
        self.orf = PerformerOrthogonalRF(d_head, num_features, seed=seed)

    # ------------------------------------------------------------------

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute FAVOR+ attention.

        Args:
            q: (B, T, d_head)
            k: (B, T, d_head)
            v: (B, T, d_head)

        Returns:
            Tensor of shape (B, T, d_head).
        """
        d = self.d_head
        scale = 1.0 / math.sqrt(d)

        omega = self.orf.get_omegas(device=q.device, dtype=q.dtype)  # (m, d)

        # Scale queries and keys
        phi_q = _positive_map(q * scale, omega, eps=self.eps)   # (B, T, m)
        phi_k = _positive_map(k * scale, omega, eps=self.eps)   # (B, T, m)

        if self.causal:
            return self._causal_forward(phi_q, phi_k, v)
        else:
            return self._noncausal_forward(phi_q, phi_k, v)

    # ------------------------------------------------------------------

    def _noncausal_forward(self, phi_q: Tensor, phi_k: Tensor, v: Tensor) -> Tensor:
        """Non-causal FAVOR+: O(T·m·d) with two matrix products."""
        # phi_k: (B, T, m)   v: (B, T, d_head)
        # kv_sum = phi_k^T @ v  → (B, m, d_head)
        kv_sum = torch.bmm(phi_k.transpose(1, 2), v)       # (B, m, d_head)
        # normaliser: phi_k^T @ ones → (B, m)
        z = phi_k.sum(dim=1)                                # (B, m)

        # out = phi_q @ kv_sum / (phi_q @ z + eps)
        # phi_q: (B, T, m)  kv_sum: (B, m, d_head) → (B, T, d_head)
        out = torch.bmm(phi_q, kv_sum)                     # (B, T, d_head)
        denom = (phi_q * z.unsqueeze(1)).sum(dim=-1, keepdim=True) + self.eps  # (B, T, 1)
        return out / denom

    def _causal_forward(self, phi_q: Tensor, phi_k: Tensor, v: Tensor) -> Tensor:
        """Causal FAVOR+ via prefix-sum recurrence (O(T) memory)."""
        B, T, m = phi_q.shape
        d_head = v.shape[-1]

        # Accumulate S_t (m × d_head) and z_t (m,) iteratively
        S = phi_q.new_zeros(B, m, d_head)   # (B, m, d_head)
        z = phi_q.new_zeros(B, m)           # (B, m)
        outputs = []

        for t in range(T):
            pk_t = phi_k[:, t, :]           # (B, m)
            v_t  = v[:, t, :]               # (B, d_head)
            pq_t = phi_q[:, t, :]           # (B, m)

            # Update accumulators: outer product pk_t^T ⊗ v_t
            S = S + torch.bmm(pk_t.unsqueeze(2), v_t.unsqueeze(1))  # (B, m, d_head)
            z = z + pk_t                                              # (B, m)

            # Query: out_t = pq_t @ S / (pq_t @ z + eps)
            out_t = torch.bmm(pq_t.unsqueeze(1), S).squeeze(1)      # (B, d_head)
            denom = (pq_t * z).sum(dim=-1, keepdim=True) + self.eps  # (B, 1)
            outputs.append(out_t / denom)

        return torch.stack(outputs, dim=1)  # (B, T, d_head)


# ---------------------------------------------------------------------------
# PerformerLayer
# ---------------------------------------------------------------------------

class PerformerLayer(nn.Module):
    """Multi-head Performer layer replacing standard MHA.

    Projects the input into Q, K, V per head, applies FAVOR+ attention
    independently to each head, concatenates, and projects back.

    Args:
        d_model:      Model dimension.
        n_heads:      Number of attention heads.
        num_features: Random feature count m (per head).
        causal:       If True, use causal (autoregressive) attention.
        seed:         Optional seed; each head gets seed+i if provided.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_features: int = 256,
        causal: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        # One linear for each of Q, K, V (combined across heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # One PerformerAttention per head
        self.heads = nn.ModuleList([
            PerformerAttention(
                d_head=self.d_head,
                num_features=num_features,
                causal=causal,
                seed=(seed + i) if seed is not None else None,
            )
            for i in range(n_heads)
        ])

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply multi-head Performer attention.

        Args:
            x:    (B, T, d_model)  input embeddings.
            mask: Ignored (causality is handled internally).  Kept for API
                  compatibility with standard attention modules.

        Returns:
            Tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape

        Q = self.q_proj(x)   # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        d = self.d_head
        head_outputs: list[Tensor] = []
        for h, attn in enumerate(self.heads):
            q_h = Q[:, :, h * d : (h + 1) * d]   # (B, T, d_head)
            k_h = K[:, :, h * d : (h + 1) * d]
            v_h = V[:, :, h * d : (h + 1) * d]
            head_outputs.append(attn(q_h, k_h, v_h))

        out = torch.cat(head_outputs, dim=-1)      # (B, T, d_model)
        return self.out_proj(out)
