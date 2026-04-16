"""
Infini-Attention — arXiv:2404.07143
"Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
Google, 2024.

Variable names mirror the paper's notation (Section 2):

  Q_s, K_s, V_s   — query / key / value for segment s, shape (B, H, T_seg, d_k)
  M_s              — compressive memory matrix, shape (B, H, d_k, d_v)   [or (H, d_k, d_v) when stored]
  z_s              — memory normaliser vector, shape (B, H, d_k)          [or (H, d_k) when stored]
  A_mem            — memory-retrieved attention, shape (B, H, T_seg, d_v)
  A_dot            — local dot-product attention, shape (B, H, T_seg, d_v)
  A_s              — gated output, shape (B, H, T_seg, d_v)
  σ                — kernel feature map: ELU(x) + 1  (ensures positivity)
  β                — learned gate scalar per head (in (0, 1) after sigmoid)
  ε                — numerical stability epsilon
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

# Memory state type: (M, z)
#   M : Tensor of shape (n_heads, d_key, d_value)
#   z : Tensor of shape (n_heads, d_key)
MemoryState = Tuple[Tensor, Tensor]


def _sigma(x: Tensor) -> Tensor:
    """Kernel feature map σ(x) = ELU(x) + 1.  Equation (2) of the paper.
    Guarantees all entries are strictly positive, making M positive-semi-definite.
    """
    return F.elu(x) + 1.0


class InfiniAttention(nn.Module):
    """
    Infini-Attention (arXiv:2404.07143).

    Augments standard causal dot-product attention with a compressive associative
    memory that accumulates context from all preceding segments.

    Parameters
    ----------
    d_model     : total model dimension (must be divisible by n_heads)
    n_heads     : number of attention heads
    segment_len : T_seg — tokens processed per memory segment (default 64)
    eps         : numerical stability constant ε (default 1e-6)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        segment_len: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads          # H
        self.segment_len = segment_len  # T_seg
        self.eps = eps

        self.d_k = d_model // n_heads   # d_key  per head
        self.d_v = d_model // n_heads   # d_value per head (equal to d_k here)

        # Standard QKV and output projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Per-head gate parameter β (Eq. 7).  One scalar per head, stored in logit
        # space; β = sigmoid(β_param) at runtime.
        self.beta_param = nn.Parameter(torch.zeros(n_heads))  # shape (H,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        """Return (M_0, z_0) initialised to zero."""
        M = torch.zeros(batch_size, self.n_heads, self.d_k, self.d_v, device=device, dtype=dtype)
        z = torch.zeros(batch_size, self.n_heads, self.d_k, device=device, dtype=dtype)
        return M, z

    def _memory_retrieve(self, Q_s: Tensor, M_prev: Tensor, z_prev: Tensor) -> Tensor:
        """
        Eq. 4: retrieve from compressive memory using linear kernel.

        Parameters
        ----------
        Q_s    : (B, H, T, d_k)
        M_prev : (B, H, d_k, d_v)
        z_prev : (B, H, d_k)

        Returns
        -------
        A_mem  : (B, H, T, d_v)
        """
        sig_Q = _sigma(Q_s)                                   # (B, H, T, d_k)
        # Numerator: σ(Q_s) @ M_{s-1}  →  (B, H, T, d_v)
        numer = torch.matmul(sig_Q, M_prev)                   # (B, H, T, d_v)
        # Denominator: σ(Q_s) @ z_{s-1}  →  (B, H, T) then unsqueeze
        denom = (sig_Q * z_prev.unsqueeze(-2)).sum(-1, keepdim=True) + self.eps  # (B, H, T, 1)
        A_mem = numer / denom                                 # (B, H, T, d_v)
        return A_mem

    def _dot_product_attention(self, Q_s: Tensor, K_s: Tensor, V_s: Tensor) -> Tensor:
        """
        Standard causal scaled dot-product attention within a segment.

        Parameters
        ----------
        Q_s, K_s, V_s : (B, H, T, d_k)

        Returns
        -------
        A_dot : (B, H, T, d_v)
        """
        scale = math.sqrt(self.d_k)
        T = Q_s.size(-2)
        # Attention logits (B, H, T, T)
        scores = torch.matmul(Q_s, K_s.transpose(-2, -1)) / scale
        # Causal mask: upper triangle → -inf
        causal_mask = torch.triu(
            torch.ones(T, T, device=Q_s.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        A_dot = torch.matmul(attn, V_s)   # (B, H, T, d_v)
        return A_dot

    def _memory_update(self, K_s: Tensor, V_s: Tensor, M_prev: Tensor, z_prev: Tensor) -> MemoryState:
        """
        Eq. 5 (delta rule variant): update compressive memory with new segment.

        Parameters
        ----------
        K_s    : (B, H, T, d_k)
        V_s    : (B, H, T, d_v)
        M_prev : (B, H, d_k, d_v)
        z_prev : (B, H, d_k)

        Returns
        -------
        M_s    : (B, H, d_k, d_v)
        z_s    : (B, H, d_k)
        """
        sig_K = _sigma(K_s)                                       # (B, H, T, d_k)
        # Recall current memory at key positions
        denom_K = (sig_K * z_prev.unsqueeze(-2)).sum(-1, keepdim=True) + self.eps  # (B, H, T, 1)
        V_prev = torch.matmul(sig_K, M_prev) / denom_K           # (B, H, T, d_v)
        # Delta update: M_s = M_{s-1} + σ(K_s)^T @ (V_s - V_prev)
        delta = V_s - V_prev                                       # (B, H, T, d_v)
        M_s = M_prev + torch.matmul(sig_K.transpose(-2, -1), delta)  # (B, H, d_k, d_v)
        # Normaliser update: z_s = z_{s-1} + Σ_t σ(K_s)_t
        z_s = z_prev + sig_K.sum(-2)                              # (B, H, d_k)
        return M_s, z_s

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[MemoryState] = None,
    ) -> Tuple[Tensor, MemoryState]:
        """
        Parameters
        ----------
        x            : (B, T, d_model) — input sequence
        memory_state : optional (M, z) from a prior call.
                       M shape: (n_heads, d_key, d_value)   [no batch dim when passed in]
                       z shape: (n_heads, d_key)
                       Pass None to zero-initialise.

        Returns
        -------
        output       : (B, T, d_model) — same shape as x
        new_memory   : (M_new, z_new) detached from graph, shapes as above (no batch dim)
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # ------ Project Q, K, V and reshape to (B, H, T, d_k) ------
        def _proj_and_split(W: nn.Linear) -> Tensor:
            out = W(x)                                  # (B, T, d_model)
            out = out.view(B, T, self.n_heads, self.d_k)
            return out.permute(0, 2, 1, 3)             # (B, H, T, d_k)

        Q = _proj_and_split(self.W_q)
        K = _proj_and_split(self.W_k)
        V = _proj_and_split(self.W_v)

        # ------ Initialise / expand memory state to (B, H, d_k, d_v) ------
        if memory_state is None:
            M, z = self._zero_memory(B, device, dtype)
        else:
            M_in, z_in = memory_state
            # Accept both batched (B, H, …) and unbatched (H, …) memory
            if M_in.dim() == 3:
                M = M_in.unsqueeze(0).expand(B, -1, -1, -1).clone()
                z = z_in.unsqueeze(0).expand(B, -1, -1).clone()
            else:
                M = M_in.clone()
                z = z_in.clone()

        # ------ Segment the sequence, padding the last if necessary ------
        T_seg = self.segment_len
        pad_len = (T_seg - T % T_seg) % T_seg
        if pad_len > 0:
            # Pad Q, K, V along the time axis
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        T_padded = T + pad_len
        n_segments = T_padded // T_seg

        # Collect per-segment outputs
        output_segments = []

        # Gate β per head — shape (1, H, 1, 1) for broadcasting
        beta = torch.sigmoid(self.beta_param).view(1, self.n_heads, 1, 1)

        for s in range(n_segments):
            t_start = s * T_seg
            t_end = t_start + T_seg

            Q_s = Q[:, :, t_start:t_end, :]   # (B, H, T_seg, d_k)
            K_s = K[:, :, t_start:t_end, :]
            V_s = V[:, :, t_start:t_end, :]

            # Memory retrieval (Eq. 4)
            A_mem = self._memory_retrieve(Q_s, M, z)  # (B, H, T_seg, d_v)

            # Local causal dot-product attention
            A_dot = self._dot_product_attention(Q_s, K_s, V_s)  # (B, H, T_seg, d_v)

            # Gated combination (Eq. 7)
            A_s = beta * A_mem + (1.0 - beta) * A_dot            # (B, H, T_seg, d_v)

            output_segments.append(A_s)

            # Memory update (Eq. 5, delta rule)
            M, z = self._memory_update(K_s, V_s, M, z)

        # ------ Concatenate all segments and strip padding ------
        out = torch.cat(output_segments, dim=2)         # (B, H, T_padded, d_v)
        out = out[:, :, :T, :]                          # (B, H, T, d_v)

        # ------ Merge heads and apply output projection ------
        out = out.permute(0, 2, 1, 3).contiguous()     # (B, T, H, d_v)
        out = out.view(B, T, self.d_model)              # (B, T, d_model)
        out = self.W_o(out)                             # (B, T, d_model)

        # ------ Return detached memory (remove batch dim → (H, d_k, d_v)) ------
        # We average across the batch for the returned state (standard practice
        # for single-sequence inference; in batched training callers typically
        # discard the state between updates).
        M_out = M.mean(0).detach()    # (H, d_k, d_v)
        z_out = z.mean(0).detach()    # (H, d_k)

        return out, (M_out, z_out)
