"""
Ring Attention with Blockwise Transformers for Near-Infinite Context.

Reference: arXiv:2310.01889
"Ring Attention with Blockwise Transformers for Near-Infinite Context"

Single-device simulation of the ring attention algorithm. Sequence is
partitioned into N_chunks = seq_len / chunk_size chunks; each round r
in [0..N_chunks-1] pairs query chunk q_i with KV chunk (k_j, v_j) where
j = (i + r) % N_chunks.  Output blocks are accumulated via the online
softmax (log-sum-exp) trick for numerical stability.

Variable notation follows the paper where possible:
  - Q, K, V   : query, key, value tensors  (B, H, T, d_k)
  - d_k        : head dimension  (= d_model / n_heads)
  - N          : number of chunks  (= T / chunk_size)
  - q_i        : query chunk for device/block i
  - k_j, v_j  : KV chunks rotating around the ring
  - S_ij       : raw attention scores q_i @ k_j^T / sqrt(d_k)
  - lse_i      : running log-sum-exp accumulator for block i
  - O_i        : running output accumulator for block i
"""

import math

import torch
import torch.nn as nn


class RingAttention(nn.Module):
    """Ring Attention (single-device simulation).

    Partitions Q, K, V into N = seq_len // chunk_size chunks and
    processes attention blockwise, rotating KV blocks in a ring.
    All P = N rounds are performed sequentially (no actual distributed
    ops), producing the same output as full causal attention.

    Args:
        d_model:    Model (embedding) dimension.
        n_heads:    Number of attention heads.
        chunk_size: Sequence chunk size (must evenly divide seq_len).
        causal:     If True, apply causal (autoregressive) masking.

    Raises:
        ValueError: If d_model is not divisible by n_heads, or if
                    seq_len is not divisible by chunk_size at forward time.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 64,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads  # H
        self.chunk_size = chunk_size  # C
        self.causal = causal
        self.d_k = d_model // n_heads  # head dimension

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self._scale = math.sqrt(self.d_k)  # scaling factor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, H, T, d_k)."""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (B, H, T, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) -> (B, T, d_model)."""
        B, H, T, d_k = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * d_k)

    # ------------------------------------------------------------------
    # Online softmax (log-sum-exp) block merge
    # ------------------------------------------------------------------

    @staticmethod
    def _online_update(
        lse_old: torch.Tensor,  # (B, H, C, 1)
        O_old: torch.Tensor,  # (B, H, C, d_k)
        S_new: torch.Tensor,  # (B, H, C, C)  raw scores (post-mask)
        V_j: torch.Tensor,  # (B, H, C, d_k)
    ):
        """Merge a new KV block into running (lse, O) accumulators.

        Uses the online softmax trick from the paper (eq. 3–5):
            lse_new = log( exp(lse_old) + sum_j exp(S_new_j) )
            O_new   = ( exp(lse_old) * O_old + exp(S_new) @ V_j )
                      / exp(lse_new)

        Args:
            lse_old: Running log-sum-exp of shape (B, H, C, 1).
            O_old:   Running output accumulator of shape (B, H, C, d_k).
            S_new:   Raw score block (B, H, C, C).
            V_j:     Value chunk (B, H, C, d_k).

        Returns:
            Tuple (lse_new, O_new) with same shapes as inputs.
        """
        # Per-row max for numerical stability.
        # Replace -inf max (all-masked row) with 0 so that exp(S - m) = 0
        # for all-masked rows, leaving those rows uncontributed.
        m_new = S_new.amax(dim=-1, keepdim=True)  # (B, H, C, 1)
        m_new_safe = m_new.clamp(min=torch.finfo(S_new.dtype).min / 2)

        # exp of shifted scores; rows where m_new=-inf safely become 0
        exp_s = torch.exp(S_new - m_new_safe)  # (B, H, C, C)

        # Sum of exp per row; all-masked rows → 0 (not NaN)
        sum_exp_s = exp_s.sum(dim=-1, keepdim=True)  # (B, H, C, 1)

        # Local lse for this block.  All-masked rows produce lse = -inf
        # because log(0 + 1e-30) ≈ -inf, which correctly contributes nothing
        # to the merge below.
        lse_block = m_new_safe + torch.log(sum_exp_s + 1e-30)  # (B, H, C, 1)

        # Merge old and new lse using the standard log-sum-exp identity.
        # When both operands are -inf the result is -inf (no contribution).
        lse_max = torch.maximum(lse_old, lse_block)
        # Guard against -inf - (-inf) = NaN by treating -inf max as finite
        lse_max_safe = lse_max.clamp(min=torch.finfo(S_new.dtype).min / 2)
        lse_new = lse_max_safe + torch.log(
            torch.exp(lse_old - lse_max_safe) + torch.exp(lse_block - lse_max_safe)
        )  # (B, H, C, 1)

        # Where both lse_old and lse_block are -inf, lse_new is effectively
        # -inf; the row has seen no valid keys yet.  Replace NaN → -inf.
        lse_new = torch.nan_to_num(lse_new, nan=float("-inf"))

        # Update output accumulator
        scale_old = torch.exp(lse_old - lse_new)  # (B, H, C, 1)
        scale_new = torch.exp(lse_block - lse_new)  # (B, H, C, 1)

        # Guard NaN from 0/0 when lse_new = -inf (no valid keys anywhere yet)
        scale_old = torch.nan_to_num(scale_old, nan=0.0)
        scale_new = torch.nan_to_num(scale_new, nan=0.0)

        # Normalise exp_s within this block (safe: all-masked rows → 0/eps = 0)
        exp_s_norm = exp_s / (sum_exp_s + 1e-30)  # (B, H, C, C)
        O_new = scale_old * O_old + scale_new * (exp_s_norm @ V_j)

        return lse_new, O_new

    # ------------------------------------------------------------------
    # Causal mask for a single (q_i, k_j) block pair
    # ------------------------------------------------------------------

    def _causal_block_mask(
        self,
        i: int,
        j: int,
        C: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute a boolean mask for the (i, j) block in causal attention.

        Position q at absolute index (i*C + qi_local) may attend to key k
        at absolute index (j*C + kj_local) iff k_abs <= q_abs.

        Returns:
            Float tensor of shape (C, C) with 0 (attend) or -inf (mask out).
        """
        q_abs = i * C + torch.arange(C, device=device)  # (C,)
        k_abs = j * C + torch.arange(C, device=device)  # (C,)
        # mask[qi, kj] = True  if  k_abs[kj] > q_abs[qi]  (future → mask)
        mask = k_abs.unsqueeze(0) > q_abs.unsqueeze(1)  # (C, C)
        return mask.to(dtype) * torch.finfo(dtype).min  # -inf where masked

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ring attention.

        Args:
            x:                Input tensor of shape (B, T, d_model).
            attention_mask:   Optional float or bool mask of shape
                              (B, T) or (B, 1, T, T). Padding positions
                              should be 0 / False (or -inf if float).
                              Currently applied as an additive bias to
                              the full score matrix; partial support.

        Returns:
            Output tensor of shape (B, T, d_model).

        Raises:
            ValueError: If T is not divisible by chunk_size.
        """
        B, T, _ = x.shape
        C = self.chunk_size  # chunk size
        if T % C != 0:
            raise ValueError(
                f"seq_len ({T}) must be divisible by chunk_size ({C}). "
                f"Pad your sequence or choose a compatible chunk_size."
            )
        N = T // C  # number of chunks (= ring size)

        # Project Q, K, V and reshape to (B, H, T, d_k)
        Q = self._split_heads(self.W_q(x))  # (B, H, T, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        # Partition into N chunks along sequence dimension
        # Q_chunks[i] : (B, H, C, d_k)
        Q_chunks = Q.chunk(N, dim=2)
        K_chunks = K.chunk(N, dim=2)
        V_chunks = V.chunk(N, dim=2)

        # Build additive attention bias from mask if provided
        # We support (B, T) padding masks (1 = keep, 0 = pad)
        attn_bias: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, T) → (B, 1, 1, T)  additive key-side mask
                pad = (~attention_mask.bool()).to(x.dtype) * torch.finfo(x.dtype).min
                attn_bias = pad.view(B, 1, 1, T)  # broadcast over H, q_pos
            elif attention_mask.dim() == 4:
                attn_bias = attention_mask  # (B, H or 1, T, T) assumed float
            # else: silently ignore unknown shapes

        # ----------------------------------------------------------------
        # Ring loop
        # ----------------------------------------------------------------
        # Per-block output accumulators: list of N tensors (B, H, C, d_k)
        O_blocks = [
            torch.zeros(B, self.n_heads, C, self.d_k, device=x.device, dtype=x.dtype)
            for _ in range(N)
        ]
        # Per-block log-sum-exp: shape (B, H, C, 1), initialised to -inf
        lse_blocks = [
            torch.full(
                (B, self.n_heads, C, 1),
                fill_value=float("-inf"),
                device=x.device,
                dtype=x.dtype,
            )
            for _ in range(N)
        ]

        for r in range(N):  # ring round
            for i in range(N):  # query block index
                j = (i + r) % N  # KV block index on this round

                q_i = Q_chunks[i]  # (B, H, C, d_k)
                k_j = K_chunks[j]  # (B, H, C, d_k)
                v_j = V_chunks[j]  # (B, H, C, d_k)

                # Raw scores S_ij = q_i @ k_j^T / sqrt(d_k)
                # (B, H, C, C)
                S_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) / self._scale

                # Causal mask
                if self.causal:
                    bias = self._causal_block_mask(i, j, C, x.device, x.dtype)
                    S_ij = S_ij + bias  # broadcast over B, H

                # External attention mask (key side, if provided)
                if attn_bias is not None:
                    k_start = j * C
                    k_end = k_start + C
                    q_start = i * C
                    q_end = q_start + C
                    # attn_bias shape: (B, 1, 1, T) key-side mask  OR
                    #                  (B, H or 1, T, T) full mask
                    ab = attn_bias
                    q_sz = ab.shape[-2]  # 1 (key-only) or T (full)
                    ab.shape[-1]  # T always

                    if q_sz == 1:
                        # (B, 1, 1, T) → slice key dimension only
                        S_ij = S_ij + ab[..., :, k_start:k_end]
                    else:
                        # (B, H or 1, T, T) → slice both dims
                        S_ij = S_ij + ab[..., q_start:q_end, k_start:k_end]

                # Online softmax update
                lse_blocks[i], O_blocks[i] = self._online_update(
                    lse_blocks[i], O_blocks[i], S_ij, v_j
                )

        # Concatenate output blocks: (B, H, T, d_k)
        O = torch.cat(O_blocks, dim=2)  # noqa: E741
        # Merge heads: (B, T, d_model)
        out = self._merge_heads(O)  # noqa: E741
        # Output projection
        return self.W_o(out)
