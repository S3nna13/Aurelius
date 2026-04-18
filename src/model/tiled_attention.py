"""
Tiled / memory-efficient attention in pure PyTorch.

Implements the Flash-Attention-style tiling algorithm that avoids
materialising the full N×N attention matrix by accumulating the output
with an online softmax (log-sum-exp trick).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Online-softmax accumulator
# ---------------------------------------------------------------------------

class OnlineSoftmax:
    """
    Stateless helper that performs one step of the online-softmax
    accumulation used in the tiled forward pass.

    Given the running state (m_prev, l_prev, O_prev) from previous
    K/V tiles, incorporate a new (K_block, V_block) pair for a query
    slice Q_i and return the updated state.
    """

    def __init__(self) -> None:
        pass  # stateless – all logic lives in update()

    @staticmethod
    def update(
        m_prev: torch.Tensor,   # [B, H, Tq, 1]
        l_prev: torch.Tensor,   # [B, H, Tq, 1]
        O_prev: torch.Tensor,   # [B, H, Tq, d_head]
        Q_i: torch.Tensor,      # [B, H, Tq, d_head]
        K_block: torch.Tensor,  # [B, H, Tk, d_head]
        V_block: torch.Tensor,  # [B, H, Tk, d_head]
        scale: float,
    ):
        """
        One step of online-softmax accumulation.

        Returns:
            m_new : [B, H, Tq, 1]
            l_new : [B, H, Tq, 1]
            O_new : [B, H, Tq, d_head]
        """
        # scores: [B, H, Tq, Tk]
        scores = torch.matmul(Q_i, K_block.transpose(-2, -1)) * scale

        # new running maximum
        m_new = torch.max(m_prev, scores.max(dim=-1, keepdim=True).values)

        # unnormalised weights shifted by new max
        P = torch.exp(scores - m_new)                                    # [B, H, Tq, Tk]

        # update normalisation factor
        correction = torch.exp(m_prev - m_new)                           # [B, H, Tq, 1]
        l_new = correction * l_prev + P.sum(dim=-1, keepdim=True)        # [B, H, Tq, 1]

        # update output
        O_new = (correction * l_prev * O_prev + torch.matmul(P, V_block)) / l_new  # [B, H, Tq, d_head]

        return m_new, l_new, O_new


# ---------------------------------------------------------------------------
# Custom autograd Function
# ---------------------------------------------------------------------------

class TiledAttention(torch.autograd.Function):
    """
    Flash-Attention-style tiled forward + backward pass.

    Avoids allocating the full [B, H, T, T] attention matrix by
    iterating over blocks of size tile_size and accumulating with the
    online-softmax trick.
    """

    @staticmethod
    def forward(ctx, Q, K, V, scale, causal, tile_size):  # type: ignore[override]
        """
        Args:
            Q, K, V : [B, H, T, d_head]
            scale   : float  – pre-computed 1/sqrt(d_head)
            causal  : bool
            tile_size: int

        Returns:
            O : [B, H, T, d_head]
        """
        B, H, T, d_head = Q.shape
        device, dtype = Q.device, Q.dtype

        O = torch.zeros_like(Q)
        # running max / denominator stored for backward reuse
        logsumexp = torch.full((B, H, T), -float("inf"), device=device, dtype=dtype)

        online = OnlineSoftmax()

        # iterate over query tiles
        for q_start in range(0, T, tile_size):
            q_end = min(q_start + tile_size, T)
            Q_tile = Q[:, :, q_start:q_end, :]       # [B, H, tq, d]
            tq = q_end - q_start

            m_i = torch.full((B, H, tq, 1), -float("inf"), device=device, dtype=dtype)
            l_i = torch.zeros((B, H, tq, 1), device=device, dtype=dtype)
            O_i = torch.zeros((B, H, tq, d_head), device=device, dtype=dtype)

            # iterate over key/value tiles
            for kv_start in range(0, T, tile_size):
                kv_end = min(kv_start + tile_size, T)

                # causal mask: skip KV tiles that are entirely in the future
                if causal and kv_start >= q_end:
                    break

                K_tile = K[:, :, kv_start:kv_end, :]  # [B, H, tk, d]
                V_tile = V[:, :, kv_start:kv_end, :]

                # compute raw scores before mask
                scores = torch.matmul(Q_tile, K_tile.transpose(-2, -1)) * scale  # [B, H, tq, tk]

                # apply causal mask within a partial tile
                if causal:
                    q_indices = torch.arange(q_start, q_end, device=device).unsqueeze(1)   # [tq, 1]
                    k_indices = torch.arange(kv_start, kv_end, device=device).unsqueeze(0) # [1, tk]
                    mask = k_indices > q_indices   # True where key is in the future
                    scores = scores.masked_fill(mask, float("-inf"))

                # online-softmax update
                m_new = torch.max(m_i, scores.max(dim=-1, keepdim=True).values)
                P = torch.exp(scores - m_new)
                correction = torch.exp(m_i - m_new)
                l_new = correction * l_i + P.sum(dim=-1, keepdim=True)
                O_i = (correction * l_i * O_i + torch.matmul(P, V_tile)) / (l_new + 1e-10)
                m_i = m_new
                l_i = l_new

            O[:, :, q_start:q_end, :] = O_i
            # logsumexp = m + log(l), used in backward
            logsumexp[:, :, q_start:q_end] = (
                m_i.squeeze(-1) + torch.log(l_i.squeeze(-1) + 1e-10)
            )

        ctx.save_for_backward(Q, K, V, O, logsumexp)
        ctx.scale = scale
        ctx.causal = causal
        ctx.tile_size = tile_size
        return O

    @staticmethod
    def backward(ctx, dO):  # type: ignore[override]
        """
        Recompute attention weights from saved Q, K, V and compute
        dQ, dK, dV via a tiled backward pass.

        Returns: (dQ, dK, dV, None, None, None)
        """
        Q, K, V, O, logsumexp = ctx.saved_tensors
        scale = ctx.scale
        causal = ctx.causal
        tile_size = ctx.tile_size

        B, H, T, d_head = Q.shape
        device = Q.device

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D_i = rowsum(dO * O)  [B, H, T]
        D = (dO * O).sum(dim=-1)  # [B, H, T]

        for q_start in range(0, T, tile_size):
            q_end = min(q_start + tile_size, T)
            Q_tile = Q[:, :, q_start:q_end, :]   # [B, H, tq, d]
            dO_tile = dO[:, :, q_start:q_end, :]
            lse_tile = logsumexp[:, :, q_start:q_end]   # [B, H, tq]
            D_tile = D[:, :, q_start:q_end]             # [B, H, tq]

            dQ_tile = torch.zeros_like(Q_tile)

            for kv_start in range(0, T, tile_size):
                kv_end = min(kv_start + tile_size, T)

                if causal and kv_start >= q_end:
                    break

                K_tile = K[:, :, kv_start:kv_end, :]
                V_tile = V[:, :, kv_start:kv_end, :]

                scores = torch.matmul(Q_tile, K_tile.transpose(-2, -1)) * scale  # [B, H, tq, tk]

                if causal:
                    q_indices = torch.arange(q_start, q_end, device=device).unsqueeze(1)
                    k_indices = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)
                    mask = k_indices > q_indices
                    scores = scores.masked_fill(mask, float("-inf"))

                # P = softmax recomputed from logsumexp
                P = torch.exp(scores - lse_tile.unsqueeze(-1))  # [B, H, tq, tk]

                # dV accumulation
                dV[:, :, kv_start:kv_end, :] += torch.matmul(P.transpose(-2, -1), dO_tile)

                # dP = dO @ V^T
                dP = torch.matmul(dO_tile, V_tile.transpose(-2, -1))  # [B, H, tq, tk]

                # dS = P * (dP - D_tile[..., None])
                dS = P * (dP - D_tile.unsqueeze(-1))  # [B, H, tq, tk]

                dQ_tile += torch.matmul(dS, K_tile) * scale
                dK[:, :, kv_start:kv_end, :] += torch.matmul(dS.transpose(-2, -1), Q_tile) * scale

            dQ[:, :, q_start:q_end, :] = dQ_tile

        return dQ, dK, dV, None, None, None


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class TiledAttentionModule(nn.Module):
    """
    Multi-head attention backed by the tiled forward/backward kernel.

    Args:
        d_model   : total model dimension
        n_heads   : number of attention heads
        tile_size : tile size for the tiled kernel (default 32)
        causal    : whether to apply a causal mask (default True)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        tile_size: int = 32,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.tile_size = tile_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, H, T, d_head]"""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, d_head] -> [B, T, d_model]"""
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        Q = self._split_heads(self.W_q(x))   # [B, H, T, d_head]
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        O = TiledAttention.apply(Q, K, V, self.scale, self.causal, self.tile_size)

        out = self._merge_heads(O)            # [B, T, d_model]
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Reference / equivalence checker
# ---------------------------------------------------------------------------

class AttentionEquivalenceChecker:
    """
    Provides a standard (non-tiled) attention implementation and a
    utility for comparing two output tensors.
    """

    @staticmethod
    def standard_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Reference full-matrix softmax attention.

        Args:
            Q, K, V : [B, H, T, d_head]
            causal  : bool

        Returns:
            [B, H, T, d_head]
        """
        d_head = Q.size(-1)
        scale = 1.0 / math.sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, T, T]

        if causal:
            T = Q.size(-2)
            mask = torch.triu(
                torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    @staticmethod
    def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
        """Maximum absolute difference between two tensors."""
        return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TiledAttentionConfig:
    d_model: int = 32
    n_heads: int = 4
    tile_size: int = 4
    causal: bool = True
