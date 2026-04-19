"""Lambda Networks attention (Bello 2021, arXiv:2102.08602).

Lambda layers compute "lambda functions" from context, then apply them to
queries. This yields linear complexity O(n * |m|) instead of the quadratic
O(n * m) self-attention softmax, by avoiding the pairwise attention map.

Given inputs X ∈ R^{B, S, D}, we derive:
    Q = X W_Q   shape [B, S, |k| * H]  (split into H heads of size |k|)
    K = X W_K   shape [B, S, |k|]
    V = X W_V   shape [B, S, |v|]

Apply BatchNorm on V (and optionally on Q), softmax K across the sequence
dimension (context), then form:

    content_lambda  = K_softmax^T @ V                 [B, |k|, |v|]
    position_lambda = E @ V    (E ∈ R^{S, S, |k|})    [B, S, |k|, |v|]

Output per head: y_n = Q_n @ (content_lambda + position_lambda_n) ∈ R^{|v|}.
Concatenate H heads and project back to D.

Complexity: O(B * S * |k| * |v|) for the content path, and
O(B * S * |k| * |v|) for the position path given a learned relative position
embedding E of size [S, S, |k|] (no quadratic softmax over sequence).

This module is self-contained and pure PyTorch (no einops).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaAttention(nn.Module):
    """Lambda layer (intra-context, global receptive field).

    Args:
        d_model: model dimension D (= input & output channels).
        n_heads: number of query heads H. Each head shares the same K, V.
        head_dim_key: |k|, per-head key dimension.
        head_dim_value: |v|, per-head value dimension. Must satisfy
            ``d_model == n_heads * head_dim_value`` so we can project back.
        n_positions: maximum sequence length the positional lambda supports.
            Sequence S <= n_positions at forward time; S > n_positions raises.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim_key: int,
        head_dim_value: int,
        n_positions: int,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or head_dim_key <= 0 \
                or head_dim_value <= 0 or n_positions <= 0:
            raise ValueError(
                "d_model, n_heads, head_dim_key, head_dim_value, n_positions "
                "must all be positive integers."
            )
        if d_model != n_heads * head_dim_value:
            raise ValueError(
                f"d_model ({d_model}) must equal n_heads * head_dim_value "
                f"({n_heads} * {head_dim_value} = {n_heads * head_dim_value})."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = head_dim_key
        self.dv = head_dim_value
        self.n_positions = n_positions

        # Linear projections (no bias, as in the paper).
        self.to_q = nn.Linear(d_model, n_heads * head_dim_key, bias=False)
        self.to_k = nn.Linear(d_model, head_dim_key, bias=False)
        self.to_v = nn.Linear(d_model, head_dim_value, bias=False)

        # BatchNorm on Q and V along the channel axis, per paper.
        self.norm_q = nn.BatchNorm1d(n_heads * head_dim_key)
        self.norm_v = nn.BatchNorm1d(head_dim_value)

        # Learned relative position embedding E ∈ R^{S_max, S_max, |k|}.
        # Used as E @ V to form the positional lambda per query position.
        # Zero-init so that at init positional lambda contributes nothing;
        # the layer then starts as pure content lambda.
        self.pos_embed = nn.Parameter(
            torch.zeros(n_positions, n_positions, head_dim_key)
        )

        # Final projection back to d_model. Zero-init weight so that, at
        # initialisation, the lambda layer outputs zeros — which makes it a
        # pass-through when wrapped by a residual connection.
        self.to_out = nn.Linear(n_heads * head_dim_value, d_model, bias=False)
        nn.init.zeros_(self.to_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the lambda layer.

        Args:
            x: input tensor of shape [B, S, D].

        Returns:
            Tensor of shape [B, S, D] with the same dtype as ``x``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"LambdaAttention expects a 3-D [B, S, D] tensor, got shape "
                f"{tuple(x.shape)}."
            )
        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"Expected channel dim {self.d_model}, got {D}."
            )
        if S > self.n_positions:
            raise ValueError(
                f"Sequence length {S} exceeds n_positions={self.n_positions}. "
                "Increase n_positions or truncate the input."
            )

        H, Dk, Dv = self.n_heads, self.dk, self.dv
        in_dtype = x.dtype

        # Project Q, K, V.
        q = self.to_q(x)                           # [B, S, H*Dk]
        k = self.to_k(x)                           # [B, S, Dk]
        v = self.to_v(x)                           # [B, S, Dv]

        # BatchNorm expects [B, C, S] layout.
        q = self.norm_q(q.transpose(1, 2)).transpose(1, 2)
        v = self.norm_v(v.transpose(1, 2)).transpose(1, 2)

        # Reshape Q into heads: [B, S, H, Dk]
        q = q.view(B, S, H, Dk)

        # Softmax K over the sequence dim so columns of K_soft sum to 1.
        k_soft = F.softmax(k, dim=1)               # [B, S, Dk]

        # Content lambda: [B, Dk, Dv] = K_soft^T @ V
        content_lambda = torch.matmul(k_soft.transpose(1, 2), v)

        # Position lambda: use the [S, S, Dk] top-left slice of pos_embed.
        # E_n @ V gives per-position [B, S, Dk, Dv] after einsum-style matmul.
        # We compute it via a single batched matmul:
        #   E : [S, S, Dk]  (rows = query pos n, cols = context pos m)
        #   V : [B, S, Dv]
        # out[b, n, k, v] = sum_m E[n, m, k] * V[b, m, v]
        E = self.pos_embed[:S, :S, :]              # [S, S, Dk]
        # Fold (n, k) -> [S*Dk, S] to leverage matmul.
        E_mat = E.permute(0, 2, 1).reshape(S * Dk, S)   # [S*Dk, S]
        # V: [B, S, Dv] -> content-lambda-like product
        pos = torch.matmul(E_mat, v)                     # [B, S*Dk, Dv]
        position_lambda = pos.view(B, S, Dk, Dv)         # [B, S, Dk, Dv]

        # Apply lambdas to Q.
        # content term: y_c[b, n, h, v] = sum_k Q[b, n, h, k] * C[b, k, v]
        # Expand C to [B, 1, Dk, Dv] and contract over Dk.
        content_out = torch.matmul(q, content_lambda.unsqueeze(1))
        # shape: [B, S, H, Dv]

        # position term: y_p[b, n, h, v] = sum_k Q[b, n, h, k] * P[b, n, k, v]
        # Broadcast position_lambda over H: [B, S, 1, Dk, Dv], Q -> [B,S,H,1,Dk]
        position_out = torch.matmul(
            q.unsqueeze(-2),                       # [B, S, H, 1, Dk]
            position_lambda.unsqueeze(2),          # [B, S, 1, Dk, Dv]
        ).squeeze(-2)                              # [B, S, H, Dv]

        y = content_out + position_out             # [B, S, H, Dv]
        y = y.reshape(B, S, H * Dv)
        y = self.to_out(y)
        return y.to(in_dtype)
