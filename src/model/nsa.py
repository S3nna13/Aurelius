"""
Native Sparse Attention (NSA) — arXiv:2502.11089
"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"

Three complementary sparse attention patterns, following the paper's notation:
  G  — block size for compressed attention branch
  r  — number of top-r blocks selected for selected-token attention branch
  w  — sliding-window half-width for local attention branch

Variable naming mirrors the paper wherever possible:
  q, k, v        — query / key / value projections
  k_c, v_c       — compressed block keys / values (Compressed branch)
  k_s, v_s       — selected token keys / values (Selected branch)
  k_l, v_l       — local-window keys / values (Sliding-window branch)
  s_b            — block-level importance score
  alpha, beta, gamma — gating weights (sum to 1 via softmax)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention (NSA), arXiv:2502.11089.

    Parameters
    ----------
    d_model    : total model dimension
    n_heads    : number of attention heads
    block_size : G — tokens per block for compression (default 16)
    r_blocks   : r — number of top-r blocks for selected attention (default 4)
    window_size: w — local window total width; each query sees w//2 tokens on
                     each side (default 64)
    causal     : if True, apply causal (autoregressive) masking (default True)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 16,
        r_blocks: int = 4,
        window_size: int = 64,
        causal: bool = True,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.G = block_size  # paper: G
        self.r = r_blocks  # paper: r
        self.w = window_size  # paper: w
        self.causal = causal
        self.scale = math.sqrt(self.head_dim)

        # Standard QKV projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Block compression: maps G*head_dim → head_dim per head
        # One shared linear layer applied across all heads (head_dim*G → head_dim)
        self.compress_k = nn.Linear(self.head_dim * self.G, self.head_dim, bias=False)
        self.compress_v = nn.Linear(self.head_dim * self.G, self.head_dim, bias=False)

        # Gate MLP: head_dim → 3 → softmax  (applied per query position)
        self.gate_mlp = nn.Linear(self.head_dim, 3, bias=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, d_model) → (B, H, T, head_dim)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, head_dim) → (B, T, d_model)"""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def _pad_to_block_multiple(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, int]:
        """
        Pad key/value sequences along T so that T_padded % G == 0.
        Returns (k_padded, v_padded, original_T).
        Shape of k, v: (B, H, T, head_dim).
        """
        T = k.size(2)
        G = self.G
        remainder = T % G
        if remainder == 0:
            return k, v, T
        pad_len = G - remainder
        # Pad with zeros at the end of the sequence dimension
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        return k, v, T

    # ------------------------------------------------------------------
    # Branch 1: Compressed attention
    # ------------------------------------------------------------------

    def _compressed_branch(
        self,
        q: Tensor,  # (B, H, T, D)
        k: Tensor,  # (B, H, T, D)
        v: Tensor,  # (B, H, T, D)
        causal_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress K/V into n_blocks block tokens via learned linear projection,
        then compute attention from all queries to the compressed block tokens.

        Returns
        -------
        out_c   : (B, H, T, D) attention output
        k_c     : (B, H, n_blocks, D) compressed block keys (reused in selected branch)
        """
        B, H, T, D = q.shape
        G = self.G

        k_pad, v_pad, T_orig = self._pad_to_block_multiple(
            k,
            v,
        )
        T_pad = k_pad.size(2)
        n_blocks = T_pad // G  # paper: n_blocks

        # Reshape to blocks: (B, H, n_blocks, G, D)
        k_blocks = k_pad.view(B, H, n_blocks, G, D)
        v_blocks = v_pad.view(B, H, n_blocks, G, D)

        # Flatten last two dims for linear compression: (B, H, n_blocks, G*D)
        k_flat = k_blocks.reshape(B * H, n_blocks, G * D)
        v_flat = v_blocks.reshape(B * H, n_blocks, G * D)

        # Apply learned compression: (B*H, n_blocks, D)
        k_c = self.compress_k(k_flat).view(B, H, n_blocks, D)  # paper: k_c
        v_c = self.compress_v(v_flat).view(B, H, n_blocks, D)  # paper: v_c

        # Compressed attention: q (B,H,T,D) × k_c (B,H,n_blocks,D)
        # Build causal mask over block tokens if needed
        if self.causal:
            # Strict causal: query at position i can attend to compressed block b
            # only if ALL tokens in block b come strictly before i.
            # Block b covers positions [b*G .. (b+1)*G - 1].
            # Condition: (b+1)*G <= i  ↔  i >= (b+1)*G
            pos_i = torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
            block_end_excl = (
                torch.arange(n_blocks, device=q.device).unsqueeze(0) + 1
            ) * G  # (1, n_blocks)
            # True = allowed to attend
            c_mask = pos_i >= block_end_excl  # (T, n_blocks)
            attn_mask_c = torch.zeros(T, n_blocks, device=q.device)
            attn_mask_c = attn_mask_c.masked_fill(~c_mask, float("-inf"))
            attn_mask_c = attn_mask_c.unsqueeze(0).unsqueeze(0)  # (1,1,T,n_blocks)
        else:
            attn_mask_c = None

        out_c = F.scaled_dot_product_attention(
            q,
            k_c,
            v_c,
            attn_mask=attn_mask_c,
            dropout_p=0.0,
        )  # (B, H, T, D)

        # Positions with no valid compressed blocks (early positions in causal mode)
        # will produce NaN via softmax(-inf); replace with zero.
        out_c = torch.nan_to_num(out_c, nan=0.0)

        return out_c, k_c

    # ------------------------------------------------------------------
    # Branch 2: Selected-token attention
    # ------------------------------------------------------------------

    def _selected_branch(
        self,
        q: Tensor,  # (B, H, T, D)
        k: Tensor,  # (B, H, T, D)
        v: Tensor,  # (B, H, T, D)
        k_c: Tensor,  # (B, H, n_blocks, D)  from compressed branch
    ) -> Tensor:
        """
        For each query position, score blocks using compressed keys, select
        top-r blocks, and attend over their full-resolution tokens.

        Returns out_s : (B, H, T, D)
        """
        B, H, T, D = q.shape
        G = self.G

        k_pad, v_pad, _ = self._pad_to_block_multiple(k, v)
        T_pad = k_pad.size(2)
        n_blocks = T_pad // G

        r = min(self.r, n_blocks)  # clamp r to available blocks

        # Block-level importance: s_b[i, b] = q[i] · k_c[b]  (mean over head_dim scaled)
        # Shape: (B, H, T, n_blocks)
        s_b = torch.matmul(q, k_c.transpose(-1, -2)) / self.scale  # paper: s_b

        if self.causal:
            # Same strict condition as compressed branch: block b fully before i
            pos_i = torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
            block_end_excl = (
                torch.arange(n_blocks, device=q.device).unsqueeze(0) + 1
            ) * G  # (1, n_blocks)
            causal_ok = pos_i >= block_end_excl  # (T, n_blocks): True = allowed
            s_b = s_b.masked_fill(~causal_ok.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Top-r block indices per query: (B, H, T, r)
        top_r_scores, top_r_idx = torch.topk(s_b, r, dim=-1)

        # Gather token positions for selected blocks
        # For each selected block index b, positions are [b*G .. (b+1)*G - 1]
        # Build position offset: (r, G) → (T, r*G) via block indices
        # top_r_idx: (B, H, T, r) — expand to token positions
        offsets = torch.arange(G, device=q.device).view(1, 1, 1, 1, G)  # (1,1,1,1,G)
        top_r_idx_exp = top_r_idx.unsqueeze(-1) * G  # (B, H, T, r, 1)
        token_positions = (top_r_idx_exp + offsets).view(B, H, T, r * G)  # (B,H,T,r*G)
        token_positions = token_positions.clamp(0, T_pad - 1)

        # Gather k_s, v_s: (B, H, T, r*G, D)
        # k_pad: (B, H, T_pad, D)
        # Expand token_positions for gathering along dim 2
        tp_exp = token_positions.unsqueeze(-1).expand(B, H, T, r * G, D)
        # k_pad expanded: (B, H, T_pad, D) → need to index dim 2
        k_pad_exp = k_pad.unsqueeze(2).expand(B, H, T, T_pad, D)
        v_pad_exp = v_pad.unsqueeze(2).expand(B, H, T, T_pad, D)
        k_s = torch.gather(k_pad_exp, 3, tp_exp)  # (B, H, T, r*G, D)
        v_s = torch.gather(v_pad_exp, 3, tp_exp)

        # Compute attention: q (B,H,T,D) vs k_s (B,H,T,r*G,D)
        # Need to do per-query-position attention; reshape to (B*H*T, 1, D) vs (B*H*T, r*G, D)
        BHT = B * H * T
        q_flat = q.reshape(BHT, 1, D)
        k_s_flat = k_s.reshape(BHT, r * G, D)
        v_s_flat = v_s.reshape(BHT, r * G, D)

        if self.causal:
            # token_positions: (B, H, T, r*G) — absolute positions of gathered tokens
            # For each query position i, mask out gathered tokens with position > i
            torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
            # Reshape token_positions to (B*H*T, r*G) for broadcasting
            tp_flat = token_positions.reshape(BHT, r * G)  # (B*H*T, r*G)
            # Query position index for each entry in BHT dimension
            # BHT = B * H * T; the T index cycles fastest in the last dim
            # Reconstruct per-query position index
            t_idx = torch.arange(T, device=q.device).repeat(B * H)  # (B*H*T,)
            t_idx_exp = t_idx.unsqueeze(-1)  # (B*H*T, 1)
            # Causal mask: allow only gathered tokens with position <= query position
            causal_sel = tp_flat > t_idx_exp  # (B*H*T, r*G): True = future, mask out
            attn_mask_s = torch.zeros(BHT, 1, r * G, device=q.device)
            attn_mask_s = attn_mask_s.masked_fill(causal_sel.unsqueeze(1), float("-inf"))
        else:
            attn_mask_s = None

        out_s_flat = F.scaled_dot_product_attention(
            q_flat,
            k_s_flat,
            v_s_flat,
            attn_mask=attn_mask_s,
            dropout_p=0.0,
        )  # (B*H*T, 1, D)

        # When all gathered tokens are masked (e.g., first few positions with causal=True),
        # the softmax produces NaN; replace with zero (no-op contribution to gated sum).
        out_s_flat = torch.nan_to_num(out_s_flat, nan=0.0)

        out_s = out_s_flat.view(B, H, T, D)
        return out_s

    # ------------------------------------------------------------------
    # Branch 3: Sliding-window attention
    # ------------------------------------------------------------------

    def _sliding_window_branch(
        self,
        q: Tensor,  # (B, H, T, D)
        k: Tensor,  # (B, H, T, D)
        v: Tensor,  # (B, H, T, D)
    ) -> Tensor:
        """
        Standard local sliding-window attention of half-width w//2.
        Each query at position i attends to positions [i - w//2, i + w//2].
        Returns out_l : (B, H, T, D)
        """
        B, H, T, D = q.shape
        half = self.w // 2

        # Build (T, T) mask: True = attend
        pos = torch.arange(T, device=q.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
        window_mask = dist <= half

        if self.causal:
            causal_m = pos.unsqueeze(1) >= pos.unsqueeze(0)  # (T, T): i >= j
            window_mask = window_mask & causal_m

        attn_mask_l = torch.zeros(T, T, device=q.device)
        attn_mask_l = attn_mask_l.masked_fill(~window_mask, float("-inf"))
        attn_mask_l = attn_mask_l.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

        out_l = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_l,
            dropout_p=0.0,
        )  # (B, H, T, D)

        return out_l

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _compute_gates(self, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply gate MLP to per-position query features (mean-pooled over heads).
        Returns (alpha, beta, gamma), each (B, T, 1) broadcastable over heads/dim.

        Paper notation: gates = softmax(MLP(q̄))  where q̄ = mean over heads.
        """
        # q: (B, H, T, D) → mean over heads → (B, T, D)
        q_mean = q.mean(dim=1)  # paper: q̄

        # gate_mlp: (B, T, D) → (B, T, 3)
        g = self.gate_mlp(q_mean)
        g = F.softmax(g, dim=-1)  # (B, T, 3)

        alpha = g[..., 0:1]  # (B, T, 1)  compressed branch weight
        beta = g[..., 1:2]  # (B, T, 1)  selected branch weight
        gamma = g[..., 2:3]  # (B, T, 1)  sliding window branch weight
        return alpha, beta, gamma

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x               : (B, T, d_model)
        attention_mask  : optional (B, T) boolean / float mask (True = keep);
                          currently used to zero-out padded positions pre-projection.

        Returns
        -------
        (B, T, d_model)
        """
        B, T, _ = x.shape

        # Apply padding mask to input if provided
        if attention_mask is not None:
            # attention_mask: (B, T) — True means valid token
            # Cast to float and unsqueeze for broadcast
            mask_f = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
            x = x * mask_f

        # Project to Q, K, V
        q = self._split_heads(self.W_q(x))  # (B, H, T, D)
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # --- Branch 1: Compressed ---
        out_c, k_c = self._compressed_branch(q, k, v, causal_mask=None)

        # --- Branch 2: Selected ---
        out_s = self._selected_branch(q, k, v, k_c)

        # --- Branch 3: Sliding window ---
        out_l = self._sliding_window_branch(q, k, v)

        # --- Gated combination ---
        # alpha, beta, gamma: (B, T, 1); outputs after merge: (B, T, d_model)
        alpha, beta, gamma = self._compute_gates(q)

        out_c_m = self._merge_heads(out_c)  # (B, T, d_model)
        out_s_m = self._merge_heads(out_s)
        out_l_m = self._merge_heads(out_l)

        # Weighted sum: α·out_c + β·out_s + γ·out_l
        out = alpha * out_c_m + beta * out_s_m + gamma * out_l_m  # (B, T, d_model)

        return self.W_o(out)
