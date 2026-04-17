"""Token Merging (ToMe) v2 — Bolya et al. 2022 adapted for LLMs.

Reduces effective sequence length by merging similar tokens via bipartite
soft matching, enabling faster attention without architectural changes.

Reference: "Token Merging: Your ViT but Faster" (Bolya et al. 2022)
           https://arxiv.org/abs/2210.09461
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Bipartite Soft Matching
# ---------------------------------------------------------------------------

class BipartiteSoftMatching:
    """Match and merge similar tokens using bipartite soft matching.

    Tokens are split into two sets A (even positions) and B (odd positions).
    For each token in A the most similar token in B is found, and the top-r
    pairs (by cosine similarity) are merged by averaging.
    """

    def __init__(self, r: int) -> None:
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        self.r = r

    def match(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Find the top-r most-similar A–B token pairs with unique B assignment.

        Each B token can be matched to at most one A token, ensuring that
        exactly r B tokens are removed during merge (so merged length = T-r).

        Args:
            x: (B, T, D) token representations.

        Returns:
            merge_indices:   (B, r, 2) — (a_idx, b_idx) pairs to merge.
            unmerge_weights: (B, T, 1) — 1/count for unmerging.
        """
        B, T, D = x.shape
        r = min(self.r, T // 2)  # cannot merge more than T//2 pairs

        # Split into even (A) and odd (B) positions
        x_a = x[:, 0::2, :]  # (B, T_a, D)  T_a = ceil(T/2)
        x_b = x[:, 1::2, :]  # (B, T_b, D)  T_b = floor(T/2)

        T_a = x_a.shape[1]
        T_b = x_b.shape[1]

        # Cosine similarity matrix (B, T_a, T_b)
        a_norm = F.normalize(x_a, p=2, dim=-1)
        b_norm = F.normalize(x_b, p=2, dim=-1)
        sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, T_a, T_b)

        # Greedy matching: select top-r (a, b) pairs from the full similarity
        # matrix, ensuring each B index is used at most once per batch item.
        # Flatten (T_a * T_b) similarities, sort descending, pick greedily.
        sim_flat = sim.view(B, T_a * T_b)  # (B, T_a*T_b)
        # Sort all pairs by similarity (descending) for each batch item
        sorted_scores, sorted_flat_idx = sim_flat.sort(dim=1, descending=True)

        top_a_idx_list = []
        top_b_idx_list = []

        for bi in range(B):
            used_a = set()
            used_b = set()
            pairs_a = []
            pairs_b = []
            for flat_i in sorted_flat_idx[bi]:
                flat_i_val = flat_i.item()
                ai = flat_i_val // T_b
                bi_tok = flat_i_val % T_b
                if ai not in used_a and bi_tok not in used_b:
                    pairs_a.append(ai)
                    pairs_b.append(bi_tok)
                    used_a.add(ai)
                    used_b.add(bi_tok)
                    if len(pairs_a) == r:
                        break
            top_a_idx_list.append(torch.tensor(pairs_a, dtype=torch.long, device=x.device))
            top_b_idx_list.append(torch.tensor(pairs_b, dtype=torch.long, device=x.device))

        top_a_idx = torch.stack(top_a_idx_list, dim=0)  # (B, r)
        top_b_idx = torch.stack(top_b_idx_list, dim=0)  # (B, r)

        # merge_indices: (B, r, 2)
        merge_indices = torch.stack([top_a_idx, top_b_idx], dim=2)  # (B, r, 2)

        # Build unmerge weights: tokens involved in a merge have count=2, rest=1
        # Positions in A: 2*a_idx; positions in B: 2*b_idx+1
        counts = x.new_ones(B, T)  # (B, T)
        a_orig = top_a_idx * 2          # (B, r)  — original positions in A
        b_orig = top_b_idx * 2 + 1      # (B, r)  — original positions in B

        # Mark merged positions with count 2
        for bi in range(B):
            counts[bi].scatter_(0, a_orig[bi],
                                2.0 * torch.ones(r, dtype=counts.dtype, device=x.device))
            counts[bi].scatter_(0, b_orig[bi],
                                2.0 * torch.ones(r, dtype=counts.dtype, device=x.device))

        unmerge_weights = (1.0 / counts).unsqueeze(-1)  # (B, T, 1)

        return merge_indices, unmerge_weights

    def merge(self, x: Tensor, merge_indices: Tensor) -> Tensor:
        """Average the r matched pairs and keep unmatched tokens.

        Args:
            x:             (B, T, D) original tokens.
            merge_indices: (B, r, 2) pairs of (a_idx, b_idx).

        Returns:
            merged_x: (B, T-r, D)
                      r merged tokens + (T - 2r) unmatched tokens = T-r total.
        """
        B, T, D = x.shape
        r = merge_indices.shape[1]

        if r == 0:
            return x

        # Build a boolean mask of which positions get merged away (removed).
        # Strategy: keep even-half tokens always; for matched A tokens, replace
        # with average; remove matched B tokens.
        #
        # More precisely:
        #   - merged token = mean(x_a[a_idx], x_b[b_idx])
        #   - unmatched A tokens stay as-is
        #   - unmatched B tokens stay as-is
        #   - matched B tokens are removed (absorbed into their A partner)
        #
        # We rebuild the sequence as:
        #   [modified_A_tokens  |  unmatched_B_tokens]
        # where modified_A = original A, except a_idx positions become averages.

        # --- even (A) and odd (B) subsets ---
        x_a = x[:, 0::2, :]  # (B, T_a, D)
        x_b = x[:, 1::2, :]  # (B, T_b, D)

        T_a = x_a.shape[1]
        T_b = x_b.shape[1]

        a_idx = merge_indices[:, :, 0]  # (B, r)
        b_idx = merge_indices[:, :, 1]  # (B, r)

        # Expand indices for gathering
        a_idx_exp = a_idx.unsqueeze(-1).expand(B, r, D)  # (B, r, D)
        b_idx_exp = b_idx.unsqueeze(-1).expand(B, r, D)  # (B, r, D)

        matched_a = x_a.gather(1, a_idx_exp)  # (B, r, D)
        matched_b = x_b.gather(1, b_idx_exp)  # (B, r, D)
        merged = (matched_a + matched_b) * 0.5  # (B, r, D)

        # Write merged values back into x_a at a_idx positions
        x_a_mod = x_a.clone()
        x_a_mod.scatter_(1, a_idx_exp, merged)

        # Remove matched B positions — build boolean keep mask for B
        keep_b_mask = torch.ones(B, T_b, dtype=torch.bool, device=x.device)
        for bi in range(B):
            keep_b_mask[bi].scatter_(0, b_idx[bi],
                                     torch.zeros(r, dtype=torch.bool, device=x.device))

        # Gather unmatched B tokens per batch item
        # Pad to uniform shape using a simple loop (T_b - r unmatched per item)
        unmatched_b_list = []
        for bi in range(B):
            unmatched_b_list.append(x_b[bi][keep_b_mask[bi]])  # (T_b-r, D)
        unmatched_b = torch.stack(unmatched_b_list, dim=0)  # (B, T_b-r, D)

        # Concatenate: [all A (with merged values) | unmatched B]
        # Total tokens: T_a + (T_b - r) = T - r  ✓  (when T_a = T_b = T//2)
        merged_x = torch.cat([x_a_mod, unmatched_b], dim=1)  # (B, T-r, D)

        return merged_x


# ---------------------------------------------------------------------------
# Token Unmerger
# ---------------------------------------------------------------------------

class TokenUnmerger:
    """Restore original sequence length after processing merged tokens."""

    def __init__(self) -> None:
        pass

    def unmerge(
        self,
        merged_x: Tensor,
        merge_indices: Tensor,
        original_T: int,
        unmerge_weights: Tensor,
    ) -> Tensor:
        """Scatter merged tokens back to their original positions.

        Args:
            merged_x:        (B, T-r, D) processed merged tokens.
            merge_indices:   (B, r, 2)  (a_idx, b_idx) pairs.
            original_T:      T — original sequence length.
            unmerge_weights: (B, T, 1) weights (unused here; averaging handled
                              by copying the merged token to both positions).

        Returns:
            output: (B, original_T, D)
        """
        B, T_merged, D = merged_x.shape
        r = merge_indices.shape[1]

        # Reconstruct original positions.
        # merged_x layout: [x_a_mod (T_a tokens) | unmatched_b (T_b-r tokens)]
        # where T_a = ceil(original_T / 2), T_b = floor(original_T / 2)
        T_a = math.ceil(original_T / 2)
        T_b = original_T // 2

        x_a_mod = merged_x[:, :T_a, :]        # (B, T_a, D)
        unmatched_b = merged_x[:, T_a:, :]    # (B, T_b-r, D)

        a_idx = merge_indices[:, :, 0]  # (B, r)
        b_idx = merge_indices[:, :, 1]  # (B, r)

        # Reconstruct x_b: place unmatched tokens back, fill matched slots with
        # the corresponding merged (a_mod) token.
        x_b_restored = torch.zeros(B, T_b, D, device=merged_x.device,
                                   dtype=merged_x.dtype)

        # Build keep_b_mask to find where unmatched B slots are
        keep_b_mask = torch.ones(B, T_b, dtype=torch.bool, device=merged_x.device)
        for bi in range(B):
            keep_b_mask[bi].scatter_(0, b_idx[bi],
                                     torch.zeros(r, dtype=torch.bool,
                                                 device=merged_x.device))

        # Place unmatched B tokens
        for bi in range(B):
            x_b_restored[bi][keep_b_mask[bi]] = unmatched_b[bi]

        # Place merged tokens at matched B positions (copy from A's merged value)
        a_idx_exp = a_idx.unsqueeze(-1).expand(B, r, D)
        merged_vals = x_a_mod.gather(1, a_idx_exp)  # (B, r, D) — merged value at A

        for bi in range(B):
            x_b_restored[bi].scatter_(0,
                                      b_idx[bi].unsqueeze(-1).expand(r, D),
                                      merged_vals[bi])

        # Interleave A and B back into original positions
        output = torch.zeros(B, original_T, D, device=merged_x.device,
                             dtype=merged_x.dtype)
        output[:, 0::2, :] = x_a_mod
        output[:, 1::2, :] = x_b_restored

        return output


# ---------------------------------------------------------------------------
# ToMe Attention
# ---------------------------------------------------------------------------

class ToMeAttention(nn.Module):
    """Multi-head self-attention with Token Merging integrated.

    Merges r token pairs before attention, runs attention on the shorter
    sequence, then unmerges back to the original length.
    """

    def __init__(self, d_model: int, n_heads: int, r: int = 2) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.r = r
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _attention(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """
        Args:
            x: (B, T, D)

        Returns:
            output:     (B, T, D)  — same shape as input
            merge_info: dict with keys 'original_T', 'merged_T', 'compression_ratio'
        """
        B, T, D = x.shape
        r = min(self.r, T // 2)

        if r == 0:
            output = self._attention(x)
            merge_info = {
                'original_T': T,
                'merged_T': T,
                'compression_ratio': 1.0,
            }
            return output, merge_info

        matcher = BipartiteSoftMatching(r)
        merge_indices, unmerge_weights = matcher.match(x)

        # Merge tokens → shorter sequence
        merged_x = matcher.merge(x, merge_indices)  # (B, T-r, D)
        T_merged = merged_x.shape[1]

        # Attention on compressed sequence
        attn_out = self._attention(merged_x)  # (B, T-r, D)

        # Unmerge back to original length
        unmerger = TokenUnmerger()
        output = unmerger.unmerge(attn_out, merge_indices, T, unmerge_weights)  # (B, T, D)

        merge_info = {
            'original_T': T,
            'merged_T': T_merged,
            'compression_ratio': T_merged / T,
        }
        return output, merge_info


# ---------------------------------------------------------------------------
# ToMe Transformer Block
# ---------------------------------------------------------------------------

class ToMeTransformerBlock(nn.Module):
    """Full transformer block using ToMe attention."""

    def __init__(self, d_model: int, n_heads: int, r: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ToMeAttention(d_model, n_heads, r)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, float]:
        """
        Args:
            x: (B, T, D)

        Returns:
            output:            (B, T, D)
            compression_ratio: float in (0, 1] (1.0 when r=0)
        """
        attn_out, merge_info = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, merge_info['compression_ratio']


# ---------------------------------------------------------------------------
# ToMe Full Model
# ---------------------------------------------------------------------------

class ToMeModel(nn.Module):
    """Full transformer with Token Merging applied at every layer."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        r_per_layer: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [ToMeTransformerBlock(d_model, n_heads, r_per_layer) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tuple[Tensor, float]:
        """
        Args:
            input_ids: (B, T) integer token ids

        Returns:
            logits:                (B, T, vocab_size)
            mean_compression_ratio: float
        """
        x = self.embedding(input_ids)
        compression_ratios = []
        for block in self.blocks:
            x, cr = block(x)
            compression_ratios.append(cr)
        x = self.norm(x)
        logits = self.lm_head(x)
        mean_cr = sum(compression_ratios) / len(compression_ratios)
        return logits, mean_cr

    def set_r(self, r: int) -> None:
        """Update r in all ToMeAttention blocks (for inference-time tuning)."""
        for block in self.blocks:
            block.attn.r = r


# ---------------------------------------------------------------------------
# Efficiency Analyzer
# ---------------------------------------------------------------------------

class ToMeEfficiencyAnalyzer:
    """Analyze theoretical efficiency gains from token merging."""

    def __init__(self) -> None:
        pass

    def theoretical_speedup(
        self, T: int, r_per_layer: int, n_layers: int
    ) -> float:
        """Compute attention-FLOPs speedup: T_initial / T_final.

        Attention cost scales as O(T^2), so speedup = (T_initial / T_final)^2
        where T_final is the sequence length at the last layer.

        Returns:
            speedup: float >= 1.0
        """
        T_current = T
        total_initial = 0.0
        total_final = 0.0
        for _ in range(n_layers):
            total_initial += T ** 2  # always compare against base T
            total_final += T_current ** 2
            T_current = max(1, T_current - r_per_layer)
        # Return ratio of initial to final cumulative attention cost
        if total_final == 0:
            return float('inf')
        return total_initial / total_final

    def compression_per_layer(
        self, T: int, r: int, n_layers: int
    ) -> List[int]:
        """Return sequence length after each layer.

        Args:
            T:        initial sequence length
            r:        tokens merged per layer
            n_layers: number of layers

        Returns:
            lengths: list of n_layers integers (non-increasing)
        """
        lengths = []
        T_current = T
        for _ in range(n_layers):
            T_current = max(1, T_current - r)
            lengths.append(T_current)
        return lengths

    def flop_reduction(
        self, T: int, r: int, d_model: int, n_heads: int
    ) -> float:
        """Fraction of attention FLOPs saved in one layer by merging r pairs.

        Attention cost ~ T^2 * d_model (QK^T is the dominant term).

        Returns:
            fraction_saved: float in (0, 1)
        """
        T_merged = max(1, T - r)
        flops_original = T ** 2 * d_model
        flops_merged = T_merged ** 2 * d_model
        saved = (flops_original - flops_merged) / flops_original
        return float(saved)
