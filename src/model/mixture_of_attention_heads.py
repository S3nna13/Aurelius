"""
Mixture of Attention Heads (MoAH)

Routes tokens to different attention pattern experts:
  - LocalAttentionHead   : sliding-window attention
  - GlobalAttentionHead  : full causal attention
  - RelativePositionHead : causal attention with relative position bias
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual attention head types
# ---------------------------------------------------------------------------

class LocalAttentionHead(nn.Module):
    """Attends only within ±window_size positions (sliding window, causal)."""

    def __init__(self, d_model: int, d_head: int, window_size: int = 4) -> None:
        super().__init__()
        self.d_head = d_head
        self.window_size = window_size
        self.scale = math.sqrt(d_head)

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_head]
        """
        B, T, _ = x.shape

        q = self.W_q(x)  # [B, T, d_head]
        k = self.W_k(x)
        v = self.W_v(x)

        # Compute full attention scores then mask
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, T, T]

        # Build combined causal + window mask  (True = keep, False = mask out)
        idx = torch.arange(T, device=x.device)
        # causal: j <= i
        causal_mask = idx.unsqueeze(0) <= idx.unsqueeze(1)          # [T, T]
        # window: i - window_size <= j
        window_mask = idx.unsqueeze(0) >= (idx.unsqueeze(1) - self.window_size)  # [T, T]
        mask = causal_mask & window_mask  # [T, T]

        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        # Replace NaN rows (all -inf) with 0 — happens for position 0 with
        # an infinitely large negative window, but shouldn't occur normally.
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.bmm(attn, v)  # [B, T, d_head]
        return out


class GlobalAttentionHead(nn.Module):
    """Standard causal full-sequence attention head."""

    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.d_head = d_head
        self.scale = math.sqrt(d_head)

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_head]
        """
        B, T, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, T, T]

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)  # [B, T, d_head]
        return out


class RelativePositionHead(nn.Module):
    """
    Causal attention with additive relative position bias.

    r_ij = rel_emb(clip(j - i, -max_relative_pos, max_relative_pos) + max_relative_pos)
    The embedding produces a scalar that is added to the attention logit.
    """

    def __init__(self, d_model: int, d_head: int, max_relative_pos: int = 8) -> None:
        super().__init__()
        self.d_head = d_head
        self.max_relative_pos = max_relative_pos
        self.scale = math.sqrt(d_head)

        vocab_size = 2 * max_relative_pos + 1
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        # Scalar bias per relative position bucket
        self.rel_emb = nn.Embedding(vocab_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_head]
        """
        B, T, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, T, T]

        # Relative position indices: shape [T, T]
        i_idx = torch.arange(T, device=x.device).unsqueeze(1)   # [T, 1]
        j_idx = torch.arange(T, device=x.device).unsqueeze(0)   # [1, T]
        rel = j_idx - i_idx                                       # [T, T]
        rel_clipped = rel.clamp(-self.max_relative_pos, self.max_relative_pos)
        rel_indices = rel_clipped + self.max_relative_pos         # shift to [0, 2*max+1)

        # Relative bias: [T, T, 1] -> [T, T]
        rel_bias = self.rel_emb(rel_indices).squeeze(-1)          # [T, T]

        scores = scores + rel_bias.unsqueeze(0)  # broadcast over batch

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)  # [B, T, d_head]
        return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class HeadRouter(nn.Module):
    """
    Routes each token to the top-k head types.

    Returns softmax-normalized gates and the selected head-type indices.
    """

    def __init__(self, d_model: int, n_head_types: int, top_k: int = 2) -> None:
        super().__init__()
        self.n_head_types = n_head_types
        self.top_k = top_k
        self.router_linear = nn.Linear(d_model, n_head_types)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            gates:   [B, T, top_k]  — softmax-normalised routing weights
            indices: [B, T, top_k]  — head-type indices in [0, n_head_types)
        """
        logits = self.router_linear(x)          # [B, T, n_head_types]
        topk_logits, indices = torch.topk(logits, self.top_k, dim=-1)  # [B, T, top_k]
        gates = F.softmax(topk_logits, dim=-1)  # [B, T, top_k]
        return gates, indices


# ---------------------------------------------------------------------------
# Mixture of Attention Heads
# ---------------------------------------------------------------------------

class MixtureOfAttentionHeads(nn.Module):
    """
    Routes each token to top-k attention head types and returns a weighted
    combination of their outputs projected back to d_model.

    Head types (in order, index 0..2):
        0 — LocalAttentionHead
        1 — GlobalAttentionHead
        2 — RelativePositionHead
    """

    HEAD_TYPE_NAMES: List[str] = ["local", "global", "relative_position"]

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads_per_type: int = 2,
        top_k: int = 2,
        window_size: int = 4,
        max_relative_pos: int = 8,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.n_heads_per_type = n_heads_per_type
        self.top_k = top_k
        n_head_types = len(self.HEAD_TYPE_NAMES)

        # Multiple heads per type — we average within a type for simplicity
        self.local_heads = nn.ModuleList(
            [LocalAttentionHead(d_model, d_head, window_size) for _ in range(n_heads_per_type)]
        )
        self.global_heads = nn.ModuleList(
            [GlobalAttentionHead(d_model, d_head) for _ in range(n_heads_per_type)]
        )
        self.relative_heads = nn.ModuleList(
            [RelativePositionHead(d_model, d_head, max_relative_pos) for _ in range(n_heads_per_type)]
        )
        # Grouped for index-based access: head_types[i] is a ModuleList
        self.head_groups = nn.ModuleList([
            self.local_heads,
            self.global_heads,
            self.relative_heads,
        ])

        self.router = HeadRouter(d_model, n_head_types, top_k)
        self.W_o = nn.Linear(d_head, d_model)

        # Running stats for routing analysis (no grad)
        self.register_buffer(
            "_route_counts",
            torch.zeros(n_head_types),
            persistent=False,
        )
        self._total_tokens: int = 0

    def _run_head_type(self, type_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Average outputs of all heads of the given type."""
        heads = self.head_groups[type_idx]
        outs = torch.stack([h(x) for h in heads], dim=0)  # [n_heads, B, T, d_head]
        return outs.mean(dim=0)                             # [B, T, d_head]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape
        n_head_types = len(self.head_groups)

        gates, indices = self.router(x)  # [B, T, top_k], [B, T, top_k]

        # Pre-compute outputs for every head type (only those referenced)
        needed = indices.unique().tolist()
        head_outputs: Dict[int, torch.Tensor] = {}
        for ti in needed:
            head_outputs[int(ti)] = self._run_head_type(int(ti), x)  # [B, T, d_head]

        # Weighted combination
        out = torch.zeros(B, T, self.d_head, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            gate_k = gates[:, :, k].unsqueeze(-1)          # [B, T, 1]
            idx_k  = indices[:, :, k]                       # [B, T]
            for ti in range(n_head_types):
                mask = (idx_k == ti).unsqueeze(-1).float()  # [B, T, 1]
                if int(ti) in head_outputs:
                    out = out + gate_k * mask * head_outputs[int(ti)]

        # Update routing stats (detached, training + eval)
        with torch.no_grad():
            for ti in range(n_head_types):
                count = (indices == ti).sum().item()
                self._route_counts[ti] += count
            self._total_tokens += B * T * self.top_k

        return self.W_o(out)  # [B, T, d_model]

    def routing_stats(self) -> Dict[str, float]:
        """Return fraction of routing slots assigned to each head type."""
        total = float(self._total_tokens) if self._total_tokens > 0 else 1.0
        return {
            name: float(self._route_counts[i].item()) / total
            for i, name in enumerate(self.HEAD_TYPE_NAMES)
        }


# ---------------------------------------------------------------------------
# Transformer block and full language model
# ---------------------------------------------------------------------------

class MoAHTransformerBlock(nn.Module):
    """Pre-norm transformer block using MixtureOfAttentionHeads."""

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads_per_type: int = 2,
        top_k: int = 2,
        window_size: int = 4,
        max_relative_pos: int = 8,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.moah = MixtureOfAttentionHeads(
            d_model=d_model,
            d_head=d_head,
            n_heads_per_type=n_heads_per_type,
            top_k=top_k,
            window_size=window_size,
            max_relative_pos=max_relative_pos,
        )
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        x = x + self.moah(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MoAHLanguageModel(nn.Module):
    """
    Simple language model stacking MoAHTransformerBlocks.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        d_head: int = 8,
        n_heads_per_type: int = 2,
        top_k: int = 2,
        window_size: int = 4,
        max_relative_pos: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MoAHTransformerBlock(
                d_model=d_model,
                d_head=d_head,
                n_heads_per_type=n_heads_per_type,
                top_k=top_k,
                window_size=window_size,
                max_relative_pos=max_relative_pos,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]  (long)
        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.embedding(input_ids)   # [B, T, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)          # [B, T, vocab_size]

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Next-token prediction cross-entropy loss.

        Args:
            input_ids: [B, T]
        Returns:
            scalar loss tensor
        """
        logits = self.forward(input_ids)          # [B, T, V]
        # Shift: predict token t+1 from token t
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MoAHConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    d_head: int = 8
    n_heads_per_type: int = 2
    top_k: int = 2
    window_size: int = 4
    max_relative_pos: int = 8
