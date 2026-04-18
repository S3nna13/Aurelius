"""
LSTM-Augmented Transformer — hybrid architecture combining transformer
attention with recurrent LSTM memory.

Each LSTMTransformerBlock:
  1. Runs MemoryAugmentedAttention using the current LSTM hidden state as
     extra key/value tokens prepended to the sequence.
  2. Mean-pools the attended output and feeds it through an LSTMCell to
     produce a new memory state.
  3. Broadcasts the new memory hidden state back into the attended output
     via addition before the FFN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LSTMMemoryCell
# ---------------------------------------------------------------------------

class LSTMMemoryCell(nn.Module):
    """Wraps nn.LSTMCell with convenience helpers."""

    def __init__(self, d_model: int, d_memory: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory
        self.lstm_cell = nn.LSTMCell(d_model, d_memory)

    def forward(
        self,
        x: torch.Tensor,                        # [B, d_model]
        state: Tuple[torch.Tensor, torch.Tensor],  # (h, c) each [B, d_memory]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return (output [B, d_memory], new_state)."""
        h, c = state
        new_h, new_c = self.lstm_cell(x, (h, c))
        return new_h, (new_h, new_c)

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero (h, c) tensors on the same device as the cell weights."""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.d_memory, device=device)
        c = torch.zeros(batch_size, self.d_memory, device=device)
        return (h, c)


# ---------------------------------------------------------------------------
# MemoryAugmentedAttention
# ---------------------------------------------------------------------------

class MemoryAugmentedAttention(nn.Module):
    """
    Multi-head attention where an LSTM hidden state supplies one additional
    key/value token prepended to the sequence keys and values.
    """

    def __init__(self, d_model: int, n_heads: int, d_memory: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.memory_key = nn.Linear(d_memory, d_model, bias=False)
        self.memory_value = nn.Linear(d_memory, d_model, bias=False)

        self.scale = math.sqrt(self.d_head)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, S, d_model] -> [B, n_heads, S, d_head]"""
        B, S, _ = t.shape
        return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,          # [B, T, d_model]
        memory_h: torch.Tensor,   # [B, d_memory]
    ) -> torch.Tensor:            # [B, T, d_model]
        B, T, _ = x.shape

        Q = self._split_heads(self.W_q(x))           # [B, H, T, d_head]
        K_seq = self._split_heads(self.W_k(x))       # [B, H, T, d_head]
        V_seq = self._split_heads(self.W_v(x))       # [B, H, T, d_head]

        # Memory provides one extra K/V token  →  [B, 1, d_model]
        mem_k = self.memory_key(memory_h).unsqueeze(1)    # [B, 1, d_model]
        mem_v = self.memory_value(memory_h).unsqueeze(1)  # [B, 1, d_model]

        K_mem = self._split_heads(mem_k)   # [B, H, 1, d_head]
        V_mem = self._split_heads(mem_v)   # [B, H, 1, d_head]

        # Prepend memory token to sequence keys/values
        K = torch.cat([K_mem, K_seq], dim=2)   # [B, H, T+1, d_head]
        V = torch.cat([V_mem, V_seq], dim=2)   # [B, H, T+1, d_head]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T+1]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # [B, H, T, d_head]

        # Recombine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(attn_out)


# ---------------------------------------------------------------------------
# LSTMTransformerBlock
# ---------------------------------------------------------------------------

class LSTMTransformerBlock(nn.Module):
    """
    Single block:
      norm1 → MemoryAugmentedAttention → residual
      mean-pool → LSTMCell → new memory
      memory broadcast + norm2 → FFN → residual
    """

    def __init__(self, d_model: int, n_heads: int, d_memory: int) -> None:
        super().__init__()
        self.attn = MemoryAugmentedAttention(d_model, n_heads, d_memory)
        self.lstm = LSTMMemoryCell(d_model, d_memory)

        # FFN: 4× expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Project memory hidden state back to d_model for broadcasting
        self.d_memory = d_memory
        self.mem_proj = nn.Linear(d_memory, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,                           # [B, T, d_model]
        state: Tuple[torch.Tensor, torch.Tensor],  # (h, c) each [B, d_memory]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        h, _ = state

        # 1. Attention with memory as extra key/value
        attended = x + self.attn(self.norm1(x), h)        # [B, T, d_model]

        # 2. Update memory: mean-pool over sequence length
        pooled = attended.mean(dim=1)                      # [B, d_model]
        _, new_state = self.lstm(pooled, state)

        new_h, _ = new_state

        # 3. Broadcast new memory hidden state into attended output
        mem_broadcast = self.mem_proj(new_h).unsqueeze(1)  # [B, 1, d_model]
        mixed = attended + mem_broadcast                   # [B, T, d_model]

        # 4. FFN with residual
        out = mixed + self.ffn(self.norm2(mixed))          # [B, T, d_model]
        out = self.norm3(out)

        return out, new_state


# ---------------------------------------------------------------------------
# LSTMTransformerModel
# ---------------------------------------------------------------------------

class LSTMTransformerModel(nn.Module):
    """Full stacked LSTM-Augmented Transformer language model."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 4,
        n_heads: int = 4,
        d_memory: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_memory = d_memory

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [LSTMTransformerBlock(d_model, n_heads, d_memory) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def init_states(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a list of zero states, one per layer."""
        return [block.lstm.init_state(batch_size) for block in self.blocks]

    def forward(
        self,
        input_ids: torch.Tensor,                  # [B, T]
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            logits:     [B, T, vocab_size]
            new_states: list of (h, c) tuples, one per layer
        """
        B = input_ids.size(0)
        if states is None:
            states = self.init_states(B)

        x = self.embedding(input_ids)   # [B, T, d_model]

        new_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block, state in zip(self.blocks, states):
            x, new_state = block(x, state)
            new_states.append(new_state)

        logits = self.lm_head(x)        # [B, T, vocab_size]
        return logits, new_states

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard next-token prediction cross-entropy loss.
        input_ids: [B, T]  (uses input[:-1] as context, input[1:] as targets)
        """
        logits, _ = self.forward(input_ids)
        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss


# ---------------------------------------------------------------------------
# SegmentedTrainer
# ---------------------------------------------------------------------------

class SegmentedTrainer:
    """
    Trains an LSTMTransformerModel with Truncated Back-Propagation Through
    Time (TBPTT).  The recurrent state is carried forward across segments but
    detached so gradients do not flow through segment boundaries.
    """

    def __init__(
        self,
        model: LSTMTransformerModel,
        lr: float = 1e-3,
        segment_len: int = 32,
    ) -> None:
        self.model = model
        self.segment_len = segment_len
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_sequence(
        self,
        input_ids: torch.Tensor,   # [B, T_total]
        labels: torch.Tensor,      # [B, T_total]
    ) -> List[float]:
        """
        Split the sequence into segments of length `segment_len`, run a
        forward/backward pass on each segment, and return per-segment losses.

        The LSTM state is propagated between segments but detached from the
        computation graph at each boundary.
        """
        B, T_total = input_ids.shape
        states = self.model.init_states(B)
        segment_losses: List[float] = []

        seg_start = 0
        while seg_start < T_total - 1:
            seg_end = min(seg_start + self.segment_len, T_total)
            seg_ids = input_ids[:, seg_start:seg_end]   # [B, seg_len]
            seg_labels = labels[:, seg_start:seg_end]   # [B, seg_len]

            # Detach states to prevent gradient flow across segment boundary
            detached_states = [(h.detach(), c.detach()) for h, c in states]

            logits, new_states = self.model(seg_ids, detached_states)

            # Shift logits/labels inside the segment for next-token prediction
            if seg_ids.size(1) < 2:
                states = new_states
                seg_start = seg_end
                continue

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = seg_labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            segment_losses.append(loss.item())
            states = new_states
            seg_start = seg_end

        return segment_losses


# ---------------------------------------------------------------------------
# LSTMTransformerConfig
# ---------------------------------------------------------------------------

@dataclass
class LSTMTransformerConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_memory: int = 16
    segment_len: int = 16
