"""Recurrent Memory Transformer v3 (Bulatov et al. 2022).

Augments a transformer with a fixed set of recurrent memory tokens that persist
across segments, enabling unbounded context via recurrence.

v3 because recurrent_memory.py and recurrent_memory_v2.py already exist with
different APIs.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RecurrentMemoryTokens(nn.Module):
    """Learnable memory state -- a fixed set of persistent tokens.

    The parameter holds the initial (learned) memory prototype. At runtime
    the prototype is expanded to the requested batch size.
    """

    def __init__(self, n_memory: int, d_model: int) -> None:
        super().__init__()
        self.n_memory = n_memory
        self.d_model = d_model
        self.memory_tokens = nn.Parameter(torch.zeros(n_memory, d_model))

    def forward(self, batch_size: int) -> Tensor:
        """Expand memory prototype to (batch_size, n_memory, d_model)."""
        return self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def detach_memory(self, memory_state: Tensor) -> Tensor:
        """Return a copy of memory_state with no gradient history (TBPTT cut)."""
        return memory_state.detach()

    def memory_size(self) -> int:
        """Return the number of memory tokens."""
        return self.n_memory


class SegmentProcessor(nn.Module):
    """Process a single segment with prepended memory tokens.

    Wraps any transformer block that accepts (B, T, D) and returns (B, T, D).
    Memory tokens are prepended before the segment so the transformer can
    attend across both, then split back out.
    """

    def __init__(self, transformer_block: nn.Module, n_memory: int) -> None:
        super().__init__()
        self.transformer_block = transformer_block
        self.n_memory = n_memory

    def forward(self, segment: Tensor, memory_in: Tensor) -> tuple[Tensor, Tensor]:
        """Process one segment with memory prepended.

        Args:
            segment:   (B, T_seg, D) -- token embeddings for this segment.
            memory_in: (B, n_memory, D) -- memory from the previous segment.

        Returns:
            output:     (B, T_seg, D) -- processed segment tokens.
            memory_out: (B, n_memory, D) -- updated memory for the next segment.
        """
        combined = torch.cat([memory_in, segment], dim=1)  # (B, n_mem+T_seg, D)
        processed = self.transformer_block(combined)       # (B, n_mem+T_seg, D)
        memory_out = processed[:, : self.n_memory, :]     # (B, n_mem, D)
        output = processed[:, self.n_memory :, :]         # (B, T_seg, D)
        return output, memory_out


class RecurrentMemoryTransformer(nn.Module):
    """Full RMT model that processes long sequences in fixed-size segments.

    A learnable token embedding, n_layers SegmentProcessors (each wrapping a
    nn.TransformerEncoderLayer), and a linear LM head.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        n_memory: int = 4,
        segment_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_memory = n_memory
        self.segment_size = segment_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                SegmentProcessor(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        batch_first=True,
                        norm_first=True,
                    ),
                    n_memory=n_memory,
                )
                for _ in range(n_layers)
            ]
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.initial_memory = RecurrentMemoryTokens(n_memory, d_model)

    def forward(
        self,
        input_ids: Tensor,
        initial_memory: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Process input_ids in segments, threading memory through each.

        Args:
            input_ids:      (B, T) integer token ids.
            initial_memory: (B, n_memory, D) or None. When None the learned
                            initial memory tokens are used.

        Returns:
            logits:       (B, T, vocab_size)
            final_memory: (B, n_memory, D) -- detached memory after last segment.
        """
        B, T = input_ids.shape
        embeddings = self.embedding(input_ids)  # (B, T, D)

        if initial_memory is None:
            # Use learned initial memory, cloned to a contiguous tensor so that
            # numerical results are identical to passing the same values explicitly.
            proto = self.initial_memory(B).clone()
            memory_per_layer = [proto.clone() for _ in self.layers]
        else:
            memory_per_layer = [initial_memory.clone() for _ in self.layers]

        segments: list[Tensor] = []
        for start in range(0, T, self.segment_size):
            end = min(start + self.segment_size, T)
            segments.append(embeddings[:, start:end, :])

        all_outputs: list[Tensor] = []

        for seg in segments:
            x = seg
            new_memory_per_layer: list[Tensor] = []
            for layer_idx, layer in enumerate(self.layers):
                x, mem_out = layer(x, memory_per_layer[layer_idx])
                new_memory_per_layer.append(mem_out)
            # Truncated BPTT: cut gradient flow through memory across segments
            memory_per_layer = [m.detach() for m in new_memory_per_layer]
            all_outputs.append(x)

        hidden = torch.cat(all_outputs, dim=1)  # (B, T, D)
        logits = self.lm_head(hidden)            # (B, T, vocab_size)
        final_memory = memory_per_layer[-1]      # already detached

        return logits, final_memory


class MemoryUpdateGate(nn.Module):
    """Gated (GRU-inspired) memory update for controlled forgetting.

    Learns a per-slot mixture of old and new memory:
        g = sigmoid(Linear([old; new]))    shape (B, n_memory, 1)
        out = g * new_memory + (1-g) * old_memory
    """

    def __init__(self, d_model: int, n_memory: int) -> None:
        super().__init__()
        self.n_memory = n_memory
        self.gate_proj = nn.Linear(2 * d_model, 1, bias=True)

    def forward(self, old_memory: Tensor, new_memory: Tensor) -> Tensor:
        """Compute gated blend of old and new memory.

        Args:
            old_memory: (B, n_memory, D)
            new_memory: (B, n_memory, D)

        Returns:
            updated:    (B, n_memory, D)
        """
        combined = torch.cat([old_memory, new_memory], dim=-1)  # (B, n_mem, 2D)
        g = torch.sigmoid(self.gate_proj(combined))              # (B, n_mem, 1)
        return g * new_memory + (1.0 - g) * old_memory


class RMTEvaluator:
    """Utilities for evaluating memory utilization in a RecurrentMemoryTransformer."""

    def __init__(self) -> None:
        pass

    def memory_utilization(self, memory_states: list[Tensor]) -> float:
        """Measure how much memory changes across segments.

        mean(||memory[t] - memory[t-1]||_F) / (||memory[0]||_F + eps)

        Returns a non-negative float.
        """
        if len(memory_states) < 2:
            return 0.0

        diffs: list[float] = []
        for t in range(1, len(memory_states)):
            diff = (memory_states[t] - memory_states[t - 1]).norm(p="fro")
            diffs.append(diff.item())

        mean_diff = sum(diffs) / len(diffs)
        norm0 = memory_states[0].norm(p="fro").item()
        return mean_diff / (norm0 + 1e-8)

    def segment_dependency(
        self, model: RecurrentMemoryTransformer, input_ids: Tensor
    ) -> float:
        """Test whether memory actually influences the model output.

        Returns mean cosine similarity of logits(learned_memory) vs
        logits(zero_memory).  Range [-1, 1].
        """
        model.train(False)
        B = input_ids.shape[0]
        with torch.no_grad():
            logits_default, _ = model(input_ids, initial_memory=None)
            zeros = torch.zeros(
                B, model.n_memory, model.d_model, device=input_ids.device
            )
            logits_zeros, _ = model(input_ids, initial_memory=zeros)

        a = logits_default.reshape(-1, logits_default.shape[-1])
        b = logits_zeros.reshape(-1, logits_zeros.shape[-1])
        sim = F.cosine_similarity(a, b, dim=-1)
        return sim.mean().item()

    def cross_segment_perplexity(
        self,
        model: RecurrentMemoryTransformer,
        input_ids: Tensor,
        labels: Tensor,
    ) -> float:
        """Compute perplexity of the model on (input_ids, labels).

        Perplexity = exp(mean_cross_entropy).  Always >= 1.0.
        """
        model.train(False)
        with torch.no_grad():
            logits, _ = model(input_ids)
        B, T, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
        )
        return math.exp(ce_loss.item())
