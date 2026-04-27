"""Recurrent Memory Transformer (RMT): learnable memory tokens that persist across segments."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RMTConfig:
    """Configuration for Recurrent Memory Transformer."""

    n_memory_tokens: int = 16
    segment_size: int = 512
    memory_layers: list[int] = field(default_factory=lambda: [0, -1])
    memory_dim: int = 64
    detach_memory: bool = False


class MemoryTokens(nn.Module):
    """Learnable memory token bank that persists across segments."""

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.memory = nn.Parameter(torch.empty(1, n_tokens, d_model))
        nn.init.normal_(self.memory, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        """Return memory expanded to (B, n_tokens, d_model)."""
        return self.memory.expand(batch_size, -1, -1)

    def update(self, new_memory: Tensor) -> None:
        """Detach and store updated memory state for recurrence.

        Args:
            new_memory: (B, n_tokens, d_model) updated hidden states.
        """
        self.memory.data = new_memory.mean(0, keepdim=True).detach()


def segment_sequence(input_ids: Tensor, segment_size: int) -> list[Tensor]:
    """Split (B, T) input into list of (B, seg_size) segments.

    The last segment may be shorter than segment_size.

    Args:
        input_ids: (B, T) token ids.
        segment_size: Maximum tokens per segment.

    Returns:
        List of (B, seg_len) tensors.
    """
    T = input_ids.shape[1]
    segments = []
    for start in range(0, T, segment_size):
        end = min(start + segment_size, T)
        segments.append(input_ids[:, start:end])
    return segments


class RMTWrapper(nn.Module):
    """Wraps an AureliusTransformer with Recurrent Memory Transformer logic.

    Memory tokens are prepended to each segment's embeddings. After processing,
    the last n_memory_tokens hidden states are extracted as memory for the next
    segment, enabling long-context processing via recurrence.
    """

    def __init__(self, base_model: nn.Module, config: RMTConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        d_model = base_model.config.d_model
        self.memory_tokens = MemoryTokens(config.n_memory_tokens, d_model)

    def process_segment(self, seg_ids: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Process a single segment with memory prepended.

        Args:
            seg_ids: (B, S) token ids for this segment.
            memory: (B, n_memory_tokens, d_model) current memory state.

        Returns:
            logits_for_seg: (B, S, vocab_size) — logits for segment tokens only.
            new_memory_hidden: (B, n_memory_tokens, d_model) — updated memory.
        """
        B, S = seg_ids.shape
        n_mem = self.config.n_memory_tokens

        # Embed segment tokens
        seg_embed = self.base_model.embed(seg_ids)  # (B, S, d_model)

        # Concatenate memory tokens before segment embeddings
        # x shape: (B, n_mem + S, d_model)
        x = torch.cat([memory, seg_embed], dim=1)

        total_len = n_mem + S
        # Get RoPE frequencies for total sequence length
        freqs_cis = self.base_model.freqs_cis[:total_len]

        # Run through transformer layers
        for layer in self.base_model.layers:
            x, _, _ = layer(x, freqs_cis, mask=None, past_kv=None)

        # Apply final norm
        x = self.base_model.norm(x)

        # Apply LM head to get logits
        all_logits = self.base_model.lm_head(x)  # (B, n_mem + S, vocab_size)

        # Extract logits for segment tokens only (skip memory positions)
        logits_for_seg = all_logits[:, n_mem:, :]  # (B, S, vocab_size)

        # Extract memory hidden states (last n_mem positions before LM head)
        # Re-run norm on raw hidden states; we already have x = norm(x)
        # so just slice x for memory
        new_memory_hidden = x[:, :n_mem, :]  # (B, n_mem, d_model)

        return logits_for_seg, new_memory_hidden

    def forward(self, input_ids: Tensor) -> tuple[None, Tensor, list]:
        """Process input in segments with recurrent memory.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            Tuple of (None, full_logits, []) matching (loss, logits, pkv) API.
            full_logits: (B, T, vocab_size) concatenated over all segments.
        """
        B = input_ids.shape[0]

        # Split input into segments
        segments = segment_sequence(input_ids, self.config.segment_size)

        # Initialize memory from parameter
        memory = self.memory_tokens(B)  # (B, n_mem, d_model)

        all_logits = []
        for seg in segments:
            logits_seg, new_memory = self.process_segment(seg, memory)
            all_logits.append(logits_seg)

            # Update memory for next segment
            if self.config.detach_memory:
                memory = new_memory.detach()
            else:
                memory = new_memory

        # Update stored memory state for future calls
        self.memory_tokens.update(memory)

        full_logits = torch.cat(all_logits, dim=1)  # (B, T, vocab_size)
        return None, full_logits, []


def compute_memory_utilization(memory_before: Tensor, memory_after: Tensor) -> float:
    """Compute cosine similarity between memory states to measure memory change.

    A value of 1.0 means the memory is unchanged (not being used).
    A value of 0.0 means the memory has fully changed (maximally used).

    Args:
        memory_before: Tensor of any shape, flattened for comparison.
        memory_after: Tensor of same shape as memory_before.

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    before_flat = memory_before.reshape(-1).float()
    after_flat = memory_after.reshape(-1).float()
    similarity = F.cosine_similarity(before_flat.unsqueeze(0), after_flat.unsqueeze(0))
    return similarity.item()
