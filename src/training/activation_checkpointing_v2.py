"""Activation checkpointing and recomputation utilities for memory-efficient training.

Trades compute for memory by recomputing activations during backward pass
instead of storing them, enabling training of larger models on limited hardware.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor


class CheckpointedModule(nn.Module):
    """Wrap a module to use gradient checkpointing during training.

    During training, activations are recomputed in the backward pass rather
    than stored, reducing peak memory at the cost of extra compute.
    During eval, checkpointing is skipped since memory saving is not needed.
    """

    def __init__(self, module: nn.Module, use_reentrant: bool = False) -> None:
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        if not self.training:
            return self.module(*args, **kwargs)
        # torch.utils.checkpoint does not support kwargs directly with
        # use_reentrant=False in all versions, so we fuse kwargs via closure.
        if kwargs:
            def _run(*a):
                return self.module(*a, **kwargs)
            return torch.utils.checkpoint.checkpoint(
                _run, *args, use_reentrant=self.use_reentrant
            )
        return torch.utils.checkpoint.checkpoint(
            self.module, *args, use_reentrant=self.use_reentrant
        )

    def memory_saved_estimate(self, input_shape: tuple) -> float:
        """Estimate bytes saved by not storing activations for this module.

        Approximation: numel(input) * 4 bytes (float32) * 0.8 heuristic factor.
        """
        numel = 1
        for dim in input_shape:
            numel *= dim
        return float(numel) * 4.0 * 0.8


class SegmentedCheckpointing(nn.Module):
    """Checkpoint transformer layers organised into contiguous segments.

    Divides an nn.ModuleList of layers into n_segments groups and wraps each
    group in a single checkpoint call, balancing memory savings and recompute
    overhead.
    """

    def __init__(self, layers: nn.ModuleList, n_segments: int = 2) -> None:
        super().__init__()
        self.layers = layers
        n_layers = len(layers)
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")
        self.n_segments = min(n_segments, n_layers)

    def _make_segment_fn(self, start: int, end: int):
        """Return a callable that runs layers[start:end] sequentially."""
        def _segment_fn(x: Tensor) -> Tensor:
            for i in range(start, end):
                x = self.layers[i](x)
            return x
        return _segment_fn

    def forward(self, x: Tensor) -> Tensor:
        n_layers = len(self.layers)
        boundaries = self.segment_boundaries()

        for seg_idx, start in enumerate(boundaries):
            end = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else n_layers
            seg_fn = self._make_segment_fn(start, end)
            if self.training:
                x = torch.utils.checkpoint.checkpoint(seg_fn, x, use_reentrant=False)
            else:
                x = seg_fn(x)
        return x

    def segment_boundaries(self) -> List[int]:
        """Return layer indices where each segment starts.

        Always starts with 0; length equals n_segments.
        """
        n_layers = len(self.layers)
        boundaries = []
        for seg in range(self.n_segments):
            idx = int(math.floor(seg * n_layers / self.n_segments))
            boundaries.append(idx)
        return boundaries


class ActivationMemoryEstimator:
    """Estimate activation memory for transformer layers.

    Provides per-layer and whole-model estimates, with and without checkpointing.
    """

    def __init__(self) -> None:
        pass

    def estimate_transformer_layer(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
    ) -> int:
        """Estimate activation memory (bytes) for one transformer layer.

        Components:
          - Attention Q/K maps: 2 * B * H * T * T * 4 bytes
          - FFN hidden states: 2 * B * T * 4*D * 4 bytes
          - Layer input:        B * T * D * 4 bytes
        """
        B, T, D, H = batch_size, seq_len, d_model, n_heads
        attn_bytes = 2 * B * H * T * T * 4
        ffn_bytes = 2 * B * T * 4 * D * 4
        input_bytes = B * T * D * 4
        return attn_bytes + ffn_bytes + input_bytes

    def full_model_memory(self, n_layers: int, **layer_kwargs) -> int:
        """Total activation memory (bytes) across all layers without checkpointing."""
        per_layer = self.estimate_transformer_layer(**layer_kwargs)
        return n_layers * per_layer

    def checkpointed_memory(
        self, n_layers: int, n_segments: int, **layer_kwargs
    ) -> int:
        """Activation memory with segmented checkpointing.

        With checkpointing we only store:
          - The segment-boundary activation tensors (one per segment boundary)
          - The activations of one segment at a time during recompute

        Boundaries: n_segments values (segment 0 start is always kept as input).
        Stored boundary tensors: n_segments (the input to each segment).
        Plus the working memory of the largest segment.
        """
        per_layer = self.estimate_transformer_layer(**layer_kwargs)
        # Each segment start activation is a single layer input
        input_bytes = layer_kwargs["batch_size"] * layer_kwargs["seq_len"] * layer_kwargs["d_model"] * 4
        # Store one activation per segment boundary
        boundary_memory = n_segments * input_bytes
        # Max segment size in layers
        layers_per_segment = math.ceil(n_layers / n_segments)
        # Working memory for one segment recompute
        working_memory = layers_per_segment * per_layer
        return boundary_memory + working_memory


class SelectiveCheckpointing:
    """Apply gradient checkpointing selectively to attention and/or FFN submodules.

    This is useful when only certain expensive operations need memory savings,
    avoiding the recompute overhead on cheaper layers.
    """

    def __init__(
        self,
        checkpoint_attention: bool = True,
        checkpoint_ffn: bool = False,
    ) -> None:
        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_ffn = checkpoint_ffn

    def wrap_attention(self, attn_module: nn.Module):
        """Return CheckpointedModule if checkpoint_attention=True, else original."""
        if self.checkpoint_attention:
            return CheckpointedModule(attn_module)
        return attn_module

    def wrap_ffn(self, ffn_module: nn.Module):
        """Return CheckpointedModule if checkpoint_ffn=True, else original."""
        if self.checkpoint_ffn:
            return CheckpointedModule(ffn_module)
        return ffn_module

    def apply_to_transformer(self, transformer_blocks: nn.ModuleList) -> None:
        """In-place: replace attention/FFN submodules in each block.

        Looks for common attribute names used in transformer implementations:
          - attention: 'attn', 'attention', 'self_attn', 'self_attention'
          - ffn: 'ffn', 'mlp', 'feed_forward', 'ff'
        """
        attn_names = ("attn", "attention", "self_attn", "self_attention")
        ffn_names = ("ffn", "mlp", "feed_forward", "ff")

        for block in transformer_blocks:
            for name in attn_names:
                if hasattr(block, name):
                    original = getattr(block, name)
                    setattr(block, name, self.wrap_attention(original))
                    break
            for name in ffn_names:
                if hasattr(block, name):
                    original = getattr(block, name)
                    setattr(block, name, self.wrap_ffn(original))
                    break


class CheckpointingScheduler:
    """Recommend checkpointing strategy based on memory budget.

    Computes the minimum number of segments required to keep activation memory
    within a specified budget, using ActivationMemoryEstimator internally.
    """

    def __init__(self, memory_budget_gb: float = 8.0) -> None:
        self.memory_budget_gb = memory_budget_gb
        self._estimator = ActivationMemoryEstimator()

    def recommend_segments(
        self,
        n_layers: int,
        batch_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
    ) -> int:
        """Return the minimum n_segments so activation memory fits the budget.

        Starts from 1 segment and increases until memory fits or we reach
        full per-layer checkpointing (n_segments == n_layers).
        """
        budget_bytes = self.memory_budget_gb * (1024 ** 3)
        layer_kwargs = dict(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
        )
        for n_seg in range(1, n_layers + 1):
            mem = self._estimator.checkpointed_memory(n_layers, n_seg, **layer_kwargs)
            if mem <= budget_bytes:
                return n_seg
        return n_layers

    def should_checkpoint_layer(
        self, layer_idx: int, n_layers: int, n_segments: int
    ) -> bool:
        """Return True if layer_idx falls on a segment boundary.

        Segment boundaries are computed the same way as SegmentedCheckpointing.
        """
        boundaries = set()
        for seg in range(n_segments):
            idx = int(math.floor(seg * n_layers / n_segments))
            boundaries.add(idx)
        return layer_idx in boundaries
