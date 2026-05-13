"""Memory manager for CPU-offloaded KV tiering (InfLLM-style).

InfLLM (Xiao et al., 2023) observes that for very long sequences the full
KV cache often exceeds GPU memory.  The solution is to keep only a small
"hot" subset on GPU:

    * sink tokens   — first few tokens (high attention weight)
    * local window  — most recent tokens
    * top-K blocks  — blocks retrieved by an importance heuristic

All remaining KV pages live in CPU pinned memory and are prefetched to GPU
on demand.  This module provides the bookkeeping layer; it does not contain
kernels or model-specific logic.

Side-effect-free beyond torch tensor movement.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger("ark.longcontext.memory_manager")


class MemoryManager:
    """Manages GPU/CPU KV tiering for long-context inference.

    Args:
        cpu_kv_offload: Whether offloading is enabled at all.
        cpu_kv_sink_size: Number of sink tokens kept permanently on GPU.
        cpu_kv_recent_size: Size of the local (recent) window on GPU.
        cpu_kv_topk_blocks: Number of top-K retrieved blocks to keep on GPU.
    """

    def __init__(
        self,
        cpu_kv_offload: bool = False,
        cpu_kv_sink_size: int = 4,
        cpu_kv_recent_size: int = 512,
        cpu_kv_topk_blocks: int = 32,
    ) -> None:
        self.cpu_kv_offload = cpu_kv_offload
        self.sink_size = cpu_kv_sink_size
        self.recent_size = cpu_kv_recent_size
        self.topk_blocks = cpu_kv_topk_blocks

        # CPU cache holds offloaded tensors in pinned memory.
        self.cpu_cache: dict[str, Tensor] = {}
        # GPU cache holds tensors that are currently resident on device.
        self.gpu_cache: dict[str, Tensor] = {}

        # Dedicated CUDA stream for async H2D transfers so prefetch overlaps
        # with GPU compute on the default stream.
        self._prefetch_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    def should_offload(self, seq_len: int, gpu_budget: int) -> bool:
        """Return ``True`` if the sequence length exceeds the GPU budget.

        Args:
            seq_len: Total number of tokens in the sequence.
            gpu_budget: Maximum tokens the GPU tier can hold.
        """
        return self.cpu_kv_offload and seq_len > gpu_budget

    # ------------------------------------------------------------------
    # Tier movement
    # ------------------------------------------------------------------

    def offload_to_cpu(self, key: str, tensor: Tensor) -> None:
        """Move a tensor from GPU to CPU pinned memory.

        Args:
            key: Unique identifier for the KV block / tensor.
            tensor: The tensor to offload.  If already on CPU it is stored
                directly; otherwise ``pin_memory()`` is used.
        """
        if not self.cpu_kv_offload:
            return
        if tensor.device.type == "cpu":
            self.cpu_cache[key] = tensor
        else:
            self.cpu_cache[key] = tensor.pin_memory().cpu()
        logger.debug("Offloaded %s to CPU (shape=%s)", key, tuple(tensor.shape))

    def prefetch_to_gpu(self, key: str) -> Tensor | None:
        """Fetch a tensor from CPU to GPU using a dedicated CUDA stream.

        The copy runs on ``self._prefetch_stream`` so it can overlap with
        compute already queued on the default stream. Callers that need the
        tensor immediately should call ``synchronize_prefetch()`` first.

        Args:
            key: Identifier of the tensor to fetch.

        Returns:
            The GPU-resident tensor, or ``None`` if the key is unknown.
        """
        if key in self.cpu_cache:
            if not torch.cuda.is_available():
                # No GPU — return CPU tensor directly; caller must handle device placement.
                logger.debug(
                    "prefetch_to_gpu called without CUDA; returning CPU tensor for %s", key
                )
                return self.cpu_cache[key]
            if self._prefetch_stream is not None:
                with torch.cuda.stream(self._prefetch_stream):
                    t = self.cpu_cache[key].cuda(non_blocking=True)
            else:
                t = self.cpu_cache[key].cuda(non_blocking=True)
            self.gpu_cache[key] = t
            logger.debug("Prefetched %s to GPU (shape=%s)", key, tuple(t.shape))
            return t
        return self.gpu_cache.get(key)

    def synchronize_prefetch(self) -> None:
        """Block the default stream until all pending prefetch transfers finish."""
        if self._prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)

    def evict_from_gpu(self, key: str, *, keep_on_cpu: bool = True) -> None:
        """Remove a tensor from the GPU cache, optionally preserving CPU copy.

        Args:
            key: Identifier of the tensor to evict.
            keep_on_cpu: If ``True`` and the tensor is not already in
                ``cpu_cache``, it is offloaded before eviction.
        """
        if key not in self.gpu_cache:
            return
        if keep_on_cpu and key not in self.cpu_cache:
            self.offload_to_cpu(key, self.gpu_cache[key])
        del self.gpu_cache[key]
        logger.debug("Evicted %s from GPU", key)

    def promote_to_gpu(self, key: str, tensor: Tensor) -> None:
        """Explicitly place a tensor into the GPU tier.

        Args:
            key: Identifier for the tensor.
            tensor: Tensor to store on GPU.
        """
        self.gpu_cache[key] = tensor.cuda() if tensor.device.type != "cuda" else tensor
        logger.debug("Promoted %s to GPU (shape=%s)", key, tuple(tensor.shape))

    # ------------------------------------------------------------------
    # Tier queries
    # ------------------------------------------------------------------

    def get_gpu_tensors(self) -> dict[str, Tensor]:
        """Return a shallow copy of the GPU-resident tensor map."""
        return dict(self.gpu_cache)

    def get_cpu_tensors(self) -> dict[str, Tensor]:
        """Return a shallow copy of the CPU-offloaded tensor map."""
        return dict(self.cpu_cache)

    def get_tensor(self, key: str) -> Tensor | None:
        """Return the tensor for *key*, preferring GPU over CPU."""
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        if key in self.cpu_cache:
            return self.prefetch_to_gpu(key)
        return None

    def is_on_gpu(self, key: str) -> bool:
        """Return ``True`` if the given key is resident on GPU."""
        return key in self.gpu_cache

    def is_on_cpu(self, key: str) -> bool:
        """Return ``True`` if the given key is resident on CPU."""
        return key in self.cpu_cache

    # ------------------------------------------------------------------
    # Budget-aware eviction
    # ------------------------------------------------------------------

    def enforce_gpu_budget(self, gpu_budget_bytes: int) -> int:
        """Evict GPU tensors (LRU heuristic) until the budget is met.

        Args:
            gpu_budget_bytes: Maximum bytes allowed in ``gpu_cache``.

        Returns:
            Number of tensors evicted.
        """
        if not self.cpu_kv_offload:
            return 0

        current_bytes = sum(t.numel() * t.element_size() for t in self.gpu_cache.values())
        if current_bytes <= gpu_budget_bytes:
            return 0

        evicted = 0
        # Simple FIFO eviction — real systems may use access counters.
        for key in list(self.gpu_cache.keys()):
            self.evict_from_gpu(key, keep_on_cpu=True)
            evicted += 1
            current_bytes = sum(t.numel() * t.element_size() for t in self.gpu_cache.values())
            if current_bytes <= gpu_budget_bytes:
                break

        logger.info("Enforced GPU budget: evicted %d tensors", evicted)
        return evicted

    # ------------------------------------------------------------------
    # InfLLM-specific helpers
    # ------------------------------------------------------------------

    def build_gpu_tier_mask(
        self,
        seq_len: int,
        block_size: int = 64,
    ) -> list[int]:
        """Return the list of block indices that should stay on GPU.

        The GPU tier always contains:
        1. Sink tokens (first ``sink_size`` tokens).
        2. Recent tokens (last ``recent_size`` tokens).
        3. Top-K blocks (placeholder — caller must supply real scores).

        Args:
            seq_len: Total sequence length in tokens.
            block_size: Tokens per block.

        Returns:
            Sorted list of block indices to keep on GPU.
        """
        n_blocks = (seq_len + block_size - 1) // block_size
        gpu_blocks: set[int] = set()

        # Sink blocks
        sink_blocks = (self.sink_size + block_size - 1) // block_size
        gpu_blocks.update(range(min(sink_blocks, n_blocks)))

        # Recent blocks
        recent_start = max(0, seq_len - self.recent_size)
        recent_start_block = recent_start // block_size
        gpu_blocks.update(range(recent_start_block, n_blocks))

        # Top-K blocks placeholder — caller can refine with real scores.
        # We reserve capacity by not adding anything here.

        return sorted(gpu_blocks)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Drop all cached tensors (both GPU and CPU)."""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        logger.info("MemoryManager cleared")

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state (tensors are moved to CPU)."""
        return {
            "cpu_kv_offload": self.cpu_kv_offload,
            "sink_size": self.sink_size,
            "recent_size": self.recent_size,
            "topk_blocks": self.topk_blocks,
            "cpu_cache": {k: v.cpu() for k, v in self.cpu_cache.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a serialisable state."""
        self.cpu_kv_offload = state.get("cpu_kv_offload", self.cpu_kv_offload)
        self.sink_size = state.get("sink_size", self.sink_size)
        self.recent_size = state.get("recent_size", self.recent_size)
        self.topk_blocks = state.get("topk_blocks", self.topk_blocks)
        self.cpu_cache = {k: v.pin_memory() for k, v in state.get("cpu_cache", {}).items()}
        self.gpu_cache.clear()


__all__ = ["MemoryManager"]
