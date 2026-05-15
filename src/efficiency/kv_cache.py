"""Aurelius v2 KV Cache — paged allocation, quantization, prefix caching.

Implements:
- PagedKVCache: Non-contiguous KV cache blocks (like vLLM paged attention)
- KVCacheQuantizer: KIVI-style 2-bit/4-bit KV quantization
- PrefixCache: Caches common prefixes across requests
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from typing import Any


class QuantBits(enum.StrEnum):
    BITS_2 = "2bit"
    BITS_4 = "4bit"
    BITS_8 = "8bit"
    FP8 = "fp8"
    NONE = "none"


@dataclass
class KVBlock:
    """A single paged block of KV cache."""
    block_id: int
    layer: int
    head: int
    k: Any  # numpy/tensor placeholder
    v: Any  # numpy/tensor placeholder
    quantized: bool = False
    quant_bits: QuantBits = QuantBits.NONE
    pinned: bool = False  # Prevents eviction


@dataclass
class PrefixCacheEntry:
    """A cached prefix."""
    prefix_hash: str
    token_ids: list[int]
    kv_blocks: list[int]  # references to blocks in the KV cache
    ref_count: int = 0
    last_accessed: float = 0.0


class PagedKVCache:
    """Paged KV cache manager with non-contiguous block allocation.

    Inspired by vLLM paged attention:
    - Allocates KV cache in fixed-size blocks
    - Blocks can be shared across sequences (prefix caching)
    - Supports eviction when memory pressure rises
    """

    def __init__(
        self,
        total_blocks: int = 1024,
        block_size: int = 512,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> None:
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Block pool
        self._free_blocks: list[int] = list(range(total_blocks))
        self._allocated: dict[int, KVBlock] = {}
        self._next_block_id = 0

        # Prefix cache
        self._prefix_cache: dict[str, PrefixCacheEntry] = {}

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self._free_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def utilization(self) -> float:
        return self.used_blocks / max(1, self.total_blocks)

    def allocate_block(self, layer: int, head: int) -> int | None:
        """Allocate a KV cache block. Returns block_id or None if full."""
        if not self._free_blocks:
            return None
        
        block_id = self._free_blocks.pop(0)
        self._allocated[block_id] = KVBlock(
            block_id=block_id,
            layer=layer,
            head=head,
            k=None,  # Would be numpy/tensor in real implementation
            v=None,
        )
        return block_id

    def free_block(self, block_id: int) -> bool:
        """Free a KV cache block. Returns False if block doesn't exist."""
        if block_id not in self._allocated:
            return False
        del self._allocated[block_id]
        self._free_blocks.append(block_id)
        return True

    def get_block(self, block_id: int) -> KVBlock | None:
        return self._allocated.get(block_id)

    def evict_lru(self, count: int = 1) -> int:
        """Evict least-recently-used unpinned blocks. Returns count evicted."""
        evicted = 0
        # Sort allocated blocks by a simple heuristic (higher block_id = older)
        candidates = [b for b in self._allocated.values() if not b.pinned]
        candidates.sort(key=lambda b: b.block_id, reverse=True)
        for block in candidates[:count]:
            self.free_block(block.block_id)
            evicted += 1
        return evicted


class KVCacheQuantizer:
    """Quantizes KV cache using KIVI-style 2/4-bit quantization.

    Reduces KV cache memory by 2x-4x at the cost of some precision.
    Applied step 4 in the degradation ladder.
    """

    def __init__(self, bits: QuantBits = QuantBits.BITS_4) -> None:
        self.bits = bits

    def quantize(self, k_tensor: Any, v_tensor: Any) -> tuple[Any, Any]:
        """Quantize KV tensors. Returns (quantized_k, quantized_v, scale)."""
        # Placeholder for actual quantization logic
        # In production: numpy/torch-based KIVI implementation
        if self.bits == QuantBits.BITS_4:
            # 4-bit: ~4x compression
            return k_tensor, v_tensor  # Would apply 4-bit quantization
        elif self.bits == QuantBits.BITS_2:
            # 2-bit: ~8x compression (with more precision loss)
            return k_tensor, v_tensor
        elif self.bits == QuantBits.FP8:
            # FP8: ~2x compression (E4M3 format)
            return k_tensor, v_tensor
        return k_tensor, v_tensor  # None quantization

    @property
    def compression_ratio(self) -> float:
        if self.bits == QuantBits.BITS_4:
            return 4.0
        elif self.bits == QuantBits.BITS_2:
            return 8.0
        elif self.bits == QuantBits.FP8:
            return 2.0
        return 1.0

    def savings_gb(self, original_gb: float) -> float:
        """Estimate memory savings from quantization."""
        return original_gb * (1.0 - 1.0 / self.compression_ratio)


class PrefixCache:
    """Caches common KV prefixes across requests.

    Used for:
    - System prompts (shared across all requests with same system prompt)
    - Long document prefixes (shared across queries about same document)
    - Conversation history prefixes
    """

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self._entries: list[PrefixCacheEntry] = []

    def compute_hash(self, tokens: list[int], prefix_len: int = 0) -> str:
        """Compute a hash of a token prefix for caching."""
        prefix = tokens[:prefix_len] if prefix_len > 0 else tokens
        return hashlib.sha256(str(prefix).encode()).hexdigest()[:16]

    def lookup(self, tokens: list[int]) -> PrefixCacheEntry | None:
        """Look up prefix cache by tokens."""
        token_hash = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
        for entry in self._entries:
            if entry.prefix_hash == token_hash:
                import time
                entry.last_accessed = time.time()
                entry.ref_count += 1
                return entry
        return None

    def store(self, tokens: list[int], kv_block_refs: list[int]) -> PrefixCacheEntry:
        """Store a prefix in the cache."""
        import time
        token_hash = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
        if len(self._entries) >= self.max_entries:
            # Evict LRU
            self._entries.sort(key=lambda e: e.last_accessed)
            self._entries.pop(0)
        entry = PrefixCacheEntry(
            prefix_hash=token_hash,
            token_ids=tokens,
            kv_blocks=kv_block_refs,
            ref_count=1,
            last_accessed=time.time(),
        )
        self._entries.append(entry)
        return entry

    def evict_all(self) -> int:
        """Evict all prefix cache entries (step 6 in degradation ladder)."""
        count = len(self._entries)
        self._entries.clear()
        return count
