"""Radix Cache: prefix-sharing KV cache via a radix tree (SGLang RadixAttention, 2024).

Instead of per-request KV caches, a radix tree shares KV cache blocks across
requests with common prefixes. When two requests share a long system prompt or
few-shot examples, the prefix KV is computed once and reused, dramatically
reducing redundant computation for batched inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheBlock:
    """A single KV cache block associated with a token sequence."""

    block_id: int
    token_ids: tuple[int, ...]  # token sequence for this block
    ref_count: int = 0          # how many requests currently reference this block
    last_access: float = 0.0    # timestamp for LRU eviction
    metadata: dict = field(default_factory=dict)


@dataclass
class RadixCacheConfig:
    """Configuration for the radix cache."""

    max_blocks: int = 1024
    block_size: int = 16         # tokens per block
    eviction_policy: str = "lru" # "lru" or "lfu"


class RadixNode:
    """Internal trie node mapping token_id → child RadixNode."""

    def __init__(self) -> None:
        self.children: dict[int, RadixNode] = {}
        self.block: Optional[CacheBlock] = None  # None for internal nodes without a block


# ---------------------------------------------------------------------------
# Main cache
# ---------------------------------------------------------------------------


class RadixCache:
    """Radix tree KV cache for prefix sharing across batched inference requests.

    The cache maps token ID tuples (prefixes) to CacheBlock references using a
    trie structure.  Nodes represent shared prefixes; leaves represent
    per-request extensions.  Eviction respects reference counts so in-flight
    requests are never displaced.
    """

    def __init__(self, config: Optional[RadixCacheConfig] = None) -> None:
        self._config = config or RadixCacheConfig()
        self._root = RadixNode()
        # Flat index of all blocks by block_id for O(1) ref/deref/eviction.
        self._blocks: dict[int, CacheBlock] = {}
        self._last_hit_rate: float = 0.0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def match_prefix(self, token_ids: list[int]) -> tuple[int, Optional[CacheBlock]]:
        """Find the longest prefix of token_ids that exists in the cache.

        Returns:
            (matched_len, last_block) where matched_len is the number of tokens
            matched and last_block is the deepest CacheBlock found, or None if
            no match exists.
        """
        node = self._root
        matched_len = 0
        last_block: Optional[CacheBlock] = None

        for tok in token_ids:
            if tok not in node.children:
                break
            node = node.children[tok]
            matched_len += 1
            if node.block is not None:
                last_block = node.block

        return matched_len, last_block

    def insert(self, token_ids: list[int], block: CacheBlock) -> None:
        """Insert token_ids → block into the radix tree.

        Walks (and creates) nodes for each token in token_ids, placing block at
        the terminal node.  If a node already has a block, the block reference is
        updated.
        """
        node = self._root
        for tok in token_ids:
            if tok not in node.children:
                node.children[tok] = RadixNode()
            node = node.children[tok]

        if node.block is not None:
            # Remove the old block from the flat index before replacing.
            old_id = node.block.block_id
            self._blocks.pop(old_id, None)

        node.block = block
        block.last_access = time.monotonic()
        self._blocks[block.block_id] = block

    def evict(self, n_blocks: int = 1) -> list[CacheBlock]:
        """Evict up to n_blocks blocks using the configured eviction policy.

        Only blocks with ref_count == 0 are eligible for eviction.

        LRU: evict the least recently accessed blocks.

        Returns the list of evicted CacheBlocks.
        """
        evicted: list[CacheBlock] = []
        for _ in range(n_blocks):
            candidate = self._select_eviction_candidate()
            if candidate is None:
                break
            self._remove_block(candidate)
            evicted.append(candidate)
        return evicted

    def ref(self, block_id: int) -> bool:
        """Increment ref_count for the block identified by block_id.

        Returns True if the block was found, False otherwise.
        """
        block = self._blocks.get(block_id)
        if block is None:
            return False
        block.ref_count += 1
        return True

    def deref(self, block_id: int) -> bool:
        """Decrement ref_count for the block identified by block_id.

        Clamps to 0 to avoid negative counts.
        Returns True if the block was found, False otherwise.
        """
        block = self._blocks.get(block_id)
        if block is None:
            return False
        block.ref_count = max(0, block.ref_count - 1)
        return True

    def cache_hit_rate(self, queries: list[list[int]]) -> float:
        """Compute the fraction of queries with at least one partial cache hit.

        For each query, check whether any prefix exists in the cache.
        Updates internal last_hit_rate used by stats().

        Returns the fraction [0.0, 1.0] of queries that had a hit.
        """
        if not queries:
            self._last_hit_rate = 0.0
            return 0.0

        hits = 0
        for token_ids in queries:
            matched_len, _ = self.match_prefix(token_ids)
            if matched_len > 0:
                hits += 1

        rate = hits / len(queries)
        self._last_hit_rate = rate
        return rate

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    def total_blocks(self) -> int:
        """Return the total number of blocks currently stored in the cache."""
        return len(self._blocks)

    def free_blocks(self) -> int:
        """Return the number of blocks that could still be inserted.

        Defined as max_blocks minus the number of blocks with ref_count > 0.
        Unreferenced blocks occupy space but are evictable on demand.
        """
        referenced = sum(1 for b in self._blocks.values() if b.ref_count > 0)
        return self._config.max_blocks - referenced

    def stats(self) -> dict:
        """Return a summary dict with cache accounting information."""
        return {
            "total_blocks": self.total_blocks(),
            "free_blocks": self.free_blocks(),
            "hit_rate": self._last_hit_rate,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_eviction_candidate(self) -> Optional[CacheBlock]:
        """Pick the best eviction candidate according to the configured policy.

        Returns None if no evictable block exists (all have ref_count > 0).
        """
        eligible = [b for b in self._blocks.values() if b.ref_count == 0]
        if not eligible:
            return None

        if self._config.eviction_policy == "lfu":
            # LFU via metadata hit_count; fall back to last_access for tiebreak.
            return min(
                eligible,
                key=lambda b: (b.metadata.get("hit_count", 0), b.last_access),
            )
        # Default: LRU
        return min(eligible, key=lambda b: b.last_access)

    def _remove_block(self, block: CacheBlock) -> None:
        """Remove a block from the flat index and from the trie."""
        self._blocks.pop(block.block_id, None)
        self._remove_from_trie(self._root, block.token_ids, 0)

    def _remove_from_trie(
        self, node: RadixNode, token_ids: tuple[int, ...], depth: int
    ) -> bool:
        """Recursively remove the leaf node for token_ids from the trie.

        Returns True if the node should be pruned from its parent (it has no
        children and no block after the removal).
        """
        if depth == len(token_ids):
            node.block = None
            return not node.children  # prune if leaf

        tok = token_ids[depth]
        child = node.children.get(tok)
        if child is None:
            return False

        should_prune = self._remove_from_trie(child, token_ids, depth + 1)
        if should_prune:
            del node.children[tok]

        return not node.children and node.block is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

try:
    DECODER_REGISTRY  # type: ignore[name-defined]
except NameError:
    DECODER_REGISTRY: dict[str, object] = {}  # type: ignore[assignment]

DECODER_REGISTRY["radix_cache"] = RadixCache  # type: ignore[name-defined]
