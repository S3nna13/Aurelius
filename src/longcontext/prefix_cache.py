"""Prefix cache for prompt-prefix sharing across requests.

vLLM-style automatic prefix caching. Given a new prompt, finds the
longest cached prefix match (by token-id sequence prefix) and returns
``(prefix_length, cached_kv_ref)``. Uses a block-level trie for prefix
lookup and LRU eviction when capacity is reached.

Pure stdlib (``hashlib``, ``time``, ``collections``). No torch
required. Side-effect-free import.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PrefixEntry:
    """Metadata for a cached prefix.

    Attributes:
        tokens_hash: Stable hash over the token-id prefix.
        prefix_length: Number of tokens covered by the cached prefix.
        kv_ref: Opaque reference to the backing KV storage (pages, tensors,
            etc.). This module never dereferences it.
        last_access: Monotonic timestamp of the most recent access.
        refcount: Number of active users. Entries with ``refcount > 0``
            are pinned and cannot be evicted.
    """

    tokens_hash: str
    prefix_length: int
    kv_ref: Any
    last_access: float
    refcount: int = 0


def _hash_block(block: Tuple[int, ...]) -> str:
    """Deterministic hash of a block of token ids.

    Uses BLAKE2b (stdlib) for speed + collision resistance; the block is
    encoded as little-endian 8-byte ints so the hash is independent of
    Python's platform-dependent ``hash()`` salt.
    """
    h = hashlib.blake2b(digest_size=16)
    for tok in block:
        h.update(int(tok).to_bytes(8, "little", signed=True))
    return h.hexdigest()


class _TrieNode:
    __slots__ = ("children", "entry")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.entry: Optional[PrefixEntry] = None


class PrefixCache:
    """Block-level prefix cache with LRU eviction.

    Args:
        max_entries: Maximum number of stored ``PrefixEntry`` objects.
        min_prefix_tokens: Shortest matchable prefix in tokens. Lookups
            below this threshold return ``(0, None)``.
        block_size: Token count per trie edge. Must be a positive int.

    The cache stores one ``PrefixEntry`` per *full-block* prefix. On
    ``insert`` the token stream is chopped into ``block_size`` blocks;
    each intermediate full-block prefix (whose length is a multiple of
    ``block_size``) is added to the trie. Partial trailing tokens are
    ignored -- the unit of sharing is a block.
    """

    def __init__(
        self,
        max_entries: int = 128,
        min_prefix_tokens: int = 16,
        block_size: int = 16,
    ) -> None:
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(
                f"block_size must be a positive int, got {block_size!r}"
            )
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive int, got {max_entries!r}"
            )
        if not isinstance(min_prefix_tokens, int) or min_prefix_tokens < 0:
            raise ValueError(
                f"min_prefix_tokens must be a non-negative int, "
                f"got {min_prefix_tokens!r}"
            )

        self.max_entries = max_entries
        self.min_prefix_tokens = min_prefix_tokens
        self.block_size = block_size

        self._root = _TrieNode()
        # OrderedDict acts as the LRU index: oldest access first.
        # Keyed by tokens_hash -> PrefixEntry.
        self._lru: "OrderedDict[str, PrefixEntry]" = OrderedDict()
        # tokens_hash -> trie node holding the entry (for O(1) removal).
        self._nodes: Dict[str, _TrieNode] = {}
        # tokens_hash -> path of (parent_node, edge_key) for pruning.
        self._paths: Dict[str, List[Tuple[_TrieNode, str]]] = {}

        self._hits = 0
        self._misses = 0
        self._inserts = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def find_longest_prefix(
        self, tokens: List[int]
    ) -> Tuple[int, Optional[PrefixEntry]]:
        """Return ``(matched_token_count, entry)`` for the longest prefix.

        Only block-aligned prefixes whose length is ``>= min_prefix_tokens``
        are considered. If no prefix matches, returns ``(0, None)``.
        """
        if not tokens:
            self._misses += 1
            return 0, None

        node = self._root
        best_entry: Optional[PrefixEntry] = None
        best_len = 0
        n_full_blocks = len(tokens) // self.block_size

        for i in range(n_full_blocks):
            start = i * self.block_size
            block = tuple(tokens[start : start + self.block_size])
            key = _hash_block(block)
            child = node.children.get(key)
            if child is None:
                break
            node = child
            if node.entry is not None:
                matched = (i + 1) * self.block_size
                if matched >= self.min_prefix_tokens:
                    best_entry = node.entry
                    best_len = matched

        if best_entry is None:
            self._misses += 1
            return 0, None

        # Touch LRU + timestamp.
        best_entry.last_access = time.monotonic()
        self._lru.move_to_end(best_entry.tokens_hash)
        self._hits += 1
        return best_len, best_entry

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert(self, tokens: List[int], kv_ref: Any) -> None:
        """Install ``tokens`` -> ``kv_ref`` into the cache.

        The full token list is split into ``block_size`` chunks. Each
        block-aligned prefix is inserted (deduping against existing
        entries). Trailing partial tokens are ignored -- the sharing
        unit is the block.
        """
        if not tokens:
            return

        node = self._root
        path: List[Tuple[_TrieNode, str]] = []
        n_full_blocks = len(tokens) // self.block_size

        for i in range(n_full_blocks):
            start = i * self.block_size
            block = tuple(tokens[start : start + self.block_size])
            key = _hash_block(block)
            child = node.children.get(key)
            if child is None:
                child = _TrieNode()
                node.children[key] = child
            path.append((node, key))
            node = child

            prefix_len = (i + 1) * self.block_size
            if node.entry is None:
                tokens_hash = _hash_block(tuple(tokens[:prefix_len]))
                entry = PrefixEntry(
                    tokens_hash=tokens_hash,
                    prefix_length=prefix_len,
                    kv_ref=kv_ref,
                    last_access=time.monotonic(),
                    refcount=0,
                )
                node.entry = entry
                self._lru[tokens_hash] = entry
                self._nodes[tokens_hash] = node
                self._paths[tokens_hash] = list(path)
                self._inserts += 1
                # Enforce capacity after each addition.
                while len(self._lru) > self.max_entries:
                    if self.evict_lru() is None:
                        break
            else:
                # Duplicate: refresh LRU / timestamp, do not double-store.
                existing = node.entry
                existing.last_access = time.monotonic()
                self._lru.move_to_end(existing.tokens_hash)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------
    def evict_lru(self) -> Optional[PrefixEntry]:
        """Evict the least-recently-used unpinned entry.

        Entries with ``refcount > 0`` are skipped. Returns the evicted
        entry or ``None`` if every entry is pinned / the cache is empty.
        """
        victim_hash: Optional[str] = None
        for tokens_hash, entry in self._lru.items():
            if entry.refcount <= 0:
                victim_hash = tokens_hash
                break
        if victim_hash is None:
            return None

        entry = self._lru.pop(victim_hash)
        node = self._nodes.pop(victim_hash, None)
        path = self._paths.pop(victim_hash, [])
        if node is not None:
            node.entry = None
            # Prune empty tail nodes (leaf-ward to root).
            for parent, key in reversed(path):
                child = parent.children.get(key)
                if child is None:
                    continue
                if child.entry is None and not child.children:
                    del parent.children[key]
                else:
                    break
        self._evictions += 1
        return entry

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        """Return aggregate counters for this cache instance."""
        return {
            "entries": len(self._lru),
            "max_entries": self.max_entries,
            "block_size": self.block_size,
            "min_prefix_tokens": self.min_prefix_tokens,
            "hits": self._hits,
            "misses": self._misses,
            "inserts": self._inserts,
            "evictions": self._evictions,
        }

    def __len__(self) -> int:
        return len(self._lru)

    def __contains__(self, tokens_hash: str) -> bool:
        return tokens_hash in self._lru
