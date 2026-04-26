"""
PagedAttention-style KV cache manager for efficient memory use during inference.

Instead of pre-allocating a contiguous KV cache for max_seq_len, pages are
allocated on demand, allowing memory proportional to actual token count and
enabling page sharing between sequences for prefix caching.
"""

from __future__ import annotations

from collections import deque

import torch

PAGE_SIZE = 16  # tokens per page (small for testability)


class PagePool:
    """Manages a pool of KV pages.

    Pre-allocates all storage up front as a single tensor of shape
    (n_pages, 2, page_size, n_kv_heads, head_dim), where dim-1 indexes
    keys (0) vs values (1).

    Args:
        n_pages: total pages in pool
        n_kv_heads: number of KV heads
        head_dim: head dimension
        page_size: tokens per page (default PAGE_SIZE)
        device: torch device
    """

    def __init__(
        self,
        n_pages: int,
        n_kv_heads: int,
        head_dim: int,
        page_size: int = PAGE_SIZE,
        device: str | torch.device = "cpu",
    ):
        self.n_pages = n_pages
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = torch.device(device)

        # Pre-allocate all pages: (n_pages, 2, page_size, n_kv_heads, head_dim)
        # Index 0 along dim 1 = keys, index 1 = values
        self.storage = torch.zeros(
            n_pages,
            2,
            page_size,
            n_kv_heads,
            head_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # All pages start free
        self._free_pages: deque[int] = deque(range(n_pages))

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_page(self) -> int:
        """Get a free page_id. Raises RuntimeError if pool is exhausted."""
        if not self._free_pages:
            raise RuntimeError(f"PagePool exhausted: all {self.n_pages} pages are in use.")
        return self._free_pages.popleft()

    def free_page(self, page_id: int) -> None:
        """Return a page to the pool."""
        if page_id < 0 or page_id >= self.n_pages:
            raise ValueError(f"Invalid page_id {page_id}")
        self._free_pages.append(page_id)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def write_token(
        self,
        page_id: int,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write key/value to *slot* within *page_id*.

        Args:
            page_id: target page
            slot: position within the page (0 <= slot < page_size)
            key: (n_kv_heads, head_dim)
            value: (n_kv_heads, head_dim)
        """
        self.storage[page_id, 0, slot] = key
        self.storage[page_id, 1, slot] = value

    def read_page(self, page_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) for *page_id*, each (page_size, n_kv_heads, head_dim)."""
        return self.storage[page_id, 0], self.storage[page_id, 1]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def available_pages(self) -> int:
        """Number of free pages."""
        return len(self._free_pages)


class SequenceKVCache:
    """Manages paged KV cache for a single sequence.

    Tracks which pages belong to this sequence and at which positions.

    Args:
        pool: shared PagePool
        n_layers: number of transformer layers
    """

    def __init__(self, pool: PagePool, n_layers: int):
        self._pool = pool
        self._n_layers = n_layers

        # page_table[layer] = list of page_ids for that layer (in order)
        self._page_table: list[list[int]] = [[] for _ in range(n_layers)]

        # n_filled_per_layer[layer] = total tokens appended so far
        self._n_filled: list[int] = [0] * n_layers

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append_token(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append one token's KV to the sequence cache for *layer*.

        Args:
            layer: transformer layer index
            key: (n_kv_heads, head_dim)
            value: (n_kv_heads, head_dim)
        """
        token_idx = self._n_filled[layer]
        slot = token_idx % self._pool.page_size

        # Allocate a new page when we start a new page boundary
        if slot == 0:
            new_page_id = self._pool.allocate_page()
            self._page_table[layer].append(new_page_id)

        # Write into the current (last) page
        current_page_id = self._page_table[layer][-1]
        self._pool.write_token(current_page_id, slot, key, value)
        self._n_filled[layer] += 1

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def gather_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect all stored K, V for *layer* into contiguous tensors.

        Returns:
            (keys, values) each of shape (total_tokens, n_kv_heads, head_dim)
        """
        n_tokens = self._n_filled[layer]
        if n_tokens == 0:
            empty = torch.zeros(
                0,
                self._pool.n_kv_heads,
                self._pool.head_dim,
                dtype=torch.float32,
                device=self._pool.device,
            )
            return empty, empty.clone()

        page_size = self._pool.page_size
        all_keys: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for page_idx, page_id in enumerate(self._page_table[layer]):
            k_page, v_page = self._pool.read_page(page_id)
            # Determine how many slots are filled in this page
            tokens_before = page_idx * page_size
            tokens_in_page = min(page_size, n_tokens - tokens_before)
            all_keys.append(k_page[:tokens_in_page])
            all_values.append(v_page[:tokens_in_page])

        return torch.cat(all_keys, dim=0), torch.cat(all_values, dim=0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def free(self) -> None:
        """Release all pages back to the pool."""
        for layer in range(self._n_layers):
            for page_id in self._page_table[layer]:
                self._pool.free_page(page_id)
            self._page_table[layer].clear()
            self._n_filled[layer] = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        """Number of tokens stored (same for all layers after consistent appends)."""
        return self._n_filled[0] if self._n_layers > 0 else 0


class PagedKVCacheManager:
    """High-level manager for multiple sequences.

    Wraps PagePool and creates/destroys SequenceKVCache per request.

    Args:
        n_pages: total pages in pool
        n_layers: number of transformer layers
        n_kv_heads: number of KV heads
        head_dim: head dimension
        page_size: tokens per page (default PAGE_SIZE)
        device: torch device
    """

    def __init__(
        self,
        n_pages: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        page_size: int = PAGE_SIZE,
        device: str | torch.device = "cpu",
    ):
        self._pool = PagePool(
            n_pages=n_pages,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            device=device,
        )
        self._n_layers = n_layers
        self._sequences: dict[int, SequenceKVCache] = {}

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    def create_sequence(self, seq_id: int) -> SequenceKVCache:
        """Create a new sequence cache and register it under *seq_id*."""
        if seq_id in self._sequences:
            raise ValueError(f"Sequence {seq_id} already exists.")
        cache = SequenceKVCache(pool=self._pool, n_layers=self._n_layers)
        self._sequences[seq_id] = cache
        return cache

    def free_sequence(self, seq_id: int) -> None:
        """Free all pages for *seq_id* and remove it from the manager."""
        if seq_id not in self._sequences:
            raise KeyError(f"Sequence {seq_id} not found.")
        self._sequences[seq_id].free()
        del self._sequences[seq_id]

    def available_pages(self) -> int:
        """How many pages are free in the underlying pool."""
        return self._pool.available_pages()
