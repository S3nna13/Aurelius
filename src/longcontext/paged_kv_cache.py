"""Paged KV cache (vLLM-style, Kwon 2023 arXiv:2309.06180).

Simulation/library-only single-device KV page manager. No kernel; callers
manage logical -> physical page maps via `PageTable`.

This module is dependency-free beyond ``torch`` and is side-effect-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


class PagedKVOutOfMemory(RuntimeError):
    """Raised when an allocation cannot find enough free physical pages."""


@dataclass
class PageTable:
    """Maps logical page positions to physical page ids for one request.

    The i-th entry of ``logical_pages`` is the physical page id backing the
    i-th logical page of the request (each page holds ``page_size`` tokens).
    """

    request_id: str
    logical_pages: list[int] = field(default_factory=list)


class PagedKVCache:
    """Fixed-size-page KV cache.

    Storage layout: K and V are ``[num_pages, page_size, n_heads, head_dim]``.
    A request's tokens occupy consecutive logical pages; each logical page
    maps to any physical page via its ``PageTable``.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        page_size: int = 16,
        num_pages: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError(f"n_heads must be positive int, got {n_heads!r}")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be positive int, got {head_dim!r}")
        if not isinstance(page_size, int) or page_size <= 0:
            raise ValueError(f"page_size must be positive int, got {page_size!r}")
        if not isinstance(num_pages, int) or num_pages <= 0:
            raise ValueError(f"num_pages must be positive int, got {num_pages!r}")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = num_pages
        self.dtype = dtype

        shape = (num_pages, page_size, n_heads, head_dim)
        self.K: Tensor = torch.zeros(shape, dtype=dtype)
        self.V: Tensor = torch.zeros(shape, dtype=dtype)

        # Free list is maintained in ascending order so allocations are
        # deterministic given identical call sequences.
        self._free_pages: list[int] = list(range(num_pages))
        # request_id -> PageTable
        self._tables: dict[str, PageTable] = {}
        # request_id -> token count (write-visible length)
        self._lengths: dict[str, int] = {}
        # physical page id -> reference count (for prefix sharing)
        self._refcount: list[int] = [0] * num_pages

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    def _pages_needed(self, n_tokens: int) -> int:
        if n_tokens <= 0:
            return 0
        return (n_tokens + self.page_size - 1) // self.page_size

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def _take_pages(self, n: int) -> list[int]:
        if n > len(self._free_pages):
            raise PagedKVOutOfMemory(f"need {n} pages, only {len(self._free_pages)} free")
        taken = self._free_pages[:n]
        self._free_pages = self._free_pages[n:]
        for p in taken:
            self._refcount[p] += 1
        return taken

    def _release_page(self, page_id: int) -> None:
        self._refcount[page_id] -= 1
        if self._refcount[page_id] == 0:
            # Return to free list; keep sorted for determinism.
            self._free_pages.append(page_id)
            self._free_pages.sort()
        elif self._refcount[page_id] < 0:  # pragma: no cover - defensive
            raise RuntimeError(f"page {page_id} refcount went negative")

    def allocate(self, request_id: str, n_tokens_needed: int) -> PageTable:
        if request_id in self._tables:
            raise ValueError(f"request_id {request_id!r} already allocated")
        if n_tokens_needed < 0:
            raise ValueError("n_tokens_needed must be non-negative")
        n_pages = self._pages_needed(n_tokens_needed)
        pages = self._take_pages(n_pages)
        table = PageTable(request_id=request_id, logical_pages=pages)
        self._tables[request_id] = table
        self._lengths[request_id] = 0
        return table

    def deallocate(self, request_id: str) -> None:
        if request_id not in self._tables:
            raise KeyError(f"unknown request_id {request_id!r}")
        table = self._tables.pop(request_id)
        self._lengths.pop(request_id, None)
        for p in table.logical_pages:
            self._release_page(p)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _locate(self, request_id: str, token_position: int) -> tuple[int, int]:
        if request_id not in self._tables:
            raise KeyError(f"unknown request_id {request_id!r}")
        if token_position < 0:
            raise IndexError(f"token_position must be non-negative, got {token_position}")
        table = self._tables[request_id]
        logical_idx = token_position // self.page_size
        offset = token_position % self.page_size
        if logical_idx >= len(table.logical_pages):
            raise IndexError(
                f"token_position {token_position} exceeds allocated pages for {request_id!r}"
            )
        return table.logical_pages[logical_idx], offset

    def write(
        self,
        request_id: str,
        token_position: int,
        k_vec: Tensor,
        v_vec: Tensor,
    ) -> None:
        expected_shape = (self.n_heads, self.head_dim)
        if tuple(k_vec.shape) != expected_shape:
            raise ValueError(f"k_vec shape {tuple(k_vec.shape)} != expected {expected_shape}")
        if tuple(v_vec.shape) != expected_shape:
            raise ValueError(f"v_vec shape {tuple(v_vec.shape)} != expected {expected_shape}")
        phys, offset = self._locate(request_id, token_position)
        # Copy-on-write: if this physical page is shared, clone it before
        # writing so we don't corrupt other requests sharing the prefix.
        if self._refcount[phys] > 1:
            new_pages = self._take_pages(1)
            new_phys = new_pages[0]
            self.K[new_phys].copy_(self.K[phys])
            self.V[new_phys].copy_(self.V[phys])
            # Swap this logical slot to the private copy; drop old ref.
            table = self._tables[request_id]
            logical_idx = token_position // self.page_size
            table.logical_pages[logical_idx] = new_phys
            self._release_page(phys)
            phys = new_phys
        self.K[phys, offset].copy_(k_vec.to(self.dtype))
        self.V[phys, offset].copy_(v_vec.to(self.dtype))
        length = self._lengths.get(request_id, 0)
        if token_position + 1 > length:
            self._lengths[request_id] = token_position + 1

    def read(self, request_id: str, start: int, end: int) -> tuple[Tensor, Tensor]:
        if request_id not in self._tables:
            raise KeyError(f"unknown request_id {request_id!r}")
        if start < 0 or end < start:
            raise IndexError(f"invalid read range [{start}, {end})")
        length = self._lengths.get(request_id, 0)
        if end > length:
            raise IndexError(f"read end {end} exceeds written length {length} for {request_id!r}")
        n = end - start
        out_k = torch.empty((n, self.n_heads, self.head_dim), dtype=self.dtype)
        out_v = torch.empty((n, self.n_heads, self.head_dim), dtype=self.dtype)
        table = self._tables[request_id]
        for i in range(n):
            pos = start + i
            logical_idx = pos // self.page_size
            offset = pos % self.page_size
            phys = table.logical_pages[logical_idx]
            out_k[i] = self.K[phys, offset]
            out_v[i] = self.V[phys, offset]
        return out_k, out_v

    # ------------------------------------------------------------------
    # Prefix sharing
    # ------------------------------------------------------------------
    def prefix_share(
        self,
        source_rid: str,
        target_rid: str,
        n_shared_tokens: int,
    ) -> PageTable:
        """Create ``target_rid`` that shares the first ``n_shared_tokens`` of
        ``source_rid``. Shared pages are reference-counted; writes past the
        shared region allocate new pages, and writes *into* a shared page
        trigger copy-on-write.
        """
        if source_rid not in self._tables:
            raise KeyError(f"unknown source request_id {source_rid!r}")
        if target_rid in self._tables:
            raise ValueError(f"target request_id {target_rid!r} already exists")
        if n_shared_tokens < 0:
            raise ValueError("n_shared_tokens must be non-negative")
        src_len = self._lengths.get(source_rid, 0)
        if n_shared_tokens > src_len:
            raise ValueError(
                f"source has only {src_len} written tokens, cannot share {n_shared_tokens}"
            )
        src_table = self._tables[source_rid]
        n_shared_pages = self._pages_needed(n_shared_tokens)
        shared = list(src_table.logical_pages[:n_shared_pages])
        # Bump refcount on each shared physical page.
        for p in shared:
            self._refcount[p] += 1
        table = PageTable(request_id=target_rid, logical_pages=shared)
        self._tables[target_rid] = table
        self._lengths[target_rid] = n_shared_tokens
        return table

    def extend(self, request_id: str, n_extra_tokens: int) -> None:
        """Grow a request's page table by enough pages to hold
        ``n_extra_tokens`` additional tokens past its current length.
        """
        if request_id not in self._tables:
            raise KeyError(f"unknown request_id {request_id!r}")
        if n_extra_tokens <= 0:
            return
        table = self._tables[request_id]
        current_len = self._lengths.get(request_id, 0)
        new_total = current_len + n_extra_tokens
        needed_pages = self._pages_needed(new_total)
        have_pages = len(table.logical_pages)
        if needed_pages > have_pages:
            extra = self._take_pages(needed_pages - have_pages)
            table.logical_pages.extend(extra)
