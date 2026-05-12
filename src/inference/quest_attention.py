"""Quest — Page-Level Sparse Attention.

Implements the Quest algorithm (arXiv:2310.01356-style page-level sparse
attention) on top of a :class:`PagedKVCache`.  For each physical page we
store per-channel K statistics (min / max).  At inference time an
upper-bound attention score is computed for every page using only those
statistics; only the top-``page_budget`` pages are loaded into the dense
attention kernel.

This module is side-effect-free and has no runtime dependencies beyond
``torch``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from src.longcontext.paged_kv_cache import PagedKVCache

logger = logging.getLogger("ark.quest_attention")


class QuestAttention:
    """Page-level sparse attention wrapper around a :class:`PagedKVCache`.

    Args:
        paged_kv: The underlying paged KV cache.
        page_budget: Maximum number of pages to attend to per query step.
        device: Torch device on which page-statistics tensors live.
    """

    def __init__(
        self,
        paged_kv: PagedKVCache,
        page_budget: int = 64,
        device: torch.device | str | None = None,
    ) -> None:
        if page_budget <= 0:
            raise ValueError(f"page_budget must be positive, got {page_budget}")
        self.paged_kv = paged_kv
        self.page_budget = page_budget
        self.device = device or (torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        # page_stats[physical_page_id] -> {"k_min": Tensor, "k_max": Tensor}
        self.page_stats: dict[int, dict[str, Tensor]] = {}

    # ------------------------------------------------------------------
    # Statistics maintenance
    # ------------------------------------------------------------------

    def update_page_stats(self, physical_page_id: int, k_page: Tensor) -> None:
        """Update min/max K statistics for a single physical page.

        Args:
            physical_page_id: Integer id of the physical page in the cache.
            k_page: Key tensor for the full page, shape
                ``(page_size, n_heads, head_dim)``.
        """
        if k_page.dim() != 3:
            raise ValueError(f"k_page must be 3-D, got shape {tuple(k_page.shape)}")
        expected_page_size = self.paged_kv.page_size
        if k_page.shape[0] != expected_page_size:
            raise ValueError(
                f"k_page page_size {k_page.shape[0]} != expected {expected_page_size}"
            )

        k_min = k_page.amin(dim=0)  # (n_heads, head_dim)
        k_max = k_page.amax(dim=0)  # (n_heads, head_dim)

        self.page_stats[physical_page_id] = {
            "k_min": k_min.to(device=self.device),
            "k_max": k_max.to(device=self.device),
        }

    def invalidate_page_stats(self, physical_page_id: int) -> None:
        """Remove cached statistics for a physical page (e.g. after eviction)."""
        self.page_stats.pop(physical_page_id, None)

    def clear_stats(self) -> None:
        """Drop all page statistics."""
        self.page_stats.clear()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_page_scores(self, query: Tensor) -> dict[int, float]:
        """Compute an upper-bound attention score for every tracked page.

        The bound is derived from Hölder's inequality applied to the dot
        product ``q @ k``:

            score <= sum(|q| * max(|k_min|, |k_max|))

        Args:
            query: Query vector of shape ``(n_heads, head_dim)``.

        Returns:
            Mapping from ``physical_page_id`` to scalar score (averaged over
            heads).
        """
        if query.dim() != 2:
            raise ValueError(f"query must be 2-D (n_heads, head_dim), got shape {tuple(query.shape)}")

        scores: dict[int, float] = {}
        q_abs = query.abs().to(device=self.device)  # (n_heads, head_dim)

        for page_id, stats in self.page_stats.items():
            k_min = stats["k_min"]
            k_max = stats["k_max"]
            # Per-channel bound on |k|
            k_bound = torch.maximum(k_max.abs(), k_min.abs())  # (n_heads, head_dim)
            # Upper bound on dot product per head
            bound = (q_abs * k_bound).sum(dim=-1)  # (n_heads,)
            scores[page_id] = bound.mean().item()

        return scores

    def select_pages(self, query: Tensor) -> list[int]:
        """Return the top-``page_budget`` page ids for the given query.

        Args:
            query: Query vector of shape ``(n_heads, head_dim)``.

        Returns:
            List of ``physical_page_id`` values sorted by descending score.
            Length is at most ``page_budget``.
        """
        scores = self.compute_page_scores(query)
        sorted_pages = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [page_id for page_id, _ in sorted_pages[: self.page_budget]]

    # ------------------------------------------------------------------
    # Attention helpers
    # ------------------------------------------------------------------

    def gather_kv_for_pages(
        self,
        request_id: str,
        page_ids: list[int],
    ) -> tuple[Tensor, Tensor]:
        """Gather K/V tensors for a specific set of physical pages.

        .. note::
            This is a convenience helper for testing / prototyping.  In a
            production system the gather would be fused into the attention
            kernel to avoid materialising the full sequence.

        Args:
            request_id: Identifier of the request whose logical pages we read.
            page_ids: Physical page ids to gather.

        Returns:
            ``(K, V)`` tensors of shape ``(n_tokens, n_heads, head_dim)``.
        """
        table = self.paged_kv._tables.get(request_id)
        if table is None:
            raise KeyError(f"unknown request_id {request_id!r}")

        # Map physical pages back to logical offsets.  Since multiple logical
        # positions may map to the same physical page (prefix sharing), we
        # simply iterate in logical order and keep pages that are in the
        # selected set.
        selected = set(page_ids)
        k_chunks: list[Tensor] = []
        v_chunks: list[Tensor] = []

        for logical_idx, phys in enumerate(table.logical_pages):
            if phys not in selected:
                continue
            k_chunks.append(self.paged_kv.K[phys])
            v_chunks.append(self.paged_kv.V[phys])

        if not k_chunks:
            # Return empty tensors with correct shape
            empty = torch.empty(
                (0, self.paged_kv.n_heads, self.paged_kv.head_dim),
                dtype=self.paged_kv.dtype,
            )
            return empty, empty

        return torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)

    def forward(
        self,
        request_id: str,
        query: Tensor,
    ) -> tuple[Tensor, Tensor, list[int]]:
        """End-to-end page selection + KV gather for a single query.

        Args:
            request_id: Request whose KV cache we read from.
            query: Query vector of shape ``(n_heads, head_dim)``.

        Returns:
            ``(K_selected, V_selected, selected_page_ids)`` where the KV
            tensors contain only tokens from the selected pages.
        """
        selected_pages = self.select_pages(query)
        k_sel, v_sel = self.gather_kv_for_pages(request_id, selected_pages)
        return k_sel, v_sel, selected_pages

    # ------------------------------------------------------------------
    # State serialization (useful for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of page statistics."""
        return {
            "page_budget": self.page_budget,
            "page_stats": {
                pid: {"k_min": s["k_min"].cpu(), "k_max": s["k_max"].cpu()}
                for pid, s in self.page_stats.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore page statistics from a snapshot."""
        self.page_budget = state.get("page_budget", self.page_budget)
        self.page_stats = {
            int(pid): {
                "k_min": t["k_min"].to(device=self.device),
                "k_max": t["k_max"].to(device=self.device),
            }
            for pid, t in state.get("page_stats", {}).items()
        }


__all__ = ["QuestAttention"]
