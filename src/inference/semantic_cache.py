"""Semantic similarity cache for inference.

Cache previous responses keyed by semantic similarity of the prompt embedding,
not exact string match. Enables fast retrieval for semantically similar queries.

Architecture:
  embed_fn(prompt_ids) -> (d_embed,) normalized tensor
  EmbeddingIndex: batched cosine-similarity search over cached embeddings
  SemanticCache: LRU-evicting prompt-response store with similarity lookup
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached prompt → response pair."""

    prompt_embedding: torch.Tensor  # (d_embed,) normalized
    response_ids: torch.Tensor      # (seq_len,) token IDs
    response_text: str = ""         # decoded response (optional)
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Fast cosine similarity search over cached embeddings.

    Stores embeddings as a (N, d_embed) matrix for batched similarity.
    Assumes all embeddings are L2-normalized, so dot product == cosine similarity.
    """

    def __init__(self, d_embed: int, max_entries: int = 1000) -> None:
        self.d_embed = d_embed
        self.max_entries = max_entries
        self._embeddings: torch.Tensor = torch.zeros(0, d_embed)
        self._entry_ids: list[int] = []

    # ------------------------------------------------------------------
    def add(self, embedding: torch.Tensor, entry_id: int) -> None:
        """Add a normalized embedding to the index.

        Args:
            embedding: (d_embed,) L2-normalized tensor.
            entry_id:  integer identifier for this entry.
        """
        emb = embedding.detach().float().unsqueeze(0)  # (1, d_embed)
        if self._embeddings.shape[0] == 0:
            self._embeddings = emb
        else:
            self._embeddings = torch.cat([self._embeddings, emb], dim=0)
        self._entry_ids.append(entry_id)

    # ------------------------------------------------------------------
    def search(
        self, query: torch.Tensor, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Return top_k (entry_id, similarity) pairs sorted descending.

        Uses batched dot product (cosine similarity for normalized embeddings).
        Returns [] if the index is empty.

        Args:
            query:  (d_embed,) L2-normalized query tensor.
            top_k:  number of results to return.
        """
        if self._embeddings.shape[0] == 0:
            return []

        q = query.detach().float().unsqueeze(0)          # (1, d_embed)
        sims = (q @ self._embeddings.T).squeeze(0)       # (N,)
        k = min(top_k, sims.shape[0])
        topk_vals, topk_idxs = torch.topk(sims, k)

        return [
            (self._entry_ids[idx.item()], topk_vals[i].item())
            for i, idx in enumerate(topk_idxs)
        ]

    # ------------------------------------------------------------------
    def remove(self, entry_id: int) -> None:
        """Remove an entry from the index by its entry_id."""
        if entry_id not in self._entry_ids:
            return
        pos = self._entry_ids.index(entry_id)
        self._entry_ids.pop(pos)
        if self._embeddings.shape[0] == 1:
            self._embeddings = torch.zeros(0, self.d_embed)
        else:
            self._embeddings = torch.cat(
                [self._embeddings[:pos], self._embeddings[pos + 1 :]], dim=0
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entry_ids)


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Prompt-response cache with semantic similarity lookup.

    Args:
        embed_fn:             callable(token_ids: torch.Tensor) -> (d_embed,)
                              Function to compute a normalized prompt embedding.
        similarity_threshold: float in [0, 1] — minimum cosine similarity for a hit.
        max_entries:          maximum cache size; LRU eviction when full.
        d_embed:              embedding dimension.
    """

    def __init__(
        self,
        embed_fn: callable,
        similarity_threshold: float = 0.95,
        max_entries: int = 100,
        d_embed: int = 64,
    ) -> None:
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.d_embed = d_embed

        self._entries: dict[int, CacheEntry] = {}
        self._next_id: int = 0
        self._index = EmbeddingIndex(d_embed, max_entries)
        self._access_order: list[int] = []   # front = LRU, back = MRU

        # Stats counters
        self._total_lookups: int = 0
        self._total_hits: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized embedding for prompt_ids."""
        emb = self.embed_fn(prompt_ids)
        return F.normalize(emb.float(), dim=0)

    def _touch(self, entry_id: int) -> None:
        """Move entry_id to back (MRU position) of access order."""
        if entry_id in self._access_order:
            self._access_order.remove(entry_id)
        self._access_order.append(entry_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, prompt_ids: torch.Tensor) -> CacheEntry | None:
        """Search for a cache hit.

        Returns CacheEntry if best similarity >= threshold, else None.
        Updates hit_count and access order on hit.

        Args:
            prompt_ids: (seq_len,) token ID tensor.
        """
        self._total_lookups += 1

        if len(self._index) == 0:
            return None

        query_emb = self._embed(prompt_ids)
        results = self._index.search(query_emb, top_k=1)
        if not results:
            return None

        best_id, best_sim = results[0]
        if best_sim < self.similarity_threshold:
            return None

        # Cache hit
        entry = self._entries[best_id]
        entry.hit_count += 1
        self._total_hits += 1
        self._touch(best_id)
        return entry

    def store(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        response_text: str = "",
    ) -> None:
        """Store a new cache entry.

        Evicts the LRU entry if at capacity before inserting.

        Args:
            prompt_ids:    (seq_len,) prompt token IDs.
            response_ids:  (seq_len,) response token IDs.
            response_text: optional decoded response string.
        """
        if len(self._entries) >= self.max_entries:
            self.evict_lru()

        emb = self._embed(prompt_ids)
        entry_id = self._next_id
        self._next_id += 1

        self._entries[entry_id] = CacheEntry(
            prompt_embedding=emb,
            response_ids=response_ids.detach().clone(),
            response_text=response_text,
            created_at=time.time(),
        )
        self._index.add(emb, entry_id)
        self._touch(entry_id)

    def evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._access_order:
            return
        lru_id = self._access_order.pop(0)
        self._entries.pop(lru_id, None)
        self._index.remove(lru_id)

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            dict with keys:
              'n_entries'  — current number of cached entries
              'total_hits' — cumulative hit count across all lookups
              'hit_rate'   — fraction of lookups that were hits (0.0 if no lookups)
        """
        hit_rate = (
            self._total_hits / self._total_lookups
            if self._total_lookups > 0
            else 0.0
        )
        return {
            "n_entries": len(self._entries),
            "total_hits": self._total_hits,
            "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._entries.clear()
        self._index = EmbeddingIndex(self.d_embed, self.max_entries)
        self._access_order.clear()
        self._total_lookups = 0
        self._total_hits = 0
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_mean_pool_embed_fn(model: nn.Module) -> callable:
    """Create an embedding function using mean pooling of model hidden states.

    Attaches a forward hook to capture the output of the model's final
    normalization layer (``model.norm``), mean-pools over the sequence
    dimension, and L2-normalizes the result.

    Args:
        model: AureliusTransformer (or any nn.Module with a ``norm`` attribute).

    Returns:
        embed_fn(prompt_ids: torch.Tensor) -> (d_model,) normalized tensor
    """
    _hidden: list[torch.Tensor] = []

    def hook_fn(
        module: nn.Module, input: tuple, output: torch.Tensor
    ) -> None:
        _hidden.clear()
        _hidden.append(output)

    hook = model.norm.register_forward_hook(hook_fn)

    def embed_fn(prompt_ids: torch.Tensor) -> torch.Tensor:
        """Embed a single prompt (no batch dimension expected).

        Args:
            prompt_ids: (seq_len,) token ID tensor.

        Returns:
            (d_model,) L2-normalized embedding.
        """
        _hidden.clear()
        ids = prompt_ids.unsqueeze(0)  # (1, seq_len)
        with torch.no_grad():
            model(ids)

        hidden = _hidden[0]            # (1, seq_len, d_model)
        emb = hidden.squeeze(0).mean(dim=0)  # (d_model,)
        return F.normalize(emb.float(), dim=0)

    # Expose cleanup if needed
    embed_fn._hook = hook  # type: ignore[attr-defined]
    return embed_fn
