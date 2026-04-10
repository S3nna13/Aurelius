"""Semantic similarity cache for inference.

Cache inference results keyed by semantic similarity of the prompt embedding,
enabling fast retrieval for semantically similar queries without exact string matching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SemanticCacheConfig:
    """Configuration for SemanticCache."""
    max_size: int = 1000
    similarity_threshold: float = 0.85
    embedding_dim: int = 64
    eviction_policy: str = "lru"   # "lru" | "lfu" | "ttl"
    ttl_seconds: float = 3600.0


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached prompt → response pair."""
    key_text: str
    key_embedding: torch.Tensor          # (d_model,) normalized
    value: str
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Text embedder
# ---------------------------------------------------------------------------

class TextEmbedder:
    """Embeds text by mean-pooling the final transformer layer hidden states.

    Uses a forward hook on model.layers[-1] to capture hidden states,
    mean-pools over the sequence dimension, and L2-normalizes.
    """

    def __init__(self, model: nn.Module, d_model: int = 64) -> None:
        self.model = model
        self.d_model = d_model
        self._hidden: list[torch.Tensor] = []

        # Register hook on last layer
        def hook_fn(module: nn.Module, input: tuple, output) -> None:
            self._hidden.clear()
            # output may be a tuple (x, kv) from TransformerBlock or just a tensor
            if isinstance(output, tuple):
                self._hidden.append(output[0])
            else:
                self._hidden.append(output)

        self._hook = model.layers[-1].register_forward_hook(hook_fn)

    def embed(self, text: str, tokenizer_encode) -> torch.Tensor:
        """Encode text, run model forward, mean-pool last layer hidden states.

        Args:
            text: Input text string.
            tokenizer_encode: Callable(str) -> list[int] or Tensor of token IDs.

        Returns:
            (d_model,) L2-normalized embedding tensor.
        """
        token_ids = tokenizer_encode(text)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # (1, seq_len)

        self._hidden.clear()
        with torch.no_grad():
            self.model(token_ids)

        if not self._hidden:
            raise RuntimeError("Forward hook did not capture hidden states.")

        hidden = self._hidden[0]  # (1, seq_len, d_model) or (seq_len, d_model)
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)  # (seq_len, d_model)

        emb = hidden.mean(dim=0).float()   # (d_model,)
        return F.normalize(emb, dim=0)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D tensors.

    Args:
        a: 1D tensor.
        b: 1D tensor.

    Returns:
        Cosine similarity as a Python float in [-1, 1].
    """
    a = a.float()
    b = b.float()
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    return (a_norm * b_norm).sum().item()


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Prompt-response cache keyed by semantic similarity.

    Args:
        config: SemanticCacheConfig controlling cache behavior.
    """

    def __init__(self, config: SemanticCacheConfig) -> None:
        self.config = config
        self._entries: list[CacheEntry] = []
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    def lookup(self, query_embedding: torch.Tensor) -> Optional[CacheEntry]:
        """Find entry with cosine similarity >= threshold.

        If multiple entries qualify, returns the one with highest similarity.
        Updates hit count and last_accessed on hit; updates miss counter on miss.

        Args:
            query_embedding: (d_model,) query embedding tensor.

        Returns:
            Matching CacheEntry or None.
        """
        if not self._entries:
            self._misses += 1
            return None

        now = time.time()
        best_entry: Optional[CacheEntry] = None
        best_sim: float = -1.0

        for entry in self._entries:
            sim = compute_cosine_similarity(query_embedding, entry.key_embedding)
            if sim >= self.config.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None:
            best_entry.hits += 1
            best_entry.last_accessed = now
            self._hits += 1
            return best_entry

        self._misses += 1
        return None

    # ------------------------------------------------------------------
    def insert(self, key_text: str, key_embedding: torch.Tensor, value: str) -> None:
        """Add a new entry, evicting if at capacity.

        Eviction policies:
          lru: remove entry with oldest last_accessed.
          lfu: remove entry with lowest hit count.
          ttl: remove all expired entries first, then fall back to lru if still full.

        Args:
            key_text:      Original prompt text.
            key_embedding: (d_model,) embedding tensor.
            value:         Generated response string.
        """
        if len(self._entries) >= self.config.max_size:
            self._evict()

        entry = CacheEntry(
            key_text=key_text,
            key_embedding=key_embedding.detach().clone().float(),
            value=value,
        )
        self._entries.append(entry)

    def _evict(self) -> None:
        """Remove one entry according to eviction policy."""
        policy = self.config.eviction_policy

        if policy == "ttl":
            now = time.time()
            expired = [
                e for e in self._entries
                if (now - e.created_at) > self.config.ttl_seconds
            ]
            if expired:
                # Remove all expired
                self._entries = [
                    e for e in self._entries
                    if (now - e.created_at) <= self.config.ttl_seconds
                ]
                # If still at capacity after removing expired, fall through to LRU
                if len(self._entries) < self.config.max_size:
                    return
            # Fall back to LRU
            lru = min(self._entries, key=lambda e: e.last_accessed)
            self._entries.remove(lru)

        elif policy == "lfu":
            lfu = min(self._entries, key=lambda e: e.hits)
            self._entries.remove(lfu)

        else:  # default: lru
            lru = min(self._entries, key=lambda e: e.last_accessed)
            self._entries.remove(lru)

    # ------------------------------------------------------------------
    def invalidate(self, key_text: str) -> bool:
        """Remove entry with exact matching key_text.

        Args:
            key_text: Exact text to look up.

        Returns:
            True if found and removed, False otherwise.
        """
        for entry in self._entries:
            if entry.key_text == key_text:
                self._entries.remove(entry)
                return True
        return False

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            dict with keys: 'size', 'hits', 'misses', 'hit_rate'.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Cached inference engine
# ---------------------------------------------------------------------------

class CachedInferenceEngine:
    """Wraps a model with a semantic cache for fast repeated inference.

    Args:
        model:            AureliusTransformer (or compatible) model.
        cache:            SemanticCache instance.
        tokenizer_encode: Callable(str) -> list[int] | Tensor.
        tokenizer_decode: Callable(list[int] | Tensor) -> str.
        embedder:         TextEmbedder instance.
    """

    def __init__(
        self,
        model: nn.Module,
        cache: SemanticCache,
        tokenizer_encode,
        tokenizer_decode,
        embedder: TextEmbedder,
    ) -> None:
        self.model = model
        self.cache = cache
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.embedder = embedder

    # ------------------------------------------------------------------
    def generate(
        self, prompt: str, max_new_tokens: int = 64
    ) -> tuple[str, bool]:
        """Generate a response, checking the cache first.

        Args:
            prompt:         Input prompt string.
            max_new_tokens: Maximum tokens to generate if cache miss.

        Returns:
            Tuple of (result_text, cache_hit).
            cache_hit is True if result came from cache.
        """
        # Compute embedding for the prompt
        query_emb = self.embedder.embed(prompt, self.tokenizer_encode)

        # Check cache
        entry = self.cache.lookup(query_emb)
        if entry is not None:
            return entry.value, True

        # Generate with model
        result = self._run_generation(prompt, max_new_tokens)

        # Store in cache
        self.cache.insert(prompt, query_emb, result)
        return result, False

    def _run_generation(self, prompt: str, max_new_tokens: int) -> str:
        """Run greedy autoregressive generation.

        Args:
            prompt:         Input prompt string.
            max_new_tokens: Number of tokens to generate.

        Returns:
            Decoded generated text (continuation only).
        """
        token_ids = self.tokenizer_encode(prompt)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # (1, seq_len)

        generated: list[int] = []
        with torch.no_grad():
            input_ids = token_ids
            past_key_values = None
            for _ in range(max_new_tokens):
                loss, logits, past_key_values = self.model(
                    input_ids, past_key_values=past_key_values
                )
                next_token_id = logits[:, -1, :].argmax(dim=-1)  # (1,)
                generated.append(next_token_id.item())
                input_ids = next_token_id.unsqueeze(0)  # (1, 1)

        return self.tokenizer_decode(generated)

    # ------------------------------------------------------------------
    def batch_generate(
        self, prompts: list[str], max_new_tokens: int = 64
    ) -> list[tuple[str, bool]]:
        """Generate responses for a list of prompts, using cache where possible.

        Args:
            prompts:        List of input prompt strings.
            max_new_tokens: Maximum tokens to generate per prompt on miss.

        Returns:
            List of (result_text, cache_hit) tuples, one per prompt.
        """
        return [self.generate(p, max_new_tokens) for p in prompts]
