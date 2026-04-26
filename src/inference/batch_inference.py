"""Batch inference optimization: dynamic padding, bucket batching, continuous batching,
throughput measurement, and prefix KV caching. Pure PyTorch only."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# DynamicPadder
# ---------------------------------------------------------------------------


class DynamicPadder:
    """Pad variable-length sequences to a common length within a batch."""

    def __init__(self, pad_token_id: int = 0, padding_side: str = "right") -> None:
        if padding_side not in ("right", "left"):
            raise ValueError("padding_side must be 'right' or 'left'")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def pad_batch(self, sequences: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Pad a list of 1-D token tensors to (B, max_len).

        Returns:
            padded       : (B, max_len) long tensor
            attention_mask: (B, max_len) bool tensor — True for real tokens
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")
        max_len = max(s.size(0) for s in sequences)
        B = len(sequences)
        padded = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            L = seq.size(0)
            if self.padding_side == "right":
                padded[i, :L] = seq
                attention_mask[i, :L] = True
            else:  # left
                padded[i, max_len - L :] = seq
                attention_mask[i, max_len - L :] = True

        return padded, attention_mask

    def unpad(self, padded: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """Remove padding, returning a list of 1-D tensors of the original lengths."""
        result: list[Tensor] = []
        for i in range(padded.size(0)):
            mask = attention_mask[i]  # (max_len,)
            result.append(padded[i][mask])
        return result

    def padding_overhead(self, sequences: list[Tensor]) -> float:
        """Fraction of pad tokens in the batch: total_padding / (B * max_len)."""
        if not sequences:
            raise ValueError("sequences must be non-empty")
        max_len = max(s.size(0) for s in sequences)
        total_slots = max_len * len(sequences)
        real_tokens = sum(s.size(0) for s in sequences)
        padding = total_slots - real_tokens
        return padding / total_slots


# ---------------------------------------------------------------------------
# BucketBatcher
# ---------------------------------------------------------------------------


class BucketBatcher:
    """Group sequences into length buckets for efficient batching."""

    def __init__(self, bucket_boundaries: list[int], batch_size: int = 8) -> None:
        if not bucket_boundaries:
            raise ValueError("bucket_boundaries must be non-empty")
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.batch_size = batch_size
        # Each bucket stores a list of (seq_ids, metadata) tuples
        # Extra bucket for sequences longer than the largest boundary
        self._buckets: list[list[tuple[Tensor, dict | None]]] = [
            [] for _ in range(len(self.bucket_boundaries) + 1)
        ]

    def _bucket_index(self, length: int) -> int:
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)  # overflow bucket

    def add(self, seq_ids: Tensor, metadata: dict | None = None) -> None:
        """Add a sequence to the appropriate bucket."""
        idx = self._bucket_index(seq_ids.size(0))
        self._buckets[idx].append((seq_ids, metadata))

    def get_batch(self) -> tuple[Tensor, Tensor, list] | None:
        """Return a full batch from the fullest bucket, or None if no bucket is full."""
        # Find the bucket with the most items that has >= batch_size items
        best_idx = -1
        best_count = 0
        for i, bucket in enumerate(self._buckets):
            if len(bucket) >= self.batch_size and len(bucket) > best_count:
                best_count = len(bucket)
                best_idx = i

        if best_idx == -1:
            return None

        batch_items = self._buckets[best_idx][: self.batch_size]
        self._buckets[best_idx] = self._buckets[best_idx][self.batch_size :]

        sequences = [item[0] for item in batch_items]
        metadata_list = [item[1] for item in batch_items]

        # Pad to the bucket boundary length (or max in batch for overflow)
        if best_idx < len(self.bucket_boundaries):
            target_len = self.bucket_boundaries[best_idx]
        else:
            target_len = max(s.size(0) for s in sequences)

        B = len(sequences)
        padded = torch.zeros(B, target_len, dtype=torch.long)
        attention_mask = torch.zeros(B, target_len, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            L = seq.size(0)
            padded[i, :L] = seq
            attention_mask[i, :L] = True

        return padded, attention_mask, metadata_list

    def n_waiting(self) -> int:
        """Total number of sequences waiting across all buckets."""
        return sum(len(b) for b in self._buckets)

    def clear(self) -> None:
        """Remove all queued sequences."""
        for bucket in self._buckets:
            bucket.clear()


# ---------------------------------------------------------------------------
# ContinuousBatcher
# ---------------------------------------------------------------------------


class _ActiveRequest:
    """Internal state for a single in-flight generation request."""

    def __init__(self, request_id: int, input_ids: Tensor, max_new_tokens: int) -> None:
        self.request_id = request_id
        # Keep all tokens including prompt for the running context
        self.token_ids: Tensor = input_ids.clone()
        self.max_new_tokens = max_new_tokens
        self.generated: list[int] = []

    @property
    def is_done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


EOS_TOKEN_ID = 2  # arbitrary sentinel used for completion detection


class ContinuousBatcher:
    """Dynamic batch composition — iteration-level batching."""

    def __init__(self, max_batch_size: int = 4, max_seq_len: int = 512) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._next_id: int = 0
        self._active: dict[int, _ActiveRequest] = {}
        self._completed: dict[int, Tensor] = {}

    def add_request(self, input_ids: Tensor, max_new_tokens: int = 64) -> int:
        """Register a new request and return its request_id."""
        rid = self._next_id
        self._next_id += 1
        self._active[rid] = _ActiveRequest(rid, input_ids.long(), max_new_tokens)
        return rid

    def step(self, model: nn.Module) -> dict[int, int]:
        """Run one generation step across all active requests.

        Returns mapping {request_id: next_token_id}.
        Requests that complete (EOS or max_new_tokens) are moved to completed.
        """
        if not self._active:
            return {}

        # Take up to max_batch_size active requests
        active_items = list(self._active.items())[: self.max_batch_size]

        # Find max length among current contexts
        max_len = max(req.token_ids.size(0) for _, req in active_items)
        max_len = min(max_len, self.max_seq_len)

        B = len(active_items)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        for i, (_, req) in enumerate(active_items):
            L = min(req.token_ids.size(0), max_len)
            padded[i, :L] = req.token_ids[-L:]

        model.train(False)
        with torch.no_grad():
            logits = model(padded)  # (B, max_len, vocab) or (B, vocab)

        # Accept both (B, vocab) and (B, T, vocab) output shapes
        if logits.dim() == 3:
            next_logits = logits[:, -1, :]  # (B, vocab)
        else:
            next_logits = logits  # (B, vocab)

        next_tokens = next_logits.argmax(dim=-1)  # (B,)

        result: dict[int, int] = {}
        done_ids: list[int] = []

        for i, (rid, req) in enumerate(active_items):
            tok = int(next_tokens[i].item())
            req.generated.append(tok)
            req.token_ids = torch.cat([req.token_ids, torch.tensor([tok], dtype=torch.long)])
            result[rid] = tok

            if req.is_done or tok == EOS_TOKEN_ID:
                done_ids.append(rid)

        for rid in done_ids:
            req = self._active.pop(rid)
            self._completed[rid] = req.token_ids

        return result

    def get_completed(self) -> dict[int, Tensor]:
        """Return {request_id: full_output_ids} for finished requests."""
        return dict(self._completed)

    def active_count(self) -> int:
        """Number of currently active (in-flight) requests."""
        return len(self._active)


# ---------------------------------------------------------------------------
# ThroughputBenchmark
# ---------------------------------------------------------------------------


class ThroughputBenchmark:
    """Measure inference throughput and latency."""

    def __init__(self) -> None:
        pass

    def tokens_per_second(self, model: nn.Module, input_ids: Tensor, n_tokens: int = 50) -> float:
        """Time autoregressive generation of n_tokens new tokens.

        Returns tokens_generated / elapsed_seconds.
        """
        model.train(False)
        current_ids = input_ids.clone()
        if current_ids.dim() == 1:
            current_ids = current_ids.unsqueeze(0)  # (1, T)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = model(current_ids)
                if logits.dim() == 3:
                    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
                else:
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_tok], dim=1)

        elapsed = time.perf_counter() - start
        return n_tokens / elapsed

    def batch_efficiency(self, padded_tokens: int, real_tokens: int) -> float:
        """Compute utilization: real_tokens / padded_tokens. In (0, 1]."""
        if padded_tokens <= 0:
            raise ValueError("padded_tokens must be positive")
        if real_tokens <= 0:
            raise ValueError("real_tokens must be positive")
        return real_tokens / padded_tokens

    def latency_breakdown(self, model: nn.Module, input_ids: Tensor) -> dict:
        """Measure prefill latency and per-token decode latency.

        Returns dict with 'prefill_ms' and 'decode_ms_per_token'.
        """
        model.train(False)
        ids = input_ids.clone()
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        # Prefill: one forward pass on the full prompt
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(ids)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        # Decode: average over a few steps
        n_decode = 5
        if logits.dim() == 3:
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        current_ids = torch.cat([ids, next_tok], dim=1)

        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_decode):
                logits = model(current_ids)
                if logits.dim() == 3:
                    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                else:
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_tok], dim=1)
        decode_elapsed = (time.perf_counter() - t1) * 1000.0

        return {
            "prefill_ms": prefill_ms,
            "decode_ms_per_token": decode_elapsed / n_decode,
        }


# ---------------------------------------------------------------------------
# PrefixCacheManager
# ---------------------------------------------------------------------------


def _hash_ids(ids: Tensor) -> str:
    """Deterministic hash of a token id tensor (no NumPy required)."""
    # Convert each element to 4-byte little-endian and concatenate
    flat = ids.cpu().contiguous().view(-1)
    parts: list[bytes] = []
    for i in range(flat.size(0)):
        parts.append(int(flat[i].item()).to_bytes(4, byteorder="little", signed=False))
    data = b"".join(parts)
    return hashlib.sha256(data).hexdigest()


class PrefixCacheManager:
    """Cache shared prefix KV-pairs (first-layer) across requests, with LRU eviction."""

    def __init__(self, model: nn.Module, max_cache_entries: int = 8) -> None:
        self.model = model
        self.max_cache_entries = max_cache_entries
        # OrderedDict for LRU: key -> (keys_tensor, values_tensor)
        self._cache: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def _register_first_layer_hook(self) -> tuple:
        """Attach a forward hook to capture the output of the first linear layer."""
        captured: dict[str, Tensor] = {}

        def hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
            captured["output"] = output.detach().clone()

        # Find the first nn.Linear in the model
        first_linear: nn.Linear | None = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break

        if first_linear is None:
            raise RuntimeError("Model has no nn.Linear layers to hook")

        handle = first_linear.register_forward_hook(hook)
        return handle, captured

    def compute_prefix_kv(self, prefix_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run model on prefix, capture and store first-layer KV representations.

        Uses a forward hook on the first nn.Linear to capture activations.
        Stores under the hash of prefix_ids with LRU eviction.

        Returns (keys, values) where both are the captured activation split in half
        along the last dimension to simulate K and V tensors.
        """
        h = _hash_ids(prefix_ids)

        handle, captured = self._register_first_layer_hook()
        self.model.train(False)
        ids = prefix_ids.clone()
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        with torch.no_grad():
            self.model(ids)

        handle.remove()

        if "output" not in captured:
            raise RuntimeError("Hook did not capture any output")

        activation = captured["output"]  # (1, T, d) or (1, d)
        # Split along last dim to produce K and V tensors
        half = activation.size(-1) // 2
        keys = activation[..., :half].clone()
        values = activation[..., half:].clone()

        # LRU eviction before storing
        if h not in self._cache and len(self._cache) >= self.max_cache_entries:
            self.evict_lru()

        # Move to end (most recently used)
        if h in self._cache:
            self._cache.move_to_end(h)
        self._cache[h] = (keys, values)

        return keys, values

    def lookup(self, prefix_ids: Tensor) -> tuple[Tensor, Tensor] | None:
        """Return cached KV pair for prefix_ids, or None on cache miss."""
        h = _hash_ids(prefix_ids)
        if h in self._cache:
            self._hits += 1
            self._cache.move_to_end(h)  # mark as recently used
            return self._cache[h]
        self._misses += 1
        return None

    def cache_hit_rate(self) -> float:
        """hits / (hits + misses). Returns 0.0 if no lookups have occurred."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def evict_lru(self) -> None:
        """Remove the least recently used cache entry."""
        if self._cache:
            self._cache.popitem(last=False)
