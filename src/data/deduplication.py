"""Text deduplication via MinHash LSH: shingling, banding, Jaccard estimation, and streaming pipeline."""

from __future__ import annotations

import hashlib
import sys


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------

def shingle(text: str, k: int = 5) -> set[str]:
    """Extract k-character shingles from text.

    Returns a set of shingles. Returns empty set if text is shorter than k.
    """
    if len(text) < k:
        return set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


# ---------------------------------------------------------------------------
# MinHash Signature
# ---------------------------------------------------------------------------

_MAX_INT = sys.maxsize


def minhash_signature(shingles: set[str], n_hashes: int = 128, seed: int = 42) -> list[int]:
    """Compute MinHash signature for a set of shingles.

    For each hash function h_i, computes min h_i(shingle) over all shingles.
    Uses hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest()[:8] converted to int.

    Returns list of n_hashes integers. All sys.maxsize if shingles is empty.
    """
    if not shingles:
        return [_MAX_INT] * n_hashes

    signature: list[int] = []
    for i in range(n_hashes):
        min_val = _MAX_INT
        for s in shingles:
            h = int(hashlib.sha256(f"{i}:{s}".encode()).hexdigest()[:8], 16)
            if h < min_val:
                min_val = h
        signature.append(min_val)
    return signature


# ---------------------------------------------------------------------------
# Jaccard Estimation
# ---------------------------------------------------------------------------

def estimate_jaccard(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity as fraction of positions where sig1[i] == sig2[i].

    Returns float in [0, 1].
    """
    if not sig1 or not sig2:
        return 0.0
    matches = sum(a == b for a, b in zip(sig1, sig2))
    return matches / len(sig1)


# ---------------------------------------------------------------------------
# LSH Index
# ---------------------------------------------------------------------------

class LSHIndex:
    """Locality-Sensitive Hashing index using banding technique."""

    def __init__(self, n_hashes: int = 128, n_bands: int = 32, threshold: float = 0.8) -> None:
        self.n_hashes = n_hashes
        self.n_bands = n_bands
        self.threshold = threshold
        self.rows_per_band = n_hashes // n_bands
        # buckets[band_idx][bucket_key] = list of doc_ids
        self._buckets: list[dict[int, list[str]]] = [{} for _ in range(n_bands)]
        # Store signatures for candidate verification
        self._signatures: dict[str, list[int]] = {}

    def _band_hash(self, band_slice: list[int]) -> int:
        """Hash a band slice to a bucket key."""
        return hash(tuple(band_slice))

    def add(self, doc_id: str, signature: list[int]) -> list[str]:
        """Add document to index; return list of candidate near-duplicate doc_ids.

        Candidates are docs that share at least one band hash bucket.
        """
        candidates: set[str] = set()

        for band_idx in range(self.n_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            bucket_key = self._band_hash(band_slice)

            bucket = self._buckets[band_idx]
            if bucket_key in bucket:
                for cid in bucket[bucket_key]:
                    if cid != doc_id:
                        candidates.add(cid)
                bucket[bucket_key].append(doc_id)
            else:
                bucket[bucket_key] = [doc_id]

        self._signatures[doc_id] = signature
        return list(candidates)


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------

def is_duplicate(sig1: list[int], sig2: list[int], threshold: float) -> bool:
    """Return True if estimate_jaccard(sig1, sig2) >= threshold."""
    return estimate_jaccard(sig1, sig2) >= threshold


# ---------------------------------------------------------------------------
# StreamingDeduplicator
# ---------------------------------------------------------------------------

class StreamingDeduplicator:
    """Streaming deduplication pipeline using MinHash LSH."""

    def __init__(
        self,
        n_hashes: int = 128,
        n_bands: int = 32,
        threshold: float = 0.8,
        shingle_k: int = 5,
    ) -> None:
        self.n_hashes = n_hashes
        self.n_bands = n_bands
        self.threshold = threshold
        self.shingle_k = shingle_k
        self._index = LSHIndex(n_hashes=n_hashes, n_bands=n_bands, threshold=threshold)
        self._n_seen = 0
        self._n_kept = 0
        self._n_dropped = 0

    def process(self, doc_id: str, text: str) -> bool:
        """Process a document through the deduplication pipeline.

        Computes shingles → signature → queries LSH → checks candidates via Jaccard.
        Returns True (kept) if no duplicates found, False (dropped) if duplicate exists.
        """
        self._n_seen += 1
        shingles = shingle(text, k=self.shingle_k)
        sig = minhash_signature(shingles, n_hashes=self.n_hashes)
        candidates = self._index.add(doc_id, sig)

        for candidate_id in candidates:
            candidate_sig = self._index._signatures.get(candidate_id)
            if candidate_sig is not None and is_duplicate(sig, candidate_sig, self.threshold):
                self._n_dropped += 1
                return False

        self._n_kept += 1
        return True

    def stats(self) -> dict:
        """Return deduplication statistics."""
        duplicate_rate = self._n_dropped / self._n_seen if self._n_seen > 0 else 0.0
        return {
            "n_seen": self._n_seen,
            "n_kept": self._n_kept,
            "n_dropped": self._n_dropped,
            "duplicate_rate": duplicate_rate,
        }
