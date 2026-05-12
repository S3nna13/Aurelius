"""Verifiable memory store — MemMachine-inspired ground-truth preservation layer.

Reference: MemMachine (arXiv:2604.04853) introduces a "ground-truth-preserving
memory system" for personalized AI agents. The key insight: agent memory systems
often lose or distort the original data through compression/processing. MemMachine
preserves the ground truth alongside processed versions so that the original
content is never irrecoverably lost.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class VerificationRecord:
    """A record of a verification event applied to a VerifiableFact."""

    verified_by: str
    """Who or what performed the verification (e.g. 'user', 'system', 'agent-1')."""

    timestamp: float
    """When the verification occurred (Unix timestamp)."""

    method: str
    """How the verification was performed (e.g. 'direct_observation',
    'cross_reference', 'user_confirmation')."""

    confidence: float = 1.0
    """Confidence level assigned by this verification (0.0 — 1.0)."""


@dataclass
class VerifiableFact:
    """A ground-truth-preserving fact with verification provenance.

    The core concept from MemMachine: the *original* content (ground truth) is
    always stored and never replaced by compressed or processed versions.
    """

    fact_id: str
    """Unique identifier (UUID string)."""

    source: str
    """Origin of this fact. One of:
    - "direct_observation"
    - "inference"
    - "user_provided"
    - "external"
    """

    content: str
    """The actual fact content — this is the ground truth."""

    timestamp: float
    """Unix timestamp of when the fact was first recorded."""

    confidence: float = 1.0
    """Overall confidence in this fact (0.0 — 1.0)."""

    metadata: dict = field(default_factory=dict)
    """Extra context (e.g. ``{"turn_number": 42, "agent_id": "agent-1"}``)."""

    verifications: list[VerificationRecord] = field(default_factory=list)
    """Ordered list of verification records applied to this fact."""

    compressed_version: str | None = None
    """Optional compressed/processed form of the content.
    *Never* replaces the original ``content`` field — this is preserved
    alongside the ground truth."""

    is_active: bool = True
    """Whether this fact is still considered active/valid."""


class VerifiableMemoryStore:
    """A memory store that preserves ground truth alongside processed versions.

    Wraps any existing memory store (e.g. EpisodicMemory, WorkingMemory) and
    adds fact-level tracking, verification provenance, and compression-safe
    ground-truth retrieval.

    All facts are stored in an in-memory ``dict[fact_id, VerifiableFact]``.
    The original ``content`` is always retrievable via ``get_ground_truth()``
    even if a compressed version has been stored.
    """

    def __init__(self, memory_store: object | None = None) -> None:
        """Initialise the verifiable memory store.

        Parameters
        ----------
        memory_store : object, optional
            An existing memory store to wrap (may be used for delegated
            persistence in future extensions).  Defaults to ``None``.
        """
        self._memory_store = memory_store
        self._facts: dict[str, VerifiableFact] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store_fact(self, fact: VerifiableFact) -> str:
        """Store a fact.  If ``fact.fact_id`` is empty a new UUID is assigned.

        Returns the ``fact_id`` of the stored fact.
        """
        if not fact.fact_id:
            fact.fact_id = str(uuid.uuid4())
        if not fact.timestamp:
            fact.timestamp = time.time()
        self._facts[fact.fact_id] = fact
        return fact.fact_id

    def get_fact(self, fact_id: str) -> VerifiableFact | None:
        """Retrieve a fact by its ``fact_id``, or ``None`` if not found."""
        return self._facts.get(fact_id)

    def search(self, query: str, top_k: int = 5) -> list[VerifiableFact]:
        """Basic keyword search over fact content (case-insensitive).

        Only active facts are returned.  Results are ranked by a simple
        substring-match score.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        def score(fact: VerifiableFact) -> float:
            if not fact.is_active:
                return -1.0
            content_lower = fact.content.lower()
            # Exact substring match
            substring_score = 2.0 if query_lower in content_lower else 0.0
            # Term-frequency score
            term_score = sum(1 for term in query_terms if term in content_lower)
            return substring_score + term_score + fact.confidence

        ranked = sorted(self._facts.values(), key=score, reverse=True)
        return [f for f in ranked if score(f) >= 0][:top_k]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_fact(
        self,
        fact_id: str,
        verification: VerificationRecord,
    ) -> None:
        """Append a verification record to a fact and update its confidence.

        The fact's ``confidence`` is updated to the **minimum** between its
        current value and the verification's confidence, reflecting that a
        lower-confidence verification can reduce trust in a fact.
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(f"Fact {fact_id!r} not found in store.")
        fact.verifications.append(verification)
        fact.confidence = min(fact.confidence, verification.confidence)

    def get_unverified(self, min_confidence: float = 0.3) -> list[VerifiableFact]:
        """Return active facts whose confidence is below *min_confidence*.

        These facts have not been adequately verified and may need attention.
        """
        return [f for f in self._facts.values() if f.is_active and f.confidence < min_confidence]

    # ------------------------------------------------------------------
    # Compression-safe ground truth
    # ------------------------------------------------------------------

    def store_compressed(self, fact_id: str, compressed: str) -> None:
        """Store a compressed/processed version of a fact's content.

        The original ``content`` (ground truth) is **never** replaced.
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(f"Fact {fact_id!r} not found in store.")
        fact.compressed_version = compressed

    def get_ground_truth(self, fact_id: str) -> str:
        """Always returns the original content (ground truth).

        This is the key MemMachine guarantee: even if a compressed version
        has been stored via ``store_compressed()``, the original content is
        never lost.
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(f"Fact {fact_id!r} not found in store.")
        return fact.content

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of facts in the store (active or not)."""
        return len(self._facts)

    def clear(self) -> None:
        """Remove all facts from the store."""
        self._facts.clear()
