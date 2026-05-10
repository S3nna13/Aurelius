from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class VectorEntry:
    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Vector database with pluggable backends.

    Currently supports:
    - In-memory cosine similarity (default, for dev/testing)
    - Qdrant (when qdrant-client is installed and VECTOR_QDRANT_URL is set)

    The in-memory backend is a drop-in that uses the same interface as Qdrant,
    so switching backends requires no code changes.
    """

    def __init__(self, collection: str = "default", dim: int = 1536) -> None:
        self.collection = collection
        self.dim = dim
        self._entries: dict[str, VectorEntry] = {}
        self._lock = threading.Lock()
        self._qdrant_client = None
        self._use_qdrant = False
        self._init_backend()

    def _init_backend(self) -> None:
        url = os.environ.get("VECTOR_QDRANT_URL") or os.environ.get("ARK_QDRANT_URL")
        if not url:
            return
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]
            from qdrant_client.http import models  # type: ignore[import-untyped]

            client: Any = QdrantClient(url=url)
            try:
                client.get_collection(self.collection)
            except Exception:
                client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.dim, distance=models.Distance.COSINE
                    ),
                )
            self._qdrant_client = client
            self._use_qdrant = True
        except ImportError:
            pass

    def upsert(
        self, entry_id: str, vector: list[float], payload: dict[str, Any] | None = None
    ) -> None:
        if self._use_qdrant and self._qdrant_client is not None:
            from qdrant_client.http import models  # type: ignore[import-untyped]

            self._qdrant_client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(id=entry_id, vector=vector, payload=payload or {})],
            )
            return
        with self._lock:
            self._entries[entry_id] = VectorEntry(id=entry_id, vector=vector, payload=payload or {})

    def search(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        if self._use_qdrant and self._qdrant_client is not None:
            results = self._qdrant_client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
            )
            return [
                VectorSearchResult(id=str(r.id), score=float(r.score), payload=r.payload or {})
                for r in results
            ]
        with self._lock:
            scored: list[tuple[float, VectorEntry]] = []
            for entry in self._entries.values():
                sim = self._cosine(query_vector, entry.vector)
                if sim > 0:
                    scored.append((sim, entry))
            scored.sort(key=lambda x: -x[0])
            return [
                VectorSearchResult(id=e.id, score=s, payload=e.payload) for s, e in scored[:top_k]
            ]

    def delete(self, entry_id: str) -> None:
        if self._use_qdrant and self._qdrant_client is not None:
            self._qdrant_client.delete(
                collection_name=self.collection,
                points_selector=[entry_id],
            )
            return
        with self._lock:
            self._entries.pop(entry_id, None)

    def count(self) -> int:
        if self._use_qdrant and self._qdrant_client is not None:
            try:
                info = self._qdrant_client.get_collection(self.collection)
                return info.points_count or 0
            except Exception:
                return 0
        with self._lock:
            return len(self._entries)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
