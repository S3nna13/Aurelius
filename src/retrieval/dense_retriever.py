"""Dense retriever: embedding store, ANN-style lookup, in-memory index."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DenseDocument:
    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


class DenseRetriever:
    """In-memory dense retriever with stub embedding and cosine similarity search."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        self._store: dict[str, DenseDocument] = {}

    def _embed_stub(self, text: str) -> list[float]:
        words = text.lower().split()[: self.embedding_dim]
        vec = [hash(word) % 256 / 255.0 for word in words]
        # Pad to embedding_dim
        while len(vec) < self.embedding_dim:
            vec.append(0.0)
        return vec

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> DenseDocument:
        embedding = self._embed_stub(text)
        doc = DenseDocument(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata if metadata is not None else {},
        )
        self._store[doc_id] = doc
        return doc

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_vec = self._embed_stub(query)
        scored = [
            (doc_id, self.cosine_similarity(q_vec, doc.embedding))
            for doc_id, doc in self._store.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def remove(self, doc_id: str) -> bool:
        if doc_id in self._store:
            del self._store[doc_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._store)
