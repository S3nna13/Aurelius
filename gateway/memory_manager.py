"""Persistent cross-session semantic memory for the Aurelius platform."""

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

try:
    from torch import Tensor
except ImportError:
    Tensor = object  # type: ignore


@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: list[float] | None
    created_at: str
    tags: list[str]
    importance: float = 1.0


class MemoryStore:
    """Persist and retrieve memory entries as a JSON file on disk."""

    def __init__(self, storage_path: str = "~/.aurelius/memory.json") -> None:
        self.storage_path = os.path.expanduser(storage_path)
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            entry = MemoryEntry(
                id=item["id"],
                content=item["content"],
                embedding=item.get("embedding"),
                created_at=item["created_at"],
                tags=item.get("tags", []),
                importance=item.get("importance", 1.0),
            )
            self._entries[entry.id] = entry

    def add(
        self,
        content: str,
        embedding: list[float] | None = None,
        tags: list[str] = None,
        importance: float = 1.0,
    ) -> str:
        entry_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            created_at=datetime.now(UTC).isoformat(),
            tags=tags if tags is not None else [],
            importance=importance,
        )
        self._entries[entry_id] = entry
        return entry_id

    def get(self, id: str) -> MemoryEntry | None:
        return self._entries.get(id)

    def delete(self, id: str) -> bool:
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    def save(self) -> None:
        data = [asdict(e) for e in self._entries.values()]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def search_by_tag(self, tag: str) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if tag in e.tags]


class SemanticMemory:
    """Embed and retrieve memory entries by cosine similarity."""

    def __init__(self, store: MemoryStore, embed_dim: int = 64) -> None:
        self.store = store
        self.embed_dim = embed_dim

    def _mean_embed(self, tokens: list[int], embed_weight: "Tensor") -> list[float]:
        rows = [embed_weight[t].tolist() for t in tokens]
        dim = len(rows[0])
        mean = [sum(r[i] for r in rows) / len(rows) for i in range(dim)]
        norm = math.sqrt(sum(v * v for v in mean)) or 1.0
        return [v / norm for v in mean]

    def remember(
        self,
        content: str,
        tokens: list[int],
        embed_weight: "Tensor",
        tags: list[str] = None,
    ) -> str:
        embedding = self._mean_embed(tokens, embed_weight)
        return self.store.add(content, embedding=embedding, tags=tags)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(y * y for y in b) if False else math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)

    def recall(
        self,
        query_tokens: list[int],
        embed_weight: "Tensor",
        top_k: int = 3,
    ) -> list[MemoryEntry]:
        candidates = [e for e in self.store.list_all() if e.embedding is not None]
        if not candidates:
            return []
        query_emb = self._mean_embed(query_tokens, embed_weight)
        scored = [(self.cosine_similarity(query_emb, e.embedding), e) for e in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def build_context(
        self,
        query_tokens: list[int],
        embed_weight: "Tensor",
        top_k: int = 3,
    ) -> str:
        entries = self.recall(query_tokens, embed_weight, top_k=top_k)
        body = "\n".join(e.content for e in entries)
        return "Relevant memories:\n" + body
