"""Conversation memory store for Aurelius chat surface.

Persists structured facts extracted from conversations across sessions.
Supports:

* pluggable storage backends (in-memory, JSON-on-disk with atomic writes),
* fact CRUD with importance scores, creation timestamps and optional TTL,
* namespace/session isolation,
* retrieval by BM25 (via ``src.retrieval.bm25_retriever``) when a retriever
  is supplied, otherwise a deterministic case-insensitive substring fall-back.

The goal is to give an agentic coding LLM a place to record durable
preferences, environment state, and tool outputs-of-note that should
survive across turns and process restarts without pulling in any
third-party dependency. Pure stdlib only.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

__all__ = [
    "Fact",
    "MemoryStore",
    "InMemoryStore",
    "JSONFileStore",
    "ConversationMemory",
    "MemoryStoreError",
    "MalformedMemoryFileError",
]


class MemoryStoreError(Exception):
    """Base class for conversation memory errors."""


class MalformedMemoryFileError(MemoryStoreError):
    """Raised when a JSONFileStore backing file cannot be parsed.

    Never swallowed into a silent fallback -- corrupted state is a bug
    that callers must surface.
    """


def _validate_importance(importance: float) -> float:
    if not isinstance(importance, (int, float)) or isinstance(importance, bool):
        raise ValueError(f"importance must be a float in [0, 1], got {importance!r}")
    f = float(importance)
    if f != f:  # NaN
        raise ValueError("importance must not be NaN")
    if f < 0.0 or f > 1.0:
        raise ValueError(f"importance must be in [0, 1], got {f}")
    return f


@dataclass
class Fact:
    """A single durable fact recorded in conversation memory."""

    id: str
    namespace: str
    content: str
    created_at: float
    importance: float = 0.5
    ttl_seconds: float | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Fact.id must be a non-empty string")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("Fact.namespace must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("Fact.content must be a string")
        if not isinstance(self.created_at, (int, float)):
            raise TypeError("Fact.created_at must be numeric")
        self.created_at = float(self.created_at)
        self.importance = _validate_importance(self.importance)
        if self.ttl_seconds is not None:
            if not isinstance(self.ttl_seconds, (int, float)) or isinstance(self.ttl_seconds, bool):
                raise TypeError("Fact.ttl_seconds must be numeric or None")
            if self.ttl_seconds <= 0:
                raise ValueError("Fact.ttl_seconds must be > 0 when set")
            self.ttl_seconds = float(self.ttl_seconds)
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise TypeError("Fact.tags must be list[str]")

    def is_expired(self, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return now >= self.created_at + self.ttl_seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Fact:
        required = {"id", "namespace", "content", "created_at"}
        missing = required - data.keys()
        if missing:
            raise MalformedMemoryFileError(f"Fact record missing required keys: {sorted(missing)}")
        return cls(
            id=data["id"],
            namespace=data["namespace"],
            content=data["content"],
            created_at=data["created_at"],
            importance=data.get("importance", 0.5),
            ttl_seconds=data.get("ttl_seconds"),
            tags=list(data.get("tags", [])),
        )


# --------------------------------------------------------------------------- #
# Storage backends                                                            #
# --------------------------------------------------------------------------- #


class MemoryStore(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def add(self, fact: Fact) -> None: ...

    @abstractmethod
    def get(self, fact_id: str) -> Fact | None: ...

    @abstractmethod
    def delete(self, fact_id: str) -> bool: ...

    @abstractmethod
    def list_by_namespace(self, namespace: str) -> list[Fact]: ...

    @abstractmethod
    def query(self, namespace: str, query: str, k: int = 5) -> list[Fact]: ...


class InMemoryStore(MemoryStore):
    """Process-local store. Thread-unsafe; callers own locking."""

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    def add(self, fact: Fact) -> None:
        if fact.id in self._facts:
            raise MemoryStoreError(f"duplicate fact id: {fact.id}")
        self._facts[fact.id] = fact

    def get(self, fact_id: str) -> Fact | None:
        return self._facts.get(fact_id)

    def delete(self, fact_id: str) -> bool:
        return self._facts.pop(fact_id, None) is not None

    def list_by_namespace(self, namespace: str) -> list[Fact]:
        return [f for f in self._facts.values() if f.namespace == namespace]

    def query(self, namespace: str, query: str, k: int = 5) -> list[Fact]:
        # Default backend implements substring fall-back; BM25 ranking
        # is handled by ConversationMemory, which may bypass this path.
        return _substring_query(self.list_by_namespace(namespace), query, k)

    # Introspection -- used by ConversationMemory.prune_expired.
    def _all(self) -> list[Fact]:
        return list(self._facts.values())


class JSONFileStore(MemoryStore):
    """Durable store backed by a single JSON file.

    Writes are atomic: serialize to a temporary file in the same
    directory, ``fsync``, then ``os.replace`` over the target. A crash
    mid-write leaves either the previous file intact or the new file
    fully written -- never a truncated target.

    A malformed backing file raises :class:`MalformedMemoryFileError`
    instead of silently resetting to an empty store; corruption is a
    caller-visible event.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = os.fspath(path)
        self._facts: dict[str, Fact] = {}
        self._load()

    # ---- persistence ---------------------------------------------------- #

    def _load(self) -> None:
        if not os.path.exists(self._path):
            self._facts = {}
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise MalformedMemoryFileError(
                f"failed to parse memory file {self._path!r}: {exc}"
            ) from exc
        except OSError as exc:
            raise MemoryStoreError(f"failed to read memory file {self._path!r}: {exc}") from exc

        if not isinstance(data, dict) or "facts" not in data:
            raise MalformedMemoryFileError(f"memory file {self._path!r} has no 'facts' key")
        records = data.get("facts")
        if not isinstance(records, list):
            raise MalformedMemoryFileError(f"memory file {self._path!r}: 'facts' must be a list")
        facts: dict[str, Fact] = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise MalformedMemoryFileError(
                    f"memory file {self._path!r}: non-object fact record"
                )
            fact = Fact.from_dict(rec)
            facts[fact.id] = fact
        self._facts = facts

    def _flush(self) -> None:
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "facts": [f.to_dict() for f in self._facts.values()],
        }
        directory = os.path.dirname(os.path.abspath(self._path)) or "."
        os.makedirs(directory, exist_ok=True)

        # Write to a uniquely-named sibling temp file, fsync, then rename.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".conversation_memory.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._path)
        except BaseException:
            # Best-effort cleanup; swallow unlink errors because the
            # original exception is what matters.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- MemoryStore API ------------------------------------------------ #

    def add(self, fact: Fact) -> None:
        if fact.id in self._facts:
            raise MemoryStoreError(f"duplicate fact id: {fact.id}")
        self._facts[fact.id] = fact
        self._flush()

    def get(self, fact_id: str) -> Fact | None:
        return self._facts.get(fact_id)

    def delete(self, fact_id: str) -> bool:
        if fact_id not in self._facts:
            return False
        del self._facts[fact_id]
        self._flush()
        return True

    def list_by_namespace(self, namespace: str) -> list[Fact]:
        return [f for f in self._facts.values() if f.namespace == namespace]

    def query(self, namespace: str, query: str, k: int = 5) -> list[Fact]:
        return _substring_query(self.list_by_namespace(namespace), query, k)

    def _all(self) -> list[Fact]:
        return list(self._facts.values())

    # Used by ConversationMemory.prune_expired for bulk deletes w/ one
    # flush instead of O(n) flushes.
    def _bulk_delete(self, fact_ids: Iterable[str]) -> int:
        removed = 0
        for fid in fact_ids:
            if fid in self._facts:
                del self._facts[fid]
                removed += 1
        if removed:
            self._flush()
        return removed


# --------------------------------------------------------------------------- #
# Retrieval helpers                                                           #
# --------------------------------------------------------------------------- #


def _substring_query(facts: list[Fact], query: str, k: int) -> list[Fact]:
    """Deterministic case-insensitive substring fall-back.

    Ranks by: number of query-token hits DESC, then importance DESC,
    then created_at DESC, then id ASC for a fully deterministic order.
    Returns at most ``k`` facts. An empty query returns ``[]`` -- we
    never silently degrade to "return everything".
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    q = query.strip().lower()
    if not q:
        return []
    tokens = [t for t in q.split() if t]
    if not tokens:
        return []
    scored: list[tuple[int, Fact]] = []
    for f in facts:
        text = f.content.lower()
        hits = sum(1 for t in tokens if t in text)
        if hits > 0:
            scored.append((hits, f))
    scored.sort(
        key=lambda pair: (
            -pair[0],
            -pair[1].importance,
            -pair[1].created_at,
            pair[1].id,
        )
    )
    return [f for _, f in scored[:k]]


# --------------------------------------------------------------------------- #
# Facade                                                                      #
# --------------------------------------------------------------------------- #


class ConversationMemory:
    """High-level facade over a :class:`MemoryStore`.

    Parameters
    ----------
    store:
        Backing :class:`MemoryStore`. Owns persistence.
    bm25_retriever:
        Optional factory or instance for BM25 ranking. If a zero-arg
        callable is provided it is invoked per-query to build a fresh
        retriever (BM25 indexes are immutable after ``add_documents``).
        If an instance with ``add_documents`` + ``query`` methods is
        provided, it is treated as a factory type and instantiated
        per-query as well. Pass ``None`` to use the deterministic
        substring fall-back.
    clock:
        Zero-arg callable returning a monotonic-ish wall-clock float.
        Defaults to :func:`time.time`; override for deterministic tests.
    id_factory:
        Zero-arg callable returning a fresh string id. Defaults to
        ``lambda: uuid.uuid4().hex``; override for deterministic tests.
    """

    def __init__(
        self,
        store: MemoryStore,
        bm25_retriever=None,
        *,
        clock=None,
        id_factory=None,
    ) -> None:
        if not isinstance(store, MemoryStore):
            raise TypeError("store must be a MemoryStore instance")
        self.store = store
        self._bm25 = bm25_retriever
        self._clock = clock if clock is not None else time.time
        self._id_factory = id_factory if id_factory is not None else (lambda: uuid.uuid4().hex)

    # ---- CRUD ---------------------------------------------------------- #

    def record_fact(
        self,
        namespace: str,
        content: str,
        importance: float = 0.5,
        ttl: float | None = None,
        tags: list[str] | None = None,
    ) -> Fact:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        _validate_importance(importance)
        fact = Fact(
            id=self._id_factory(),
            namespace=namespace,
            content=content,
            created_at=float(self._clock()),
            importance=float(importance),
            ttl_seconds=ttl,
            tags=list(tags) if tags is not None else [],
        )
        self.store.add(fact)
        return fact

    def delete(self, fact_id: str) -> bool:
        return self.store.delete(fact_id)

    def get(self, fact_id: str) -> Fact | None:
        return self.store.get(fact_id)

    # ---- Retrieval ----------------------------------------------------- #

    def retrieve(self, namespace: str, query: str, k: int = 5) -> list[Fact]:
        """Return up to ``k`` live (non-expired) facts ranked for ``query``.

        Uses BM25 when a retriever is configured, otherwise substring.
        Expired facts are filtered out of the candidate set before
        scoring so stale data never surfaces.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        now = float(self._clock())
        candidates = [f for f in self.store.list_by_namespace(namespace) if not f.is_expired(now)]
        if not candidates or not query.strip():
            return []

        if self._bm25 is None:
            return _substring_query(candidates, query, k)

        retriever = self._build_bm25()
        retriever.add_documents([f.content for f in candidates])
        hits = retriever.query(query, k=k)
        return [candidates[doc_id] for doc_id, _score in hits]

    def _build_bm25(self):
        """Instantiate a fresh BM25 retriever for the current query.

        Accepts either a zero-arg factory (class or callable returning a
        retriever) or a fully-constructed retriever-shaped instance
        whose ``__class__`` can be re-instantiated with no arguments.
        """
        candidate = self._bm25
        if isinstance(candidate, type):
            return candidate()
        if callable(candidate):
            built = candidate()
            if hasattr(built, "add_documents") and hasattr(built, "query"):
                return built
        if hasattr(candidate, "add_documents") and hasattr(candidate, "query"):
            # Instance supplied: rebuild a sibling of the same class so
            # indexes remain immutable per-query.
            cls = candidate.__class__
            return cls()
        raise TypeError("bm25_retriever must be a BM25-shaped class, factory, or instance")

    # ---- Maintenance --------------------------------------------------- #

    def prune_expired(self, namespace: str | None = None) -> int:
        """Delete expired facts, optionally scoped to one namespace.

        Returns the number of facts deleted.
        """
        now = float(self._clock())
        store = self.store
        if hasattr(store, "_all"):
            all_facts = store._all()  # type: ignore[attr-defined]
        else:  # pragma: no cover - defensive path for custom backends
            raise MemoryStoreError("store does not expose iteration for prune_expired")
        victims = [
            f.id
            for f in all_facts
            if f.is_expired(now) and (namespace is None or f.namespace == namespace)
        ]
        if not victims:
            return 0
        if isinstance(store, JSONFileStore):
            return store._bulk_delete(victims)
        removed = 0
        for fid in victims:
            if store.delete(fid):
                removed += 1
        return removed

    def top_by_importance(self, namespace: str, k: int = 5) -> list[Fact]:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        now = float(self._clock())
        facts = [f for f in self.store.list_by_namespace(namespace) if not f.is_expired(now)]
        facts.sort(key=lambda f: (-f.importance, -f.created_at, f.id))
        return facts[:k]
