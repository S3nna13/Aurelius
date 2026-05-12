from __future__ import annotations

import hashlib
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    n_workers: int = 4
    max_sessions_per_worker: int = 32
    eviction_policy: str = "lru"

    def __post_init__(self) -> None:
        if self.n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if self.max_sessions_per_worker < 1:
            raise ValueError("max_sessions_per_worker must be at least 1")
        if self.eviction_policy != "lru":
            raise ValueError("eviction_policy must be 'lru'")


@dataclass
class Session:
    session_id: str
    worker_id: int
    last_active: float
    n_turns: int = 0
    metadata: dict = field(default_factory=dict)


class ConsistentHashRouter:
    def __init__(
        self,
        config: SessionConfig,
        load_source: Callable[[], Iterable[Session]] | None = None,
    ):
        self.config = config
        self._load_source = load_source
        self._ring: list[tuple] = []
        n_virtual = 100
        for w in range(config.n_workers):
            for i in range(n_virtual):
                h = self._hash(f"worker_{w}_{i}")
                self._ring.append((h, w))
        self._ring.sort(key=lambda x: x[0])

    def _hash(self, key: str) -> int:
        return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16)

    def route(self, session_id: str) -> int:
        h = self._hash(session_id)
        for pos, worker_id in self._ring:
            if pos >= h:
                return worker_id
        return self._ring[0][1]

    def worker_load(self) -> dict[int, int]:
        load = {w: 0 for w in range(self.config.n_workers)}
        if self._load_source is None:
            return load
        for session in self._load_source():
            if 0 <= session.worker_id < self.config.n_workers:
                load[session.worker_id] += 1
        return load


class SessionManager:
    def __init__(self, config: SessionConfig):
        self.config = config
        self._sessions: dict[str, Session] = {}
        self._router = ConsistentHashRouter(config, load_source=self._iter_sessions)

    def _iter_sessions(self) -> Iterable[Session]:
        return list(self._sessions.values())

    def _worker_sessions(self, worker_id: int) -> list[Session]:
        return [s for s in self._sessions.values() if s.worker_id == worker_id]

    def _enforce_worker_capacity(self, worker_id: int) -> None:
        while len(self._worker_sessions(worker_id)) >= self.config.max_sessions_per_worker:
            if self.evict_lru(worker_id) is None:
                break

    def create_session(self, session_id: str | None = None) -> Session:
        if session_id is None:
            session_id = str(uuid.uuid4())
        existing = self._sessions.get(session_id)
        if existing is not None:
            existing.last_active = time.time()
            return existing
        worker_id = self._router.route(session_id)
        self._enforce_worker_capacity(worker_id)
        session = Session(
            session_id=session_id,
            worker_id=worker_id,
            last_active=time.time(),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def update_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.n_turns += 1
            session.last_active = time.time()

    def evict_lru(self, worker_id: int) -> str | None:
        worker_sessions = self._worker_sessions(worker_id)
        if not worker_sessions:
            return None
        lru = min(worker_sessions, key=lambda s: (s.last_active, s.session_id))
        del self._sessions[lru.session_id]
        return lru.session_id

    def list_sessions(self, worker_id: int | None = None) -> list[Session]:
        if worker_id is None:
            return list(self._sessions.values())
        return [s for s in self._sessions.values() if s.worker_id == worker_id]

    def worker_load(self) -> dict[int, int]:
        return self._router.worker_load()

    def session_count(self) -> int:
        return len(self._sessions)


_module = sys.modules[__name__]
sys.modules["serving.session_router"] = _module
sys.modules["src.serving.session_router"] = _module
sys.modules["aurelius.serving.session_router"] = _module
