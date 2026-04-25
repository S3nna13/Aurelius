import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionConfig:
    n_workers: int = 4
    max_sessions_per_worker: int = 32
    eviction_policy: str = 'lru'


@dataclass
class Session:
    session_id: str
    worker_id: int
    last_active: float
    n_turns: int = 0
    metadata: Dict = field(default_factory=dict)


class ConsistentHashRouter:
    def __init__(self, config: SessionConfig):
        self.config = config
        self._ring: List[tuple] = []
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

    def worker_load(self) -> Dict[int, int]:
        load = {w: 0 for w in range(self.config.n_workers)}
        return load


class SessionManager:
    def __init__(self, config: SessionConfig):
        self.config = config
        self._router = ConsistentHashRouter(config)
        self._sessions: Dict[str, Session] = {}

    def create_session(self, session_id: Optional[str] = None) -> Session:
        if session_id is None:
            session_id = str(uuid.uuid4())
        worker_id = self._router.route(session_id)
        session = Session(
            session_id=session_id,
            worker_id=worker_id,
            last_active=time.time(),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def update_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.n_turns += 1
            session.last_active = time.time()

    def evict_lru(self, worker_id: int) -> Optional[str]:
        worker_sessions = [s for s in self._sessions.values() if s.worker_id == worker_id]
        if not worker_sessions:
            return None
        lru = min(worker_sessions, key=lambda s: s.last_active)
        del self._sessions[lru.session_id]
        return lru.session_id

    def list_sessions(self, worker_id: Optional[int] = None) -> List[Session]:
        if worker_id is None:
            return list(self._sessions.values())
        return [s for s in self._sessions.values() if s.worker_id == worker_id]

    def session_count(self) -> int:
        return len(self._sessions)
