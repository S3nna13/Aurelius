"""Session-oriented streaming response handler for Aurelius."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass
class StreamChunk:
    chunk_id: int
    text: str
    is_final: bool
    token_count: int
    finish_reason: str | None = None


@dataclass
class StreamSession:
    session_id: str
    request_id: str
    created_at: float
    chunks: list[StreamChunk]
    is_complete: bool = False
    total_tokens: int = 0


class StreamingHandler:
    """Manages streaming response sessions — collect, buffer, and finalize chunks."""

    def __init__(self, max_sessions: int = 1000, buffer_size: int = 256) -> None:
        self._max_sessions = max_sessions
        self._buffer_size = buffer_size
        self._sessions: dict[str, StreamSession] = {}

    def create_session(self, request_id: str) -> StreamSession:
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError("Session limit reached")
        session = StreamSession(
            session_id=str(uuid.uuid4()),
            request_id=request_id,
            created_at=time.time(),
            chunks=[],
        )
        self._sessions[session.session_id] = session
        return session

    def push_chunk(
        self,
        session_id: str,
        text: str,
        is_final: bool = False,
        finish_reason: str | None = None,
    ) -> StreamChunk:
        session = self._sessions[session_id]
        chunk_id = len(session.chunks)
        token_count = max(1, int(len(text.split()) * 1.3))
        chunk = StreamChunk(
            chunk_id=chunk_id,
            text=text,
            is_final=is_final,
            token_count=token_count,
            finish_reason=finish_reason,
        )
        session.chunks.append(chunk)
        session.total_tokens += token_count
        if is_final:
            session.is_complete = True
        return chunk

    def get_session(self, session_id: str) -> StreamSession | None:
        return self._sessions.get(session_id)

    def collect(self, session_id: str) -> str:
        session = self._sessions[session_id]
        return "".join(c.text for c in session.chunks)

    def finalize(self, session_id: str) -> StreamSession:
        session = self._sessions[session_id]
        if session.is_complete:
            raise ValueError(f"Session {session_id} is already complete")
        session.is_complete = True
        return session

    def prune_completed(self, max_age_s: float = 300.0) -> int:
        cutoff = time.time() - max_age_s
        to_remove = [
            sid for sid, s in self._sessions.items() if s.is_complete and s.created_at < cutoff
        ]
        for sid in to_remove:
            del self._sessions[sid]
        return len(to_remove)

    def iter_chunks(self, session_id: str) -> list[StreamChunk]:
        return list(self._sessions[session_id].chunks)

    def active_sessions(self) -> int:
        return len(self._sessions)
