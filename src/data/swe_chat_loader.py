"""SWE-chat dataset — 6K real coding agent sessions, 63K prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SWEChatSession:
    session_id: str
    user_prompts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    code_survival_rate: float = 0.0


class SWEChatLoader:
    def __init__(self):
        self.sessions: dict[str, SWEChatSession] = {}

    def load(self, data: list[dict]) -> list[SWEChatSession]:
        for item in data:
            session = SWEChatSession(
                session_id=item.get("id", ""),
                user_prompts=item.get("prompts", []),
                tool_calls=item.get("tool_calls", []),
                code_survival_rate=item.get("survival", 0.0),
            )
            self.sessions[session.session_id] = session
        return list(self.sessions.values())

    def survival_stats(self) -> dict[str, float]:
        rates = [s.code_survival_rate for s in self.sessions.values()]
        return {
            "mean": sum(rates) / max(len(rates), 1),
            "sessions": len(self.sessions),
        }
