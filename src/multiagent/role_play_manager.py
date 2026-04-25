"""Role-play session manager for multiagent conversations."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRole:
    name: str
    persona: str
    turn_order: int = 0


@dataclass
class RolePlayConfig:
    roles: list[AgentRole] = field(default_factory=list)
    max_turns: int = 10
    audience: str = "general"


@dataclass(frozen=True)
class Utterance:
    role_name: str
    content: str
    turn: int


_AUDIENCE_PREFIXES: dict[str, str] = {
    "expert": "You are speaking to domain experts. Be precise and technical. ",
    "child": "You are speaking to a child. Use simple words and be friendly. ",
    "general": "You are speaking to a general audience. Be clear and accessible. ",
}


class RolePlayManager:
    def __init__(self, config: RolePlayConfig | None = None) -> None:
        self._config = config or RolePlayConfig()
        self._transcript: list[Utterance] = []
        self._turn: int = 0

    def add_role(self, role: AgentRole) -> None:
        self._config.roles.append(role)
        self._config.roles.sort(key=lambda r: r.turn_order)

    def current_role(self) -> AgentRole | None:
        roles = self._config.roles
        if not roles:
            return None
        return roles[self._turn % len(roles)]

    def record(self, role_name: str, content: str) -> None:
        self._transcript.append(Utterance(role_name=role_name, content=content, turn=self._turn))
        self._turn += 1

    def transcript(self) -> list[Utterance]:
        return list(self._transcript)

    def reset(self) -> None:
        self._transcript = []
        self._turn = 0

    def format_for_agent(self, role: AgentRole) -> str:
        audience = self._config.audience
        prefix = _AUDIENCE_PREFIXES.get(audience, _AUDIENCE_PREFIXES["general"])
        return f"{prefix}You are playing the role of {role.name}. {role.persona}"


ROLE_PLAY_REGISTRY: dict[str, type[RolePlayManager]] = {"default": RolePlayManager}
