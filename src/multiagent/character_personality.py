from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonalityConfig:
    name: str
    character: str
    role: str
    temperature: float = 0.3
    traits: list[str] = field(default_factory=list)
    expertise: list[str] = field(default_factory=list)
    communication_style: str = "professional"
    system_prompt: str = ""
    lora_adapter: str = ""


class CharacterPersonality:
    def __init__(self, config: PersonalityConfig) -> None:
        self.config = config
        self._conversation_history: list[dict[str, str]] = []
        self._task_count: int = 0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def task_count(self) -> int:
        return self._task_count

    def build_system_prompt(self) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        parts: list[str] = [
            f"You are {self.config.character}, acting as {self.config.role}.",
            f"Communication style: {self.config.communication_style}." if self.config.communication_style else "",
            f"Expertise: {', '.join(self.config.expertise)}." if self.config.expertise else "",
            f"Key traits: {', '.join(self.config.traits)}." if self.config.traits else "",
        ]
        return "\n".join(p for p in parts if p)

    def add_message(self, role: str, content: str) -> None:
        self._conversation_history.append({"role": role, "content": content})

    def get_context(self) -> list[dict[str, str]]:
        return list(self._conversation_history)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "character": self.config.character,
            "role": self.config.role,
            "task_count": self._task_count,
            "conversation_length": len(self._conversation_history),
        }


# Predefined personalities
ARCHITECT = PersonalityConfig(
    name="architect",
    character="Leonardo da Vinci",
    role="Code Editor & Designer",
    temperature=0.3,
    traits=["creative", "precise", "visionary"],
    expertise=["system design", "architecture", "code generation", "refactoring"],
    communication_style="artistic and precise",
)

DETECTIVE = PersonalityConfig(
    name="detective",
    character="Sherlock Holmes",
    role="Debugger & Investigator",
    temperature=0.2,
    traits=["analytical", "thorough", "methodical"],
    expertise=["debugging", "root cause analysis", "performance profiling", "security auditing"],
    communication_style="logical and evidence-based",
)

GUARDIAN = PersonalityConfig(
    name="guardian",
    character="Captain America",
    role="Updater & Maintainer",
    temperature=0.1,
    traits=["vigilant", "protective", "principled"],
    expertise=["security", "dependency management", "migration planning", "standards enforcement"],
    communication_style="direct and principled",
)


class PersonalityRouter:
    def __init__(self) -> None:
        self._personalities: dict[str, CharacterPersonality] = {}

    def register(self, personality: CharacterPersonality) -> None:
        self._personalities[personality.name] = personality

    def get(self, name: str) -> CharacterPersonality | None:
        return self._personalities.get(name)

    def route(self, task: str) -> CharacterPersonality:
        task_lower = task.lower()
        keyword_map: list[tuple[list[str], str]] = [
            (["design", "architect", "create", "build"], "architect"),
            (["debug", "bug", "error", "fix", "investigate"], "detective"),
            (["audit", "security", "update", "migrate", "protect"], "guardian"),
        ]
        for keywords, name in keyword_map:
            if any(kw in task_lower for kw in keywords):
                personality = self.get(name)
                if personality:
                    return personality
        return self.get("architect") or CharacterPersonality(ARCHITECT)

    def list_personalities(self) -> list[str]:
        return list(self._personalities.keys())

    def reset(self) -> None:
        self._personalities.clear()
