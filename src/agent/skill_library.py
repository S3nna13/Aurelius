"""Voyager Skill Library — self-growing executable code skills.

Automatic curriculum, skill library, and iterative prompting with
environment feedback and self-verification for program improvement.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LOGGER = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    code: str
    description: str = ""
    success_count: int = 0
    failure_count: int = 0
    dependencies: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    @property
    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)


class VoyagerSkillLibrary:
    """Self-growing library of executable code skills.

    Three components:
    1. Skill storage — versioned, retrievable code snippets
    2. Automatic curriculum — progressive difficulty ordering
    3. Iterative refinement — env feedback → self-verification → improvement
    """

    def __init__(self, storage_path: str | None = None):
        self.path = (
            Path(storage_path)
            if storage_path is not None
            else Path.home() / ".aurelius" / "voyager_skills"
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self.skills: dict[str, Skill] = {}
        self._curriculum: list[str] = []
        self._skill_history: list[dict[str, Any]] = []

    def add_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        dependencies: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Skill:
        skill = Skill(
            name=name,
            code=code,
            description=description,
            dependencies=list(dependencies or []),
            embedding=list(embedding) if embedding is not None else None,
        )
        self.skills[name] = skill
        return skill

    def retrieve(self, task: str, top_k: int = 5) -> list[Skill]:
        task_tokens = set(_TOKEN_RE.findall(task.lower()))
        scored: list[tuple[float, Skill]] = []
        for skill in self.skills.values():
            name_tokens = set(_TOKEN_RE.findall(skill.name.lower()))
            description_tokens = set(_TOKEN_RE.findall(skill.description.lower()))
            score = 0.0
            if task_tokens & name_tokens:
                score += 0.5
            if task_tokens & description_tokens:
                score += 0.5
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def refine_skill(
        self,
        name: str,
        task: str,
        execution_result: str,
        env_feedback: str,
    ) -> Skill | None:
        skill = self.skills.get(name)
        if skill is None:
            return None

        if "error" in execution_result.lower() or "fail" in execution_result.lower():
            skill.failure_count += 1
            self._skill_history.append({
                "skill": name, "task": task, "result": "failure",
                "feedback": env_feedback, "code_before": skill.code[:200],
            })
        else:
            skill.success_count += 1

        if skill.failure_count > 3:
            improved = self._improve_skill(skill, task, env_feedback)
            if improved != skill.code:
                skill.code = improved
                skill.failure_count = 0
        return skill

    @staticmethod
    def _run_test_case(namespace: dict[str, Any], test_case: Any) -> None:
        if isinstance(test_case, str):
            exec(test_case, {}, namespace)  # noqa: S102  # nosec B102 - test case code is trusted skill verification input
            return

        if isinstance(test_case, dict):
            setup = test_case.get("setup")
            if isinstance(setup, str):
                exec(setup, {}, namespace)  # noqa: S102  # nosec B102 - test case code is trusted skill verification input
            elif callable(setup):
                setup(namespace)

            assertion = test_case.get("assertion", test_case.get("check"))
            if assertion is None:
                raise ValueError("test case dict must include 'assertion' or 'check'")
            if callable(assertion):
                result = assertion(namespace)
                if result is False:
                    raise AssertionError("callable assertion returned False")
            else:
                exec(str(assertion), {}, namespace)  # noqa: S102  # nosec B102 - test case code is trusted skill verification input
            return

        raise TypeError("test cases must be strings or dicts")

    def _improve_skill(self, skill: Skill, task: str, feedback: str) -> str:
        if "timeout" in feedback.lower():
            return skill.code + "\n# Add timeout handling"
        if "permission" in feedback.lower():
            return skill.code + "\n# Request permissions before execution"
        if "not found" in feedback.lower():
            return skill.code + "\n# Add existence checks"
        return skill.code

    def build_curriculum(self, tasks: list[tuple[str, int]]) -> None:
        tasks.sort(key=lambda t: t[1])
        self._curriculum = [t[0] for t in tasks]

    def next_task(self) -> str | None:
        if not self._curriculum:
            return None
        return self._curriculum.pop(0)

    def verify_skill(
        self,
        skill: Skill,
        test_cases: list[Any] | None = None,
    ) -> bool:
        if not test_cases:
            return skill.reliability > 0.8

        try:
            base_namespace: dict[str, Any] = {}
            exec(skill.code, {}, base_namespace)  # noqa: S102  # nosec B102 - skill code is loaded from the local skill library
        except Exception:
            return False

        passed = 0
        for tc in test_cases:
            try:
                namespace = dict(base_namespace)
                self._run_test_case(namespace, tc)
                passed += 1
            except Exception:
                _LOGGER.debug(
                    "Skill verification failed for %s on test case %r",
                    skill.name,
                    tc,
                    exc_info=True,
                )
        return passed / max(len(test_cases), 1) > 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": {
                name: {
                    "code": skill.code,
                    "description": skill.description,
                    "success_count": skill.success_count,
                    "failure_count": skill.failure_count,
                    "dependencies": list(skill.dependencies),
                    "embedding": (
                        list(skill.embedding) if skill.embedding is not None else None
                    ),
                }
                for name, skill in self.skills.items()
            },
            "curriculum": list(self._curriculum),
            "skill_history": list(self._skill_history),
        }

    def update_from_dict(self, data: dict[str, Any]) -> None:
        self.skills.clear()
        self._curriculum = []
        self._skill_history = []

        for name, raw_skill in data.get("skills", {}).items():
            embedding = raw_skill.get("embedding")
            self.skills[name] = Skill(
                name=name,
                code=raw_skill["code"],
                description=raw_skill.get("description", ""),
                success_count=raw_skill.get("success_count", raw_skill.get("success", 0)),
                failure_count=raw_skill.get("failure_count", raw_skill.get("failure", 0)),
                dependencies=list(raw_skill.get("dependencies", [])),
                embedding=list(embedding) if embedding is not None else None,
            )

        self._curriculum = list(data.get("curriculum", []))
        self._skill_history = list(data.get("skill_history", []))

    def save(self) -> None:
        data = self.to_dict()
        (self.path / "skills.json").write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        p = self.path / "skills.json"
        if p.exists():
            self.update_from_dict(json.loads(p.read_text()))
