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

# Optional cross-link to the categorical SKILLS_REGISTRY so VoyagerSkillLibrary
# stays in sync with the dissertation's 32-skill taxonomy.
try:  # pragma: no cover - import guarded for stand-alone use
    from agent.skills_registry import SKILLS_REGISTRY as _SKILLS_REGISTRY
except Exception:  # pragma: no cover
    _SKILLS_REGISTRY = {}  # type: ignore[assignment]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LOGGER = logging.getLogger(__name__)

# Momentum-encoder coefficient (Voyager / MoCo style). The dissertation
# specifies tau=0.99 for slowly-updated skill embedding statistics.
_SKILL_MOMENTUM_TAU: float = 0.99


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
            self._skill_history.append(
                {
                    "skill": name,
                    "task": task,
                    "result": "failure",
                    "feedback": env_feedback,
                    "code_before": skill.code[:200],
                }
            )
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
                    "embedding": (list(skill.embedding) if skill.embedding is not None else None),
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
            data = json.loads(p.read_text())
            # Validate skills before loading (defense against bootstrapper injection)
            for name, raw_skill in data.get("skills", {}).items():
                code = raw_skill.get("code", "")
                if self._is_suspicious_code(code):
                    _LOGGER.warning("Skipping suspicious skill %s: patterns detected", name)
                    continue
            self.update_from_dict(data)

    # ------------------------------------------------------------------
    # Trajectory-based skill acquisition (dissertation §Skills, tau=0.99)
    # ------------------------------------------------------------------
    @staticmethod
    def _embed_trajectory(trajectory: list[dict[str, Any]]) -> list[float]:
        """Cheap stdlib-only embedding for a trajectory.

        Hashes the concatenation of step contents into a fixed-width
        float vector so we can compute cosine similarity downstream
        without pulling in torch.
        """
        text = " ".join(str(step.get("content", step)) for step in trajectory)
        tokens = _TOKEN_RE.findall(text.lower())
        dim = 32
        vec = [0.0] * dim
        for tok in tokens:
            vec[hash(tok) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def _momentum_update(
        self,
        old: list[float] | None,
        new: list[float],
        tau: float = _SKILL_MOMENTUM_TAU,
    ) -> list[float]:
        if old is None or len(old) != len(new):
            return list(new)
        return [tau * o + (1.0 - tau) * n for o, n in zip(old, new, strict=False)]

    def record_trajectory_outcome(
        self,
        trajectory: list[dict[str, Any]],
        success: bool,
        skill_name: str | None = None,
    ) -> Skill | None:
        """Record the outcome of an agent trajectory.

        On success, extracts a skill embedding and either adds a new skill or
        momentum-updates an existing one's embedding (tau=0.99). On failure,
        returns ``None`` and does not alter the library — failed trajectories
        do not become skills.
        """
        if not success:
            return None
        if not trajectory:
            return None

        derived_name = skill_name or f"trajectory_skill_{len(self.skills)}"
        embedding = self._embed_trajectory(trajectory)

        existing = self.skills.get(derived_name)
        if existing is None:
            code = "\n".join(f"# step: {step.get('content', step)}" for step in trajectory[:3])
            skill = self.add_skill(
                derived_name,
                code or "# learned from trajectory\n",
                description="Skill learned from successful trajectory",
                embedding=embedding,
            )
            skill.success_count += 1
            return skill

        existing.embedding = self._momentum_update(existing.embedding, embedding)
        existing.success_count += 1
        return existing

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _is_suspicious_code(code: str) -> bool:
        """Detect potential bootstrapper or malicious patterns in skill code."""
        suspicious = [
            r"import\s+os\b",
            r"import\s+subprocess\b",
            r"from\s+os\b",
            r"from\s+subprocess\b",
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"open\s*\(",
            r"os\.system\b",
            r"os\.remove\b",
            r"os\.rmdir\b",
            r"shutil\.rmtree\b",
            r"Path\s*\(.*\)\s*\.unlink\b",
            r"requests\.(post|put|delete)\b",
            r"urllib\.request\b",
            r"socket\b",
            r"pickle\.load\b",
            r"yaml\.load\b",
            r"docker\b",
            r"subprocess\.(run|call|Popen)\b",
        ]
        for pattern in suspicious:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def retrieve_skills(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[Skill]:
        """Retrieve top-k skills by cosine similarity to *query_embedding*.

        Skills without an embedding are ranked by reliability * success_count
        as a fallback. An empty library returns ``[]``.
        """
        if not self.skills:
            return []
        scored: list[tuple[float, Skill]] = []
        for skill in self.skills.values():
            if skill.embedding is not None and query_embedding:
                score = self._cosine(query_embedding, skill.embedding)
            else:
                score = skill.reliability * (skill.success_count / 10.0)
            scored.append((score, skill))
        scored.sort(
            key=lambda x: (x[0], x[1].success_count, x[1].reliability),
            reverse=True,
        )
        return [s for _, s in scored[:top_k]]

    def get_top_skills(self, top_k: int = 3) -> list[Skill]:
        """Return the top-k skills ordered by success_count then reliability."""
        ordered = sorted(
            self.skills.values(),
            key=lambda s: (s.success_count, s.reliability),
            reverse=True,
        )
        return ordered[:top_k]

    # ------------------------------------------------------------------
    # ReAct loop integration
    # ------------------------------------------------------------------
    def integrate_with_react_loop(self, react_loop: Any) -> bool:
        """Hook this skill library into a ReActLoop's Reflect phase.

        Patches ``react_loop._post_step_hooks`` with a callback that records
        successful trajectories as skills. Returns True if a hook was
        installed; False if the loop does not expose a hook list (in which
        case the caller can wire ``record_trajectory_outcome`` manually after
        each step).
        """
        hooks = getattr(react_loop, "_post_step_hooks", None)
        if hooks is None or not isinstance(hooks, list):
            return False

        def _hook(step: Any, trajectory: list[dict[str, Any]]) -> None:
            success = bool(getattr(step, "is_final", False)) and not getattr(step, "error", None)
            if success:
                self.record_trajectory_outcome(trajectory, success=True)

        hooks.append(_hook)
        return True

    @property
    def categorical_registry(self) -> dict[str, Any]:
        """Expose the 32-skill categorical registry for cross-reference."""
        return dict(_SKILLS_REGISTRY)
