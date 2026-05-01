"""Tests for src.agent.skill_registry and skill persistence round-trips."""

from __future__ import annotations

from src.agent.agent_persistence import AgentPersistence
from src.agent.skill_registry import SkillRegistry, SkillSpec


def _make_skill(
    skill_id: str,
    *,
    name: str | None = None,
    version: str = "1.0.0",
    description: str = "",
    prompt_template: str = "",
    tools: list[str] | None = None,
    dependencies: list[str] | None = None,
    enabled: bool = True,
) -> SkillSpec:
    return SkillSpec(
        id=skill_id,
        name=name or skill_id,
        version=version,
        description=description,
        prompt_template=prompt_template,
        tools=tools or [],
        dependencies=dependencies or [],
        enabled=enabled,
    )


def test_register_find_and_dependency_resolution() -> None:
    registry = SkillRegistry()
    registry.register(_make_skill("root", dependencies=["mid"]))
    registry.register(_make_skill("mid", dependencies=["leaf"]))
    registry.register(_make_skill("leaf"))

    assert registry.count == 3
    assert registry.get("root").id == "root"
    assert registry.find_by_name("ROO")[0].id == "root"
    assert registry.resolve_dependencies("root") == ["leaf", "mid", "root"]


def test_enable_disable_updates_state() -> None:
    registry = SkillRegistry()
    registry.register(_make_skill("toggle", enabled=False))

    assert registry.get("toggle").enabled is False

    registry.enable("toggle")
    assert registry.get("toggle").enabled is True

    registry.disable("toggle")
    assert registry.get("toggle").enabled is False


def test_compose_copies_input_list() -> None:
    registry = SkillRegistry()
    steps = ["extract", "transform", "load"]

    registry.compose("etl", steps)
    steps.append("audit")

    assert registry.get_composition("etl") == ["extract", "transform", "load"]


def test_to_dict_and_from_dict_round_trip() -> None:
    registry = SkillRegistry()
    registry.register(
        _make_skill(
            "review",
            name="Code Review",
            description="Reviews code.",
            prompt_template="review carefully and cite evidence",
            tools=["git", "rg"],
            dependencies=["lint"],
            enabled=False,
        )
    )
    registry.register(_make_skill("lint"))
    registry.compose("quality", ["lint", "review"])

    snapshot = registry.to_dict()
    restored = SkillRegistry.from_dict(snapshot)

    restored_skill = restored.get("review")
    assert restored_skill is not None
    assert restored_skill.name == "Code Review"
    assert restored_skill.description == "Reviews code."
    assert restored_skill.prompt_template == "review carefully and cite evidence"
    assert restored_skill.tools == ["git", "rg"]
    assert restored_skill.dependencies == ["lint"]
    assert restored_skill.enabled is False
    assert restored.get_composition("quality") == ["lint", "review"]


def test_agent_persistence_round_trip_preserves_full_prompt_template(
    tmp_path,
) -> None:
    registry = SkillRegistry()
    registry.register(
        _make_skill(
            "longform",
            description="A long-form skill.",
            prompt_template="x" * 400,
            tools=["bash"],
            dependencies=["base"],
        )
    )
    registry.register(_make_skill("base"))
    registry.compose("bundle", ["base", "longform"])

    persistence = AgentPersistence(base_path=str(tmp_path))
    saved_path = persistence.save_skills(registry, name="skills")

    loaded_registry = SkillRegistry()
    assert persistence.load_skills(loaded_registry, name="skills") is True

    loaded_skill = loaded_registry.get("longform")
    assert loaded_skill is not None
    assert len(loaded_skill.prompt_template) == 400
    assert loaded_skill.prompt_template == "x" * 400
    assert loaded_skill.tools == ["bash"]
    assert loaded_skill.dependencies == ["base"]
    assert loaded_registry.get_composition("bundle") == ["base", "longform"]
    assert saved_path.endswith("skills.json")
