"""Tests for src.agent.skill_library."""

from __future__ import annotations

from src.agent.skill_library import Skill, VoyagerSkillLibrary


def _make_library(tmp_path) -> VoyagerSkillLibrary:
    return VoyagerSkillLibrary(storage_path=str(tmp_path))


def test_retrieve_uses_token_overlap(tmp_path) -> None:
    library = _make_library(tmp_path)
    library.add_skill(
        "file_tool",
        "def run():\n    return 'file'\n",
        description="read and write files",
    )
    library.add_skill(
        "calculator",
        "def run():\n    return 1\n",
        description="math arithmetic",
    )

    results = library.retrieve("please read a file from disk")
    assert [skill.name for skill in results] == ["file_tool"]


def test_verify_skill_uses_test_cases(tmp_path) -> None:
    library = _make_library(tmp_path)
    skill = library.add_skill(
        "adder",
        "def add_one(x):\n    return x + 1\n",
        description="increment values",
    )

    assert library.verify_skill(
        skill,
        [
            "assert add_one(1) == 2",
            {"assertion": "assert add_one(3) == 4"},
        ],
    ) is True

    assert library.verify_skill(skill, ["assert add_one(1) == 99"]) is False


def test_save_and_load_preserve_metadata(tmp_path) -> None:
    library = _make_library(tmp_path)
    skill = library.add_skill(
        "planner",
        "def plan():\n    return ['step']\n",
        description="planning skill",
        dependencies=["base"],
        embedding=[0.1, 0.2, 0.3],
    )
    skill.success_count = 4
    skill.failure_count = 1
    library.build_curriculum([("slow", 2), ("fast", 1)])
    library._skill_history.append({"skill": "planner", "result": "failure"})

    library.save()

    restored = _make_library(tmp_path)
    restored.load()

    restored_skill = restored.skills["planner"]
    assert isinstance(restored_skill, Skill)
    assert restored_skill.description == "planning skill"
    assert restored_skill.dependencies == ["base"]
    assert restored_skill.embedding == [0.1, 0.2, 0.3]
    assert restored_skill.success_count == 4
    assert restored_skill.failure_count == 1
    assert restored.next_task() == "fast"
    assert restored._skill_history == [{"skill": "planner", "result": "failure"}]
