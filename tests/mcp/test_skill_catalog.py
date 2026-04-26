"""Tests for src.mcp.skill_catalog — SkillCatalog, SkillMetadata, and friends."""

from __future__ import annotations

import time

import pytest

from src.mcp.skill_catalog import (
    DEFAULT_SKILL_CATALOG,
    SKILL_CATALOG_REGISTRY,
    SkillCatalog,
    SkillCatalogError,
    SkillInvocationRecord,
    SkillMetadata,
    SkillTrigger,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(
    skill_id: str = "test-skill",
    version: str = "1.0.0",
    tags: list[str] | None = None,
    enabled: bool = True,
    trigger_pattern: str = "test trigger",
) -> SkillMetadata:
    return SkillMetadata(
        skill_id=skill_id,
        name=f"Test Skill ({skill_id})",
        version=version,
        description="A test skill for unit tests.",
        triggers=[
            SkillTrigger(
                pattern=trigger_pattern,
                description="Fires on test trigger text.",
                examples=["example trigger phrase"],
            )
        ],
        tags=tags or ["test"],
        enabled=enabled,
    )


def _fresh_catalog() -> SkillCatalog:
    """Return a blank catalog for each test."""
    return SkillCatalog()


# ---------------------------------------------------------------------------
# 1. DEFAULT_SKILL_CATALOG has 3 pre-registered skills
# ---------------------------------------------------------------------------


def test_default_catalog_has_three_skills():
    skills = DEFAULT_SKILL_CATALOG.list_skills(enabled_only=False)
    assert len(skills) == 3


def test_default_catalog_skill_ids():
    ids = {s.skill_id for s in DEFAULT_SKILL_CATALOG.list_skills(enabled_only=False)}
    assert ids == {"code-review", "test-generator", "doc-writer"}


# ---------------------------------------------------------------------------
# 2. register() valid skill → get() returns it
# ---------------------------------------------------------------------------


def test_register_and_get():
    catalog = _fresh_catalog()
    skill = _make_skill()
    catalog.register(skill)
    retrieved = catalog.get("test-skill")
    assert retrieved.skill_id == "test-skill"
    assert retrieved.version == "1.0.0"


# ---------------------------------------------------------------------------
# 3. register() empty skill_id → SkillCatalogError
# ---------------------------------------------------------------------------


def test_register_empty_skill_id_raises():
    catalog = _fresh_catalog()
    skill = _make_skill(skill_id="")
    with pytest.raises(SkillCatalogError, match="skill_id"):
        catalog.register(skill)


def test_register_whitespace_skill_id_raises():
    catalog = _fresh_catalog()
    skill = _make_skill(skill_id="   ")
    with pytest.raises(SkillCatalogError, match="skill_id"):
        catalog.register(skill)


# ---------------------------------------------------------------------------
# 4. register() bad version → SkillCatalogError
# ---------------------------------------------------------------------------


def test_register_bad_version_v1_raises():
    catalog = _fresh_catalog()
    skill = _make_skill(version="v1")
    with pytest.raises(SkillCatalogError, match="version"):
        catalog.register(skill)


def test_register_bad_version_partial_raises():
    catalog = _fresh_catalog()
    skill = _make_skill(version="1.0")
    with pytest.raises(SkillCatalogError, match="version"):
        catalog.register(skill)


def test_register_bad_version_alpha_raises():
    catalog = _fresh_catalog()
    skill = _make_skill(version="1.0.0-beta")
    with pytest.raises(SkillCatalogError, match="version"):
        catalog.register(skill)


# ---------------------------------------------------------------------------
# 5. unregister() → skill gone; unknown id → SkillCatalogError
# ---------------------------------------------------------------------------


def test_unregister_removes_skill():
    catalog = _fresh_catalog()
    catalog.register(_make_skill())
    catalog.unregister("test-skill")
    with pytest.raises(SkillCatalogError):
        catalog.get("test-skill")


def test_unregister_unknown_id_raises():
    catalog = _fresh_catalog()
    with pytest.raises(SkillCatalogError, match="skill not found"):
        catalog.unregister("no-such-skill")


# ---------------------------------------------------------------------------
# 6. find_by_trigger("review") → returns "code-review" skill
# ---------------------------------------------------------------------------


def test_find_by_trigger_code_review():
    results = DEFAULT_SKILL_CATALOG.find_by_trigger("review")
    ids = {s.skill_id for s in results}
    assert "code-review" in ids


def test_find_by_trigger_case_insensitive():
    results = DEFAULT_SKILL_CATALOG.find_by_trigger("REVIEW THIS CODE")
    ids = {s.skill_id for s in results}
    assert "code-review" in ids


def test_find_by_trigger_no_match():
    catalog = _fresh_catalog()
    catalog.register(_make_skill(trigger_pattern="unique-xyz-999"))
    results = catalog.find_by_trigger("nothing matching here")
    assert results == []


# ---------------------------------------------------------------------------
# 7. find_by_tag("generation") → returns matching skills
# ---------------------------------------------------------------------------


def test_find_by_tag_generation():
    results = DEFAULT_SKILL_CATALOG.find_by_tag("generation")
    ids = {s.skill_id for s in results}
    # "test-generator" and "doc-writer" both carry "generation"
    assert "test-generator" in ids
    assert "doc-writer" in ids


def test_find_by_tag_no_match():
    catalog = _fresh_catalog()
    catalog.register(_make_skill(tags=["alpha", "beta"]))
    results = catalog.find_by_tag("gamma")
    assert results == []


# ---------------------------------------------------------------------------
# 8. list_skills(enabled_only=True) returns only enabled
# ---------------------------------------------------------------------------


def test_list_skills_enabled_only():
    catalog = _fresh_catalog()
    catalog.register(_make_skill(skill_id="enabled-one", enabled=True))
    catalog.register(_make_skill(skill_id="disabled-one", enabled=False))
    enabled = catalog.list_skills(enabled_only=True)
    ids = {s.skill_id for s in enabled}
    assert "enabled-one" in ids
    assert "disabled-one" not in ids


def test_list_skills_all():
    catalog = _fresh_catalog()
    catalog.register(_make_skill(skill_id="enabled-one", enabled=True))
    catalog.register(_make_skill(skill_id="disabled-one", enabled=False))
    all_skills = catalog.list_skills(enabled_only=False)
    ids = {s.skill_id for s in all_skills}
    assert "enabled-one" in ids
    assert "disabled-one" in ids


# ---------------------------------------------------------------------------
# 9. record_invocation + invocation_history round-trip
# ---------------------------------------------------------------------------


def test_record_and_retrieve_invocation():
    catalog = _fresh_catalog()
    catalog.register(_make_skill())
    record = SkillInvocationRecord(
        skill_id="test-skill",
        invoked_at=time.time(),
        args={"input": "hello"},
        result={"output": "world"},
        duration_ms=12.5,
    )
    catalog.record_invocation(record)
    history = catalog.invocation_history()
    assert len(history) == 1
    assert history[0].skill_id == "test-skill"
    assert history[0].result == {"output": "world"}


# ---------------------------------------------------------------------------
# 10. invocation_history(skill_id="code-review") filters correctly
# ---------------------------------------------------------------------------


def test_invocation_history_filtered():
    catalog = _fresh_catalog()
    catalog.register(_make_skill(skill_id="skill-a"))
    catalog.register(_make_skill(skill_id="skill-b"))

    catalog.record_invocation(
        SkillInvocationRecord(skill_id="skill-a", invoked_at=time.time(), args={})
    )
    catalog.record_invocation(
        SkillInvocationRecord(skill_id="skill-b", invoked_at=time.time(), args={})
    )
    catalog.record_invocation(
        SkillInvocationRecord(skill_id="skill-a", invoked_at=time.time(), args={})
    )

    a_history = catalog.invocation_history(skill_id="skill-a")
    assert len(a_history) == 2
    assert all(r.skill_id == "skill-a" for r in a_history)

    b_history = catalog.invocation_history(skill_id="skill-b")
    assert len(b_history) == 1


def test_invocation_history_all_when_skill_id_none():
    catalog = _fresh_catalog()
    catalog.register(_make_skill(skill_id="skill-x"))
    catalog.register(_make_skill(skill_id="skill-y"))
    catalog.record_invocation(
        SkillInvocationRecord(skill_id="skill-x", invoked_at=time.time(), args={})
    )
    catalog.record_invocation(
        SkillInvocationRecord(skill_id="skill-y", invoked_at=time.time(), args={})
    )
    assert len(catalog.invocation_history(skill_id=None)) == 2


# ---------------------------------------------------------------------------
# 11. to_dict() returns dict with skills and invocations
# ---------------------------------------------------------------------------


def test_to_dict_structure():
    catalog = _fresh_catalog()
    catalog.register(_make_skill())
    catalog.record_invocation(
        SkillInvocationRecord(skill_id="test-skill", invoked_at=1234.5, args={"x": 1})
    )
    d = catalog.to_dict()
    assert isinstance(d, dict)
    assert "skills" in d
    assert "invocations" in d
    assert "test-skill" in d["skills"]
    assert d["invocations"][0]["skill_id"] == "test-skill"


# ---------------------------------------------------------------------------
# 12. SKILL_CATALOG_REGISTRY contains "default"
# ---------------------------------------------------------------------------


def test_skill_catalog_registry_default():
    assert "default" in SKILL_CATALOG_REGISTRY
    assert SKILL_CATALOG_REGISTRY["default"] is DEFAULT_SKILL_CATALOG
