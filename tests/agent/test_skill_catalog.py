"""Unit tests for ``src.agent.skill_catalog``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.skill_catalog import SkillCatalog, SkillCatalogEntry
from src.model import InterfaceFrameworkError


def _repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "skills").mkdir(parents=True)
    return root


def _global_root(tmp_path: Path) -> Path:
    root = tmp_path / "global"
    root.mkdir()
    return root


def _write_skill_pack(
    root: Path,
    *,
    skill_id: str,
    body: str,
    name: str | None = None,
    scope: str | None = None,
    frontmatter: bool = True,
) -> Path:
    skill_root = root / "skills" / skill_id
    skill_root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if frontmatter:
        lines.append("---")
        lines.append(f"skill_id: {skill_id}")
        lines.append(f"name: {name or skill_id}")
        if scope is not None:
            lines.append(f"scope: {scope}")
        lines.append("---")
    lines.extend([body, ""])
    (skill_root / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return skill_root


def test_skill_catalog_discovers_body_only_skill_and_normalizes_scope_layers(tmp_path):
    repo_root = _repo_root(tmp_path)
    global_root = _global_root(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    _write_skill_pack(
        repo_root,
        skill_id="body-only",
        body="Body-only skill description.\n\nMore detail.",
        frontmatter=False,
    )
    _write_skill_pack(
        repo_root,
        skill_id="workspace-scope",
        body="Workspace scoped skill body.",
        scope="workspace",
    )
    _write_skill_pack(
        repo_root,
        skill_id="duplicate-skill",
        body="Repo local copy.",
        scope="repo",
    )
    _write_skill_pack(
        global_root,
        skill_id="duplicate-skill",
        body="Global copy.",
        scope="global",
    )
    (repo_root / "AGENTS.md").write_text("repo instructions", encoding="utf-8")
    (workspace_root / "SOUL.md").write_text("workspace soul", encoding="utf-8")
    (workspace_root / "TOOLS.md").write_text("workspace tools", encoding="utf-8")

    catalog = SkillCatalog(
        repo_root,
        global_root=global_root,
        state_dir=tmp_path / ".aurelius" / "skills",
    )
    entries = catalog.discover()
    body_entry = catalog.get("body-only")
    scoped_entry = catalog.get("workspace-scope")
    duplicate_entry = catalog.get("duplicate-skill")
    bundle = catalog.resolve_skill_bundle("body-only")
    layers = catalog.instruction_layers_for(
        workspace=workspace_root,
        repo_root=repo_root,
        skill_ids=("body-only",),
        mode_name="code",
        memory_summary="remember me",
    )

    assert isinstance(body_entry, SkillCatalogEntry)
    assert entries[0].skill_id == "body-only"
    assert body_entry.description == "Body-only skill description."
    assert body_entry.scope == "thread"
    assert bundle.scope == "thread"
    assert bundle.metadata["declared_scope"] == "thread"
    assert bundle.metadata["normalized_scope"] == "thread"
    assert scoped_entry.scope == "thread"
    assert duplicate_entry.description == "Repo local copy."
    assert duplicate_entry.provenance == "repo"
    assert catalog.list(scope="workspace")[0].skill_id == "body-only"
    assert any(layer.startswith("repo instructions:") for layer in layers)
    assert any(layer.startswith("workspace instructions:") for layer in layers)
    assert any(layer.startswith("skill instructions: body-only") for layer in layers)
    assert "mode instructions: code" in layers
    assert "thread memory summary: remember me" in layers
    json.dumps(catalog.describe())


def test_skill_catalog_install_activate_search_and_deactivate(tmp_path):
    repo_root = _repo_root(tmp_path)
    source_root = tmp_path / "source-skill"
    source_root.mkdir()
    (source_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: installed-skill",
                "name: Installed Skill",
                "scope: workspace",
                "---",
                "Installed skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    catalog = SkillCatalog(repo_root, state_dir=tmp_path / ".aurelius" / "skills")
    installed = catalog.install(source_root, scope="repo")
    activated = catalog.activate("installed-skill")
    matches = catalog.search("Installed Skill")
    resolved = catalog.resolve_skill_bundle("installed-skill")
    deactivated = catalog.deactivate("installed-skill")

    assert installed.skill_id == "installed-skill"
    assert installed.scope == "thread"
    assert activated.metadata["active"] is True
    assert matches and matches[0].skill_id == "installed-skill"
    assert resolved.skill_id == "installed-skill"
    assert deactivated.metadata["active"] is False
    assert any(entry.skill_id == "installed-skill" for entry in catalog.list(active=False))


def test_skill_catalog_archive_and_provenance_summary(tmp_path):
    repo_root = _repo_root(tmp_path)
    source_root = tmp_path / "archive-source"
    source_root.mkdir()
    (source_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: archive-me",
                "name: Archive Me",
                "scope: repo",
                "---",
                "Archiveable skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    catalog = SkillCatalog(repo_root, state_dir=tmp_path / ".aurelius" / "skills")
    installed = catalog.install(source_root, scope="repo")
    catalog.activate(installed.skill_id)
    summary_before = catalog.provenance_summary()
    archived = catalog.archive(installed.skill_id, reason="superseded")
    archived_entry = catalog.get(installed.skill_id)
    summary_after = catalog.provenance_summary()
    reloaded = SkillCatalog(repo_root, state_dir=tmp_path / ".aurelius" / "skills")
    reloaded_entry = reloaded.get(installed.skill_id)

    assert summary_before["active_count"] == 1
    assert archived.skill_id == installed.skill_id
    assert archived.metadata["archived"] is True
    assert archived.metadata["archived_reason"] == "superseded"
    assert archived_entry is not None
    assert archived_entry.archived is True
    assert archived_entry.active is False
    assert archived_entry.source_kind == "archive"
    assert reloaded_entry is not None
    assert reloaded_entry.archived is True
    assert reloaded_entry.source_kind == "archive"
    assert summary_after["archived_count"] == 1
    assert summary_after["active_count"] == 0
    assert summary_after["source_kinds"]["archive"] == 1


def test_skill_catalog_rejects_unknown_scope(tmp_path):
    repo_root = _repo_root(tmp_path)
    bad_root = repo_root / "skills" / "bad-scope"
    bad_root.mkdir(parents=True)
    (bad_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: bad-scope",
                "name: Bad Scope",
                "scope: impossible",
                "---",
                "Bad scope body.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    catalog = SkillCatalog(repo_root, state_dir=tmp_path / ".aurelius" / "skills")

    with pytest.raises(InterfaceFrameworkError, match="unsupported skill scope"):
        catalog.discover()
