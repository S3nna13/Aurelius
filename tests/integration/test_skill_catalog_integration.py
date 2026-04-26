"""Integration tests for ``src.agent.skill_catalog``."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import SkillCatalog, SkillCatalogEntry


def _repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "skills").mkdir(parents=True)
    return root


def _global_root(tmp_path: Path) -> Path:
    root = tmp_path / "global"
    root.mkdir()
    return root


def _write_skill(
    root: Path, *, skill_id: str, body: str, scope: str | None = None, frontmatter: bool = True
) -> None:
    skill_root = root / "skills" / skill_id
    skill_root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if frontmatter:
        lines.append("---")
        lines.append(f"skill_id: {skill_id}")
        lines.append(f"name: {skill_id.replace('-', ' ').title()}")
        if scope is not None:
            lines.append(f"scope: {scope}")
        lines.append("---")
    lines.extend([body, ""])
    (skill_root / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_skill_catalog_discovery_precedence_and_round_trip(tmp_path):
    repo_root = _repo_root(tmp_path)
    global_root = _global_root(tmp_path)

    _write_skill(
        repo_root, skill_id="body-only", body="Body-only integration skill.", frontmatter=False
    )
    _write_skill(
        repo_root, skill_id="normalized-scope", body="Normalized scope body.", scope="workspace"
    )
    _write_skill(repo_root, skill_id="duplicate", body="Repo version.", scope="repo")
    _write_skill(global_root, skill_id="duplicate", body="Global version.", scope="global")

    catalog = SkillCatalog(
        repo_root,
        global_root=global_root,
        state_dir=tmp_path / ".aurelius" / "skills",
    )
    entries = catalog.discover()
    body_entry = catalog.get("body-only")
    duplicate_entry = catalog.get("duplicate")
    resolved = catalog.resolve_skill_bundle("body-only")
    visible = catalog.list(scope="workspace")

    assert isinstance(body_entry, SkillCatalogEntry)
    assert entries[0].skill_id == "body-only"
    assert body_entry.scope == "thread"
    assert body_entry.description == "Body-only integration skill."
    assert duplicate_entry.description == "Repo version."
    assert duplicate_entry.provenance == "repo"
    assert resolved.metadata["normalized_scope"] == "thread"
    assert visible[0].skill_id == "body-only"
    json.dumps(catalog.describe())
