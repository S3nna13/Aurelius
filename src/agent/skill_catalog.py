"""Local-first skill catalog and bundle loader for Aurelius.

The catalog indexes skill-pack folders from repository-local, workspace,
global, and archived locations. It keeps provenance explicit, uses the
repo's existing skill scanner for lightweight safety scoring, and never
talks to a network service.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

_SAFE_SKILL_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

from src.model.interface_framework import InterfaceFrameworkError, SkillBundle
from src.safety.skill_scanner import SkillScanner, _parse_yaml_frontmatter

__all__ = [
    "SkillCatalogEntry",
    "SkillCatalog",
]


_SKILL_FILENAMES = ("skill.md", "SKILL.md")
_INSTRUCTION_FILENAMES = ("AGENTS.md", "SOUL.md", "TOOLS.md")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
    return value


def _validate_skill_id(skill_id: str) -> str:
    """Validate *skill_id* is a safe filesystem identifier.

    Rejects path separators, parent-directory references, and shell
    metacharacters to prevent directory-traversal and command-injection
    attacks.
    """
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise InterfaceFrameworkError("skill_id must be a non-empty string")
    if not _SAFE_SKILL_ID_RE.match(skill_id):
        raise InterfaceFrameworkError(
            f"skill_id {skill_id!r} contains invalid characters; "
            "only alphanumeric characters, hyphens, and underscores are allowed"
        )
    # Defence-in-depth: explicit rejection of traversal segments
    lower = skill_id.lower()
    if ".." in lower or "//" in skill_id or "\\" in skill_id:
        raise InterfaceFrameworkError(
            f"skill_id {skill_id!r} contains path-traversal patterns"
        )
    return skill_id


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raise InterfaceFrameworkError("expected a sequence of strings, got bare str")
    try:
        items = tuple(value)
    except TypeError as exc:  # pragma: no cover - defensive
        raise InterfaceFrameworkError(f"expected an iterable, got {type(value).__name__}") from exc
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise InterfaceFrameworkError("sequence entries must be non-empty strings")
    return items


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_text(path: Path) -> str:
    if not path.exists():
        raise InterfaceFrameworkError(f"missing skill artifact: {path}")
    if not path.is_file():
        raise InterfaceFrameworkError(f"skill artifact is not a file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise InterfaceFrameworkError(f"skill artifact is empty: {path}")
    return text


def _first_existing(root: Path, filenames: Iterable[str]) -> Path | None:
    for name in filenames:
        path = root / name
        if path.exists():
            return path
    return None


def _normalize_skill_scope(scope: Any) -> str:
    if scope is None:
        return "thread"
    value = str(scope).strip().casefold().replace("_", "-")
    if value in {"global", "org", "repo", "thread"}:
        return value
    if value in {"workspace", "workspace-local", "local", "session"}:
        return "thread"
    if value in {"repository", "repo-local", "project"}:
        return "repo"
    if value == "organization":
        return "org"
    raise InterfaceFrameworkError(f"unsupported skill scope: {scope!r}")


def _derive_description(body: str, fallback_name: str) -> str:
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        return line
    return fallback_name


@dataclass(frozen=True)
class SkillCatalogEntry:
    """Indexed skill pack metadata."""

    skill_id: str
    name: str
    description: str
    scope: str
    instructions: str
    scripts: tuple[str, ...]
    resources: tuple[str, ...]
    entrypoints: tuple[str, ...]
    version: str | None
    provenance: str | None
    source_path: str
    source_kind: str
    archived: bool
    active: bool
    risk_score: float
    allow_level: str
    metadata: dict[str, Any] = field(default_factory=dict)
    findings: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_skill_id(self.skill_id)
        for field_name in ("name", "description", "scope", "instructions", "source_path", "source_kind", "allow_level"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.scope not in {"global", "org", "repo", "thread"}:
            raise InterfaceFrameworkError(
                f"scope must be one of ['global', 'org', 'repo', 'thread'], got {self.scope!r}"
            )
        for tuple_name in ("scripts", "resources", "entrypoints"):
            value = getattr(self, tuple_name)
            if not isinstance(value, tuple):
                raise InterfaceFrameworkError(f"{tuple_name} must be a tuple")
            if not all(isinstance(item, str) and item for item in value):
                raise InterfaceFrameworkError(f"{tuple_name} entries must be non-empty strings")
        if self.version is not None and not isinstance(self.version, str):
            raise InterfaceFrameworkError("version must be str or None")
        if self.provenance is not None and not isinstance(self.provenance, str):
            raise InterfaceFrameworkError("provenance must be str or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")
        if not isinstance(self.findings, tuple):
            raise InterfaceFrameworkError("findings must be a tuple")

    def to_bundle(self) -> SkillBundle:
        return SkillBundle(
            skill_id=self.skill_id,
            name=self.name,
            description=self.description,
            scope=self.scope,
            instructions=self.instructions,
            scripts=self.scripts,
            resources=self.resources,
            entrypoints=self.entrypoints,
            version=self.version,
            provenance=self.provenance,
            source_path=self.source_path,
            metadata={
                **self.metadata,
                "source_kind": self.source_kind,
                "archived": self.archived,
                "active": self.active,
                "risk_score": self.risk_score,
                "allow_level": self.allow_level,
                "declared_scope": self.metadata.get("declared_scope", self.scope),
                "normalized_scope": self.scope,
                "findings": list(self.findings),
            },
        )


class SkillCatalog:
    """Index, search, install, and activate skill packs locally."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        global_root: str | Path | None = None,
        archive_roots: Iterable[str | Path] | None = None,
        state_dir: str | Path | None = None,
        scanner: SkillScanner | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.exists():
            raise InterfaceFrameworkError(f"repo_root does not exist: {self.repo_root}")
        self.global_root = Path(global_root).expanduser().resolve() if global_root is not None else (Path.home() / ".aurelius" / "skills")
        default_archive_roots = (
            self.repo_root / ".aurelius" / "archive" / "skills",
            Path.home() / ".aurelius" / "archive" / "skills",
        )
        roots = archive_roots if archive_roots is not None else default_archive_roots
        self.archive_roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self.state_dir = Path(state_dir).expanduser().resolve() if state_dir is not None else (self.repo_root / ".aurelius" / "skills")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.scanner = scanner or SkillScanner()
        self._entries: dict[str, SkillCatalogEntry] = {}
        self._active_skill_ids: set[str] = set(self._load_active_skill_ids())

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def discover(self, *, refresh: bool = True) -> list[SkillCatalogEntry]:
        if refresh:
            self._entries.clear()
            for source_kind, root in self._iter_roots():
                self._discover_root(root, source_kind=source_kind)
        return sorted(self._entries.values(), key=lambda item: (item.skill_id, item.source_kind))

    def list(
        self,
        *,
        scope: str | None = None,
        include_archived: bool = True,
        active: bool | None = None,
    ) -> list[SkillCatalogEntry]:
        if not self._entries:
            self.discover()
        entries = list(self._entries.values())
        if scope is not None:
            entries = [entry for entry in entries if entry.scope == _normalize_skill_scope(scope)]
        if not include_archived:
            entries = [entry for entry in entries if not entry.archived]
        if active is not None:
            entries = [entry for entry in entries if entry.active is active]
        return sorted(entries, key=lambda item: (item.skill_id, item.source_kind))

    def search(self, query: str) -> list[SkillCatalogEntry]:
        if not isinstance(query, str) or not query.strip():
            raise InterfaceFrameworkError("query must be a non-empty string")
        needle = query.casefold()
        matches = []
        for entry in self.list():
            haystack = " ".join(
                [
                    entry.skill_id,
                    entry.name,
                    entry.description,
                    entry.scope,
                    entry.instructions,
                    entry.provenance or "",
                    entry.source_kind,
                    entry.source_path,
                    entry.metadata.get("declared_scope", ""),
                    entry.metadata.get("normalized_scope", ""),
                ]
            ).casefold()
            if needle in haystack:
                matches.append(entry)
        return matches

    def provenance_summary(self) -> dict[str, Any]:
        entries = self.list()
        source_kinds: dict[str, int] = {}
        provenances: dict[str, int] = {}
        allow_levels: dict[str, int] = {}
        scopes: dict[str, int] = {}
        for entry in entries:
            source_kinds[entry.source_kind] = source_kinds.get(entry.source_kind, 0) + 1
            provenance_key = entry.provenance or "unknown"
            provenances[provenance_key] = provenances.get(provenance_key, 0) + 1
            allow_levels[entry.allow_level] = allow_levels.get(entry.allow_level, 0) + 1
            scopes[entry.scope] = scopes.get(entry.scope, 0) + 1
        return {
            "count": len(entries),
            "active_count": sum(1 for entry in entries if entry.active),
            "archived_count": sum(1 for entry in entries if entry.archived),
            "source_kinds": dict(sorted(source_kinds.items())),
            "provenances": dict(sorted(provenances.items())),
            "allow_levels": dict(sorted(allow_levels.items())),
            "scopes": dict(sorted(scopes.items())),
        }

    def get(self, skill_id: str) -> SkillCatalogEntry | None:
        if not self._entries:
            self.discover()
        return self._entries.get(skill_id)

    def load_bundle(self, skill_id: str) -> SkillBundle:
        entry = self.get(skill_id)
        if entry is None:
            raise InterfaceFrameworkError(f"unknown skill: {skill_id!r}")
        return entry.to_bundle()

    def resolve_skill_bundle(self, skill_id: str) -> SkillBundle:
        return self.load_bundle(skill_id)

    def resolve_skill_bundles(self, skill_ids: Iterable[str]) -> tuple[SkillBundle, ...]:
        bundles: list[SkillBundle] = []
        for skill_id in skill_ids:
            entry = self.get(skill_id)
            if entry is None:
                raise InterfaceFrameworkError(f"unknown skill: {skill_id!r}")
            bundles.append(entry.to_bundle())
        return tuple(bundles)

    def activate(self, skill_id: str) -> SkillBundle:
        entry = self.get(skill_id)
        if entry is None:
            raise InterfaceFrameworkError(f"unknown skill: {skill_id!r}")
        self._active_skill_ids.add(skill_id)
        self._persist_active_skill_ids()
        updated = SkillCatalogEntry(
            **{
                **entry.__dict__,
                "active": True,
            }
        )
        self._entries[skill_id] = updated
        return updated.to_bundle()

    def deactivate(self, skill_id: str) -> SkillBundle:
        entry = self.get(skill_id)
        if entry is None:
            raise InterfaceFrameworkError(f"unknown skill: {skill_id!r}")
        self._active_skill_ids.discard(skill_id)
        self._persist_active_skill_ids()
        updated = SkillCatalogEntry(
            **{
                **entry.__dict__,
                "active": False,
            }
        )
        self._entries[skill_id] = updated
        return updated.to_bundle()

    def archive(self, skill_id: str, *, reason: str | None = None) -> SkillBundle:
        entry = self.get(skill_id)
        if entry is None:
            raise InterfaceFrameworkError(f"unknown skill: {skill_id!r}")
        if entry.archived:
            raise InterfaceFrameworkError(f"skill is already archived: {skill_id!r}")
        source_root = Path(entry.source_path).expanduser().resolve()
        if not source_root.exists() or not source_root.is_dir():
            raise InterfaceFrameworkError(f"skill source does not exist: {source_root}")
        archive_root = self._archive_install_root()
        archive_root.mkdir(parents=True, exist_ok=True)
        destination = archive_root / Path(entry.skill_id).name
        destination = destination.resolve()
        archive_root_resolved = archive_root.resolve()
        try:
            destination.relative_to(archive_root_resolved)
        except ValueError as exc:
            raise InterfaceFrameworkError(
                f"skill_id {entry.skill_id!r} resolves outside archive root"
            ) from exc
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source_root), str(destination))
        metadata_path = destination / "metadata.json"
        metadata = dict(entry.metadata)
        metadata.update(
            {
                "archived": True,
                "archived_at": _utc_now(),
                "archived_from": str(source_root),
                "archived_reason": reason,
                "source_kind": entry.source_kind,
                "provenance": entry.provenance,
            }
        )
        metadata_path.write_text(
            json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._active_skill_ids.discard(skill_id)
        self._persist_active_skill_ids()
        archived = self._build_entry(destination, source_kind="archive", archived=True)
        self._entries[skill_id] = archived
        return archived.to_bundle()

    def install(self, source_path: str | Path, *, scope: str = "repo") -> SkillBundle:
        source = Path(source_path).expanduser().resolve()
        if not source.exists() or not source.is_dir():
            raise InterfaceFrameworkError(f"skill source must be a directory: {source}")
        entry = self._build_entry(source, source_kind=scope, archived=False)
        destination_root = self._install_root(scope)
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / Path(entry.skill_id).name
        destination = destination.resolve()
        root_resolved = destination_root.resolve()
        try:
            destination.relative_to(root_resolved)
        except ValueError as exc:
            raise InterfaceFrameworkError(
                f"skill_id {entry.skill_id!r} resolves outside install root"
            ) from exc
        if destination.exists():
            shutil.rmtree(destination)
        # Preserve symlinks as links instead of following them; if a symlink
        # points outside the source tree it is a potential info-disclosure
        # vector, so we validate below.
        shutil.copytree(source, destination, symlinks=True)
        # Validate no symlink escapes the skill root
        for item in destination.rglob("*"):
            if item.is_symlink():
                try:
                    target = item.resolve()
                    target.relative_to(destination)
                except ValueError as exc:
                    # Symlink escapes the skill root — remove the tree and reject
                    shutil.rmtree(destination)
                    raise InterfaceFrameworkError(
                        f"skill contains symlink escaping root: {item.name} -> {item.readlink()}"
                    ) from exc
        refreshed = self._build_entry(destination, source_kind=scope, archived=False)
        self._entries[refreshed.skill_id] = refreshed
        return refreshed.to_bundle()

    def describe(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "global_root": str(self.global_root),
            "state_dir": str(self.state_dir),
            "archive_roots": [str(root) for root in self.archive_roots],
            "skill_count": len(self.list()),
            "active_skill_ids": sorted(self._active_skill_ids),
            "archived_skill_ids": sorted(entry.skill_id for entry in self.list() if entry.archived),
        }

    def instruction_layers_for(
        self,
        *,
        workspace: str | Path | None = None,
        repo_root: str | Path | None = None,
        skill_ids: Iterable[str] = (),
        mode_name: str | None = None,
        memory_summary: str | None = None,
    ) -> tuple[str, ...]:
        layers: list[str] = []
        root = Path(repo_root).expanduser().resolve() if repo_root is not None else self.repo_root
        if root.exists():
            layers.extend(self._read_instruction_files(root, label="repo"))
        if workspace is not None:
            workspace_root = Path(workspace).expanduser().resolve()
            if workspace_root.exists():
                layers.extend(self._read_instruction_files(workspace_root, label="workspace"))
        if mode_name:
            layers.append(f"mode instructions: {mode_name}")
        for skill_id in skill_ids:
            entry = self.get(skill_id)
            if entry is None:
                raise InterfaceFrameworkError(f"unknown skill: {skill_id!r}")
            layers.append(f"skill instructions: {entry.skill_id}: {entry.instructions}")
        if memory_summary:
            layers.append(f"thread memory summary: {memory_summary}")
        return tuple(layers)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _iter_roots(self) -> Iterable[tuple[str, Path]]:
        yield ("repo", self.repo_root / "skills")
        yield ("local", self.state_dir)
        yield ("global", self.global_root)
        for root in self.archive_roots:
            yield ("archive", root)

    def _install_root(self, scope: str) -> Path:
        if scope == "global":
            return self.global_root
        if scope == "archive":
            return self._archive_install_root()
        return self.state_dir

    def _archive_install_root(self) -> Path:
        return self.archive_roots[0] if self.archive_roots else (self.state_dir / "archive")

    def _discover_root(self, root: Path, *, source_kind: str) -> None:
        if not root.exists() or not root.is_dir():
            return
        candidates = []
        for filename in _SKILL_FILENAMES:
            candidates.extend(root.rglob(filename))
        for skill_md in sorted(set(candidates)):
            skill_root = skill_md.parent
            entry = self._build_entry(skill_root, source_kind=source_kind, archived=(source_kind == "archive"))
            self._entries.setdefault(entry.skill_id, entry)
            if entry.active:
                self._active_skill_ids.add(entry.skill_id)

    def _build_entry(self, root: Path, *, source_kind: str, archived: bool) -> SkillCatalogEntry:
        skill_md = _first_existing(root, _SKILL_FILENAMES)
        if skill_md is None:
            raise InterfaceFrameworkError(f"no skill.md or SKILL.md found under {root}")
        text = _read_text(skill_md)
        frontmatter, body = _parse_yaml_frontmatter(text)
        metadata_path = root / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(_read_text(metadata_path))
            except json.JSONDecodeError as exc:
                raise InterfaceFrameworkError(f"metadata.json must contain valid JSON: {metadata_path}") from exc
            if not isinstance(metadata, dict):
                raise InterfaceFrameworkError(f"metadata.json must decode to an object: {metadata_path}")
        merged = {**metadata, **frontmatter}
        skill_id = _validate_skill_id(str(merged.get("skill_id") or root.name))
        name = str(merged.get("name") or root.name)
        description = str(merged.get("description") or _derive_description(body, name))
        scope = _normalize_skill_scope(merged.get("scope"))
        instructions = str(merged.get("instructions") or body or description)
        scripts = _as_tuple(merged.get("scripts"))
        resources = _as_tuple(merged.get("resources"))
        entrypoints = _as_tuple(merged.get("entrypoints"))
        version = None if merged.get("version") is None else str(merged.get("version"))
        provenance = str(merged.get("provenance") or source_kind)
        scan = self.scanner.scan_markdown(text)
        findings = tuple(_json_safe(asdict(finding)) for finding in scan.findings)
        active = skill_id in self._active_skill_ids or bool(merged.get("active"))
        return SkillCatalogEntry(
            skill_id=skill_id,
            name=name,
            description=description,
            scope=scope,
            instructions=instructions,
            scripts=scripts,
            resources=resources,
            entrypoints=entrypoints,
            version=version,
            provenance=provenance,
            source_path=str(root),
            source_kind=source_kind,
            archived=archived or bool(merged.get("archived", False)),
            active=active,
            risk_score=float(scan.risk_score),
            allow_level=str(scan.allow_level),
            metadata={
                **merged,
                "skill_path": str(skill_md),
                "body": body,
                "allow_level": scan.allow_level,
                "risk_score": scan.risk_score,
            },
            findings=findings,
        )

    def _load_active_skill_ids(self) -> set[str]:
        path = self.state_dir / "active_skills.json"
        if not path.exists():
            return set()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InterfaceFrameworkError(f"active_skills.json must contain valid JSON: {path}") from exc
        if not isinstance(payload, list):
            raise InterfaceFrameworkError(f"active_skills.json must be a list: {path}")
        return {str(item) for item in payload if isinstance(item, str)}

    def _persist_active_skill_ids(self) -> None:
        path = self.state_dir / "active_skills.json"
        path.write_text(
            json.dumps(sorted(self._active_skill_ids), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _read_instruction_files(self, root: Path, *, label: str) -> list[str]:
        layers: list[str] = []
        for name in _INSTRUCTION_FILENAMES:
            path = root / name
            if not path.exists() or not path.is_file():
                continue
            text = _read_text(path)
            layers.append(f"{label} instructions: {path}\n{text}")
        return layers
