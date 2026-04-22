"""Runtime loader for the canonical Aurelius interface contract artifacts.

The interface contract lives in ``docs/plans`` as four companion artifacts:

    - markdown spec
    - machine-readable JSON contract
    - JSON schema for the contract
    - prompt YAML consumed as raw text

This module resolves those artifacts relative to the repository root, loads
them with stdlib-only helpers, validates a small set of runtime-critical
invariants, and returns a frozen bundle for downstream callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "InterfaceContractBundle",
    "InterfaceContractPaths",
    "InterfaceContractError",
    "resolve_interface_contract_paths",
    "load_interface_contract_bundle",
    "load_interface_contract_json",
    "load_interface_contract_schema",
    "load_interface_contract_markdown",
    "load_interface_contract_prompt_yaml_text",
    "validate_interface_contract",
]


_CONTRACT_STEM = "2026-04-21-aurelius-canonical-interface-contract"
_CONTRACT_DIR = Path("docs/plans")

_REQUIRED_TOP_LEVEL_SECTIONS: tuple[str, ...] = (
    "metadata",
    "purpose",
    "contract_principles",
    "canonical_nouns",
    "lifecycle",
    "mode_semantics",
    "skill_contract",
    "approval_contract",
    "checkpoint_contract",
    "host_adapters",
    "source_product_patterns",
    "aurelius_substrate",
    "non_goals",
    "implementation_sequence",
    "open_questions",
)

_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "schema_version",
    "title",
    "date",
    "status",
    "scope",
    "doc_path",
)

_REQUIRED_SOURCE_PRODUCT_KEYS: tuple[str, ...] = (
    "cline",
    "codex",
    "codex_plugin_cc",
    "openai_skills",
    "openclaw",
    "openclaw_clawhub",
    "openclaw_skills_archive",
    "openclaw_acpx",
    "openclaw_lobster",
    "openclaw_ansible",
    "ironclaw",
)

_REQUIRED_PROMPT_YAML_PATH_KEYS: tuple[str, ...] = (
    "doc_path",
    "schema_path",
    "json_path",
)


class InterfaceContractError(Exception):
    """Raised when interface contract artifacts are missing or malformed."""


@dataclass(frozen=True)
class InterfaceContractPaths:
    """Resolved filesystem locations for the interface contract artifacts."""

    repo_root: Path
    markdown_path: Path
    json_path: Path
    schema_path: Path
    prompt_yaml_path: Path

    @classmethod
    def resolve(
        cls,
        repo_root: str | Path | None = None,
    ) -> "InterfaceContractPaths":
        """Resolve canonical artifact locations relative to the repo root."""
        return resolve_interface_contract_paths(repo_root=repo_root)


@dataclass(frozen=True)
class InterfaceContractBundle:
    """Loaded contract artifacts plus their resolved paths."""

    paths: InterfaceContractPaths
    markdown_text: str
    contract_json: dict[str, Any]
    schema_json: dict[str, Any]
    prompt_yaml_text: str


def resolve_interface_contract_paths(
    *,
    repo_root: str | Path | None = None,
) -> InterfaceContractPaths:
    """Resolve the canonical contract artifact paths relative to the repo root."""
    root = _resolve_repo_root(repo_root)
    artifact_root = root / _CONTRACT_DIR / _CONTRACT_STEM
    return InterfaceContractPaths(
        repo_root=root,
        markdown_path=artifact_root.with_suffix(".md"),
        json_path=artifact_root.with_suffix(".json"),
        schema_path=artifact_root.with_suffix(".schema.json"),
        prompt_yaml_path=artifact_root.with_suffix(".prompt.yaml"),
    )


def load_interface_contract_markdown(
    *,
    repo_root: str | Path | None = None,
    paths: InterfaceContractPaths | None = None,
) -> str:
    """Load the canonical markdown contract as UTF-8 text."""
    resolved_paths = _coerce_paths(paths=paths, repo_root=repo_root)
    return _read_text_file(resolved_paths.markdown_path, "interface contract markdown")


def load_interface_contract_json(
    *,
    repo_root: str | Path | None = None,
    paths: InterfaceContractPaths | None = None,
) -> dict[str, Any]:
    """Load the machine-readable interface contract JSON."""
    resolved_paths = _coerce_paths(paths=paths, repo_root=repo_root)
    return _read_json_file(resolved_paths.json_path, "interface contract JSON")


def load_interface_contract_schema(
    *,
    repo_root: str | Path | None = None,
    paths: InterfaceContractPaths | None = None,
) -> dict[str, Any]:
    """Load the JSON schema companion for the interface contract."""
    resolved_paths = _coerce_paths(paths=paths, repo_root=repo_root)
    return _read_json_file(
        resolved_paths.schema_path,
        "interface contract schema JSON",
    )


def load_interface_contract_prompt_yaml_text(
    *,
    repo_root: str | Path | None = None,
    paths: InterfaceContractPaths | None = None,
) -> str:
    """Load the prompt YAML companion as raw UTF-8 text."""
    resolved_paths = _coerce_paths(paths=paths, repo_root=repo_root)
    return _read_text_file(
        resolved_paths.prompt_yaml_path,
        "interface contract prompt YAML",
    )


def validate_interface_contract(
    contract_json: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    paths: InterfaceContractPaths | None = None,
    schema_json: Mapping[str, Any] | None = None,
    markdown_text: str | None = None,
    prompt_yaml_text: str | None = None,
) -> None:
    """Validate runtime-critical invariants for the canonical contract."""
    if not isinstance(contract_json, Mapping):
        raise InterfaceContractError(
            "interface contract must be a mapping, got "
            f"{type(contract_json).__name__}"
        )

    resolved_paths = _coerce_paths(paths=paths, repo_root=repo_root)
    _validate_required_sections(contract_json)
    _validate_metadata(contract_json, resolved_paths)
    _validate_source_product_patterns(contract_json)
    _validate_companion_artifacts(
        resolved_paths,
        schema_json=schema_json,
        markdown_text=markdown_text,
        prompt_yaml_text=prompt_yaml_text,
    )


def load_interface_contract_bundle(
    *,
    repo_root: str | Path | None = None,
    paths: InterfaceContractPaths | None = None,
) -> InterfaceContractBundle:
    """Load and validate the canonical interface contract artifact bundle."""
    resolved_paths = _coerce_paths(paths=paths, repo_root=repo_root)
    markdown_text = load_interface_contract_markdown(paths=resolved_paths)
    contract_json = load_interface_contract_json(paths=resolved_paths)
    schema_json = load_interface_contract_schema(paths=resolved_paths)
    prompt_yaml_text = load_interface_contract_prompt_yaml_text(paths=resolved_paths)

    validate_interface_contract(
        contract_json,
        paths=resolved_paths,
        schema_json=schema_json,
        markdown_text=markdown_text,
        prompt_yaml_text=prompt_yaml_text,
    )

    return InterfaceContractBundle(
        paths=resolved_paths,
        markdown_text=markdown_text,
        contract_json=contract_json,
        schema_json=schema_json,
        prompt_yaml_text=prompt_yaml_text,
    )


def _coerce_paths(
    *,
    paths: InterfaceContractPaths | None,
    repo_root: str | Path | None,
) -> InterfaceContractPaths:
    if paths is None:
        return InterfaceContractPaths.resolve(repo_root=repo_root)
    if repo_root is not None:
        raise InterfaceContractError(
            "pass either paths or repo_root when resolving interface contract artifacts"
        )
    if not isinstance(paths, InterfaceContractPaths):
        raise InterfaceContractError(
            f"paths must be InterfaceContractPaths, got {type(paths).__name__}"
        )
    return paths


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(repo_root).expanduser().resolve()
    if not root.exists():
        raise InterfaceContractError(f"repo root does not exist: {root}")
    if not root.is_dir():
        raise InterfaceContractError(f"repo root is not a directory: {root}")
    return root


def _read_text_file(path: Path, label: str) -> str:
    _ensure_readable_file(path, label)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InterfaceContractError(
            f"failed to read {label} at {path}: {exc}"
        ) from exc
    if not text.strip():
        raise InterfaceContractError(f"{label} is empty: {path}")
    return text


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    text = _read_text_file(path, label)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InterfaceContractError(
            f"failed to parse {label} at {path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise InterfaceContractError(
            f"{label} must decode to a JSON object, got {type(payload).__name__}"
        )
    return payload


def _ensure_readable_file(path: Path, label: str) -> None:
    if not path.exists():
        raise InterfaceContractError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise InterfaceContractError(f"{label} is not a file: {path}")


def _validate_required_sections(contract_json: Mapping[str, Any]) -> None:
    missing = [
        section
        for section in _REQUIRED_TOP_LEVEL_SECTIONS
        if section not in contract_json
    ]
    if missing:
        raise InterfaceContractError(
            f"interface contract missing required top-level sections: {missing}"
        )


def _validate_metadata(
    contract_json: Mapping[str, Any],
    paths: InterfaceContractPaths,
) -> None:
    metadata = _require_mapping(contract_json, "metadata", "interface contract")
    missing = [field for field in _REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise InterfaceContractError(
            f"interface contract metadata missing required fields: {missing}"
        )
    doc_path = _require_non_empty_string(
        metadata,
        "doc_path",
        "interface contract metadata",
    )
    expected_doc_path = _relative_repo_path(paths.repo_root, paths.markdown_path)
    if doc_path != expected_doc_path:
        raise InterfaceContractError(
            "interface contract metadata.doc_path must match the canonical markdown "
            f"path {expected_doc_path!r}, got {doc_path!r}"
        )
    _ensure_readable_file(paths.repo_root / doc_path, "metadata.doc_path")


def _validate_source_product_patterns(contract_json: Mapping[str, Any]) -> None:
    patterns = _require_mapping(
        contract_json,
        "source_product_patterns",
        "interface contract",
    )
    missing = [key for key in _REQUIRED_SOURCE_PRODUCT_KEYS if key not in patterns]
    if missing:
        raise InterfaceContractError(
            "interface contract source_product_patterns missing required projects: "
            f"{missing}"
        )
    for key in _REQUIRED_SOURCE_PRODUCT_KEYS:
        entry = _require_mapping(patterns, key, "source_product_patterns")
        reference_url = _require_non_empty_string(
            entry,
            "reference_url",
            f"source_product_patterns.{key}",
        )
        if not reference_url.startswith(("https://", "http://")):
            raise InterfaceContractError(
                f"source_product_patterns.{key}.reference_url must be an HTTP URL, "
                f"got {reference_url!r}"
            )
        pattern_list = _require_string_sequence(
            entry.get("patterns"),
            f"source_product_patterns.{key}.patterns",
        )
        if not pattern_list:
            raise InterfaceContractError(
                f"source_product_patterns.{key}.patterns must not be empty"
            )


def _validate_companion_artifacts(
    paths: InterfaceContractPaths,
    *,
    schema_json: Mapping[str, Any] | None,
    markdown_text: str | None,
    prompt_yaml_text: str | None,
) -> None:
    for label, artifact_path in (
        ("interface contract markdown", paths.markdown_path),
        ("interface contract JSON", paths.json_path),
        ("interface contract schema JSON", paths.schema_path),
        ("interface contract prompt YAML", paths.prompt_yaml_path),
    ):
        _ensure_readable_file(artifact_path, label)

    if markdown_text is not None and not markdown_text.strip():
        raise InterfaceContractError("interface contract markdown is empty")

    if schema_json is not None:
        schema_required = _require_string_sequence(
            schema_json.get("required"),
            "interface contract schema.required",
        )
        missing = [
            section
            for section in _REQUIRED_TOP_LEVEL_SECTIONS
            if section not in schema_required
        ]
        if missing:
            raise InterfaceContractError(
                "interface contract schema.required missing top-level sections: "
                f"{missing}"
            )

    if prompt_yaml_text is None:
        return
    if not prompt_yaml_text.strip():
        raise InterfaceContractError("interface contract prompt YAML is empty")

    prompt_metadata_paths = _extract_prompt_yaml_metadata_paths(prompt_yaml_text)
    missing_prompt_paths = [
        key for key in _REQUIRED_PROMPT_YAML_PATH_KEYS if key not in prompt_metadata_paths
    ]
    if missing_prompt_paths:
        raise InterfaceContractError(
            "interface contract prompt YAML metadata missing companion paths: "
            f"{missing_prompt_paths}"
        )

    expected_paths = {
        "doc_path": _relative_repo_path(paths.repo_root, paths.markdown_path),
        "schema_path": _relative_repo_path(paths.repo_root, paths.schema_path),
        "json_path": _relative_repo_path(paths.repo_root, paths.json_path),
    }
    for key, expected_value in expected_paths.items():
        actual_value = prompt_metadata_paths[key]
        if actual_value != expected_value:
            raise InterfaceContractError(
                f"interface contract prompt YAML metadata.{key} must be "
                f"{expected_value!r}, got {actual_value!r}"
            )
        _ensure_readable_file(paths.repo_root / actual_value, f"prompt YAML metadata.{key}")


def _extract_prompt_yaml_metadata_paths(prompt_yaml_text: str) -> dict[str, str]:
    metadata_paths: dict[str, str] = {}
    in_metadata_block = False
    metadata_indent: int | None = None

    for raw_line in prompt_yaml_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if not in_metadata_block:
            if stripped == "metadata:":
                in_metadata_block = True
                metadata_indent = indent
            continue

        if metadata_indent is not None and indent <= metadata_indent:
            break
        if ":" not in stripped:
            continue

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in _REQUIRED_PROMPT_YAML_PATH_KEYS and value:
            metadata_paths[key] = value

    return metadata_paths


def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise InterfaceContractError(
            f"{context}.{key} must be a mapping, got {type(value).__name__}"
        )
    return value


def _require_non_empty_string(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InterfaceContractError(
            f"{context}.{key} must be a non-empty string, got {value!r}"
        )
    return value


def _require_string_sequence(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise InterfaceContractError(
            f"{context} must be a sequence of strings, got bare str"
        )
    if value is None:
        raise InterfaceContractError(f"{context} must be present")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise InterfaceContractError(
            f"{context} must be iterable, got {type(value).__name__}"
        ) from exc
    for item in items:
        if not isinstance(item, str) or not item:
            raise InterfaceContractError(
                f"{context} entries must be non-empty strings, got {item!r}"
            )
    return items


def _relative_repo_path(repo_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise InterfaceContractError(
            f"path {path} is not inside repo root {repo_root}"
        ) from exc
    return relative.as_posix()
