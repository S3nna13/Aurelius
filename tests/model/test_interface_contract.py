"""Unit tests for ``src.model.interface_contract``."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.model.interface_contract import (
    InterfaceContractBundle,
    InterfaceContractError,
    InterfaceContractPaths,
    load_interface_contract_bundle,
    load_interface_contract_json,
    load_interface_contract_markdown,
    load_interface_contract_prompt_yaml_text,
    load_interface_contract_schema,
    resolve_interface_contract_paths,
    validate_interface_contract,
)


_REQUIRED_TOP_LEVEL_SECTIONS = (
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

_REQUIRED_SOURCE_PRODUCT_KEYS = (
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


def _build_contract_payload() -> dict:
    return {
        "metadata": {
            "schema_version": "1.0",
            "title": "Aurelius Canonical Interface Contract",
            "date": "2026-04-21",
            "status": "draft",
            "scope": ["cli", "ide"],
            "doc_path": "docs/plans/2026-04-21-aurelius-canonical-interface-contract.md",
        },
        "purpose": {
            "summary": "Canonical agent interface.",
            "target_hosts": ["terminal-cli"],
            "shared_contract": ["instruction layering"],
        },
        "contract_principles": [{"id": "P1", "statement": "No silent fallbacks."}],
        "canonical_nouns": {
            "task_thread": {
                "description": "top-level task",
                "fields": ["thread_id"],
                "status_values": ["draft"],
            },
            "mode": {
                "description": "policy preset",
                "canonical_names": ["ask"],
            },
            "instruction_layer": {
                "description": "instruction source",
                "precedence_order": ["system policy"],
            },
            "skill": {
                "description": "capability bundle",
                "fields": ["skill_id"],
                "scope_values": ["repo"],
            },
            "approval": {
                "description": "human decision",
                "categories": ["file write"],
            },
            "checkpoint": {
                "description": "durable snapshot",
                "minimum_contents": ["thread metadata"],
            },
            "subagent": {
                "description": "nested task",
                "inherits": ["task context"],
                "inherits_by_default": ["mutable execution state"],
            },
            "background_job": {
                "description": "detached task",
                "required_capabilities": ["status polling"],
            },
            "tool_call": {
                "description": "structured tool invocation",
                "normalized_fields": ["tool_name"],
            },
        },
        "lifecycle": {
            "create": ["create thread"],
            "plan": ["record plan"],
            "approve": ["request approval"],
            "act": ["emit tool call"],
            "observe": ["record output"],
            "checkpoint": ["save state"],
            "resume": ["restore state"],
            "complete": ["write final artifact"],
        },
        "mode_semantics": {
            "ask": {"intent": "clarify", "tool_policy": "minimal", "default_constraints": ["no writes"]},
            "code": {"intent": "implement", "tool_policy": "edit", "default_constraints": ["approval"]},
            "debug": {"intent": "debug", "tool_policy": "inspect", "default_constraints": ["approval"]},
            "architect": {"intent": "design", "tool_policy": "read", "default_constraints": ["plan"]},
            "review": {"intent": "audit", "tool_policy": "read", "default_constraints": ["risks"]},
            "background": {"intent": "run", "tool_policy": "persistent", "default_constraints": ["checkpoint"]},
            "chat": {"intent": "chat", "tool_policy": "minimal", "default_constraints": ["escalate"]},
        },
        "skill_contract": {
            "packaging_shape": ["skill-name/SKILL.md"],
            "minimum_expectations": ["portable instructions"],
            "loading_order": ["system policy"],
            "openclaw_extensions": ["workspace-scoped skills"],
        },
        "approval_contract": {
            "required_fields": ["action summary"],
            "decision_values": ["allow"],
            "hard_rule": "Never infer approval from silence.",
        },
        "checkpoint_contract": {
            "capabilities": ["serialization"],
            "recommended_state": ["thread metadata"],
        },
        "host_adapters": {
            "cli": {"must_support": ["mode selection"]},
            "ide": {"must_support": ["inline approvals"]},
            "web": {"must_support": ["thread list"]},
            "plugin": {"must_support": ["status"]},
            "gateway": {"must_support": ["message routing"]},
        },
        "source_product_patterns": {
            key: {
                "reference_url": f"https://example.com/{key}",
                "patterns": [f"{key} pattern"],
            }
            for key in _REQUIRED_SOURCE_PRODUCT_KEYS
        },
        "aurelius_substrate": {
            "agent": ["src/agent/react_loop.py"],
            "serving": ["src/serving/api_server.py"],
            "ui": ["src/ui/ui_surface.py"],
            "model_governance": ["src/model/manifest.py"],
        },
        "non_goals": ["model architecture"],
        "implementation_sequence": ["thread dataclasses"],
        "open_questions": ["Should modes allow aliases?"],
    }


def _build_schema_payload() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Aurelius Canonical Interface Contract Schema",
        "type": "object",
        "required": list(_REQUIRED_TOP_LEVEL_SECTIONS),
    }


def _build_prompt_yaml_text() -> str:
    return "\n".join([
        "metadata:",
        "  title: Aurelius Canonical Interface Contract",
        "  doc_path: docs/plans/2026-04-21-aurelius-canonical-interface-contract.md",
        "  schema_path: docs/plans/2026-04-21-aurelius-canonical-interface-contract.schema.json",
        "  json_path: docs/plans/2026-04-21-aurelius-canonical-interface-contract.json",
        "",
        "purpose:",
        "  summary: canonical interface",
        "",
    ])


def _write_artifacts(
    repo_root: Path,
    *,
    contract_payload: dict | None = None,
    schema_payload: dict | None = None,
    prompt_yaml_text: str | None = None,
    markdown_text: str = "# Aurelius Canonical Interface Contract\n",
) -> InterfaceContractPaths:
    plans_dir = repo_root / "docs" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    paths = InterfaceContractPaths.resolve(repo_root=repo_root)
    paths.markdown_path.write_text(markdown_text, encoding="utf-8")
    paths.json_path.write_text(
        json.dumps(contract_payload or _build_contract_payload(), indent=2),
        encoding="utf-8",
    )
    paths.schema_path.write_text(
        json.dumps(schema_payload or _build_schema_payload(), indent=2),
        encoding="utf-8",
    )
    paths.prompt_yaml_path.write_text(
        prompt_yaml_text or _build_prompt_yaml_text(),
        encoding="utf-8",
    )
    return paths


def test_paths_resolve_relative_to_repo_root(tmp_path):
    paths = InterfaceContractPaths.resolve(repo_root=tmp_path)
    direct_paths = resolve_interface_contract_paths(repo_root=tmp_path)
    assert paths.repo_root == tmp_path.resolve()
    assert direct_paths == paths
    assert paths.markdown_path == tmp_path / "docs" / "plans" / (
        "2026-04-21-aurelius-canonical-interface-contract.md"
    )
    assert paths.json_path.name.endswith(".json")
    assert paths.schema_path.name.endswith(".schema.json")
    assert paths.prompt_yaml_path.name.endswith(".prompt.yaml")


def test_load_bundle_reads_and_validates_all_contract_artifacts(tmp_path):
    _write_artifacts(tmp_path)

    bundle = load_interface_contract_bundle(repo_root=tmp_path)

    assert isinstance(bundle, InterfaceContractBundle)
    assert bundle.contract_json["metadata"]["title"] == "Aurelius Canonical Interface Contract"
    assert bundle.schema_json["required"] == list(_REQUIRED_TOP_LEVEL_SECTIONS)
    assert "schema_path" in bundle.prompt_yaml_text
    assert bundle.markdown_text.startswith("# Aurelius Canonical Interface Contract")

    assert load_interface_contract_markdown(repo_root=tmp_path) == bundle.markdown_text
    assert load_interface_contract_json(repo_root=tmp_path) == bundle.contract_json
    assert load_interface_contract_schema(repo_root=tmp_path) == bundle.schema_json
    assert (
        load_interface_contract_prompt_yaml_text(repo_root=tmp_path)
        == bundle.prompt_yaml_text
    )


def test_validate_interface_contract_rejects_missing_top_level_section(tmp_path):
    contract_payload = _build_contract_payload()
    del contract_payload["mode_semantics"]
    _write_artifacts(tmp_path, contract_payload=contract_payload)

    with pytest.raises(InterfaceContractError, match="missing required top-level sections"):
        load_interface_contract_bundle(repo_root=tmp_path)


def test_validate_interface_contract_rejects_missing_source_product_pattern(tmp_path):
    contract_payload = _build_contract_payload()
    del contract_payload["source_product_patterns"]["codex"]
    paths = _write_artifacts(tmp_path, contract_payload=contract_payload)

    contract_json = load_interface_contract_json(paths=paths)
    schema_json = load_interface_contract_schema(paths=paths)
    markdown_text = load_interface_contract_markdown(paths=paths)
    prompt_yaml_text = load_interface_contract_prompt_yaml_text(paths=paths)

    with pytest.raises(InterfaceContractError, match="missing required projects"):
        validate_interface_contract(
            contract_json,
            paths=paths,
            schema_json=schema_json,
            markdown_text=markdown_text,
            prompt_yaml_text=prompt_yaml_text,
        )


def test_load_interface_contract_json_fails_loudly_on_malformed_json(tmp_path):
    paths = InterfaceContractPaths.resolve(repo_root=tmp_path)
    paths.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    paths.json_path.write_text("{ bad json", encoding="utf-8")

    with pytest.raises(InterfaceContractError, match="failed to parse interface contract JSON"):
        load_interface_contract_json(repo_root=tmp_path)


def test_validate_interface_contract_rejects_prompt_yaml_with_missing_companion_path(tmp_path):
    prompt_yaml_text = "\n".join([
        "metadata:",
        "  title: Aurelius Canonical Interface Contract",
        "  doc_path: docs/plans/2026-04-21-aurelius-canonical-interface-contract.md",
        "  json_path: docs/plans/2026-04-21-aurelius-canonical-interface-contract.json",
        "",
    ])
    _write_artifacts(tmp_path, prompt_yaml_text=prompt_yaml_text)

    with pytest.raises(InterfaceContractError, match="prompt YAML metadata missing companion paths"):
        load_interface_contract_bundle(repo_root=tmp_path)


def test_validate_interface_contract_uses_canonical_metadata_doc_path(tmp_path):
    contract_payload = copy.deepcopy(_build_contract_payload())
    contract_payload["metadata"]["doc_path"] = "docs/plans/not-the-contract.md"
    _write_artifacts(tmp_path, contract_payload=contract_payload)

    with pytest.raises(InterfaceContractError, match="metadata.doc_path must match"):
        load_interface_contract_bundle(repo_root=tmp_path)
