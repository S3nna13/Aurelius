"""Integration tests for the Aurelius interface contract loader."""

from __future__ import annotations

from pathlib import Path

import src.model as model_pkg
from src.model import (
    InterfaceContractBundle,
    InterfaceContractPaths,
    load_interface_contract_bundle,
    validate_interface_contract,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_public_src_model_exports_interface_contract_api():
    for name in (
        "InterfaceContractBundle",
        "InterfaceContractPaths",
        "InterfaceContractError",
        "load_interface_contract_bundle",
        "load_interface_contract_json",
        "load_interface_contract_schema",
        "load_interface_contract_markdown",
        "load_interface_contract_prompt_yaml_text",
        "validate_interface_contract",
    ):
        assert hasattr(model_pkg, name)


def test_load_interface_contract_bundle_reads_real_repo_artifacts():
    bundle = load_interface_contract_bundle(repo_root=_repo_root())

    assert isinstance(bundle, InterfaceContractBundle)
    assert bundle.paths == InterfaceContractPaths.resolve(repo_root=_repo_root())
    assert bundle.contract_json["metadata"]["title"] == "Aurelius Canonical Interface Contract"
    assert bundle.contract_json["metadata"]["doc_path"] == (
        "docs/plans/2026-04-21-aurelius-canonical-interface-contract.md"
    )
    assert "source_product_patterns" in bundle.contract_json
    assert "codex" in bundle.contract_json["source_product_patterns"]
    assert "cline" in bundle.contract_json["source_product_patterns"]
    assert bundle.schema_json["title"] == "Aurelius Canonical Interface Contract Schema"
    assert "schema_path" in bundle.prompt_yaml_text
    assert bundle.markdown_text.startswith("# Aurelius Canonical Interface Contract")


def test_validate_interface_contract_accepts_real_contract_artifacts():
    bundle = load_interface_contract_bundle(repo_root=_repo_root())

    validate_interface_contract(
        bundle.contract_json,
        paths=bundle.paths,
        schema_json=bundle.schema_json,
        markdown_text=bundle.markdown_text,
        prompt_yaml_text=bundle.prompt_yaml_text,
    )
