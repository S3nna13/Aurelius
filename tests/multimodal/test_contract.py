"""Unit tests for ``src.multimodal.contract``."""

from __future__ import annotations

import json

import pytest

from src.multimodal import (
    AUDIO_MODALITY_CONTRACT,
    DOCUMENT_MODALITY_CONTRACT,
    VISION_MODALITY_CONTRACT,
    ModalityContract,
    ModalityContractError,
    describe_modality_registry,
    dump_modality_contract,
    get_modality_contract,
    list_modality_contracts,
    load_modality_contract,
    register_modality_contract,
)


def test_builtin_modality_registry_summaries_are_json_safe():
    names = list_modality_contracts()
    summary = describe_modality_registry()

    assert names == ("vision", "audio", "document")
    assert summary["count"] == 3
    assert summary["names"] == ["vision", "audio", "document"]
    assert json.dumps(summary)
    assert summary["contracts"][0]["kind_summary"].startswith("vision:")
    assert summary["contracts"][1]["kind_summary"].startswith("audio:")
    assert summary["contracts"][2]["kind_summary"].startswith("document:")
    assert VISION_MODALITY_CONTRACT is get_modality_contract("vision")
    assert AUDIO_MODALITY_CONTRACT is get_modality_contract("audio")
    assert DOCUMENT_MODALITY_CONTRACT is get_modality_contract("document")


def test_modality_contract_load_dump_round_trip():
    payload = {
        "name": "diagram",
        "description": "Diagram-style structured visual inputs and outputs.",
        "input_kinds": ["diagram_image", "flowchart", "screenshot"],
        "output_kinds": ["structured_text", "nodes", "edges"],
        "schema_version": "1.0.0",
    }

    contract = load_modality_contract(payload)
    dumped = dump_modality_contract(contract)

    assert contract.name == "diagram"
    assert contract.input_kinds == ("diagram_image", "flowchart", "screenshot")
    assert contract.output_kinds == ("structured_text", "nodes", "edges")
    assert dumped == {
        "name": "diagram",
        "description": "Diagram-style structured visual inputs and outputs.",
        "schema_version": "1.0.0",
        "input_kinds": ["diagram_image", "flowchart", "screenshot"],
        "output_kinds": ["structured_text", "nodes", "edges"],
    }
    assert json.dumps(contract.summary())
    assert json.dumps(dumped)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "missing fields"),
        (
            {
                "name": "vision_plus",
                "description": "Extra field should fail.",
                "input_kinds": ["image"],
                "output_kinds": ["caption"],
                "unexpected": True,
            },
            "unknown fields",
        ),
        (
            {
                "name": "bad name",
                "description": "Whitespace in the name is not allowed.",
                "input_kinds": ["image"],
                "output_kinds": ["caption"],
            },
            "name must match",
        ),
        (
            {
                "name": "empty-inputs",
                "description": "Empty inputs are invalid.",
                "input_kinds": [],
                "output_kinds": ["caption"],
            },
            "input_kinds must not be empty",
        ),
        (
            {
                "name": "duplicate-output",
                "description": "Duplicate outputs are invalid.",
                "input_kinds": ["image"],
                "output_kinds": ["caption", "caption"],
            },
            "duplicate entries",
        ),
        (
            {
                "name": "bad-version",
                "description": "Invalid schema version.",
                "input_kinds": ["image"],
                "output_kinds": ["caption"],
                "schema_version": "v1",
            },
            "schema_version must match",
        ),
    ],
)
def test_modality_contract_validation_is_loud(payload, message):
    with pytest.raises(ModalityContractError, match=message):
        load_modality_contract(payload)


def test_registry_rejects_duplicate_registration():
    duplicate = ModalityContract(
        name="vision",
        description="Duplicate registry entry.",
        input_kinds=("image",),
        output_kinds=("caption",),
    )

    with pytest.raises(ModalityContractError, match="already registered"):
        register_modality_contract(duplicate)


def test_registry_rejects_unknown_lookup():
    with pytest.raises(ModalityContractError, match="not registered"):
        get_modality_contract("missing")
