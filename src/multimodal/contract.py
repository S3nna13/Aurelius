"""Aurelius multimodal contract and registry surface.

This module keeps the multimodal surface explicit and stdlib-only:

* ``ModalityContract`` models one modality family.
* Module-level registries expose built-in contracts for vision, audio,
  and document-style inputs / outputs.
* Validation is loud and rejects malformed payloads, duplicate names,
  duplicate kind entries, and unknown registry lookups.
* Summaries are JSON-safe dicts for CLI / integration surfaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any, Mapping

__all__ = [
    "ModalityContract",
    "ModalityContractError",
    "MODALITY_CONTRACT_REGISTRY",
    "VISION_MODALITY_CONTRACT",
    "AUDIO_MODALITY_CONTRACT",
    "DOCUMENT_MODALITY_CONTRACT",
    "load_modality_contract",
    "dump_modality_contract",
    "register_modality_contract",
    "get_modality_contract",
    "list_modality_contracts",
    "describe_modality_registry",
]


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_\-]*$")


class ModalityContractError(Exception):
    """Raised when a modality contract or registry operation is malformed."""


def _coerce_kind_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ModalityContractError(
            f"{field_name} must be a sequence of strings, got bare str"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ModalityContractError(
            f"{field_name} must be iterable, got {type(value).__name__}"
        ) from exc
    if not items:
        raise ModalityContractError(f"{field_name} must not be empty")
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ModalityContractError(
                f"{field_name} entries must be non-empty strings, got {item!r}"
            )
    if len(set(items)) != len(items):
        raise ModalityContractError(f"{field_name} contains duplicate entries")
    return items


@dataclass(frozen=True)
class ModalityContract:
    """Canonical contract for a modality family."""

    name: str
    description: str
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModalityContractError("name must be a non-empty string")
        name = self.name.strip()
        if not _NAME_RE.match(name):
            raise ModalityContractError(
                f"name must match [a-z][a-z0-9_-]*, got {self.name!r}"
            )
        object.__setattr__(self, "name", name)

        if not isinstance(self.description, str) or not self.description.strip():
            raise ModalityContractError("description must be a non-empty string")
        object.__setattr__(self, "description", self.description.strip())

        object.__setattr__(
            self,
            "input_kinds",
            _coerce_kind_tuple(self.input_kinds, "input_kinds"),
        )
        object.__setattr__(
            self,
            "output_kinds",
            _coerce_kind_tuple(self.output_kinds, "output_kinds"),
        )

        if not isinstance(self.schema_version, str) or not _SEMVER_RE.match(
            self.schema_version
        ):
            raise ModalityContractError(
                f"schema_version must match semver X.Y.Z, got {self.schema_version!r}"
            )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary for CLI and integration surfaces."""
        payload = dump_modality_contract(self)
        payload["input_count"] = len(self.input_kinds)
        payload["output_count"] = len(self.output_kinds)
        payload["kind_summary"] = (
            f"{self.name}: "
            f"{', '.join(self.input_kinds)} -> {', '.join(self.output_kinds)}"
        )
        return payload


_MODALITY_FIELDS = tuple(field.name for field in fields(ModalityContract))
_OPTIONAL_MODALITY_FIELDS = frozenset({"schema_version"})


def load_modality_contract(data: Mapping[str, Any]) -> ModalityContract:
    """Validate a mapping and construct a :class:`ModalityContract`."""
    if not isinstance(data, Mapping):
        raise ModalityContractError(
            f"modality contract data must be a mapping, got {type(data).__name__}"
        )
    missing = [
        field_name
        for field_name in _MODALITY_FIELDS
        if field_name not in data and field_name not in _OPTIONAL_MODALITY_FIELDS
    ]
    if missing:
        raise ModalityContractError(
            f"modality contract missing fields: {missing}"
        )
    extra = [key for key in data.keys() if key not in _MODALITY_FIELDS]
    if extra:
        raise ModalityContractError(
            f"modality contract has unknown fields: {extra}"
        )

    kwargs = {
        field_name: data[field_name]
        for field_name in _MODALITY_FIELDS
        if field_name in data
    }
    return ModalityContract(**kwargs)


def dump_modality_contract(contract: ModalityContract) -> dict[str, Any]:
    """Serialize a contract to a JSON-safe dict."""
    if not isinstance(contract, ModalityContract):
        raise ModalityContractError(
            f"dump_modality_contract expected ModalityContract, got "
            f"{type(contract).__name__}"
        )
    return {
        "name": contract.name,
        "description": contract.description,
        "schema_version": contract.schema_version,
        "input_kinds": list(contract.input_kinds),
        "output_kinds": list(contract.output_kinds),
    }


MODALITY_CONTRACT_REGISTRY: dict[str, ModalityContract] = {}


def register_modality_contract(contract: ModalityContract) -> ModalityContract:
    """Insert ``contract`` into the module-level registry."""
    if not isinstance(contract, ModalityContract):
        raise ModalityContractError(
            f"register_modality_contract expected ModalityContract, got "
            f"{type(contract).__name__}"
        )
    if contract.name in MODALITY_CONTRACT_REGISTRY:
        raise ModalityContractError(
            f"modality contract {contract.name!r} already registered"
        )
    MODALITY_CONTRACT_REGISTRY[contract.name] = contract
    return contract


def get_modality_contract(name: str) -> ModalityContract:
    """Return the registered contract ``name`` or raise ``ModalityContractError``."""
    if not isinstance(name, str) or not name.strip():
        raise ModalityContractError("name must be a non-empty string")
    normalized = name.strip()
    if normalized not in MODALITY_CONTRACT_REGISTRY:
        raise ModalityContractError(
            f"modality contract {normalized!r} not registered; "
            f"known contracts: {sorted(MODALITY_CONTRACT_REGISTRY)}"
        )
    return MODALITY_CONTRACT_REGISTRY[normalized]


def list_modality_contracts() -> tuple[str, ...]:
    """Return registered contract names in insertion order."""
    return tuple(MODALITY_CONTRACT_REGISTRY.keys())


def describe_modality_registry() -> dict[str, Any]:
    """Return a JSON-safe summary of the registry."""
    return {
        "count": len(MODALITY_CONTRACT_REGISTRY),
        "names": list(MODALITY_CONTRACT_REGISTRY.keys()),
        "contracts": [
            contract.summary()
            for contract in MODALITY_CONTRACT_REGISTRY.values()
        ],
    }


VISION_MODALITY_CONTRACT = register_modality_contract(
    ModalityContract(
        name="vision",
        description="Vision-style inputs and outputs.",
        input_kinds=("image", "image_batch", "screenshot", "video_frame"),
        output_kinds=("caption", "ocr_text", "detections", "embeddings"),
    )
)

AUDIO_MODALITY_CONTRACT = register_modality_contract(
    ModalityContract(
        name="audio",
        description="Audio-style inputs and outputs.",
        input_kinds=("waveform", "audio_clip", "spectrogram"),
        output_kinds=("transcript", "translation", "summary", "embeddings"),
    )
)

DOCUMENT_MODALITY_CONTRACT = register_modality_contract(
    ModalityContract(
        name="document",
        description="Document-style inputs and outputs.",
        input_kinds=("pdf", "markdown", "html", "docx", "plain_text"),
        output_kinds=("summary", "extracted_text", "tables", "citations"),
    )
)
