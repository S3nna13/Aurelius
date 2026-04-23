"""Surface catalog helpers for Aurelius.

This module gathers JSON-safe summaries of the local package surfaces that
front the runtime: backends, engine adapters, multimodal registries, MCP,
computer-use, deployment, and UI surfaces.  It stays import-light by loading
the relevant modules lazily inside helper functions.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

_SURFACE_CATALOG_SCHEMA_NAME = "aurelius.interface.surface-catalog"
_SURFACE_CATALOG_SCHEMA_VERSION = "1.0"


def _sorted_names(mapping: dict[str, Any]) -> list[str]:
    return sorted(mapping.keys())


def _backend_record(adapter: Any) -> dict[str, Any]:
    contract = adapter.contract
    return {
        "backend_name": contract.backend_name,
        "adapter_class": type(adapter).__name__,
        "contract": asdict(contract),
        "runtime_info": adapter.runtime_info(),
    }


def _engine_record(adapter: Any, *, registered: bool) -> dict[str, Any]:
    contract = adapter.contract
    return {
        "backend_name": contract.backend_name,
        "adapter_class": type(adapter).__name__,
        "contract": asdict(contract),
        "available": bool(adapter.is_available()),
        "supported_ops": list(adapter.supported_ops()),
        "summary": adapter.describe(),
        "registered": registered,
    }


def _class_records(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in sorted(mapping.keys()):
        value = mapping[name]
        records.append(
            {
                "name": name,
                "class_name": type(value).__name__,
                "module": type(value).__module__,
            }
        )
    return records


def describe_backend_surface() -> dict[str, Any]:
    """Describe the registered training / math backend surface."""
    import src.backends as backends

    names = backends.list_backends()
    records = [_backend_record(backends.get_backend(name)) for name in names]
    return {
        "count": len(records),
        "names": list(names),
        "backends": records,
    }


def describe_engine_surface() -> dict[str, Any]:
    """Describe the known inference-engine adapter surface."""
    import src.backends as backends
    from src.backends.gguf_engine_adapter import GGUFEngineAdapter
    from src.backends.sglang_adapter import SGLangEngineAdapter
    from src.backends.vllm_adapter import VLLMEngineAdapter

    registry_names = backends.list_engine_adapters()
    discovered = [
        VLLMEngineAdapter(),
        SGLangEngineAdapter(),
        GGUFEngineAdapter(),
    ]
    records = [
        _engine_record(adapter, registered=adapter.contract.backend_name in registry_names)
        for adapter in discovered
    ]
    return {
        "count": len(records),
        "names": [record["backend_name"] for record in records],
        "registry_count": len(registry_names),
        "registry_names": list(registry_names),
        "engine_adapters": records,
    }


def describe_multimodal_surface() -> dict[str, Any]:
    """Describe the multimodal modality / encoder surface."""
    from src.multimodal.contract import describe_modality_registry
    from src.multimodal.multimodal_registry import (
        AUDIO_ENCODER_REGISTRY,
        MODALITY_PROJECTOR_REGISTRY,
        MODALITY_TOKENIZER_REGISTRY,
        VISION_ENCODER_REGISTRY,
    )

    return {
        "contracts": describe_modality_registry(),
        "vision_encoders": {
            "count": len(VISION_ENCODER_REGISTRY),
            "names": _sorted_names(VISION_ENCODER_REGISTRY),
        },
        "audio_encoders": {
            "count": len(AUDIO_ENCODER_REGISTRY),
            "names": _sorted_names(AUDIO_ENCODER_REGISTRY),
        },
        "modality_projectors": {
            "count": len(MODALITY_PROJECTOR_REGISTRY),
            "names": _sorted_names(MODALITY_PROJECTOR_REGISTRY),
        },
        "modality_tokenizers": {
            "count": len(MODALITY_TOKENIZER_REGISTRY),
            "names": _sorted_names(MODALITY_TOKENIZER_REGISTRY),
        },
    }


def describe_mcp_surface() -> dict[str, Any]:
    """Describe the local MCP server/client/tool-schema surface."""
    from src.mcp.mcp_client import MCP_CLIENT_REGISTRY
    from src.mcp.mcp_server import MCP_SERVER_REGISTRY
    from src.mcp.tool_schema_registry import MCP_TOOL_SCHEMA_REGISTRY

    return {
        "servers": {
            "count": len(MCP_SERVER_REGISTRY),
            "names": _sorted_names(MCP_SERVER_REGISTRY),
        },
        "clients": {
            "count": len(MCP_CLIENT_REGISTRY),
            "names": _sorted_names(MCP_CLIENT_REGISTRY),
        },
        "tool_schemas": {
            "count": len(MCP_TOOL_SCHEMA_REGISTRY),
            "names": _sorted_names(MCP_TOOL_SCHEMA_REGISTRY),
        },
    }


def describe_computer_use_surface() -> dict[str, Any]:
    """Describe the computer-use screen, action, and safety surface."""
    from src.computer_use.action_verifier import VERIFIER_DENY_LIST
    from src.computer_use.gui_action import GUI_ACTION_REGISTRY
    from src.computer_use.screen_parser import SCREEN_PARSER_REGISTRY

    return {
        "screen_parsers": {
            "count": len(SCREEN_PARSER_REGISTRY),
            "names": _sorted_names(SCREEN_PARSER_REGISTRY),
        },
        "action_predictors": {
            "count": len(GUI_ACTION_REGISTRY),
            "names": _sorted_names(GUI_ACTION_REGISTRY),
        },
        "denylist": {
            "count": len(VERIFIER_DENY_LIST),
            "patterns": sorted(VERIFIER_DENY_LIST),
        },
    }


def describe_deployment_surface() -> dict[str, Any]:
    """Describe the deployment artifact and health-check surface."""
    from src.deployment.container_builder import ARTIFACT_BUILDER_REGISTRY
    from src.deployment.healthz import DEPLOY_TARGET_REGISTRY, HEALTHZ_REGISTRY

    return {
        "deploy_targets": {
            "count": len(DEPLOY_TARGET_REGISTRY),
            "names": _sorted_names(DEPLOY_TARGET_REGISTRY),
        },
        "artifact_builders": {
            "count": len(ARTIFACT_BUILDER_REGISTRY),
            "names": _sorted_names(ARTIFACT_BUILDER_REGISTRY),
        },
        "healthz": {
            "count": len(HEALTHZ_REGISTRY),
            "names": _sorted_names(HEALTHZ_REGISTRY),
        },
    }


def describe_ui_surface() -> dict[str, Any]:
    """Describe the terminal/IDE UI helper surface."""
    from src.ui.command_palette import COMMAND_PALETTE_REGISTRY
    from src.ui.keyboard_nav import KEYBOARD_NAV_REGISTRY
    from src.ui.motion import MOTION_REGISTRY
    from src.ui.onboarding import ONBOARDING_REGISTRY
    from src.ui.panel_layout import PANEL_LAYOUT_REGISTRY
    from src.ui.status_hierarchy import STATUS_TREE_REGISTRY
    from src.ui.ui_surface import UI_SURFACE_REGISTRY

    return {
        "ui_surfaces": {
            "count": len(UI_SURFACE_REGISTRY),
            "names": _sorted_names(UI_SURFACE_REGISTRY),
        },
        "motions": {
            "count": len(MOTION_REGISTRY),
            "names": _sorted_names(MOTION_REGISTRY),
        },
        "panel_layouts": {
            "count": len(PANEL_LAYOUT_REGISTRY),
            "names": _sorted_names(PANEL_LAYOUT_REGISTRY),
        },
        "command_palette": {
            "count": len(COMMAND_PALETTE_REGISTRY),
            "names": _sorted_names(COMMAND_PALETTE_REGISTRY),
        },
        "status_hierarchy": {
            "count": len(STATUS_TREE_REGISTRY),
            "names": _sorted_names(STATUS_TREE_REGISTRY),
        },
        "keyboard_nav": {
            "count": len(KEYBOARD_NAV_REGISTRY),
            "names": _sorted_names(KEYBOARD_NAV_REGISTRY),
        },
        "onboarding": {
            "count": len(ONBOARDING_REGISTRY),
            "names": _sorted_names(ONBOARDING_REGISTRY),
        },
    }


def surface_catalog_schema() -> dict[str, Any]:
    """Return the versioned schema for surface catalog summaries."""
    return {
        "schema_name": _SURFACE_CATALOG_SCHEMA_NAME,
        "schema_version": _SURFACE_CATALOG_SCHEMA_VERSION,
        "description": "Versioned schema for Aurelius terminal surface catalogs.",
        "sections": [
            {"name": "backends", "required": True, "description": "Registered training / math backends."},
            {"name": "engine_adapters", "required": True, "description": "Known inference-engine adapters."},
            {"name": "multimodal", "required": True, "description": "Modality contracts and multimodal registries."},
            {"name": "mcp", "required": True, "description": "MCP client, server, and tool-schema registries."},
            {"name": "computer_use", "required": True, "description": "Screen parsers, action predictors, and denylist."},
            {"name": "deployment", "required": True, "description": "Deployment targets, artifact builders, and health checks."},
            {"name": "ui", "required": True, "description": "Terminal UI surfaces, motion, and onboarding helpers."},
        ],
    }


def describe_surface_catalog() -> dict[str, Any]:
    """Return a JSON-safe overview of all exposed Aurelius surfaces."""
    return {
        "schema": surface_catalog_schema(),
        "backends": describe_backend_surface(),
        "engine_adapters": describe_engine_surface(),
        "multimodal": describe_multimodal_surface(),
        "mcp": describe_mcp_surface(),
        "computer_use": describe_computer_use_surface(),
        "deployment": describe_deployment_surface(),
        "ui": describe_ui_surface(),
    }


__all__ = [
    "describe_backend_surface",
    "describe_engine_surface",
    "describe_multimodal_surface",
    "describe_mcp_surface",
    "describe_computer_use_surface",
    "describe_deployment_surface",
    "describe_ui_surface",
    "describe_surface_catalog",
    "surface_catalog_schema",
]
