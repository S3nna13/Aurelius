"""Aurelius - 1.3B decoder-only LLM."""

from __future__ import annotations

from importlib import import_module

_PUBLIC_SUBPACKAGES = (
    "agent",
    "alignment",
    "backends",
    "chat",
    "compression",
    "computer_use",
    "data",
    "deployment",
    "eval",
    "federation",
    "inference",
    "longcontext",
    "mcp",
    "memory",
    "model",
    "monitoring",
    "multiagent",
    "multimodal",
    "protocol",
    "reasoning",
    "retrieval",
    "safety",
    "search",
    "serving",
    "simulation",
    "tools",
    "training",
    "ui",
)

__all__ = list(_PUBLIC_SUBPACKAGES)


def __getattr__(name: str):
    if name in _PUBLIC_SUBPACKAGES:
        module = import_module(f"src.{name}")
        globals()[name] = module
        return module
    # Legacy top-level aliases: model → src.model
    if name == "model":
        _src_model = import_module("src.model")
        globals()["model"] = _src_model
        return _src_model
    if name == "safety":
        _src_safety = import_module("src.safety")
        globals()["safety"] = _src_safety
        return _src_safety
    raise AttributeError(f"module 'src' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC_SUBPACKAGES))
