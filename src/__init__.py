"""Aurelius - 1.3B decoder-only LLM."""

from __future__ import annotations

from importlib import import_module

_PUBLIC_SUBPACKAGES = (
    "agent",
    "alignment",
    "backends",
    "chat",
    "computer_use",
    "data",
    "deployment",
    "eval",
    "inference",
    "longcontext",
    "mcp",
    "memory",
    "model",
    "multimodal",
    "reasoning",
    "retrieval",
    "safety",
    "search",
    "serving",
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
    raise AttributeError(f"module 'src' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC_SUBPACKAGES))
