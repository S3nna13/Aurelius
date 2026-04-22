"""Integration tests for the matryoshka embedding surface."""

from __future__ import annotations

import importlib

import src
import src.model as model_pkg


def test_matryoshka_embedding_exported_from_model_package() -> None:
    assert hasattr(model_pkg, "MatryoshkaConfig")
    assert hasattr(model_pkg, "MatryoshkaEmbedding")
    assert "MatryoshkaConfig" in model_pkg.__all__
    assert "MatryoshkaEmbedding" in model_pkg.__all__


def test_matryoshka_registry_wired_through_package() -> None:
    cls = model_pkg.MODEL_COMPONENT_REGISTRY["matryoshka_embedding"]
    assert cls is model_pkg.MatryoshkaEmbedding


def test_src_namespace_exposes_public_subpackages() -> None:
    assert src.model is importlib.import_module("src.model")
    assert src.chat is importlib.import_module("src.chat")
    assert src.eval is importlib.import_module("src.eval")
    assert "model" in src.__all__
    assert "chat" in src.__all__
