"""Integration tests for the top-level ``src`` namespace."""

from __future__ import annotations

import importlib

import src


def test_src_package_lazily_exposes_model_and_chat() -> None:
    assert src.model is importlib.import_module("src.model")
    assert src.chat is importlib.import_module("src.chat")


def test_src_package_all_lists_public_surfaces() -> None:
    assert "model" in src.__all__
    assert "chat" in src.__all__
    assert "eval" in src.__all__
