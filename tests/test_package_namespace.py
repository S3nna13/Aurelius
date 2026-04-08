"""Tests for the public Aurelius package namespace."""

from importlib import import_module


def test_aurelius_model_aliases_src_model():
    """`aurelius.model` should resolve to the legacy `src.model` package."""
    aurelius_model = import_module("aurelius.model")
    src_model = import_module("src.model")

    assert aurelius_model is src_model


def test_aurelius_transformer_matches_src_transformer():
    """Deep imports through `aurelius.*` should resolve from the same source tree."""
    aurelius_transformer = import_module("aurelius.model.transformer")
    src_transformer = import_module("src.model.transformer")

    assert aurelius_transformer.__file__ == src_transformer.__file__
