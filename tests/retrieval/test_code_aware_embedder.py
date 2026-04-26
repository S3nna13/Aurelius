"""Tests for the code-aware dense embedder."""

from __future__ import annotations

import math

import pytest

from src.retrieval.code_aware_embedder import (
    CodeAwareEmbedder,
    CodeFeatures,
    split_identifier,
    stub_token_embed,
)

SNIPPET = '''\
import os
from typing import List

def get_user_name(user_id: int) -> str:
    """Return the user's name."""
    # look up the user
    return lookup(user_id)

class HTTPServer:
    pass
'''


def _fn(d=384):
    return lambda tok: stub_token_embed(tok, d)


# --------------------------------------------------------------------------- #
# split_identifier                                                            #
# --------------------------------------------------------------------------- #


def test_split_identifier_camel():
    assert split_identifier("getUserName") == ("get", "User", "Name")


def test_split_identifier_snake():
    assert split_identifier("get_user_name") == ("get", "user", "name")


def test_split_identifier_mixed():
    parts = split_identifier("my_getUserName")
    assert "my" in parts and "User" in parts and "Name" in parts


def test_split_identifier_acronym():
    assert split_identifier("parseXMLDoc") == ("parse", "XML", "Doc")


def test_split_identifier_empty():
    assert split_identifier("") == ()
    assert split_identifier("x") == ("x",)


# --------------------------------------------------------------------------- #
# extract_features                                                            #
# --------------------------------------------------------------------------- #


def test_extract_features_identifiers():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features("x = foo_bar + CamelCase")
    assert "foo_bar" in feats.identifiers
    assert "CamelCase" in feats.identifiers
    # Python keyword-like 'x' is not a keyword; should appear.
    assert "x" in feats.identifiers
    assert feats.n_tokens >= 3


def test_extract_features_imports():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features(SNIPPET)
    assert any("import os" in s for s in feats.imports)
    assert any("from typing import List" in s for s in feats.imports)


def test_extract_features_signatures():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features(SNIPPET)
    assert any("def get_user_name" in s for s in feats.signatures)
    assert any("class HTTPServer" in s for s in feats.signatures)


def test_extract_features_signature_no_body():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features("class Foo:\nclass Bar")
    assert any("class Foo" in s for s in feats.signatures)
    assert any("class Bar" in s for s in feats.signatures)


def test_extract_features_comments():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features(SNIPPET)
    assert any("# look up the user" in c for c in feats.comments)
    assert any('"""Return' in c for c in feats.comments)


def test_features_is_dataclass_with_tuples():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features("a=1")
    assert isinstance(feats, CodeFeatures)
    assert isinstance(feats.identifiers, tuple)
    assert isinstance(feats.imports, tuple)
    assert isinstance(feats.signatures, tuple)
    assert isinstance(feats.comments, tuple)


# --------------------------------------------------------------------------- #
# embed                                                                       #
# --------------------------------------------------------------------------- #


def test_embed_dim():
    emb = CodeAwareEmbedder(_fn(384), d_embed=384)
    out = emb.embed(SNIPPET)
    assert isinstance(out, list)
    assert len(out) == 384
    assert all(isinstance(x, float) for x in out)


def test_embed_custom_dim():
    emb = CodeAwareEmbedder(_fn(64), d_embed=64)
    assert len(emb.embed(SNIPPET)) == 64


def test_embed_l2_normalized():
    emb = CodeAwareEmbedder(_fn())
    v = emb.embed(SNIPPET)
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_embed_empty_returns_zero_vector():
    emb = CodeAwareEmbedder(_fn(), d_embed=384)
    v = emb.embed("")
    assert v == [0.0] * 384


def test_embed_whitespace_returns_zero_vector():
    emb = CodeAwareEmbedder(_fn(), d_embed=384)
    v = emb.embed("   \n\t")
    assert all(x == 0.0 for x in v)


def test_weights_respected_changes_output():
    base = CodeAwareEmbedder(_fn())
    boosted = CodeAwareEmbedder(_fn(), identifier_weight=10.0)
    v1 = base.embed(SNIPPET)
    v2 = boosted.embed(SNIPPET)
    # Both normalized; direction should differ when identifier weight changes.
    dot = sum(a * b for a, b in zip(v1, v2))
    assert dot < 0.9999  # not identical


def test_embed_batch_length_matches():
    emb = CodeAwareEmbedder(_fn())
    outs = emb.embed_batch(["a=1", "def f(): pass", ""])
    assert len(outs) == 3
    assert len(outs[0]) == 384
    assert all(x == 0.0 for x in outs[2])


def test_unicode_identifiers():
    emb = CodeAwareEmbedder(_fn())
    feats = emb.extract_features("naïve_var = 1; café = 2")
    assert "naïve_var" in feats.identifiers
    assert "café" in feats.identifiers
    v = emb.embed("naïve_var = 1")
    assert len(v) == 384


def test_determinism_with_stub_embed_fn():
    emb1 = CodeAwareEmbedder(_fn())
    emb2 = CodeAwareEmbedder(_fn())
    assert emb1.embed(SNIPPET) == emb2.embed(SNIPPET)


def test_wrong_dim_raises():
    def bad_fn(tok):
        return [0.0] * 7  # wrong dim

    emb = CodeAwareEmbedder(bad_fn, d_embed=384)
    with pytest.raises(ValueError):
        emb.embed("x = 1")


def test_very_long_code_handled():
    emb = CodeAwareEmbedder(_fn())
    huge = (f"def f_{0}(): pass\n") * 50_000
    v = emb.embed(huge)
    assert len(v) == 384
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_stub_token_embed_deterministic():
    a = stub_token_embed("hello", 32)
    b = stub_token_embed("hello", 32)
    assert a == b
    assert len(a) == 32
    assert all(-1.0 <= x <= 1.0 for x in a)


def test_invalid_d_embed_raises():
    with pytest.raises(ValueError):
        CodeAwareEmbedder(_fn(), d_embed=0)


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        CodeAwareEmbedder(_fn(), identifier_weight=-1.0)


def test_non_callable_raises():
    with pytest.raises(TypeError):
        CodeAwareEmbedder("not callable")  # type: ignore[arg-type]
