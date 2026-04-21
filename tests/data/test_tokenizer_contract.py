"""Tests for tokenizer_contract (v5 FamilyManifest family-adjacent)."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from src.data.tokenizer_contract import (
    TOKENIZER_IDENTITY_REGISTRY,
    ContractMismatch,
    ContractVerdict,
    TokenizerContractValidator,
    TokenizerIdentity,
    compute_tokenizer_hash,
    register_identity,
)


def _make_identity(
    name: str = "bpe-v1",
    vocab_size: int = 1024,
    version: str = "1.0.0",
) -> TokenizerIdentity:
    return TokenizerIdentity(
        name=name,
        vocab_size=vocab_size,
        vocab_sample=(("a", 0), ("b", 1), ("z", vocab_size - 1)),
        merges_sample=("a b", "b c"),
        special_tokens=(("<pad>", 0), ("<eos>", 1)),
        version=version,
    )


@pytest.fixture(autouse=True)
def _clear_registry():
    snapshot = dict(TOKENIZER_IDENTITY_REGISTRY)
    TOKENIZER_IDENTITY_REGISTRY.clear()
    try:
        yield
    finally:
        TOKENIZER_IDENTITY_REGISTRY.clear()
        TOKENIZER_IDENTITY_REGISTRY.update(snapshot)


def test_compute_tokenizer_hash_deterministic():
    identity = _make_identity()
    h1 = compute_tokenizer_hash(identity)
    h2 = compute_tokenizer_hash(identity)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_different_identities_produce_different_hashes():
    a = _make_identity(name="bpe-v1")
    b = _make_identity(name="bpe-v2")
    assert compute_tokenizer_hash(a) != compute_tokenizer_hash(b)


def test_none_expected_hash_matches_true():
    identity = _make_identity()
    validator = TokenizerContractValidator()
    verdict = validator.validate(identity, expected_hash=None)
    assert verdict.matches is True
    assert verdict.expected_hash is None
    assert verdict.differences == ()
    assert isinstance(verdict, ContractVerdict)


def test_matching_hash_matches_true():
    identity = _make_identity()
    validator = TokenizerContractValidator()
    expected = compute_tokenizer_hash(identity)
    verdict = validator.validate(identity, expected_hash=expected)
    assert verdict.matches is True
    assert verdict.differences == ()


def test_mismatch_sets_differences():
    identity = _make_identity()
    validator = TokenizerContractValidator()
    verdict = validator.validate(identity, expected_hash="0" * 64)
    assert verdict.matches is False
    assert "hash_mismatch" in verdict.differences


def test_diff_identities_catches_vocab_size_change():
    a = _make_identity(vocab_size=1024)
    b = _make_identity(vocab_size=2048)
    validator = TokenizerContractValidator()
    diffs = validator.diff_identities(a, b)
    assert "vocab_size" in diffs


def test_diff_identities_catches_version_change():
    a = _make_identity(version="1.0.0")
    b = _make_identity(version="2.0.0")
    validator = TokenizerContractValidator()
    assert "version" in validator.diff_identities(a, b)


def test_diff_identities_catches_special_tokens_change():
    a = _make_identity()
    b = TokenizerIdentity(
        name=a.name,
        vocab_size=a.vocab_size,
        vocab_sample=a.vocab_sample,
        merges_sample=a.merges_sample,
        special_tokens=(("<pad>", 0),),
        version=a.version,
    )
    validator = TokenizerContractValidator()
    assert "special_tokens" in validator.diff_identities(a, b)


def test_register_identity_populates_registry():
    identity = _make_identity(name="tok-reg")
    register_identity(identity)
    assert TOKENIZER_IDENTITY_REGISTRY["tok-reg"] is identity


def test_register_identity_duplicate_raises():
    identity = _make_identity(name="tok-dup")
    register_identity(identity)
    with pytest.raises(ContractMismatch):
        register_identity(identity)


def test_hash_uses_sha256_by_default():
    identity = _make_identity()
    default = compute_tokenizer_hash(identity)
    explicit = compute_tokenizer_hash(identity, digest_algo="sha256")
    assert default == explicit
    assert len(default) == 64


def test_alternative_digest_algo_sha1():
    identity = _make_identity()
    sha1 = compute_tokenizer_hash(identity, digest_algo="sha1")
    assert len(sha1) == 40
    assert sha1 != compute_tokenizer_hash(identity)


def test_unicode_in_name_handled():
    identity = _make_identity(name="токенизатор-日本語")
    h = compute_tokenizer_hash(identity)
    assert len(h) == 64


def test_empty_vocab_sample_permitted():
    identity = TokenizerIdentity(
        name="empty",
        vocab_size=0,
        vocab_sample=(),
        merges_sample=(),
        special_tokens=(),
        version="0.0.1",
    )
    h = compute_tokenizer_hash(identity)
    assert isinstance(h, str)
    verdict = TokenizerContractValidator().validate(identity, expected_hash=None)
    assert verdict.matches is True


def test_identity_json_serializable_after_asdict():
    identity = _make_identity()
    # asdict converts tuples to lists; json.dumps must not raise.
    data = asdict(identity)
    dumped = json.dumps(data)
    assert "bpe-v1" in dumped


def test_invalid_digest_algo_raises():
    identity = _make_identity()
    with pytest.raises(ValueError):
        compute_tokenizer_hash(identity, digest_algo="not-a-real-algo-xyz")
