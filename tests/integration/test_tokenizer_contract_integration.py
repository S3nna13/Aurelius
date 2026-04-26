"""Integration: tokenizer_contract wired into src.data package."""

from __future__ import annotations

import src.data as data_pkg
from src.data.tokenizer_contract import (
    TOKENIZER_IDENTITY_REGISTRY,
    TokenizerContractValidator,
    TokenizerIdentity,
    compute_tokenizer_hash,
)


def test_identity_registry_exposed_from_package():
    assert hasattr(data_pkg, "TOKENIZER_IDENTITY_REGISTRY")
    assert data_pkg.TOKENIZER_IDENTITY_REGISTRY is TOKENIZER_IDENTITY_REGISTRY


def test_validator_and_identity_exposed_from_package():
    assert hasattr(data_pkg, "TokenizerContractValidator")
    assert hasattr(data_pkg, "TokenizerIdentity")
    assert hasattr(data_pkg, "compute_tokenizer_hash")


def test_end_to_end_validate_round_trip():
    identity = TokenizerIdentity(
        name="integration-tok",
        vocab_size=256,
        vocab_sample=(("a", 0), ("z", 255)),
        merges_sample=("a b",),
        special_tokens=(("<pad>", 0),),
        version="1.2.3",
    )
    expected = compute_tokenizer_hash(identity)
    verdict = TokenizerContractValidator().validate(identity, expected)
    assert verdict.matches is True
    assert verdict.actual_hash == expected
