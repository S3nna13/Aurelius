"""Tokenizer contract + hash validator (v5 FamilyManifest family-adjacent).

Computes a stable identity hash over a tokenizer's vocabulary/merges/specials,
validates manifests, and diffs identities to surface breaking changes across
variants. Pure stdlib; additive-within-file only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenizerIdentity:
    """Identity-defining data for a tokenizer version.

    vocab_sample is a deterministic subsample (first N + last N of sorted
    entries) to keep hashing cheap on large vocabularies.
    """

    name: str
    vocab_size: int
    vocab_sample: tuple[tuple[str, int], ...]
    merges_sample: tuple[str, ...]
    special_tokens: tuple[tuple[str, int], ...]
    version: str


class ContractMismatch(Exception):
    """Raised when a tokenizer contract cannot be honored."""


@dataclass(frozen=True)
class ContractVerdict:
    matches: bool
    expected_hash: str | None
    actual_hash: str
    differences: tuple[str, ...] = field(default_factory=tuple)


def _canonical_json(identity: TokenizerIdentity) -> str:
    """Serialize identity deterministically for hashing."""
    payload = {
        "name": identity.name,
        "vocab_size": identity.vocab_size,
        "vocab_sample": [list(p) for p in identity.vocab_sample],
        "merges_sample": list(identity.merges_sample),
        "special_tokens": [list(p) for p in identity.special_tokens],
        "version": identity.version,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_tokenizer_hash(identity: TokenizerIdentity, digest_algo: str = "sha256") -> str:
    """Canonicalize identity to JSON and hash via hashlib."""
    if digest_algo not in hashlib.algorithms_available:
        raise ValueError(f"unsupported digest algorithm: {digest_algo}")
    canonical = _canonical_json(identity).encode("utf-8")
    h = hashlib.new(digest_algo)
    h.update(canonical)
    return h.hexdigest()


TOKENIZER_IDENTITY_REGISTRY: dict[str, TokenizerIdentity] = {}


def register_identity(identity: TokenizerIdentity) -> None:
    """Register an identity by name; duplicate names raise ContractMismatch."""
    if identity.name in TOKENIZER_IDENTITY_REGISTRY:
        raise ContractMismatch(f"tokenizer identity already registered: {identity.name}")
    TOKENIZER_IDENTITY_REGISTRY[identity.name] = identity


class TokenizerContractValidator:
    """Validates tokenizer identities against manifest hashes."""

    def validate(
        self,
        identity: TokenizerIdentity,
        expected_hash: str | None,
    ) -> ContractVerdict:
        actual = compute_tokenizer_hash(identity)
        if expected_hash is None:
            return ContractVerdict(
                matches=True,
                expected_hash=None,
                actual_hash=actual,
                differences=(),
            )
        if actual == expected_hash:
            return ContractVerdict(
                matches=True,
                expected_hash=expected_hash,
                actual_hash=actual,
                differences=(),
            )
        registered = TOKENIZER_IDENTITY_REGISTRY.get(identity.name)
        diffs: tuple[str, ...] = ("hash_mismatch",)
        if registered is not None:
            diffs = ("hash_mismatch",) + self.diff_identities(registered, identity)
        return ContractVerdict(
            matches=False,
            expected_hash=expected_hash,
            actual_hash=actual,
            differences=diffs,
        )

    def diff_identities(self, a: TokenizerIdentity, b: TokenizerIdentity) -> tuple[str, ...]:
        diffs: list[str] = []
        for field_name in (
            "name",
            "vocab_size",
            "vocab_sample",
            "merges_sample",
            "special_tokens",
            "version",
        ):
            if getattr(a, field_name) != getattr(b, field_name):
                diffs.append(field_name)
        return tuple(diffs)


__all__ = [
    "TokenizerIdentity",
    "ContractMismatch",
    "ContractVerdict",
    "TokenizerContractValidator",
    "compute_tokenizer_hash",
    "register_identity",
    "TOKENIZER_IDENTITY_REGISTRY",
]
