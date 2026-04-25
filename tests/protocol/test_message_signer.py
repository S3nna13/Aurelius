"""Tests for src/protocol/message_signer.py"""

import hashlib
import hmac
import json

import pytest
from src.protocol.message_signer import (
    MessageSigner,
    SignedMessage,
    SignerConfig,
    MESSAGE_SIGNER_REGISTRY,
)


# ---------------------------------------------------------------------------
# sign()
# ---------------------------------------------------------------------------


def test_sign_returns_signed_message():
    signer = MessageSigner()
    msg = signer.sign(b"hello")
    assert isinstance(msg, SignedMessage)


def test_sign_payload_preserved():
    signer = MessageSigner()
    msg = signer.sign(b"payload-data")
    assert msg.payload == b"payload-data"


def test_sign_algorithm_stored():
    signer = MessageSigner()
    msg = signer.sign(b"x")
    assert msg.algorithm == "sha256"


def test_sign_signed_at_positive():
    signer = MessageSigner()
    msg = signer.sign(b"x")
    assert msg.signed_at > 0


def test_sign_signature_is_hex_string():
    signer = MessageSigner()
    msg = signer.sign(b"abc")
    # sha256 hex digest is 64 chars
    assert isinstance(msg.signature, str)
    assert len(msg.signature) == 64


def test_sign_signature_matches_manual_hmac():
    key    = b"test-key"
    config = SignerConfig(key=key)
    signer = MessageSigner(config)
    payload = b"important data"
    msg     = signer.sign(payload)
    expected = hmac.new(key, payload, digestmod=hashlib.sha256).hexdigest()
    assert msg.signature == expected


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


def test_verify_correct_payload_returns_true():
    signer = MessageSigner()
    payload = b"verify me"
    msg     = signer.sign(payload)
    assert signer.verify(msg, payload) is True


def test_verify_tampered_payload_returns_false():
    signer = MessageSigner()
    msg    = signer.sign(b"original")
    assert signer.verify(msg, b"tampered") is False


def test_verify_empty_payload():
    signer = MessageSigner()
    msg    = signer.sign(b"")
    assert signer.verify(msg, b"") is True


def test_verify_empty_payload_tampered():
    signer = MessageSigner()
    msg    = signer.sign(b"")
    assert signer.verify(msg, b"x") is False


def test_verify_different_key_returns_false():
    signer_a = MessageSigner(SignerConfig(key=b"key-a"))
    signer_b = MessageSigner(SignerConfig(key=b"key-b"))
    payload  = b"data"
    msg      = signer_a.sign(payload)
    assert signer_b.verify(msg, payload) is False


# ---------------------------------------------------------------------------
# rotate_key()
# ---------------------------------------------------------------------------


def test_rotate_key_changes_signature():
    signer  = MessageSigner(SignerConfig(key=b"old-key"))
    payload = b"rotate test"
    msg_old = signer.sign(payload)
    signer.rotate_key(b"new-key")
    msg_new = signer.sign(payload)
    assert msg_old.signature != msg_new.signature


def test_rotate_key_old_verify_fails():
    signer  = MessageSigner(SignerConfig(key=b"old-key"))
    payload = b"some data"
    msg     = signer.sign(payload)
    signer.rotate_key(b"new-key")
    # Old message no longer verifies with new key
    assert signer.verify(msg, payload) is False


def test_rotate_key_new_verify_succeeds():
    signer  = MessageSigner(SignerConfig(key=b"old"))
    payload = b"data"
    signer.rotate_key(b"new")
    msg = signer.sign(payload)
    assert signer.verify(msg, payload) is True


def test_rotate_key_updates_config():
    signer = MessageSigner()
    signer.rotate_key(b"rotated")
    assert signer.config.key == b"rotated"


# ---------------------------------------------------------------------------
# sign_dict() / verify_dict()
# ---------------------------------------------------------------------------


def test_sign_dict_returns_signed_message():
    signer = MessageSigner()
    msg    = signer.sign_dict({"a": 1})
    assert isinstance(msg, SignedMessage)


def test_sign_dict_consistent_with_manual_json():
    key    = b"dict-key"
    signer = MessageSigner(SignerConfig(key=key))
    data   = {"z": 3, "a": 1, "m": 2}
    msg    = signer.sign_dict(data)
    canonical = json.dumps(data, sort_keys=True).encode()
    expected  = hmac.new(key, canonical, digestmod=hashlib.sha256).hexdigest()
    assert msg.signature == expected


def test_verify_dict_true_for_same_data():
    signer = MessageSigner()
    data   = {"key": "value", "num": 42}
    msg    = signer.sign_dict(data)
    assert signer.verify_dict(msg, data) is True


def test_verify_dict_false_for_different_data():
    signer = MessageSigner()
    msg    = signer.sign_dict({"key": "value"})
    assert signer.verify_dict(msg, {"key": "other"}) is False


def test_sign_dict_key_order_invariant():
    signer = MessageSigner()
    msg_ab = signer.sign_dict({"a": 1, "b": 2})
    msg_ba = signer.sign_dict({"b": 2, "a": 1})
    assert msg_ab.signature == msg_ba.signature


# ---------------------------------------------------------------------------
# SignedMessage frozen
# ---------------------------------------------------------------------------


def test_signed_message_is_frozen():
    msg = SignedMessage(
        payload=b"x",
        signature="abc",
        algorithm="sha256",
        signed_at=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        msg.signature = "new"  # type: ignore[misc]


def test_signed_message_payload_frozen():
    msg = SignedMessage(payload=b"data", signature="s", algorithm="sha256", signed_at=1.0)
    with pytest.raises((AttributeError, TypeError)):
        msg.payload = b"other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Custom algorithm field (stored, not used for digest selection here)
# ---------------------------------------------------------------------------


def test_custom_algorithm_string_stored():
    cfg    = SignerConfig(algorithm="sha512")
    signer = MessageSigner(cfg)
    msg    = signer.sign(b"algo test")
    assert msg.algorithm == "sha512"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in MESSAGE_SIGNER_REGISTRY


def test_registry_default_is_class():
    assert MESSAGE_SIGNER_REGISTRY["default"] is MessageSigner


def test_registry_default_is_instantiable():
    cls    = MESSAGE_SIGNER_REGISTRY["default"]
    signer = cls()
    assert isinstance(signer, MessageSigner)
