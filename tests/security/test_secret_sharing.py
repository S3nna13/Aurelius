"""Tests for additive secret sharing."""

from __future__ import annotations

import torch
import pytest

from src.security.secret_sharing import SecretSharing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ss3() -> SecretSharing:
    """SecretSharing instance with 3 parties."""
    return SecretSharing(n_parties=3)


@pytest.fixture
def secret_1d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(16)


@pytest.fixture
def secret_2d() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(4, 8)


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

def test_instantiation():
    ss = SecretSharing(n_parties=4)
    assert ss.n_parties == 4


# ---------------------------------------------------------------------------
# 2. share() returns list of n_parties tensors
# ---------------------------------------------------------------------------

def test_share_returns_n_tensors(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    assert isinstance(shares, list)
    assert len(shares) == 3


# ---------------------------------------------------------------------------
# 3. Each share has the same shape as the secret
# ---------------------------------------------------------------------------

def test_share_shapes_match(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    for s in shares:
        assert s.shape == secret_1d.shape


# ---------------------------------------------------------------------------
# 4. Shares sum exactly to the original secret
# ---------------------------------------------------------------------------

def test_shares_sum_to_secret(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    total = sum(shares)
    assert torch.allclose(total, secret_1d, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Individual shares do not equal the secret (privacy)
# ---------------------------------------------------------------------------

def test_individual_shares_differ_from_secret(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    # At least one share must differ from the secret (with overwhelming probability)
    any_differs = any(not torch.allclose(s, secret_1d) for s in shares)
    assert any_differs, "All shares were equal to the secret — no privacy"


# ---------------------------------------------------------------------------
# 6. reconstruct() returns the correct tensor
# ---------------------------------------------------------------------------

def test_reconstruct_correct(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    recovered = ss3.reconstruct(shares)
    assert torch.allclose(recovered, secret_1d, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. verify_reconstruction returns True for valid shares
# ---------------------------------------------------------------------------

def test_verify_reconstruction_true(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    assert ss3.verify_reconstruction(secret_1d, shares) is True


# ---------------------------------------------------------------------------
# 8. verify_reconstruction returns False for corrupted shares
# ---------------------------------------------------------------------------

def test_verify_reconstruction_false_on_corruption(ss3, secret_1d):
    shares = ss3.share(secret_1d)
    corrupted = [s.clone() for s in shares]
    corrupted[0] = corrupted[0] + 999.0  # corrupt first share
    assert ss3.verify_reconstruction(secret_1d, corrupted) is False


# ---------------------------------------------------------------------------
# 9. Works with float32 tensors
# ---------------------------------------------------------------------------

def test_works_with_float32(ss3):
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    shares = ss3.share(t)
    assert shares[0].dtype == torch.float32
    assert torch.allclose(ss3.reconstruct(shares), t, atol=1e-5)


# ---------------------------------------------------------------------------
# 10. Works with 2D tensor (matrix)
# ---------------------------------------------------------------------------

def test_works_with_2d_tensor(ss3, secret_2d):
    shares = ss3.share(secret_2d)
    assert all(s.shape == secret_2d.shape for s in shares)
    assert torch.allclose(ss3.reconstruct(shares), secret_2d, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. share_gradients returns list of n dicts
# ---------------------------------------------------------------------------

def test_share_gradients_returns_n_dicts(ss3):
    state_dict = {
        "weight": torch.randn(4, 4),
        "bias": torch.randn(4),
    }
    party_dicts = ss3.share_gradients(state_dict)
    assert isinstance(party_dicts, list)
    assert len(party_dicts) == 3
    for pd in party_dicts:
        assert isinstance(pd, dict)
        assert set(pd.keys()) == set(state_dict.keys())


# ---------------------------------------------------------------------------
# 12. reconstruct_gradients returns correct state dict
# ---------------------------------------------------------------------------

def test_reconstruct_gradients_returns_state_dict(ss3):
    state_dict = {
        "weight": torch.randn(4, 4),
        "bias": torch.randn(4),
    }
    party_dicts = ss3.share_gradients(state_dict)
    recovered = ss3.reconstruct_gradients(party_dicts)
    assert isinstance(recovered, dict)
    assert set(recovered.keys()) == set(state_dict.keys())


# ---------------------------------------------------------------------------
# 13. reconstruct_gradients result matches original state dict values
# ---------------------------------------------------------------------------

def test_reconstruct_gradients_matches_original(ss3):
    torch.manual_seed(42)
    state_dict = {
        "weight": torch.randn(8, 8),
        "bias": torch.randn(8),
    }
    party_dicts = ss3.share_gradients(state_dict)
    recovered = ss3.reconstruct_gradients(party_dicts)
    for key in state_dict:
        assert torch.allclose(recovered[key], state_dict[key], atol=1e-5), (
            f"Mismatch on key '{key}'"
        )
