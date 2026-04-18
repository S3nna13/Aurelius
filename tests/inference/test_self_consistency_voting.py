"""Tests for self-consistency voting."""

from __future__ import annotations

import pytest
import torch

from src.inference.self_consistency_voting import SelfConsistencyVoting
from src.model.config import AureliusConfig


@pytest.fixture
def tiny_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def model(tiny_config: AureliusConfig) -> SelfConsistencyVoting:
    torch.manual_seed(0)
    return SelfConsistencyVoting(tiny_config)


def _make_inputs(
    batch_size: int = 2,
    n_paths: int = 3,
    x_len: int = 5,
    z_len: int = 4,
    vocab_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, x_len), dtype=torch.long)
    z = torch.randint(0, vocab_size, (batch_size, n_paths, z_len), dtype=torch.long)
    return x, z


def test_forward_shape_and_dtype_tiny_config(model: SelfConsistencyVoting):
    x, z = _make_inputs()
    output = model(x, z)

    assert output.log_p_a_given_x.shape == (2, 256)
    assert output.p_a_given_x.shape == (2, 256)
    assert output.log_p_z_given_x.shape == (2, 3)
    assert output.p_z_given_x.shape == (2, 3)
    assert output.log_p_a_given_x_z.shape == (2, 3, 256)
    assert output.log_p_a_given_x.dtype == torch.float32
    assert output.p_a_given_x.dtype == torch.float32


def test_probabilities_are_normalized(model: SelfConsistencyVoting):
    x, z = _make_inputs()
    output = model(x, z)

    assert torch.allclose(output.p_a_given_x.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.p_z_given_x.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_loss_backward_produces_finite_grads_on_all_trainable_params(model: SelfConsistencyVoting):
    x, z = _make_inputs()
    a = torch.tensor([3, 7], dtype=torch.long)

    loss = model.loss(x, z, a)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_determinism_under_manual_seed(tiny_config: AureliusConfig):
    torch.manual_seed(123)
    model_a = SelfConsistencyVoting(tiny_config)
    x = torch.randint(0, tiny_config.vocab_size, (2, 5), dtype=torch.long)
    z = torch.randint(0, tiny_config.vocab_size, (2, 4, 3), dtype=torch.long)
    output_a = model_a(x, z)

    torch.manual_seed(123)
    model_b = SelfConsistencyVoting(tiny_config)
    x_b = torch.randint(0, tiny_config.vocab_size, (2, 5), dtype=torch.long)
    z_b = torch.randint(0, tiny_config.vocab_size, (2, 4, 3), dtype=torch.long)
    output_b = model_b(x_b, z_b)

    assert torch.equal(x, x_b)
    assert torch.equal(z, z_b)
    assert torch.allclose(output_a.log_p_a_given_x, output_b.log_p_a_given_x, atol=1e-6)
    assert torch.allclose(output_a.log_p_z_given_x, output_b.log_p_z_given_x, atol=1e-6)


def test_batch_one_seq_len_one(model: SelfConsistencyVoting):
    x = torch.tensor([[5]], dtype=torch.long)
    z = torch.tensor([[[7]]], dtype=torch.long)
    output = model(x, z)

    assert output.log_p_a_given_x.shape == (1, 256)
    assert output.log_p_z_given_x.shape == (1, 1)


def test_masked_x_matches_trimmed_x(model: SelfConsistencyVoting):
    x_trimmed = torch.tensor([[11, 12, 13]], dtype=torch.long)
    z = torch.tensor([[[21, 22], [23, 24]]], dtype=torch.long)
    trimmed = model(x_trimmed, z)

    x_padded = torch.tensor([[11, 12, 13, 99, 98]], dtype=torch.long)
    x_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    masked = model(x_padded, z, x_mask=x_mask)

    assert torch.allclose(trimmed.log_p_a_given_x, masked.log_p_a_given_x, atol=1e-5)
    assert torch.allclose(trimmed.log_p_z_given_x, masked.log_p_z_given_x, atol=1e-5)


def test_masked_z_matches_trimmed_z(model: SelfConsistencyVoting):
    x = torch.tensor([[11, 12, 13]], dtype=torch.long)
    z_trimmed = torch.tensor([[[21, 22], [23, 24]]], dtype=torch.long)
    trimmed = model(x, z_trimmed)

    z_padded = torch.tensor([[[21, 22, 99, 98], [23, 24, 97, 96]]], dtype=torch.long)
    z_mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0]]], dtype=torch.bool)
    masked = model(x, z_padded, z_mask=z_mask)

    assert torch.allclose(trimmed.log_p_a_given_x, masked.log_p_a_given_x, atol=1e-5)
    assert torch.allclose(trimmed.log_p_z_given_x, masked.log_p_z_given_x, atol=1e-5)


def test_invalid_reasoning_path_is_ignored(model: SelfConsistencyVoting):
    x = torch.tensor([[3, 4]], dtype=torch.long)
    z = torch.tensor([[[10, 11], [12, 13], [99, 98]]], dtype=torch.long)
    z_mask = torch.tensor([[[1, 1], [1, 1], [0, 0]]], dtype=torch.bool)

    with_invalid = model(x, z, z_mask=z_mask)
    without_invalid = model(x, z[:, :2])

    assert torch.allclose(with_invalid.log_p_a_given_x, without_invalid.log_p_a_given_x, atol=1e-5)
    assert torch.allclose(
        with_invalid.log_p_z_given_x[:, :2], without_invalid.log_p_z_given_x, atol=1e-5
    )
    assert with_invalid.p_z_given_x[0, 2].item() == pytest.approx(0.0, abs=1e-7)


def test_numerical_stability_on_extreme_inputs(tiny_config: AureliusConfig):
    torch.manual_seed(0)
    model = SelfConsistencyVoting(tiny_config)
    for param in model.parameters():
        torch.nn.init.constant_(param, 50.0)

    x = torch.full((2, 6), tiny_config.vocab_size - 1, dtype=torch.long)
    z = torch.full((2, 4, 5), tiny_config.vocab_size - 1, dtype=torch.long)
    output = model(x, z)

    assert torch.isfinite(output.log_p_a_given_x).all()
    assert torch.isfinite(output.p_a_given_x).all()
    assert torch.isfinite(output.log_p_z_given_x).all()
    assert torch.isfinite(output.p_z_given_x).all()


def test_reference_and_vectorized_paths_are_equivalent(model: SelfConsistencyVoting):
    x, z = _make_inputs(batch_size=2, n_paths=4, x_len=3, z_len=5)
    x_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    z_mask = torch.tensor(
        [
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]],
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    z_mask[1, 3, 0] = True

    optimized = model(x, z, x_mask=x_mask, z_mask=z_mask, use_reference=False)
    reference = model(x, z, x_mask=x_mask, z_mask=z_mask, use_reference=True)

    assert torch.allclose(optimized.log_p_a_given_x, reference.log_p_a_given_x, atol=1e-5)
    assert torch.allclose(optimized.log_p_z_given_x, reference.log_p_z_given_x, atol=1e-5)


def test_predict_returns_long_answer_ids(model: SelfConsistencyVoting):
    x, z = _make_inputs(batch_size=3, n_paths=2, x_len=4, z_len=3)
    a_hat = model.predict(x, z)

    assert a_hat.shape == (3,)
    assert a_hat.dtype == torch.long


def test_all_masked_x_raises(model: SelfConsistencyVoting):
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    z = torch.tensor([[[4, 5], [6, 7]]], dtype=torch.long)
    x_mask = torch.zeros_like(x, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one unmasked token"):
        model(x, z, x_mask=x_mask)


def test_all_masked_z_raises(model: SelfConsistencyVoting):
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    z = torch.tensor([[[4, 5], [6, 7]]], dtype=torch.long)
    z_mask = torch.zeros_like(z, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid reasoning path"):
        model(x, z, z_mask=z_mask)
