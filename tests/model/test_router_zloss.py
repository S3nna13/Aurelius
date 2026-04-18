"""Tests for Switch Transformer router z-loss."""

from __future__ import annotations

import pytest
import torch

from src.model.router_zloss import RouterZLoss, router_z_loss


N_LAYERS = 2
D_MODEL = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16
D_FF = 128
VOCAB_SIZE = 256
MAX_SEQ_LEN = 64
N_EXPERTS = 4
BATCH = 2
SEQ_LEN = 5


def make_h(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randn(batch, seq_len, D_MODEL)


def test_forward_shapes_and_dtypes_on_tiny_config() -> None:
    torch.manual_seed(0)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    h = make_h()

    out = module(h)

    assert out.x.shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert out.z.shape == (BATCH, SEQ_LEN)
    assert out.x.dtype == h.dtype
    assert out.z.dtype == h.dtype
    assert out.L_z.dtype == h.dtype
    assert out.loss.dtype == h.dtype


def test_loss_scales_with_lambda_z() -> None:
    torch.manual_seed(1)
    h = make_h()
    base = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS, lambda_z=1.0)
    scaled = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS, lambda_z=0.125)
    scaled.load_state_dict(base.state_dict())

    base_out = base(h)
    scaled_out = scaled(h)

    assert torch.allclose(base_out.L_z, scaled_out.L_z, atol=1e-6)
    assert torch.allclose(scaled_out.loss, base_out.L_z * 0.125, atol=1e-6)


def test_gradient_flow_gives_finite_grads_on_all_trainable_params() -> None:
    torch.manual_seed(2)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    h = make_h().requires_grad_(True)

    out = module(h)
    (out.loss + out.x.square().mean()).backward()

    for name, param in module.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


def test_determinism_under_manual_seed() -> None:
    h = make_h()

    torch.manual_seed(3)
    module_a = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    out_a = module_a(h)

    torch.manual_seed(3)
    module_b = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    out_b = module_b(h)

    assert torch.allclose(out_a.x, out_b.x)
    assert torch.allclose(out_a.z, out_b.z)
    assert torch.allclose(out_a.L_z, out_b.L_z)
    assert torch.allclose(out_a.loss, out_b.loss)


def test_batch1_seq1_edge_case() -> None:
    torch.manual_seed(4)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    out = module(make_h(batch=1, seq_len=1))

    assert out.x.shape == (1, 1, N_EXPERTS)
    assert out.z.shape == (1, 1)
    assert torch.isfinite(out.loss)


def test_masked_inputs_match_unpadded_reference_slice() -> None:
    torch.manual_seed(5)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    h = make_h(batch=1, seq_len=4)
    token_mask = torch.tensor([[True, True, False, False]])

    masked = module(h, token_mask=token_mask)
    reference = module(h[:, :2])

    assert torch.allclose(masked.L_z, reference.L_z, atol=1e-5)
    assert torch.allclose(masked.loss, reference.loss, atol=1e-5)
    assert torch.allclose(masked.z[:, :2], reference.z, atol=1e-5)


def test_fully_masked_inputs_return_zero_losses() -> None:
    torch.manual_seed(6)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    h = make_h(batch=1, seq_len=3)
    token_mask = torch.zeros(1, 3, dtype=torch.bool)

    out = module(h, token_mask=token_mask)

    assert out.L_z.item() == 0.0
    assert out.loss.item() == 0.0
    assert torch.isfinite(out.z).all()


def test_numerical_stability_on_extreme_inputs() -> None:
    torch.manual_seed(7)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL) * 1e5
    out = module(h)

    assert torch.isfinite(out.x).all()
    assert torch.isfinite(out.z).all()
    assert torch.isfinite(out.L_z)
    assert torch.isfinite(out.loss)


def test_router_z_loss_matches_reference_formulation() -> None:
    x = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]]],
        dtype=torch.float32,
    )
    token_mask = torch.tensor([[True, False]])

    z, L_z = router_z_loss(x, token_mask=token_mask)
    z_ref = torch.logsumexp(x, dim=-1)
    L_z_ref = z_ref[:, :1].square().mean()

    assert torch.allclose(z, z_ref, atol=1e-5)
    assert torch.allclose(L_z, L_z_ref, atol=1e-5)


def test_module_matches_function_reference() -> None:
    torch.manual_seed(8)
    module = RouterZLoss(d_model=D_MODEL, n_experts=N_EXPERTS, lambda_z=0.25)
    h = make_h(batch=1, seq_len=3)

    out = module(h)
    z_ref, L_z_ref = router_z_loss(out.x)

    assert torch.allclose(out.z, z_ref, atol=1e-5)
    assert torch.allclose(out.L_z, L_z_ref, atol=1e-5)
    assert torch.allclose(out.loss, 0.25 * L_z_ref, atol=1e-5)


def test_router_z_loss_supports_higher_rank_inputs() -> None:
    x = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[-1.0, 0.5], [1.5, -2.0]],
            ]
        ],
        dtype=torch.float32,
    )
    token_mask = torch.tensor([[[True, False], [True, True]]])

    z, L_z = router_z_loss(x, token_mask=token_mask)
    z_ref = torch.logsumexp(x, dim=-1)
    L_z_ref = z_ref[token_mask].square().mean()

    assert torch.allclose(z, z_ref, atol=1e-5)
    assert torch.allclose(L_z, L_z_ref, atol=1e-5)


def test_router_z_loss_rejects_bad_mask_shape() -> None:
    x = torch.randn(2, 3, 4)
    token_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="token_mask must have shape"):
        router_z_loss(x, token_mask=token_mask)


def test_router_z_loss_rejects_missing_expert_dimension() -> None:
    with pytest.raises(ValueError, match="expert dimension"):
        router_z_loss(torch.randn(4))
