"""Tests for ST-MoE router collapse prevention helpers."""

from __future__ import annotations

import torch

from src.model.router_collapse_prevention import (
    RouterCollapsePrevention,
    load_balancing_loss,
    router_z_loss,
)


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
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    h = make_h()

    out = module(h)

    assert out.x.shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert out.p.shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert out.indices.shape == (BATCH, SEQ_LEN, 2)
    assert out.f_i.shape == (N_EXPERTS,)
    assert out.P_i.shape == (N_EXPERTS,)
    assert out.z.shape == (BATCH, SEQ_LEN)
    assert out.x.dtype == h.dtype
    assert out.p.dtype == h.dtype
    assert out.loss.dtype == h.dtype


def test_probabilities_sum_to_one() -> None:
    torch.manual_seed(1)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    out = module(make_h())
    assert torch.allclose(out.p.sum(dim=-1), torch.ones(BATCH, SEQ_LEN), atol=1e-6)


def test_indices_are_in_range() -> None:
    torch.manual_seed(2)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    out = module(make_h())
    assert out.indices.dtype == torch.int64
    assert (out.indices >= 0).all()
    assert (out.indices < N_EXPERTS).all()


def test_gradient_flow_gives_finite_grads_on_all_trainable_params() -> None:
    torch.manual_seed(3)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    h = make_h().requires_grad_(True)

    out = module(h)
    (out.loss + out.p.square().mean()).backward()

    for name, param in module.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


def test_determinism_under_manual_seed() -> None:
    h = make_h()

    torch.manual_seed(42)
    module_a = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    out_a = module_a(h)

    torch.manual_seed(42)
    module_b = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    out_b = module_b(h)

    assert torch.allclose(out_a.x, out_b.x)
    assert torch.allclose(out_a.p, out_b.p)
    assert torch.equal(out_a.indices, out_b.indices)
    assert torch.allclose(out_a.loss, out_b.loss)


def test_batch1_seq1_edge_case() -> None:
    torch.manual_seed(4)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=1)
    out = module(make_h(batch=1, seq_len=1))
    assert out.x.shape == (1, 1, N_EXPERTS)
    assert out.indices.shape == (1, 1, 1)
    assert torch.isfinite(out.loss)


def test_masked_inputs_match_unpadded_reference_slice() -> None:
    torch.manual_seed(5)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    h = make_h(batch=1, seq_len=4)
    token_mask = torch.tensor([[True, True, False, False]])

    masked = module(h, token_mask=token_mask)
    reference = module(h[:, :2])

    assert torch.allclose(masked.P_i, reference.P_i, atol=1e-5)
    assert torch.allclose(masked.f_i, reference.f_i, atol=1e-5)
    assert torch.allclose(masked.L_aux, reference.L_aux, atol=1e-5)
    assert torch.allclose(masked.L_z, reference.L_z, atol=1e-5)


def test_fully_masked_inputs_return_zero_losses() -> None:
    torch.manual_seed(6)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    h = make_h(batch=1, seq_len=3)
    token_mask = torch.zeros(1, 3, dtype=torch.bool)

    out = module(h, token_mask=token_mask)

    assert torch.equal(out.f_i, torch.zeros_like(out.f_i))
    assert torch.equal(out.P_i, torch.zeros_like(out.P_i))
    assert out.L_aux.item() == 0.0
    assert out.L_z.item() == 0.0
    assert out.loss.item() == 0.0


def test_numerical_stability_on_extreme_inputs() -> None:
    torch.manual_seed(7)
    module = RouterCollapsePrevention(d_model=D_MODEL, n_experts=N_EXPERTS, k=2)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL) * 1e4
    out = module(h)

    assert torch.isfinite(out.x).all()
    assert torch.isfinite(out.p).all()
    assert torch.isfinite(out.z).all()
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


def test_load_balancing_loss_matches_reference_top1_formulation() -> None:
    p = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0], [1], [0]]])

    f_i, P_i, L_aux = load_balancing_loss(p, indices, n_experts=3)

    f_ref = torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0])
    P_ref = p.reshape(-1, 3).mean(dim=0)
    L_ref = 3.0 * torch.sum(f_ref * P_ref)

    assert torch.allclose(f_i, f_ref, atol=1e-5)
    assert torch.allclose(P_i, P_ref, atol=1e-5)
    assert torch.allclose(L_aux, L_ref, atol=1e-5)


def test_load_balancing_loss_matches_reference_topk_formulation() -> None:
    p = torch.tensor(
        [[[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [3, 2]]])

    f_i, P_i, L_aux = load_balancing_loss(p, indices, n_experts=4)

    assign = torch.tensor([1.0, 1.0, 1.0, 1.0]) / 4.0
    P_ref = p.reshape(-1, 4).mean(dim=0)
    L_ref = 4.0 * torch.sum(assign * P_ref)

    assert torch.allclose(f_i, assign, atol=1e-5)
    assert torch.allclose(P_i, P_ref, atol=1e-5)
    assert torch.allclose(L_aux, L_ref, atol=1e-5)


def test_masked_load_balancing_ignores_padded_tokens() -> None:
    p = torch.tensor(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.2, 0.8]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0], [0]], [[1], [1]]])
    token_mask = torch.tensor([[True, False], [True, False]])

    f_i, P_i, L_aux = load_balancing_loss(p, indices, n_experts=2, token_mask=token_mask)

    p_ref = torch.stack([p[0, 0], p[1, 0]], dim=0)
    P_ref = p_ref.mean(dim=0)
    f_ref = torch.tensor([0.5, 0.5])
    L_ref = 2.0 * torch.sum(f_ref * P_ref)

    assert torch.allclose(f_i, f_ref, atol=1e-5)
    assert torch.allclose(P_i, P_ref, atol=1e-5)
    assert torch.allclose(L_aux, L_ref, atol=1e-5)
