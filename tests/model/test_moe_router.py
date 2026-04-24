"""Tests for MoERouter."""

from __future__ import annotations

import pytest
import torch

from src.model.moe_router import MoERouter, RouterConfig, RouterOutput, RouterType


N_EXPERTS = 4
D_MODEL = 16
BATCH = 2
SEQ_LEN = 8
N_TOKENS = BATCH * SEQ_LEN


def make_router(router_type: RouterType = RouterType.TOP_K, top_k: int = 2) -> MoERouter:
    cfg = RouterConfig(n_experts=N_EXPERTS, top_k=top_k, router_type=router_type)
    return MoERouter(cfg, d_model=D_MODEL)


def make_hidden_3d() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


def make_hidden_2d() -> torch.Tensor:
    return torch.randn(N_TOKENS, D_MODEL)


def test_output_is_router_output_dataclass():
    router = make_router()
    out = router(make_hidden_3d())
    assert isinstance(out, RouterOutput)


def test_expert_indices_shape_3d_input():
    router = make_router(top_k=2)
    out = router(make_hidden_3d())
    assert out.expert_indices.shape == (N_TOKENS, 2)


def test_expert_indices_shape_2d_input():
    router = make_router(top_k=2)
    out = router(make_hidden_2d())
    assert out.expert_indices.shape == (N_TOKENS, 2)


def test_router_probs_shape():
    router = make_router(top_k=2)
    out = router(make_hidden_3d())
    assert out.router_probs.shape == (N_TOKENS, 2)


def test_router_probs_non_negative():
    router = make_router()
    out = router(make_hidden_3d())
    assert (out.router_probs >= 0).all()


def test_expert_indices_in_range():
    router = make_router()
    out = router(make_hidden_3d())
    assert (out.expert_indices >= 0).all()
    assert (out.expert_indices < N_EXPERTS).all()


def test_aux_loss_is_scalar():
    router = make_router()
    out = router(make_hidden_3d())
    assert out.aux_loss.dim() == 0


def test_aux_loss_non_negative():
    router = make_router()
    out = router(make_hidden_3d())
    assert out.aux_loss.item() >= 0.0


def test_load_distribution_shape():
    router = make_router()
    out = router(make_hidden_3d())
    assert out.load_distribution.shape == (N_EXPERTS,)


def test_load_distribution_sums_to_one():
    router = make_router()
    out = router(make_hidden_3d())
    assert out.load_distribution.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_expert_choice_router_output_shapes():
    router = make_router(router_type=RouterType.EXPERT_CHOICE)
    out = router(make_hidden_3d())
    assert out.expert_indices.shape == (N_TOKENS, 2)
    assert out.router_probs.shape == (N_TOKENS, 2)
    assert out.load_distribution.shape == (N_EXPERTS,)


def test_hash_router_expert_indices_deterministic():
    router = make_router(router_type=RouterType.HASH)
    hidden = make_hidden_2d()
    out1 = router(hidden)
    out2 = router(hidden)
    assert torch.equal(out1.expert_indices, out2.expert_indices)


def test_hash_router_aux_loss_zero():
    router = make_router(router_type=RouterType.HASH)
    out = router(make_hidden_3d())
    assert out.aux_loss.item() == 0.0


def test_hash_router_token_assignment_mod():
    router = make_router(router_type=RouterType.HASH, top_k=1)
    hidden = make_hidden_2d()
    out = router(hidden)
    expected = torch.arange(N_TOKENS) % N_EXPERTS
    assert torch.equal(out.expert_indices[:, 0], expected)


def test_top_k_equals_one():
    router = make_router(top_k=1)
    out = router(make_hidden_3d())
    assert out.expert_indices.shape == (N_TOKENS, 1)
    assert out.router_probs.shape == (N_TOKENS, 1)


def test_router_is_nn_module():
    router = make_router()
    assert isinstance(router, torch.nn.Module)


def test_router_has_gate_parameter():
    router = make_router()
    assert hasattr(router, "gate")
    assert isinstance(router.gate, torch.nn.Linear)


def test_aux_loss_coef_zero_produces_zero_aux_loss():
    cfg = RouterConfig(n_experts=N_EXPERTS, top_k=2, aux_loss_coef=0.0)
    router = MoERouter(cfg, d_model=D_MODEL)
    out = router(make_hidden_3d())
    assert out.aux_loss.item() == pytest.approx(0.0, abs=1e-9)


def test_inference_mode_produces_same_result():
    cfg = RouterConfig(n_experts=N_EXPERTS, top_k=2, jitter_noise=0.5)
    router = MoERouter(cfg, d_model=D_MODEL)
    router.train(False)
    hidden = make_hidden_2d()
    out1 = router(hidden)
    out2 = router(hidden)
    assert torch.equal(out1.expert_indices, out2.expert_indices)
