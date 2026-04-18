"""Tests for src/training/weight_decay_scheduling.py."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.weight_decay_scheduling import (
    AdamWR,
    AdamWRConfig,
    cosine_eta_t,
    normalized_weight_decay,
)


TINY = {
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "d_ff": 128,
    "vocab_size": 256,
    "max_seq_len": 64,
}


class MockModel(nn.Module):
    def __init__(self, vocab_size: int = TINY["vocab_size"], d_model: int = TINY["d_model"]):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = torch.tanh(self.proj(x))
        return self.head(x)


def masked_lm_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids, attention_mask=attention_mask)
    ignore_targets = targets.masked_fill(attention_mask == 0, -100)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ignore_targets.reshape(-1),
        ignore_index=-100,
    )


def make_optimizer(model: nn.Module, **overrides) -> AdamWR:
    config_kwargs = dict(
        alpha=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        lambda_norm=0.05,
        eta_min=0.0,
        eta_max=1.0,
        T_0=2.0,
        T_mult=2.0,
        b=4,
        B=16,
    )
    config_kwargs.update(overrides)
    config = AdamWRConfig(**config_kwargs)
    return AdamWR(model.parameters(), config=config)


def reference_adamwr_step(
    theta_prev: torch.Tensor,
    g_t: torch.Tensor,
    *,
    alpha: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    eta_t: float,
    lambda_t: float,
    t: int,
    m_prev: torch.Tensor,
    v_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_t = beta1 * m_prev + (1.0 - beta1) * g_t
    v_t = beta2 * v_prev + (1.0 - beta2) * g_t.square()
    m_hat_t = m_t / (1.0 - beta1**t)
    v_hat_t = v_t / (1.0 - beta2**t)
    theta_t = theta_prev - eta_t * (
        alpha * m_hat_t / (v_hat_t.sqrt() + epsilon) + lambda_t * theta_prev
    )
    return theta_t, m_t, v_t


def test_normalized_weight_decay_matches_paper_formula() -> None:
    value = normalized_weight_decay(lambda_norm=0.025, b=128, B=50000, T=100)
    expected = 0.025 * math.sqrt(128.0 / (50000.0 * 100.0))
    assert value == pytest.approx(expected)


def test_cosine_eta_t_matches_expected_points() -> None:
    assert cosine_eta_t(T_cur=0.0, T_i=4.0, eta_min=0.1, eta_max=1.0) == pytest.approx(1.0)
    assert cosine_eta_t(T_cur=2.0, T_i=4.0, eta_min=0.1, eta_max=1.0) == pytest.approx(0.55)
    assert cosine_eta_t(T_cur=4.0, T_i=4.0, eta_min=0.1, eta_max=1.0) == pytest.approx(0.1)


def test_adamwr_state_shapes_and_dtypes_after_one_step() -> None:
    torch.manual_seed(0)
    model = MockModel()
    opt = make_optimizer(model)

    input_ids = torch.randint(0, TINY["vocab_size"], (2, 3))
    targets = torch.randint(0, TINY["vocab_size"], (2, 3))
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    loss = masked_lm_loss(model, input_ids, targets, attention_mask)
    loss.backward()
    opt.step()

    for parameter in model.parameters():
        state = opt.state[parameter]
        assert state["m_t"].shape == parameter.shape
        assert state["v_t"].shape == parameter.shape
        assert state["m_t"].dtype == parameter.dtype
        assert state["v_t"].dtype == parameter.dtype


def test_loss_backward_produces_finite_grads_for_all_trainable_params() -> None:
    torch.manual_seed(1)
    model = MockModel()

    input_ids = torch.randint(0, TINY["vocab_size"], (2, 5))
    targets = torch.randint(0, TINY["vocab_size"], (2, 5))
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long)

    loss = masked_lm_loss(model, input_ids, targets, attention_mask)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite grad for {name}"


def test_determinism_under_torch_manual_seed() -> None:
    def run_once() -> list[torch.Tensor]:
        torch.manual_seed(1234)
        model = MockModel()
        opt = make_optimizer(model, T_0=1.5, B=12, b=3)
        for _ in range(3):
            input_ids = torch.randint(0, TINY["vocab_size"], (2, 4))
            targets = torch.randint(0, TINY["vocab_size"], (2, 4))
            attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)
            opt.zero_grad()
            loss = masked_lm_loss(model, input_ids, targets, attention_mask)
            loss.backward()
            opt.step()
        return [parameter.detach().clone() for parameter in model.parameters()]

    first = run_once()
    second = run_once()
    for left, right in zip(first, second):
        assert torch.allclose(left, right)


def test_batch1_seq1_edge_case_runs_finitely() -> None:
    torch.manual_seed(2)
    model = MockModel()
    opt = make_optimizer(model, b=1, B=1, T_0=1.0)

    input_ids = torch.randint(0, TINY["vocab_size"], (1, 1))
    targets = torch.randint(0, TINY["vocab_size"], (1, 1))
    attention_mask = torch.ones(1, 1, dtype=torch.long)

    opt.zero_grad()
    loss = masked_lm_loss(model, input_ids, targets, attention_mask)
    loss.backward()
    opt.step()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_masked_or_padded_inputs_ignore_padding_positions() -> None:
    torch.manual_seed(3)
    model = MockModel()

    input_ids = torch.randint(0, TINY["vocab_size"], (1, 4))
    targets = torch.randint(0, TINY["vocab_size"], (1, 4))
    masked = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    loss_full = masked_lm_loss(model, input_ids, targets, torch.ones_like(masked))
    loss_masked = masked_lm_loss(model, input_ids, targets, masked)

    logits = model(input_ids, attention_mask=masked)
    expected = F.cross_entropy(
        logits[:, :2].reshape(-1, logits.size(-1)), targets[:, :2].reshape(-1)
    )

    assert loss_masked.item() == pytest.approx(expected.item(), abs=1e-6)
    assert loss_full.item() != pytest.approx(loss_masked.item())


def test_numerical_stability_on_extreme_inputs() -> None:
    torch.manual_seed(4)
    parameter = nn.Parameter(torch.full((8,), 1.0e6))
    opt = AdamWR(
        [parameter],
        alpha=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        lambda_norm=0.1,
        T_0=1.0,
        T_mult=1.0,
        b=1,
        B=1,
    )

    for _ in range(3):
        opt.zero_grad()
        loss = (parameter.square().sum()) * 1.0e-6
        loss.backward()
        opt.step()

    assert torch.isfinite(parameter).all()
    state = opt.state[parameter]
    assert torch.isfinite(state["m_t"]).all()
    assert torch.isfinite(state["v_t"]).all()


def test_adamwr_matches_reference_formulation_for_one_step() -> None:
    theta_prev = torch.tensor([1.0, -2.0], dtype=torch.float32)
    g_t = torch.tensor([0.5, -0.25], dtype=torch.float32)
    parameter = nn.Parameter(theta_prev.clone())

    opt = AdamWR(
        [parameter],
        alpha=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        lambda_norm=0.1,
        eta_min=0.0,
        eta_max=1.0,
        T_0=2.0,
        T_mult=1.0,
        b=2,
        B=8,
    )
    parameter.grad = g_t.clone()
    opt.step()

    eta_t = 1.0
    lambda_t = normalized_weight_decay(lambda_norm=0.1, b=2, B=8, T=2.0)
    theta_ref, m_ref, v_ref = reference_adamwr_step(
        theta_prev,
        g_t,
        alpha=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        eta_t=eta_t,
        lambda_t=lambda_t,
        t=1,
        m_prev=torch.zeros_like(theta_prev),
        v_prev=torch.zeros_like(theta_prev),
    )

    assert torch.allclose(parameter.detach(), theta_ref, atol=1e-5)
    assert torch.allclose(opt.state[parameter]["m_t"], m_ref, atol=1e-5)
    assert torch.allclose(opt.state[parameter]["v_t"], v_ref, atol=1e-5)


def test_weight_decay_shrinks_parameters_with_zero_gradient() -> None:
    parameter = nn.Parameter(torch.full((4,), 2.0))
    opt = AdamWR(
        [parameter],
        alpha=1e-3,
        lambda_norm=0.2,
        T_0=1.0,
        T_mult=1.0,
        b=1,
        B=1,
    )

    before = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    opt.step()

    assert parameter.abs().mean() < before.abs().mean()


def test_warm_restart_updates_T_cur_and_T_i() -> None:
    parameter = nn.Parameter(torch.ones(2))
    opt = AdamWR(
        [parameter],
        alpha=1e-3,
        lambda_norm=0.0,
        T_0=1.0,
        T_mult=2.0,
        b=2,
        B=4,
    )

    parameter.grad = torch.ones_like(parameter)
    opt.step()
    state_after_first = opt.get_schedule_state()
    assert state_after_first["T_cur"] == pytest.approx(0.5)
    assert state_after_first["T_i"] == pytest.approx(1.0)

    parameter.grad = torch.ones_like(parameter)
    opt.step()
    state_after_restart = opt.get_schedule_state()
    assert state_after_restart["restart_index"] == 1
    assert state_after_restart["T_cur"] == pytest.approx(0.0)
    assert state_after_restart["T_i"] == pytest.approx(2.0)


def test_normalized_lambda_uses_current_restart_length() -> None:
    parameter = nn.Parameter(torch.ones(2))
    opt = AdamWR(
        [parameter],
        alpha=1e-3,
        lambda_norm=0.1,
        T_0=1.0,
        T_mult=2.0,
        b=2,
        B=4,
    )

    parameter.grad = torch.zeros_like(parameter)
    opt.step()
    lambda_first = opt.get_schedule_state()["lambda"]

    parameter.grad = torch.zeros_like(parameter)
    opt.step()
    parameter.grad = torch.zeros_like(parameter)
    opt.step()
    lambda_second = opt.get_schedule_state()["lambda"]

    assert lambda_second == pytest.approx(normalized_weight_decay(lambda_norm=0.1, b=2, B=4, T=2.0))
    assert lambda_second < lambda_first


def test_closure_returns_loss_value() -> None:
    torch.manual_seed(5)
    parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = AdamWR([parameter], alpha=1e-3, lambda_norm=0.0, T_0=1.0, T_mult=1.0, b=1, B=1)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert torch.isfinite(loss)


def test_invalid_normalized_weight_decay_arguments_raise() -> None:
    with pytest.raises(ValueError):
        normalized_weight_decay(lambda_norm=0.1, b=0, B=8, T=1.0)
    with pytest.raises(ValueError):
        normalized_weight_decay(lambda_norm=0.1, b=2, B=0, T=1.0)
    with pytest.raises(ValueError):
        normalized_weight_decay(lambda_norm=0.1, b=2, B=8, T=0.0)
