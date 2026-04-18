"""Tests for src/training/sam_sharpness.py."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.training.sam_sharpness import (
    SAMSharpnessConfig,
    sam_sharpness,
    sam_sharpness_loss,
)


TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)


class MockModel(nn.Module):
    def __init__(self, cfg: AureliusConfig) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ff_up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.ff_down = nn.Linear(cfg.d_ff, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_emb(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.in_proj(x)
        x = self.norm(x + self.ff_down(F.gelu(self.ff_up(x))))
        return self.lm_head(x)


def _make_model(seed: int = 0) -> MockModel:
    torch.manual_seed(seed)
    return MockModel(TINY_CFG)


def _make_batch(
    batch: int = 2,
    seq_len: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, TINY_CFG.vocab_size, (batch, seq_len), generator=generator)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.float32)
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = -100
    return input_ids, attention_mask, labels


def _ce_loss_fn(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    def loss_fn(model: nn.Module) -> torch.Tensor:
        logits = model(input_ids, attention_mask=attention_mask)
        masked_labels = labels
        if attention_mask is not None:
            masked_labels = masked_labels.masked_fill(~attention_mask.bool(), -100)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            masked_labels.reshape(-1),
            ignore_index=-100,
        )

    return loss_fn


def _quadratic_loss_fn(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scale: float = 1.0,
):
    def loss_fn(model: nn.Module) -> torch.Tensor:
        logits = model(input_ids, attention_mask=attention_mask)
        return (scale * logits).square().mean()

    return loss_fn


def _clone_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_shapes_and_dtypes_match_tiny_config_loss():
    model = _make_model(seed=1)
    input_ids, attention_mask, labels = _make_batch(seed=11)
    result = sam_sharpness(model, _ce_loss_fn(input_ids, labels, attention_mask))

    assert result.L_w.shape == ()
    assert result.L_w_plus_epsilon_w.shape == ()
    assert result.sharpness.shape == ()
    assert result.grad_norm_w.shape == ()
    assert result.L_w.dtype == torch.float32
    assert result.L_w_plus_epsilon_w.dtype == torch.float32


def test_gradient_flow_backward_gives_finite_grads_on_all_trainable_params():
    model = _make_model(seed=2)
    input_ids, attention_mask, labels = _make_batch(seed=12)
    loss = sam_sharpness_loss(model, _ce_loss_fn(input_ids, labels, attention_mask))
    loss.backward()

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_determinism_under_manual_seed():
    config = SAMSharpnessConfig(rho=0.05)

    def run() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        torch.manual_seed(123)
        model = _make_model(seed=123)
        input_ids, attention_mask, labels = _make_batch(seed=123)
        result = sam_sharpness(model, _ce_loss_fn(input_ids, labels, attention_mask), config)
        return (
            result.L_w.detach().clone(),
            result.L_w_plus_epsilon_w.detach().clone(),
            [tensor.detach().clone() for tensor in result.epsilon_w.values()],
        )

    first = run()
    second = run()
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])
    for lhs, rhs in zip(first[2], second[2]):
        assert torch.allclose(lhs, rhs)


def test_batch_one_seq_len_one_edge_case_is_finite():
    model = _make_model(seed=3)
    input_ids = torch.tensor([[7]])
    attention_mask = torch.ones(1, 1)
    result = sam_sharpness(model, _quadratic_loss_fn(input_ids, attention_mask))

    assert torch.isfinite(result.L_w)
    assert torch.isfinite(result.L_w_plus_epsilon_w)
    assert torch.isfinite(result.grad_norm_w)


def test_masked_and_padded_inputs_remain_stable():
    model = _make_model(seed=4)
    input_ids = torch.tensor(
        [
            [5, 9, 0, 0],
            [11, 4, 7, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = -100
    result = sam_sharpness(model, _ce_loss_fn(input_ids, labels, attention_mask))

    assert torch.isfinite(result.L_w)
    assert torch.isfinite(result.L_w_plus_epsilon_w)
    assert torch.isfinite(result.sharpness)


def test_extreme_inputs_do_not_create_nan_or_inf():
    model = _make_model(seed=5)
    input_ids = torch.full((2, 8), TINY_CFG.vocab_size - 1, dtype=torch.long)
    attention_mask = torch.ones(2, 8)
    result = sam_sharpness(
        model,
        _quadratic_loss_fn(input_ids, attention_mask, scale=1e3),
        SAMSharpnessConfig(rho=0.1, stability_eps=1e-12),
    )

    assert torch.isfinite(result.L_w)
    assert torch.isfinite(result.L_w_plus_epsilon_w)
    assert torch.isfinite(result.grad_norm_w)
    for epsilon_w in result.epsilon_w.values():
        assert torch.isfinite(epsilon_w).all()


def test_reference_formulation_matches_direct_perturbed_loss():
    model = _make_model(seed=6)
    reference_model = copy.deepcopy(model)
    input_ids, attention_mask, labels = _make_batch(seed=16)
    loss_fn = _ce_loss_fn(input_ids, labels, attention_mask)
    config = SAMSharpnessConfig(rho=0.05)

    result = sam_sharpness(model, loss_fn, config)

    reference_loss = loss_fn(reference_model)
    grads = torch.autograd.grad(
        reference_loss,
        [parameter for parameter in reference_model.parameters() if parameter.requires_grad],
    )
    grad_norm = torch.sqrt(sum(grad.detach().pow(2).sum() for grad in grads))
    scale = config.rho / (grad_norm + config.stability_eps)
    with torch.no_grad():
        for parameter, grad in zip(
            [parameter for parameter in reference_model.parameters() if parameter.requires_grad],
            grads,
        ):
            parameter.add_(scale * grad.detach())
    reference_perturbed_loss = loss_fn(reference_model)

    assert torch.allclose(result.L_w, reference_loss, atol=1e-5)
    assert torch.allclose(result.L_w_plus_epsilon_w, reference_perturbed_loss, atol=1e-5)


def test_epsilon_w_matches_paper_formula_parameterwise():
    model = _make_model(seed=7)
    input_ids, attention_mask, labels = _make_batch(seed=17)
    loss_fn = _ce_loss_fn(input_ids, labels, attention_mask)
    config = SAMSharpnessConfig(rho=0.05)

    loss = loss_fn(model)
    params = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    grads = torch.autograd.grad(loss, [parameter for _, parameter in params])
    grad_norm = torch.sqrt(sum(grad.detach().pow(2).sum() for grad in grads))
    expected_scale = config.rho / (grad_norm + config.stability_eps)

    result = sam_sharpness(model, loss_fn, config)

    for (name, _), grad in zip(params, grads):
        assert torch.allclose(result.epsilon_w[name], expected_scale * grad.detach(), atol=1e-5)


def test_epsilon_w_global_norm_matches_rho_with_nonzero_gradient():
    model = _make_model(seed=8)
    input_ids, attention_mask, labels = _make_batch(seed=18)
    result = sam_sharpness(model, _ce_loss_fn(input_ids, labels, attention_mask))

    epsilon_norm = torch.sqrt(
        sum(epsilon_w.detach().pow(2).sum() for epsilon_w in result.epsilon_w.values())
    )
    assert result.grad_norm_w.item() > 0.0
    assert torch.allclose(epsilon_norm, torch.tensor(0.05), atol=1e-5)


def test_zero_gradient_loss_gives_zero_epsilon_and_zero_sharpness():
    model = _make_model(seed=9)

    def zero_loss_fn(module: nn.Module) -> torch.Tensor:
        return sum((parameter * 0.0).sum() for parameter in module.parameters())

    result = sam_sharpness(model, zero_loss_fn)

    assert result.grad_norm_w.item() == 0.0
    assert result.L_w.item() == 0.0
    assert result.L_w_plus_epsilon_w.item() == 0.0
    assert result.sharpness.item() == 0.0
    for epsilon_w in result.epsilon_w.values():
        assert torch.count_nonzero(epsilon_w) == 0


def test_sam_sharpness_does_not_mutate_model_parameters():
    model = _make_model(seed=10)
    before = _clone_parameters(model)
    input_ids, attention_mask, labels = _make_batch(seed=20)

    _ = sam_sharpness(model, _ce_loss_fn(input_ids, labels, attention_mask))

    after = _clone_parameters(model)
    for lhs, rhs in zip(before, after):
        assert torch.allclose(lhs, rhs)


def test_sharpness_equals_loss_difference():
    model = _make_model(seed=11)
    input_ids, attention_mask, labels = _make_batch(seed=21)
    result = sam_sharpness(model, _ce_loss_fn(input_ids, labels, attention_mask))

    assert torch.allclose(
        result.sharpness, result.L_w_plus_epsilon_w.detach() - result.L_w.detach()
    )


def test_sam_sharpness_loss_matches_result_surrogate_objective():
    model = _make_model(seed=12)
    input_ids, attention_mask, labels = _make_batch(seed=22)
    loss_fn = _ce_loss_fn(input_ids, labels, attention_mask)

    result = sam_sharpness(model, loss_fn)
    loss = sam_sharpness_loss(model, loss_fn)

    assert torch.allclose(loss, result.L_w_plus_epsilon_w, atol=1e-6)


def test_invalid_scalar_loss_raises_value_error():
    model = _make_model(seed=13)
    input_ids, attention_mask, _ = _make_batch(seed=23)

    def vector_loss_fn(module: nn.Module) -> torch.Tensor:
        logits = module(input_ids, attention_mask=attention_mask)
        return logits.mean(dim=-1)

    with pytest.raises(ValueError, match="scalar tensor"):
        sam_sharpness(model, vector_loss_fn)
