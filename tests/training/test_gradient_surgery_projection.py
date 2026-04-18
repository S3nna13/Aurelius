"""Tests for src/training/gradient_surgery_projection.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.training.gradient_surgery_projection import (
    GradientSurgeryProjectionConfig,
    aggregate_projected_gradients,
    gradient_surgery_projection,
    pcgrad_project,
    pcgrad_project_reference,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, TINY_CFG.vocab_size, (batch, seq_len), generator=generator)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.float32)
    return input_ids, attention_mask


def _make_task_losses(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    scale: float = 1.0,
) -> list[torch.Tensor]:
    logits = model(input_ids, attention_mask=attention_mask)
    mask = attention_mask.unsqueeze(-1).to(dtype=logits.dtype)
    masked_logits = logits * mask
    shifted_targets = input_ids.roll(shifts=-1, dims=1)
    shifted_targets[:, -1] = 0
    L_0 = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_targets.reshape(-1),
    )
    L_1 = scale * masked_logits.square().mean()
    return [L_0, L_1]


def _trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def test_shapes_and_dtypes_match_tiny_config():
    model = _make_model(seed=1)
    input_ids, attention_mask = _make_batch(seed=11)
    result = gradient_surgery_projection(
        _make_task_losses(model, input_ids, attention_mask),
        model.parameters(),
    )

    total_params = sum(parameter.numel() for parameter in _trainable_parameters(model))
    assert result.loss.shape == ()
    assert result.g.shape == (total_params,)
    assert result.g_i.shape == (2, total_params)
    assert result.g_i_pc.shape == (2, total_params)
    assert result.pi.shape == (2, 2)
    assert result.loss.dtype == torch.float32
    assert result.g.dtype == torch.float32


def test_loss_backward_produces_finite_grads_on_all_trainable_params():
    model = _make_model(seed=2)
    input_ids, attention_mask = _make_batch(seed=12)
    result = gradient_surgery_projection(
        _make_task_losses(model, input_ids, attention_mask),
        model.parameters(),
    )
    result.loss.backward()

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_determinism_under_manual_seed():
    config = GradientSurgeryProjectionConfig(implementation="optimized")

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(123)
        model = _make_model(seed=123)
        input_ids, attention_mask = _make_batch(seed=123)
        result = gradient_surgery_projection(
            _make_task_losses(model, input_ids, attention_mask),
            model.parameters(),
            config=config,
        )
        return (
            result.g.detach().clone(),
            result.g_i_pc.detach().clone(),
            result.pi.detach().clone(),
            result.loss.detach().clone(),
        )

    first = run()
    second = run()
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.allclose(first[3], second[3])


def test_batch_one_seq_len_one_edge_case_is_finite():
    model = _make_model(seed=3)
    input_ids = torch.tensor([[7]])
    attention_mask = torch.ones(1, 1)
    result = gradient_surgery_projection(
        _make_task_losses(model, input_ids, attention_mask),
        model.parameters(),
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.g).all()
    assert torch.isfinite(result.g_i_pc).all()


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
    result = gradient_surgery_projection(
        _make_task_losses(model, input_ids, attention_mask),
        model.parameters(),
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.g).all()
    assert torch.isfinite(result.g_i_pc).all()


def test_extreme_inputs_do_not_create_nan_or_inf():
    model = _make_model(seed=5)
    input_ids = torch.full((2, 8), TINY_CFG.vocab_size - 1, dtype=torch.long)
    attention_mask = torch.ones(2, 8)
    result = gradient_surgery_projection(
        _make_task_losses(model, input_ids, attention_mask, scale=1e4),
        model.parameters(),
        config=GradientSurgeryProjectionConfig(stability_eps=1e-18),
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.g).all()
    assert torch.isfinite(result.g_i).all()
    assert torch.isfinite(result.g_i_pc).all()


def test_optimized_matches_reference_projection_atol_1e5():
    g_i = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [-1.5, 0.5, -2.0, 1.0],
            [0.2, 0.4, -0.6, 0.8],
        ],
        dtype=torch.float32,
    )
    pi = torch.tensor(
        [
            [2, 1, 0],
            [0, 2, 1],
            [1, 0, 2],
        ],
        dtype=torch.long,
    )

    projected = pcgrad_project(g_i, pi, stability_eps=1e-12)
    reference = pcgrad_project_reference(g_i, pi, stability_eps=1e-12)
    assert torch.allclose(projected, reference, atol=1e-5)


def test_conflicting_antiparallel_gradients_cancel_under_mean_reduction():
    g_i = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    pi = torch.tensor([[1, 0], [0, 1]])
    g_i_pc = pcgrad_project(g_i, pi)
    g = aggregate_projected_gradients(g_i_pc, reduction="mean")
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_non_conflicting_gradients_are_unchanged():
    g_i = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    pi = torch.tensor([[1, 0], [0, 1]])
    g_i_pc = pcgrad_project(g_i, pi)
    assert torch.allclose(g_i_pc, g_i)


def test_sum_reduction_matches_mean_times_task_count():
    model = _make_model(seed=6)
    input_ids, attention_mask = _make_batch(seed=16)
    task_losses = _make_task_losses(model, input_ids, attention_mask)

    torch.manual_seed(77)
    mean_result = gradient_surgery_projection(
        task_losses,
        model.parameters(),
        config=GradientSurgeryProjectionConfig(reduction="mean"),
    )

    model.zero_grad(set_to_none=True)
    task_losses = _make_task_losses(model, input_ids, attention_mask)
    torch.manual_seed(77)
    sum_result = gradient_surgery_projection(
        task_losses,
        model.parameters(),
        config=GradientSurgeryProjectionConfig(reduction="sum"),
    )

    assert torch.allclose(sum_result.g, mean_result.g * len(task_losses), atol=1e-5)


def test_reference_and_optimized_paths_match_on_model_losses():
    torch.manual_seed(8)
    model_a = _make_model(seed=8)
    input_ids, attention_mask = _make_batch(seed=18)
    losses_a = _make_task_losses(model_a, input_ids, attention_mask)
    torch.manual_seed(999)
    optimized = gradient_surgery_projection(
        losses_a,
        model_a.parameters(),
        config=GradientSurgeryProjectionConfig(implementation="optimized"),
    )

    torch.manual_seed(8)
    model_b = _make_model(seed=8)
    losses_b = _make_task_losses(model_b, input_ids, attention_mask)
    torch.manual_seed(999)
    reference = gradient_surgery_projection(
        losses_b,
        model_b.parameters(),
        config=GradientSurgeryProjectionConfig(implementation="reference"),
    )

    assert torch.equal(optimized.pi, reference.pi)
    assert torch.allclose(optimized.g_i, reference.g_i, atol=1e-5)
    assert torch.allclose(optimized.g_i_pc, reference.g_i_pc, atol=1e-5)
    assert torch.allclose(optimized.g, reference.g, atol=1e-5)


def test_invalid_reduction_raises_value_error():
    model = _make_model(seed=9)
    input_ids, attention_mask = _make_batch(seed=19)
    with pytest.raises(ValueError, match="reduction"):
        gradient_surgery_projection(
            _make_task_losses(model, input_ids, attention_mask),
            model.parameters(),
            config=GradientSurgeryProjectionConfig(reduction="median"),
        )


def test_non_scalar_task_loss_raises_value_error():
    model = _make_model(seed=10)
    input_ids, attention_mask = _make_batch(seed=20)
    logits = model(input_ids, attention_mask=attention_mask)
    with pytest.raises(ValueError, match="scalar"):
        gradient_surgery_projection([logits], model.parameters())
