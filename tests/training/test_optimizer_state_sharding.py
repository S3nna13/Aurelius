"""Tests for ZeRO Stage-1 style optimizer state sharding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.training.optimizer_state_sharding import (
    OptimizerStateSharding,
    OptimizerStateShardingConfig,
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
    batch: int = 2, seq_len: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(batch * 100 + seq_len)
    input_ids = torch.randint(1, TINY_CFG.vocab_size, (batch, seq_len), generator=generator)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.float32)
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = -100
    return input_ids, attention_mask, labels


def _loss_for_batch(
    model: MockModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = model(input_ids, attention_mask=attention_mask)
    if attention_mask is not None:
        valid = attention_mask.bool()
        labels = labels.masked_fill(~valid, -100)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _reference_adamw_step(
    theta_t: torch.Tensor,
    g_t: torch.Tensor,
    m_t_prev: torch.Tensor,
    v_t_prev: torch.Tensor,
    t: int,
    config: OptimizerStateShardingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_t = config.beta_1 * m_t_prev + (1.0 - config.beta_1) * g_t
    v_t = config.beta_2 * v_t_prev + (1.0 - config.beta_2) * g_t.square()
    if config.bias_correction:
        m_hat_t = m_t / (1.0 - config.beta_1**t)
        v_hat_t = v_t / (1.0 - config.beta_2**t)
    else:
        m_hat_t = m_t
        v_hat_t = v_t
    update_t = m_hat_t / (v_hat_t.sqrt() + config.epsilon)
    if config.lambda_ != 0.0:
        update_t = update_t + config.lambda_ * theta_t
    theta_t_next = theta_t - config.alpha * update_t
    return theta_t_next, m_t, v_t


def test_partition_sizes_cover_flat_parameter_vector():
    model = _make_model()
    optimizer = OptimizerStateSharding(model.parameters(), OptimizerStateShardingConfig(N_d=3))
    sizes = [partition.numel for partition in optimizer.P_r]
    assert sum(sizes) == optimizer.numel()
    assert max(sizes) - min(sizes) <= 1


def test_state_shapes_and_dtypes_match_partitions():
    model = _make_model()
    config = OptimizerStateShardingConfig(N_d=4, state_dtype=torch.float64)
    optimizer = OptimizerStateSharding(model.parameters(), config)
    for rank, partition in enumerate(optimizer.P_r):
        assert optimizer.m_t[rank].shape == (partition.numel,)
        assert optimizer.v_t[rank].shape == (partition.numel,)
        assert optimizer.m_t[rank].dtype == torch.float64
        assert optimizer.v_t[rank].dtype == torch.float64


def test_flatten_partition_and_gather_are_lossless():
    model = _make_model()
    optimizer = OptimizerStateSharding(model.parameters(), OptimizerStateShardingConfig(N_d=5))
    theta_t = optimizer.flatten_parameters()
    theta_r = optimizer.partition_vector(theta_t)
    theta_t_restored = optimizer.gather_vector(theta_r)
    assert torch.allclose(theta_t, theta_t_restored)


def test_step_populates_sharded_optimizer_states():
    model = _make_model()
    optimizer = OptimizerStateSharding(model.parameters(), OptimizerStateShardingConfig(N_d=2))
    input_ids, attention_mask, labels = _make_batch()
    loss = _loss_for_batch(model, input_ids, labels, attention_mask)
    loss.backward()
    optimizer.step()
    assert optimizer.t == 1
    assert sum(shard.count_nonzero().item() for shard in optimizer.m_t) > 0
    assert sum(shard.count_nonzero().item() for shard in optimizer.v_t) > 0


def test_gradient_flow_gives_finite_grads_on_all_trainable_params():
    model = _make_model()
    input_ids, attention_mask, labels = _make_batch(batch=2, seq_len=8)
    loss = _loss_for_batch(model, input_ids, labels, attention_mask)
    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_determinism_under_manual_seed():
    config = OptimizerStateShardingConfig(N_d=3, alpha=2e-3, lambda_=1e-2)

    def run() -> list[torch.Tensor]:
        model = _make_model(seed=123)
        optimizer = OptimizerStateSharding(model.parameters(), config)
        input_ids, attention_mask, labels = _make_batch(batch=2, seq_len=8)
        loss = _loss_for_batch(model, input_ids, labels, attention_mask)
        loss.backward()
        optimizer.step()
        return [param.detach().clone() for param in model.parameters()]

    first = run()
    second = run()
    for lhs, rhs in zip(first, second):
        assert torch.allclose(lhs, rhs)


def test_batch_one_seq_len_one_edge_case_is_finite():
    model = _make_model()
    optimizer = OptimizerStateSharding(model.parameters(), OptimizerStateShardingConfig(N_d=2))
    input_ids = torch.tensor([[7]])
    attention_mask = torch.ones(1, 1)
    torch.tensor([[-100]])
    logits = model(input_ids, attention_mask=attention_mask)
    loss = logits.square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_masked_and_padded_inputs_remain_stable():
    model = _make_model()
    optimizer = OptimizerStateSharding(model.parameters(), OptimizerStateShardingConfig(N_d=2))
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
    loss = _loss_for_batch(model, input_ids, labels, attention_mask)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_extreme_inputs_do_not_create_nan_or_inf():
    model = _make_model()
    optimizer = OptimizerStateSharding(
        model.parameters(),
        OptimizerStateShardingConfig(N_d=4, alpha=1e-3, epsilon=1e-6),
    )
    input_ids = torch.full((2, 8), TINY_CFG.vocab_size - 1, dtype=torch.long)
    attention_mask = torch.ones(2, 8)
    logits = model(input_ids, attention_mask=attention_mask)
    loss = (logits * 1_000.0).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert torch.isfinite(param).all()
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_single_step_matches_reference_formulation():
    model = _make_model(seed=17)
    config = OptimizerStateShardingConfig(N_d=3, alpha=1e-3, lambda_=0.05)
    optimizer = OptimizerStateSharding(model.parameters(), config)
    input_ids, attention_mask, labels = _make_batch(batch=2, seq_len=5)
    loss = _loss_for_batch(model, input_ids, labels, attention_mask)
    loss.backward()
    theta_t = optimizer.flatten_parameters().clone()
    g_t = optimizer.flatten_gradients().clone()
    theta_ref, m_ref, v_ref = _reference_adamw_step(
        theta_t,
        g_t,
        torch.zeros_like(theta_t),
        torch.zeros_like(theta_t),
        t=1,
        config=config,
    )
    theta_zero = optimizer.step()
    assert torch.allclose(theta_zero, theta_ref, atol=1e-5)
    assert torch.allclose(optimizer.gather_vector(optimizer.m_t), m_ref, atol=1e-5)
    assert torch.allclose(optimizer.gather_vector(optimizer.v_t), v_ref, atol=1e-5)


def test_two_steps_match_reference_formulation():
    model = _make_model(seed=23)
    config = OptimizerStateShardingConfig(N_d=2, alpha=2e-3, lambda_=1e-2)
    optimizer = OptimizerStateSharding(model.parameters(), config)
    m_ref = torch.zeros(optimizer.numel(), dtype=config.state_dtype)
    v_ref = torch.zeros(optimizer.numel(), dtype=config.state_dtype)

    for t in (1, 2):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = _make_batch(batch=2, seq_len=4 + t)
        loss = _loss_for_batch(model, input_ids, labels, attention_mask)
        loss.backward()
        theta_t = optimizer.flatten_parameters().clone()
        g_t = optimizer.flatten_gradients().clone()
        theta_ref, m_ref, v_ref = _reference_adamw_step(theta_t, g_t, m_ref, v_ref, t, config)
        theta_zero = optimizer.step()
        assert torch.allclose(theta_zero, theta_ref, atol=1e-5)


def test_partitioning_with_more_ranks_than_elements_creates_empty_shards():
    param = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = OptimizerStateSharding([param], OptimizerStateShardingConfig(N_d=4))
    sizes = [partition.numel for partition in optimizer.P_r]
    assert sizes == [1, 1, 0, 0]
    gathered = optimizer.gather_vector(optimizer.partition_vector(optimizer.flatten_parameters()))
    assert torch.allclose(gathered, optimizer.flatten_parameters())


def test_zero_grad_clears_existing_gradients():
    model = _make_model()
    optimizer = OptimizerStateSharding(model.parameters(), OptimizerStateShardingConfig(N_d=2))
    input_ids, attention_mask, labels = _make_batch()
    loss = _loss_for_batch(model, input_ids, labels, attention_mask)
    loss.backward()
    optimizer.zero_grad()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.count_nonzero(param.grad) == 0
