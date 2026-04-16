"""
Tests for src/training/mup.py

All tests use a tiny 2-layer transformer-like model to stay fast and
dependency-free.  Only pure PyTorch is required.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.mup import (
    MuPAdamW,
    MuPConfig,
    apply_mup_init,
    coord_check,
    get_mup_param_groups,
)


# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------


class TinyTransformer(nn.Module):
    """Minimal 2-layer transformer-like model for testing.

    Architecture:
        embed  -> layer0 (hidden) -> layer1 (hidden) -> lm_head (output)
    """

    def __init__(self, d_model: int = 64, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layer0 = nn.Linear(d_model, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = torch.relu(self.layer0(h))
        h = torch.relu(self.layer1(h))
        h = self.norm(h)
        return self.lm_head(h)


def _make_model(d_model: int = 64, vocab_size: int = 32) -> TinyTransformer:
    return TinyTransformer(d_model=d_model, vocab_size=vocab_size)


def _make_config(
    d_model: int = 128,
    d_model_base: int = 64,
    base_lr: float = 1e-3,
) -> MuPConfig:
    return MuPConfig(d_model=d_model, d_model_base=d_model_base, base_lr=base_lr)


def _dummy_input(batch: int = 2, seq_len: int = 8, vocab_size: int = 32) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch, seq_len))


# ===========================================================================
# Test 1 -- apply_mup_init runs without error
# ===========================================================================


def test_apply_mup_init_runs_without_error() -> None:
    model = _make_model()
    config = _make_config()
    apply_mup_init(model, config)  # must not raise


# ===========================================================================
# Test 2 -- After init, hidden weight stds scale with 1/fan_in
# ===========================================================================


def test_hidden_weight_std_scales_with_fan_in() -> None:
    """The std of a hidden nn.Linear weight should be ~1/fan_in after muP init."""
    torch.manual_seed(0)
    model = _make_model(d_model=64)
    config = _make_config(d_model=64, d_model_base=64)
    apply_mup_init(model, config)

    # layer0: Linear(64, 64) -> fan_in = 64 -> expected std = 1/64
    fan_in = model.layer0.weight.shape[1]  # 64
    expected_std = 1.0 / fan_in
    actual_std = model.layer0.weight.std().item()
    # With 64*64 = 4096 parameters, sampling noise is ~expected_std / sqrt(4096/2)
    # Allow 40% relative tolerance.
    assert abs(actual_std - expected_std) / expected_std < 0.40, (
        f"layer0 weight std {actual_std:.6f} far from expected {expected_std:.6f}"
    )


# ===========================================================================
# Test 3 -- get_mup_param_groups returns multiple groups
# ===========================================================================


def test_get_mup_param_groups_returns_multiple_groups() -> None:
    model = _make_model()
    config = _make_config()
    groups = get_mup_param_groups(model, config)
    assert len(groups) >= 2, f"Expected at least 2 param groups, got {len(groups)}"


# ===========================================================================
# Test 4 -- Hidden layers have lower LR than embedding when width > base
# ===========================================================================


def test_hidden_lr_lower_than_embed_lr_when_wider() -> None:
    model = _make_model(d_model=64)
    # width_multiplier = 256 / 64 = 4  ->  hidden_lr = base_lr / 4
    config = _make_config(d_model=256, d_model_base=64, base_lr=1e-3)
    groups = get_mup_param_groups(model, config)

    embed_lr = next(g["lr"] for g in groups if g.get("group_name") == "embed")
    hidden_lr = next(g["lr"] for g in groups if g.get("group_name") == "hidden")

    assert hidden_lr < embed_lr, (
        f"Hidden LR {hidden_lr} should be < embed LR {embed_lr} when width > base"
    )


# ===========================================================================
# Test 5 -- MuPAdamW step runs without error
# ===========================================================================


def test_mup_adamw_step_runs_without_error() -> None:
    model = _make_model()
    config = _make_config()
    apply_mup_init(model, config)

    param_groups = get_mup_param_groups(model, config)
    optimizer = MuPAdamW(param_groups, lr=config.base_lr)

    x = _dummy_input()
    logits = model(x)
    loss = logits.mean()
    loss.backward()
    optimizer.step()  # must not raise
    optimizer.zero_grad()


# ===========================================================================
# Test 6 -- Coord check: pre-activation std stable across 3 widths (ratio < 3x)
# ===========================================================================


class EmbedOnlyModel(nn.Module):
    """Embedding-only model for coord check.

    Under muP, the embedding init std is 1/sqrt(num_embeddings), which is
    independent of d_model, so outputs should be O(1) across all widths.
    """

    def __init__(self, d_model: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, d_model)
        return self.embed(x)


def test_coord_check_stable_across_widths() -> None:
    """muP init keeps embedding output std O(1) across widths (ratio < 3x).

    The muP property: embedding outputs are initialised with std=1/sqrt(vocab_size),
    which is independent of d_model.  The coord check verifies that mean |output|
    stays within a 3x band as d_model grows.
    """
    widths = [64, 128, 256]
    d_model_base = 64
    vocab_size = 32

    def model_fn(d_model: int) -> nn.Module:
        return EmbedOnlyModel(d_model=d_model, vocab_size=vocab_size)

    input_ids = _dummy_input(batch=4, seq_len=16, vocab_size=vocab_size)
    results = coord_check(
        model_fn=model_fn,
        widths=widths,
        d_model_base=d_model_base,
        base_lr=1e-3,
        input_ids=input_ids,
        n_seeds=10,
    )

    values = list(results.values())
    ratio = max(values) / (min(values) + 1e-12)
    assert ratio < 3.0, (
        f"Coord check failed: max/min ratio = {ratio:.2f} >= 3. "
        f"Results: {results}"
    )


# ===========================================================================
# Test 7 -- Gradient flows through model after muP init
# ===========================================================================


def test_gradient_flows_after_mup_init() -> None:
    model = _make_model()
    config = _make_config()
    apply_mup_init(model, config)

    x = _dummy_input()
    logits = model(x)
    loss = logits.mean()
    loss.backward()

    # At least one parameter must have a non-None, non-zero gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad, "No gradients flowed through the model after muP init."


# ===========================================================================
# Test 8 -- Width multiplier = 1 -> hidden LR equals base_lr
# ===========================================================================


def test_width_multiplier_one_gives_base_lr() -> None:
    model = _make_model(d_model=64)
    # d_model == d_model_base -> width_multiplier = 1 -> hidden_lr = base_lr
    config = _make_config(d_model=64, d_model_base=64, base_lr=2e-4)
    groups = get_mup_param_groups(model, config)

    hidden_group = next((g for g in groups if g.get("group_name") == "hidden"), None)
    assert hidden_group is not None, "Expected a 'hidden' param group."
    assert abs(hidden_group["lr"] - config.base_lr) < 1e-12, (
        f"Hidden LR {hidden_group['lr']} != base_lr {config.base_lr} when width_mult=1"
    )


# ===========================================================================
# Test 9 -- width > base -> hidden LR < base_lr
# ===========================================================================


def test_hidden_lr_less_than_base_lr_when_wider() -> None:
    model = _make_model(d_model=64)
    config = _make_config(d_model=512, d_model_base=64, base_lr=1e-3)
    groups = get_mup_param_groups(model, config)

    hidden_group = next((g for g in groups if g.get("group_name") == "hidden"), None)
    assert hidden_group is not None, "Expected a 'hidden' param group."
    assert hidden_group["lr"] < config.base_lr, (
        f"Hidden LR {hidden_group['lr']} should be < base_lr {config.base_lr}"
    )
    expected = config.base_lr / config.width_multiplier
    assert abs(hidden_group["lr"] - expected) < 1e-12


# ===========================================================================
# Test 10 -- Init is deterministic with same seed
# ===========================================================================


def test_init_is_deterministic_with_same_seed() -> None:
    config = _make_config()

    torch.manual_seed(42)
    model_a = _make_model()
    apply_mup_init(model_a, config)

    torch.manual_seed(42)
    model_b = _make_model()
    apply_mup_init(model_b, config)

    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.equal(param_a, param_b), (
            f"Parameter '{name_a}' differs between identical-seed inits."
        )


# ===========================================================================
# Bonus Test 11 -- MuPConfig validates inputs
# ===========================================================================


def test_mup_config_validates_inputs() -> None:
    with pytest.raises(ValueError):
        MuPConfig(d_model=0, d_model_base=64)
    with pytest.raises(ValueError):
        MuPConfig(d_model=64, d_model_base=0)
    with pytest.raises(ValueError):
        MuPConfig(d_model=64, d_model_base=64, base_lr=-1.0)


# ===========================================================================
# Bonus Test 12 -- all param groups cover all trainable parameters
# ===========================================================================


def test_all_params_covered_by_groups() -> None:
    model = _make_model()
    config = _make_config()
    groups = get_mup_param_groups(model, config)

    group_param_ids = set()
    for group in groups:
        for p in group["params"]:
            group_param_ids.add(id(p))

    model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert model_param_ids == group_param_ids, (
        "Some trainable parameters are missing from param groups."
    )


# ===========================================================================
# Bonus Test 13 -- MuPAdamW respects per-group LR (groups keep different LRs)
# ===========================================================================


def test_mup_adamw_respects_per_group_lr() -> None:
    model = _make_model(d_model=64)
    config = _make_config(d_model=256, d_model_base=64, base_lr=1e-3)
    param_groups = get_mup_param_groups(model, config)

    optimizer = MuPAdamW(param_groups, lr=config.base_lr)

    # Before stepping, check that different groups have different LRs
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert len(set(lrs)) > 1, "All param groups have the same LR; muP scaling not applied."
